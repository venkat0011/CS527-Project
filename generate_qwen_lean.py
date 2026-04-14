import gc
import os
import re
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RunConfig:
    csv_path: str                 = '/workspace/CS527-Project/c_sample.csv'
    base_output_dir: str          = '/workspace/CS527-Project/'
    properties_base_path: str     = '/workspace/sv-benchmarks/c/properties/'
    seeds: list                   = field(default_factory=lambda: [42, 123, 999])
    temperatures: list            = field(default_factory=lambda: [0.2, 0.5, 0.8])
    top_p: float                  = 0.95
    max_new_tokens: int           = 3000
    max_model_len: int            = 32768
    gpu_memory_utilization: float = 0.92

MODEL_REGISTRY = {
    "qwen2.5-32b": {
        "model_id":             "Qwen/Qwen2.5-32B-Instruct",
        "prompt_format":        "chatml",
        "dtype":                "bfloat16",
        "tensor_parallel_size": 1,
        "quantization":         None,
    },
}

cfg = RunConfig()

# ─── MODEL MANAGER ────────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self):
        self.llm: Optional[LLM] = None
        self.current_id: Optional[str] = None

    def load(
        self,
        model_id: str,
        quantization: Optional[str] = None,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
    ):
        if self.current_id == model_id:
            return
        self.unload()
        print(f"\n[ModelManager] Loading {model_id} (dtype={dtype}, tp={tensor_parallel_size})")
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        self.current_id = model_id
        print(f"[ModelManager] Ready. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def unload(self):
        if self.llm is None:
            return
        print(f"[ModelManager] Unloading {self.current_id}")
        destroy_model_parallel()
        del self.llm
        self.llm = None
        self.current_id = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[ModelManager] Freed. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

manager = ModelManager()

# ─── PROMPT ───────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert Lean 4 and Mathlib programmer."
    "You are an assistant to convert source code like C, Python, Java and its specification to its Lean 4 proof."
    "You may think through the problem freely, but your final answer "
    "MUST be placed inside the delimiters below and nowhere else:\n\n"
    "===BEGIN LEAN PROOF===\n"
    "<your complete Lean 4 source here>\n"
    "===END LEAN PROOF===\n\n"
    "The code inside the delimiters must start with `import Mathlib`. "
)

def _user_msg(c_code: str, prop_text: str) -> str:
    return (
        "Translate the C program below into a complete Lean 4 proof of the given property.\n\n"
        f"### C Code\n```c\n{c_code}\n```\n\n"
        f"### Property\n```\n{prop_text}\n```\n\n"
        "When you are done thinking, write your final Lean 4 proof between the delimiters:\n"
        "===BEGIN LEAN PROOF===\n"
        "<import Mathlib and full proof here>\n"
        "===END LEAN PROOF==="
    )

def build_prompt(c_code: str, prop_text: str, fmt: str = "chatml") -> str:
    user = _user_msg(c_code, prop_text)
    if fmt == "chatml":
        return (
            f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return f"[INST] {_SYSTEM}\n\n{user} [/INST]"

# ─── PROOF EXTRACTOR ──────────────────────────────────────────────────────────

_PROOF_RE = re.compile(
    r"===BEGIN LEAN PROOF===\s*\n(.*?)\n===END LEAN PROOF===",
    re.DOTALL
)
_LEAN_FENCE_RE = re.compile(
    r"```lean\s*\n(.*?)```",
    re.DOTALL
)

def extract_proof(generated_text: str) -> str:
    # 1. Primary: clean delimited block
    m = _PROOF_RE.search(generated_text)
    if m:
        code = m.group(1).strip()
        if not code.startswith("import Mathlib"):
            code = "import Mathlib\n" + code
        return code

    # 2. ```lean fence — split on it directly, no regex
    if "```lean" in generated_text:
        after_open = generated_text.split("```lean", 1)[1]        # everything after opening fence
        code = after_open.split("```", 1)[0].strip()               # stop at closing fence
        if not code.startswith("import Mathlib"):
            code = "import Mathlib\n" + code
        return code

    # 3. Plain ``` fence (model forgot the lang tag)
    if "```" in generated_text:
        after_open = generated_text.split("```", 1)[1]
        code = after_open.split("```", 1)[0].strip()
        if not code.startswith("import Mathlib"):
            code = "import Mathlib\n" + code
        print("[extract_proof] WARNING: no ```lean tag, grabbed from plain ``` fence.")
        return code

    # 4. No fence at all — grab from last `import Mathlib` onward
    if "import Mathlib" in generated_text:
        code = generated_text[generated_text.rfind("import Mathlib"):].strip()
        print("[extract_proof] WARNING: no fence found, grabbed from last import Mathlib.")
        return code

    # 5. Give-up stub — keeps downstream files valid
    print("[extract_proof] ERROR: could not locate any Lean code.")
    return "import Mathlib\n\n-- proof extraction failed\nexample : True := by sorry\n"

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(cfg.csv_path)
    raw: dict[int, tuple[str, str]] = {}
    valid: list[int] = []
    for idx, row in df.iterrows():
        try:
            with open(row['c_file_abs']) as f: c = f.read()
            with open(f"{cfg.properties_base_path}{row['property']}.prp") as f: p = f.read()
            raw[idx] = (c, p)
            valid.append(idx)
        except Exception as e:
            print(f"[Skip] {idx}: {e}")

    print(f"[Data] {len(valid)}/{len(df)} files ready.")

    for model_name, model_cfg in MODEL_REGISTRY.items():
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")

        manager.load(
            model_id=model_cfg["model_id"],
            quantization=model_cfg.get("quantization"),
            dtype=model_cfg.get("dtype", "float16"),
            tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
        )

        fmt = model_cfg.get("prompt_format", "chatml")
        prompts = {idx: build_prompt(c, p, fmt=fmt) for idx, (c, p) in raw.items()}

        for temp in cfg.temperatures:
            print(f"\n  temp={temp}")
            df_out = df.copy()

            for seed in cfg.seeds:
                print(f"    seed={seed}")

                sampling_params = SamplingParams(
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_new_tokens,
                    temperature=temp,
                    seed=seed,
                    n=1,
                )

                results = manager.llm.generate(
                    [prompts[i] for i in valid],
                    sampling_params,
                )

                # Each seed gets its own column
                df_out[f"lean_proof_s{seed}"] = [
                    r.outputs[0].text
                    for r in results
                ]

                del results
                gc.collect()

            # One CSV per temperature, all seeds as columns
            path = os.path.join(cfg.base_output_dir, f"{model_name}_t{temp}.csv")
            df_out.to_csv(path, index=False)
            print(f"  [Saved] {path}")

            del df_out

        manager.unload()

    print("\n[Done]")

if __name__ == "__main__":
    main()