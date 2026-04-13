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
    csv_path: str             = '/workspace/CS527-Project/c_sample.csv'
    base_output_dir: str      = '/workspace/CS527-Project/'
    properties_base_path: str = '/workspace/sv-benchmarks/c/properties/'
    seeds: list               = field(default_factory=lambda: [42, 123, 999])
    temperatures: list        = field(default_factory=lambda: [0.2, 0.5, 0.8])
    top_p: float              = 0.95
    max_new_tokens: int       = 3000
    max_model_len: int        = 32768
    gpu_memory_utilization: float = 0.92

MODEL_REGISTRY = {
    "deepseek-r1-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

cfg = RunConfig()

# ─── MODEL MANAGER ────────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self):
        self.llm: Optional[LLM] = None
        self.current_id: Optional[str] = None

    def load(self, model_id: str, quantization: Optional[str] = None):
        if self.current_id == model_id:
            return
        self.unload()
        print(f"\n[ModelManager] Loading {model_id}")
        self.llm = LLM(
            model=model_id,
            dtype="float16",
            quantization=quantization,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
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

def build_prompt(c_code: str, prop_text: str) -> str:
    system = (
        "You are an expert in formal verification, Lean 4, and Mathlib. "
        "Think through the problem, then output the complete Lean 4 proof. "
        "For any proof step you are unsure about, write `sorry` — "
        "this keeps the file compilable so the Lean checker can verify the rest."
    )
    user = (
        "Translate this C program into a Lean 4 proof of the given property.\n\n"
        f"C Code:\n```c\n{c_code}\n```\n\n"
        f"Property:\n{prop_text}\n\n"
        "Start your response with `import Mathlib`."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that DeepSeek-R1 prepends to its output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

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

    for model_name, model_id in MODEL_REGISTRY.items():
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        quant = "awq" if "awq" in model_id.lower() else None
        manager.load(model_id, quantization=quant)

        prompts = {idx: build_prompt(c, p) for idx, (c, p) in raw.items()}
        prompt_list = [prompts[i] for i in valid]

        for temp in cfg.temperatures:
            print(f"\n  temp={temp}")
            seed_outputs: dict[int, dict[int, str]] = {}

            for seed in cfg.seeds:
                print(f"    seed={seed}")

                sampling_params = SamplingParams(
                    temperature=temp,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_new_tokens,
                    n=1,
                    seed=seed,
                )

                results = manager.llm.generate(prompt_list, sampling_params)

                seed_outputs[seed] = {
                    idx: "import Mathlib\n" + strip_thinking(r.outputs[0].text)
                    for idx, r in zip(valid, results)
                }

                del results
                gc.collect()

            df_out = df.copy()
            for seed in cfg.seeds:
                df_out[f"sample_seed_{seed}"] = df_out.index.map(
                    lambda x, s=seed: seed_outputs[s].get(x, "")
                )

            path = os.path.join(cfg.base_output_dir, f"{model_name}_t{temp}.csv")
            df_out.to_csv(path, index=False)
            print(f"  [Saved] {path}")

            del seed_outputs, df_out
            gc.collect()

        manager.unload()

    print("\n[Done]")

if __name__ == "__main__":
    main()