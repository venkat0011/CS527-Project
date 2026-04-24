import asyncio
import gc
import os
import re
import time
import pandas as pd
from openai import AsyncOpenAI, RateLimitError, APIError
from dataclasses import dataclass, field

@dataclass
class RunConfig:
    csv_path: str             = '/workspace/CS527-Project/c_sample.csv'
    base_output_dir: str      = '/workspace/CS527-Project/'
    properties_base_path: str = '/workspace/sv-benchmarks/c/properties/'
    seeds: list               = field(default_factory=lambda: [42, 123, 999])
    temperatures: list        = field(default_factory=lambda: [0.2, 0.5, 0.8])
    top_p: float              = 0.95
    max_new_tokens: int       = 3000
    max_retries: int          = 5
    retry_delay: float        = 5.0
    max_concurrent: int       = 10    # ← tune this: higher = faster but more rate limit risk

MODEL_REGISTRY = {

    "gemini-2.5-flash-lite": {
        "model_id":      "google/gemini-2.5-flash-lite",
        "prompt_format": "openai",
    },
}

cfg = RunConfig()

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

# ─── PROMPT ───────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert Lean 4 and Mathlib programmer. "
    "You are an assistant to convert source code like C, Python, Java and its specification to its Lean 4 proof. "
    "You may think through the problem freely, but your final answer "
    "MUST be placed inside the delimiters below and nowhere else:\n\n"
    "===BEGIN LEAN PROOF===\n"
    "<your complete Lean 4 source here>\n"
    "===END LEAN PROOF===\n\n"
    "The code inside the delimiters must start with `import Mathlib`."
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

def build_messages(c_code: str, prop_text: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _user_msg(c_code, prop_text)},
    ]

# ─── PROOF EXTRACTOR ──────────────────────────────────────────────────────────

_PROOF_RE = re.compile(
    r"===BEGIN LEAN PROOF===\s*\n(.*?)\n===END LEAN PROOF===",
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

    # 2. ```lean fence
    if "```lean" in generated_text:
        after_open = generated_text.split("```lean", 1)[1]
        code = after_open.split("```", 1)[0].strip()
        if not code.startswith("import Mathlib"):
            code = "import Mathlib\n" + code
        print("[extract_proof] WARNING: delimiters missing, grabbed from ```lean fence.")
        return code

    # 3. Plain ``` fence
    if "```" in generated_text:
        after_open = generated_text.split("```", 1)[1]
        code = after_open.split("```", 1)[0].strip()
        if not code.startswith("import Mathlib"):
            code = "import Mathlib\n" + code
        print("[extract_proof] WARNING: grabbed from plain ``` fence.")
        return code

    # 4. Grab from last `import Mathlib` onward
    if "import Mathlib" in generated_text:
        code = generated_text[generated_text.rfind("import Mathlib"):].strip()
        print("[extract_proof] WARNING: no fence found, grabbed from last import Mathlib.")
        return code

    # 5. Give-up stub
    print("[extract_proof] ERROR: could not locate any Lean code.")
    return "import Mathlib\n\n-- proof extraction failed\nexample : True := by sorry\n"

# ─── ASYNC API CALL WITH RETRY ────────────────────────────────────────────────

async def call_api(
    semaphore: asyncio.Semaphore,
    idx: int,
    messages: list[dict],
    model_id: str,
    temperature: float,
    seed: int,
    total: int,
    counter: list[int],   # mutable counter for progress tracking
) -> tuple[int, str]:
    """Returns (idx, proof_text)."""
    async with semaphore:
        for attempt in range(1, cfg.max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=cfg.max_new_tokens,
                    top_p=cfg.top_p,
                    seed=seed,
                )
                raw_text = response.choices[0].message.content or ""
                counter[0] += 1
                status = "OK" if "===BEGIN" in raw_text else "FALLBACK"
                print(f"      [{counter[0]}/{total}] idx={idx} seed={seed} — {status}")
                return idx, raw_text

            except RateLimitError:
                wait = cfg.retry_delay * attempt
                print(f"[API] Rate limited idx={idx}. Waiting {wait}s (attempt {attempt}/{cfg.max_retries})")
                await asyncio.sleep(wait)
            except APIError as e:
                print(f"[API] Error idx={idx}: {e}. Waiting {cfg.retry_delay}s (attempt {attempt}/{cfg.max_retries})")
                await asyncio.sleep(cfg.retry_delay)

    print(f"[API] Max retries exceeded for idx={idx}. Returning sorry stub.")
    counter[0] += 1
    return idx, "import Mathlib\n\n-- API call failed after retries\nexample : True := by sorry\n"

# ─── ASYNC RUNNER ─────────────────────────────────────────────────────────────

async def run_seed(
    valid: list[int],
    raw: dict[int, tuple[str, str]],
    model_id: str,
    temperature: float,
    seed: int,
) -> dict[int, str]:
    """Fire all rows concurrently (capped by semaphore), return {idx: proof}."""
    semaphore = asyncio.Semaphore(cfg.max_concurrent)
    counter = [0]  # mutable so the coroutines can update it
    total = len(valid)

    tasks = [
        call_api(
            semaphore=semaphore,
            idx=idx,
            messages=build_messages(*raw[idx]),
            model_id=model_id,
            temperature=temperature,
            seed=seed,
            total=total,
            counter=counter,
        )
        for idx in valid
    ]

    results = await asyncio.gather(*tasks)
    return {idx: proof for idx, proof in results}

# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def async_main():
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
        model_id = model_cfg["model_id"]
        print(f"\n{'='*60}\n  {model_name}  ({model_id})\n{'='*60}")

        for temp in cfg.temperatures:
            print(f"\n  temp={temp}")
            df_out = df.copy()

            for seed in cfg.seeds:
                print(f"    seed={seed}  — firing {len(valid)} requests (max_concurrent={cfg.max_concurrent})")
                t0 = time.monotonic()

                proof_map = await run_seed(valid, raw, model_id, temp, seed)

                elapsed = time.monotonic() - t0
                print(f"    seed={seed} done in {elapsed:.1f}s")

                # Preserve original row order
                df_out[f"lean_proof_s{seed}"] = [proof_map[i] for i in valid]

            path = os.path.join(cfg.base_output_dir, f"{model_name}_t{temp}.csv")
            df_out.to_csv(path, index=False)
            print(f"  [Saved] {path}")
            del df_out

    print("\n[Done]")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()