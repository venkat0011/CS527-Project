import os
import re
import json
import pandas as pd
import shutil
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ─── APOLLO & PATHS ─────────────────────────────────────────────────────────────
APOLLO_PATH = '/workspace/APOLLO'
if APOLLO_PATH not in sys.path:
    sys.path.insert(0, APOLLO_PATH)
os.chdir(APOLLO_PATH)

from apollo import ApolloRepair
from prover.lean.verifier import verify_lean4_file

# --- CONFIGURATION ---
CSV_PATH      = '/workspace/CS527-Project/c_sample.csv'
PROP_BASE     = '/workspace/sv-benchmarks/c/properties/'
OUTPUT_DIR    = "/workspace/Project/results_timed_v10"
CONFIG_PATH   = '/workspace/APOLLO/configs/baseline_sampling_kimina_prover.py'
SEEDS         = [42, 123, 999]
MAX_ITER      = 5
TEMPERATURE   = 0.2
MAX_WORKERS   = 6 

os.makedirs(OUTPUT_DIR, exist_ok=True)
q_client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm-token")

# ─── PROMPT SEMANTICS (YOUR ORIGINAL CODE) ────────────────────────────────────

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

def _initial_user_msg(c_code: str, prop_text: str) -> str:
    return (
        "Translate the C program below into a complete Lean 4 proof of the given property.\n\n"
        f"### C Code\n```c\n{c_code}\n```\n\n"
        f"### Property\n```\n{prop_text}\n```\n\n"
        "When you are done thinking, write your final Lean 4 proof between the delimiters:\n"
        "===BEGIN LEAN PROOF===\n"
        "<import Mathlib and full proof here>\n"
        "===END LEAN PROOF==="
    )

def _feedback_user_msg(repaired_code: str, errors: dict) -> str:
    """Uses similar semantics to your original message for repair cycles."""
    return (
        "The Lean 4 proof below (partially repaired by an automated tool) still contains errors.\n\n"
        f"### Current Attemp with Errors\n```lean\n{repaired_code}\n```\n\n"
        f"### Lean 4 Diagnostics\n```json\n{json.dumps(errors, indent=2)}\n```\n\n"
        "Using this information and the original C code/property, provide a complete, fixed Lean 4 proof between the delimiters:\n"
        "===BEGIN LEAN PROOF===\n"
        "<full proof here>\n"
        "===END LEAN PROOF==="
    )

# ─── EXTRACTION ───────────────────────────────────────────────────────────────

def extract_proof(text: str) -> str:
    pattern = r"===BEGIN LEAN PROOF===\s*(.*?)\s*===END LEAN PROOF==="
    m = re.search(pattern, text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        lines = [l for l in content.splitlines() if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()
        if "import" in content:
            content = content[content.find("import"):]
        return content
    return ""

# ─── WORKER UNIT ──────────────────────────────────────────────────────────────

def process_single_seed(idx, seed, c_code, p_text):
    start_time_seed = time.time()
    success = False
    token_sum = 0
    feedback_payload = ""
    success_it = -1
    final_proof_code = ""
    
    # Apollo theorem tracking
    lemma_name = f"verify_{idx}_s{seed}"
    
    # Initialize history with your exact System/User logic
    history = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _initial_user_msg(c_code, p_text)}
    ]

    for it in range(1, MAX_ITER + 1):
        if feedback_payload:
            history.append({"role": "user", "content": feedback_payload})

        try:
            # 1. GENERATE
            response = q_client.chat.completions.create(
                model="Qwen/Qwen2.5-32B-Instruct",
                messages=history,
                temperature=TEMPERATURE,
                seed=seed
            )
            
            token_sum += response.usage.total_tokens
            raw_ans = response.choices[0].message.content
            history.append({"role": "assistant", "content": raw_ans})

            # 2. EXTRACT
            current_lean = extract_proof(raw_ans)
            if not current_lean:
                feedback_payload = "Error: Use delimiters ===BEGIN LEAN PROOF===."
                continue
            
            # Ensure naming consistency for Apollo
            if lemma_name not in current_lean:
                current_lean = current_lean.replace("theorem test", f"theorem {lemma_name}")

            # 3. REPAIR (Apollo)
            tmp = os.path.join(APOLLO_PATH, f"logs/parallel/tmp_{idx}_{seed}_{it}")
            os.makedirs(tmp, exist_ok=True)
            
            manager = ApolloRepair(code=current_lean, lemma_name=lemma_name, 
                                   config=CONFIG_PATH, rec_depth=4, log_dir=tmp)
            f_path = manager.run()
            
            # 4. VERIFY
            with open(f_path, 'r') as f:
                repaired_ver = f.read()
                v = verify_lean4_file(repaired_ver)
                
                if v['pass'] and v['complete']:
                    success = True
                    success_it = it
                    final_proof_code = repaired_ver
                    shutil.rmtree(tmp, ignore_errors=True)
                    break 
                else:
                    # Capture repairs and errors for feedback
                    diag = {"errors": v.get("errors", []), "sorries": v.get("sorries", [])}
                    feedback_payload = _feedback_user_msg(repaired_ver, diag)
            
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:
            feedback_payload = f"System Error: {str(e)}"

    seed_duration = time.time() - start_time_seed
    return {
        "seed": seed, "success": success, "success_it": success_it,
        "token_sum": token_sum, "final_code": final_proof_code, "duration": seed_duration
    }

def process_sample(item):
    idx, row = item
    start_time_sample = time.time()
    try:
        with open(row['c_file_abs']) as f: c_code = f.read()
        with open(f"{PROP_BASE}{row['property']}.prp") as f: p_text = f.read()
    except: return None

    results = []
    with ThreadPoolExecutor(max_workers=3) as seed_executor:
        futures = [seed_executor.submit(process_single_seed, idx, s, c_code, p_text) for s in SEEDS]
        for f in as_completed(futures):
            results.append(f.result())

    row_data = row.to_dict()
    seed_succ_list = []
    tokens_list = []
    
    for r in results:
        s = r['seed']
        row_data[f"s{s}_pass"] = r['success']
        row_data[f"s{s}_success_at"] = r['success_it']
        row_data[f"s{s}_tokens"] = r['token_sum']
        row_data[f"s{s}_time_sec"] = r['duration']
        row_data[f"s{s}_code"] = r['final_code']
        seed_succ_list.append(r['success'])
        tokens_list.append(r['token_sum'])

    row_data["final_prediction"] = (sum(seed_succ_list) >= 2)
    row_data["avg_tokens"] = sum(tokens_list) / len(SEEDS)
    row_data["total_process_time_sec"] = time.time() - start_time_sample
    return row_data

# ─── MAIN EXECUTION ───────────────────────────────────────────────────────────

def main():
    total_start_time = time.time()
    df = pd.read_csv(CSV_PATH)
    samples = list(df.iterrows())
    final_output = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        f_to_s = {executor.submit(process_sample, s): s for s in samples}
        for future in tqdm(as_completed(f_to_s), total=len(samples), desc="Refining Proofs"):
            res = future.result()
            if res:
                final_output.append(res)
                if len(final_output) % 3 == 0:
                    pd.DataFrame(final_output).to_csv(os.path.join(OUTPUT_DIR, "checkpoint.csv"), index=False)

    print(f"\n[Done] Runtime: {(time.time() - total_start_time)/3600:.2f} hrs")
    pd.DataFrame(final_output).to_csv(os.path.join(OUTPUT_DIR, "final_timed_results.csv"), index=False)

if __name__ == "__main__": main()