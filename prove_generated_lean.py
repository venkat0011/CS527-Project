import os
import sys
import pandas as pd
import json
import shutil
import concurrent.futures
from tqdm import tqdm

# 1. SETUP ENVIRONMENT
APOLLO_PATH = '/workspace/APOLLO'
if APOLLO_PATH not in sys.path:
    sys.path.insert(0, APOLLO_PATH)
os.chdir(APOLLO_PATH)

from apollo import ApolloRepair
from prover.lean.verifier import verify_lean4_file

# 2. PATHS (Absolute)
INPUT_DIR = "/workspace/CS527-Project/raw_lean_proofs/"
OUTPUT_DIR = "/workspace/CS527-Project/apollo_final_results"
LOG_BASE_DIR = "/workspace/APOLLO/logs/active_batch"
CONFIG_PATH = '/workspace/APOLLO/configs/baseline_sampling_kimina_prover.py'
SEEDS = ["42", "123", "999"]
TIMEOUT = 600 # 10 minutes

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_BASE_DIR, exist_ok=True)

def extract_strict_lean(text):
    if not isinstance(text, str) or "no code found" in text.lower():
        return "no proof generated"
    start_pos = text.find("import")
    if start_pos == -1: return "no proof generated"
    content = text[start_pos:].strip()
    if "```" in content:
        content = content.split("```")[0].strip()
    content = content.replace("lemmimport", "import")
    return content if len(content) > 15 else "no proof generated"

def run_seed_task(info):
    """Worker function for parallel seeds."""
    idx, seed, code, p_dir = info
    res = {"seed": seed, "pass": False, "complete": False, "errors": "unknown"}
    os.makedirs(p_dir, exist_ok=True)
    try:
        # This will now be fast because it calls the local vLLM API
        manager = ApolloRepair(code=code, lemma_name=f"p_{idx}_{seed}", 
                               config=CONFIG_PATH, rec_depth=4, log_dir=p_dir)
        final_path = manager.run()
        if os.path.exists(final_path):
            with open(final_path, 'r') as f:
                v = verify_lean4_file(f.read())
            res.update({"pass": v['pass'], "complete": v['complete'], "errors": json.dumps(v['errors'])})
    except Exception as e:
        res["errors"] = str(e)
    finally:
        if os.path.exists(p_dir): shutil.rmtree(p_dir)
    return res

# 3. GLOBAL BATCH LOOP
target_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("vllm_") and f.endswith(".csv")]

for filename in tqdm(target_files, desc="Files"):
    out_name = filename.replace(".csv", "_verified.csv")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    if os.path.exists(out_path): continue

    df = pd.read_csv(os.path.join(INPUT_DIR, filename))
    all_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=filename, leave=False):
        row_data = {"property": row.get("property"), "original_index": idx}
        
        # Build 3 parallel tasks
        tasks = []
        for s in SEEDS:
            clean_code = extract_strict_lean(row.get(f"extract_s{s}", ""))
            if clean_code != "no proof generated":
                tasks.append((idx, s, clean_code, os.path.join(LOG_BASE_DIR, f"tmp_{idx}_{s}")))
            else:
                row_data.update({f"s{s}_pass": False, f"s{s}_errors": "no code"})

        # Execute the 3 seeds at once
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            fut_to_s = {executor.submit(run_seed_task, t): t[1] for t in tasks}
            for fut in concurrent.futures.as_completed(fut_to_s):
                s = fut_to_s[fut]
                try:
                    r = fut.result(timeout=TIMEOUT)
                    row_data.update({f"s{s}_pass": r["pass"], f"s{s}_complete": r["complete"], f"s{s}_errors": r["errors"]})
                except Exception as e:
                    row_data[f"s{s}_errors"] = f"Timeout/Fail: {e}"

        all_rows.append(row_data)
        if idx % 5 == 0: pd.DataFrame(all_rows).to_csv(out_path, index=False)

    pd.DataFrame(all_rows).to_csv(out_path, index=False)