import os
import sys
import pandas as pd
import json
import shutil
import traceback
from tqdm import tqdm

# 1. ADD APOLLO TO THE PYTHON MODULE SEARCH PATH
# This is the critical step that standalone scripts need
APOLLO_PATH = '/workspace/APOLLO'
if APOLLO_PATH not in sys.path:
    sys.path.insert(0, APOLLO_PATH)

# 2. SET WORKING DIRECTORY
# This ensures Apollo can find its internal 'configs' and 'utils' folders
os.chdir(APOLLO_PATH)
print(f"Current Working Directory: {os.getcwd()}")

# 3. NOW PERFORM IMPORTS
# Now that APOLLO is in sys.path, these will work
try:
    from apollo import ApolloRepair
    from prover.lean.verifier import verify_lean4_file
    print("Imports successful!")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# 4. ABSOLUTE CONFIGURATION
INPUT_DIR = "/workspace/CS527-Project/raw_lean_proofs/"
OUTPUT_DIR = "/workspace/CS527-Project/apollo_final_results"
LOG_BASE_DIR = "logs/active_batch" 

SEEDS = ["42", "123", "999"]
MAX_ATTEMPTS = 4
CONFIG_PATH = 'configs/baseline_sampling_kimina_prover.py'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_BASE_DIR, exist_ok=True)

def extract_strict_lean(text):
    if not isinstance(text, str) or "no code found" in text.lower():
        return "no proof generated"
    start_pos = text.find("import")
    if start_pos == -1: 
        return "no proof generated"
    content = text[start_pos:].strip()
    if "```" in content:
        content = content.split("```")[0].strip()
    return content if len(content) > 15 else "no proof generated"

# --- Main Multi-File Loop ---
target_files = [f for f in os.listdir(INPUT_DIR) 
                if f.startswith("vllm_") and f.endswith(".csv")]

print(f"Found {len(target_files)} files to process.")

for filename in tqdm(target_files, desc="Overall Progress"):
    output_filename = filename.replace(".csv", "_verified.csv")
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    if os.path.exists(output_path):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(input_path)
    file_results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"File: {filename}", leave=False):
        row_result = {
            "property": row.get("property", "unknown"),
            "expected_verdict": row.get("expected_verdict", "unknown"),
            "folder": row.get('folder', "unknown"),
            "original_index": index
        }

        for seed in SEEDS:
            col_name = f"extract_s{seed}"
            raw_text = row.get(col_name, "")
            code = extract_strict_lean(raw_text)
            
            p_key, c_key, e_key = f"s{seed}_pass", f"s{seed}_complete", f"s{seed}_errors"
            
            if code == "no proof generated":
                row_result.update({p_key: False, c_key: False, e_key: "no code discovered"})
                continue

            problem_dir = os.path.join(LOG_BASE_DIR, f"tmp_{index}_{seed}")
            os.makedirs(problem_dir, exist_ok=True)

            try:
                manager = ApolloRepair(
                    code=code,
                    lemma_name=f"p_{index}_{seed}",
                    config=CONFIG_PATH,
                    rec_depth=MAX_ATTEMPTS,
                    log_dir=problem_dir
                )
                
                final_proof_path = manager.run()

                if os.path.exists(final_proof_path):
                    with open(final_proof_path, 'r') as f:
                        final_code = f.read()
                    
                    v_res = verify_lean4_file(final_code)
                    row_result[p_key] = v_res.get('pass', False)
                    row_result[c_key] = v_res.get('complete', False)
                    row_result[e_key] = json.dumps(v_res.get('errors', []))
                else:
                    row_result.update({p_key: False, c_key: False, e_key: "Apollo assembly failed"})

            except Exception as e:
                row_result.update({p_key: False, c_key: False, e_key: f"Apollo Error: {str(e)}"})
            
            finally:
                if os.path.exists(problem_dir):
                    shutil.rmtree(problem_dir)

        file_results.append(row_result)
        
        if index % 10 == 0:
            pd.DataFrame(file_results).to_csv(output_path, index=False)

    pd.DataFrame(file_results).to_csv(output_path, index=False)

print("\n✨ ALL FILES PROCESSED ✨")