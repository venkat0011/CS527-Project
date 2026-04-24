import os
import re
import json
import shutil
import sys
import time
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ─── APOLLO PATH ──────────────────────────────────────────────────────────────
APOLLO_PATH = '/workspace/APOLLO'
if APOLLO_PATH not in sys.path:
    sys.path.insert(0, APOLLO_PATH)
os.chdir(APOLLO_PATH)

from apollo import ApolloRepair
from prover.lean.verifier import verify_lean4_file

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CSV_PATH = '/workspace/CS527-Project/c_sample.csv'
PROP_BASE = '/workspace/sv-benchmarks/c/properties/'
OUTPUT_DIR = '/workspace/CS527-Project/bleu_audit_results'
CONFIG_PATH = '/workspace/APOLLO/configs/baseline_sampling_kimina_prover.py'

SEEDS = [42, 123, 999]
MAX_WORKERS = 4 
GPT_MODEL = "openai/gpt-4o-mini" # Used for back-translation reconstruction

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clients pointing to persistent vLLM servers
q_client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm-token")
gpt_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="")

# ─── SIMILARITY LOGIC (BLEU) ──────────────────────────────────────────────────

def code_tokenizer(code):
    """Tokenize C code keeping operators as distinct units."""
    return re.findall(r"\w+|==|!=|<=|>=|&&|\|\||[^\w\s]", code)

def calculate_bleu(reference, candidate):
    """Standard BLEU-4 with Smoothing for code evaluation."""
    if not candidate: return 0.0
    ref_tokens = [code_tokenizer(reference)]
    cand_tokens = code_tokenizer(candidate)
    cc = SmoothingFunction()
    score = sentence_bleu(ref_tokens, cand_tokens, 
                          weights=(0.25, 0.25, 0.25, 0.25), 
                          smoothing_function=cc.method1)
    return round(score, 4)

# ─── BACK-TRANSLATION ─────────────────────────────────────────────────────────

def lean_to_c_backtranslate(lean_code):
    """Uses GPT to reconstruct C logic from Lean for cross-check."""
    try:
        resp = gpt_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{
                "role": "user", 
                "content": f"Reconstruct only the C code logic from this Lean proof. Keep math identical:\n\n{lean_code}"
            }],
            temperature=0.2
        )
        ans = resp.choices[0].message.content
        m = re.search(r"```c\n(.*?)\n```", ans, re.DOTALL)
        return m.group(1).strip() if m else ans.strip()
    except: return ""

# ─── SEED WORKER ──────────────────────────────────────────────────────────────

def run_seed_audit(idx, seed, c_orig, p_text):
    start_t = time.time()
    res = {"seed": seed, "orig_idx": idx}
    lemma_name = f"audit_seed_{idx}_{seed}"
    
    # 1. GENERATE
    resp = q_client.chat.completions.create(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ",
        messages=[{"role": "system", "content": "You are a Lean 4 expert. Wrap code in ===BEGIN LEAN PROOF===."},
                  {"role": "user", "content": f"C Code:\n{c_orig}\nProperty: {p_text}"}],
        temperature=0.8, seed=seed
    )
    raw_text = resp.choices[0].message.content
    m = re.search(r"===BEGIN LEAN PROOF===\s*(.*?)\s*===END LEAN PROOF===", raw_text, re.DOTALL)
    raw_lean = m.group(1).strip() if m else ""
    if lemma_name not in raw_lean: raw_lean = raw_lean.replace("theorem test", f"theorem {lemma_name}")

    # 2. APOLLO REPAIR + STRICT CLEANUP
    repaired_lean = ""
    log_dir = os.path.join(APOLLO_PATH, f"logs/bleu_audit/tmp_{idx}_{seed}")
    os.makedirs(log_dir, exist_ok=True)
    try:
        manager = ApolloRepair(code=raw_lean, lemma_name=lemma_name, config=CONFIG_PATH, rec_depth=1, log_dir=log_dir)
        path = manager.run()
        with open(path, 'r') as f:
            repaired_lean = f.read()
    finally:
        # CLEANUP: Delete directory after extraction
        shutil.rmtree(log_dir, ignore_errors=True)

    # 3. BACK-TRANSLATION
    c_recon_raw = lean_to_c_backtranslate(raw_lean)
    c_recon_rep = lean_to_c_backtranslate(repaired_lean)
    print('this is reparied lean', repaired_lean)
    print('this is recon_raw', c_recon_raw)
    
    # 4. MEASURE BLEU
    res["bleu_raw"] = calculate_bleu(c_orig, c_recon_raw)
    res["bleu_rep"] = calculate_bleu(c_orig, c_recon_rep)
    res["bleu_drift"] = round(res["bleu_rep"] - res["bleu_raw"], 4)
    
    return res

# ─── BATCH ORCHESTRATOR ───────────────────────────────────────────────────────

def process_sample(item):
    idx, row = item
    try:
        abs_path = row["c_file_abs"]
        # CAPTURE THE C FOLDER NAME
        c_folder = os.path.basename(os.path.dirname(abs_path))
        
        with open(abs_path) as f: c_orig = f.read()
        with open(f"{PROP_BASE}{row['property']}.prp") as f: p_text = f.read()
        
        results = []
        for s in SEEDS:
            audit_res = run_seed_audit(idx, s, c_orig, p_text)
            audit_res.update({
                "property": row['property'], 
                "c_folder": row['folder'],
            })
            results.append(audit_res)
        return results
    except Exception as e:
        print(f"Error {idx}: {e}")
        return []

def main():
    start_total = time.time()
    # df = pd.read_csv(CSV_PATH).sample(1, random_state=42)
    df = pd.read_csv(CSV_PATH)
    
    all_final_data = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_sample, item) for item in df.iterrows()]
        for f in tqdm(as_completed(futures), total=len(df), desc="BLEU Audit Batch"):
            res_list = f.result()
            all_final_data.extend(res_list)
            
            if len(all_final_data) % 15 == 0:
                pd.DataFrame(all_final_data).to_csv(os.path.join(OUTPUT_DIR, "bleu_audit_checkpoint.csv"), index=False)

    df_out = pd.DataFrame(all_final_data)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "final_bleu_audit_results.csv"), index=False)
    
    print(f"\n[Summary] Mean BLEU Drift: {df_out['bleu_drift'].mean():.4f}")
    print(f"[Done] Total Runtime: {(time.time() - start_total)/3600:.2f} hours")

if __name__ == "__main__":
    main()