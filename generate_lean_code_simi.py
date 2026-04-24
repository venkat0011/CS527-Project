import os
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
INPUT_DIR = "/workspace/CS527-Project/raw_lean_proofs"
OUTPUT_DIR = "/workspace/CS527-Project/representativeness_results"
GPT_MODEL = "openai/gpt-5.1-codex-mini"
API_KEY = ""  # Keep your key secret
SEEDS = ["42", "123", "999"]
MAX_WORKERS = 20  # Increase this based on your OpenRouter tier limits

os.makedirs(OUTPUT_DIR, exist_ok=True)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# ─── UTILS ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=128)
def get_original_c(c_path):
    """Cache file reads so we don't hit the disk for every row."""
    try:
        with open(c_path, 'r') as f:
            return f.read()
    except Exception:
        return None

def parse_filename_metadata(filename):
    pattern = r"vllm_processed_(.+)_t([\d\.]+)\.csv"
    match = re.search(pattern, filename)
    return match.groups() if match else ("unknown", "unknown")

def code_tokenizer(code):
    if not code or not isinstance(code, str):
        return []
    return re.findall(r"\w+|==|!=|<=|>=|&&|\|\||[^\w\s]", code)

def calculate_bleu(reference, candidate):
    if not candidate or not reference:
        return 0.0
    ref_tokens = [code_tokenizer(reference)]
    cand_tokens = code_tokenizer(candidate)
    if not cand_tokens:
        return 0.0
    cc = SmoothingFunction()
    score = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method1)
    return round(score, 4)

def backtranslate_lean_to_c(lean_code):
    """Individual API call worker."""
    if not lean_code or not isinstance(lean_code, str) or len(lean_code) < 10:
        return ""
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a specialized decompiler. Convert Lean 4 verification code back into its original C source code. Output only the C code in a markdown block."},
                {"role": "user", "content": f"Lean 4 Proof:\n{lean_code}"}
            ],
            temperature=0.2,
            timeout=30 # Prevent hanging forever
        )
        content = response.choices[0].message.content
        match = re.search(r"```c\n(.*?)\n```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()
    except Exception as e:
        return ""

# ─── PROCESSING ───────────────────────────────────────────────────────────────

def process_file(csv_file, model_name, temp):
    input_path = os.path.join(INPUT_DIR, csv_file)
    df = pd.read_csv(input_path)
    
    # 1. Flatten all tasks into a list to process in parallel
    tasks = []
    records = df.to_dict('records')

    for row in records:
        # Resolve C path once
        c_path = str(row.get('c_file_abs'))
        if not os.path.exists(c_path):
            c_path = os.path.join(os.path.dirname(INPUT_DIR), "c_files", str(row.get('c_file')))
        
        if not os.path.exists(c_path):
            continue

        for s in SEEDS:
            lean_code = row.get(f'extract_s{s}')
            if pd.notna(lean_code) and len(str(lean_code)) > 10:
                # Store metadata needed for the final result
                tasks.append({
                    "lean_code": lean_code,
                    "c_path": c_path,
                    "seed": s,
                    "folder": row.get('folder'),
                    "model": model_name,
                    "temp": temp
                })

    if not tasks:
        return []

    # 2. Execute API calls in parallel using a global pool
    audit_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a mapping of future -> task metadata
        future_to_task = {
            executor.submit(backtranslate_lean_to_c, t['lean_code']): t 
            for t in tasks
        }

        # Use tqdm to track progress within the file
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"  -> {csv_file[:20]}", leave=False):
            task = future_to_task[future]
            try:
                reconstructed_c = future.result()
                original_c = get_original_c(task['c_path'])
                bleu_score = calculate_bleu(original_c, reconstructed_c)

                audit_results.append({
                    "model": task['model'],
                    "temperature": task['temp'],
                    "seed": task['seed'],
                    "bleu_representativeness": bleu_score,
                    "folder": task['folder']
                })
            except Exception as e:
                print(f"Error processing task: {e}")

    return audit_results

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    target_files = sorted([f for f in os.listdir(INPUT_DIR) if f.startswith("vllm_") and f.endswith(".csv")])
    if not target_files:
        print(f"No files found in {INPUT_DIR}")
        return

    all_data = []
    for csv_file in tqdm(target_files, desc="Overall Progress"):
        m, t = parse_filename_metadata(csv_file)
        res = process_file(csv_file, m, t)
        
        if res:
            all_data.extend(res)
            # Save intermediate file results
            pd.DataFrame(res).to_csv(os.path.join(OUTPUT_DIR, f"audit_{csv_file}"), index=False)

    if not all_data:
        print("\nError: No data was collected.")
        return

    final_df = pd.DataFrame(all_data)
    final_df.to_csv(os.path.join(OUTPUT_DIR, "final_representativeness_audit.csv"), index=False)

    print("\n" + "="*50)
    print("AVERAGE BLEU BY MODEL AND TEMP")
    print("="*50)
    summary = final_df.groupby(["model", "temperature"])["bleu_representativeness"].mean().reset_index()
    print(summary)

if __name__ == "__main__":
    main()
