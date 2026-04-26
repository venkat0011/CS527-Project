#!/usr/bin/env python3
"""
pipeline1_direct.py

Direct Lean 4 generation: model sees C code + paraphrased property → outputs proof.
Iterative self-repair loop feeds Lean verifier errors back to the same model.

Models  : GPT-4.1-mini | Gemini Flash | DeepSeek-V3   (via OpenRouter)
Temps   : 0.2 | 0.5 | 0.8
Seeds   : 42  | 123 | 999
Max iter: 10
"""

import os, re, sys, json, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pathlib import Path

# ── Apollo verifier ───────────────────────────────────────────────────────────
APOLLO_PATH = "/workspace/APOLLO"
if APOLLO_PATH not in sys.path:
    sys.path.insert(0, APOLLO_PATH)
os.chdir(APOLLO_PATH)
from prover.lean.verifier import verify_lean4_file

# ==============================================================================
# CONFIGURATION
# ==============================================================================
env_path = Path('/workspace/CS527-Project/.env') # Or use your specific absolute path
load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VALIDATION_MODEL   = "openai/gpt-5-mini"   # GPT-5-mini placeholder — swap when available

CSV_PATH     = "/workspace/CS527-Project/sample_df.csv"
PROP_BASE    = "/workspace/sv-benchmarks/c/properties/"
OUTPUT_DIR   = "/workspace/CS527-Project/results/pipeline1_direct"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DIRECT_MODELS = [
    "openai/gpt-4.1-mini",
    "google/gemini-3.1-flash-lite-preview",
    "deepseek/deepseek-v4-flash",
]
TEMPERATURES  = [0.2, 0.5, 0.8]
SEEDS         = [42, 123, 999]
MAX_ITER      = 10
VERIFY_TIMEOUT = 120
MAX_WORKERS   = 4     # outer sample parallelism; inner combos each use their own threads

TEST_MODE = False
TEST_N    = 2

# ==============================================================================
# PROPERTY PARAPHRASING
# Key insight: natural language descriptions outperform raw SV-COMP spec strings
# ==============================================================================

def paraphrase_property(prop_text: str) -> str:
    p = prop_text.lower()
    if "overflow" in p:
        return (
            "Prove using Lean 4 and Mathlib that this C program does not cause any "
            "signed integer overflow. Signed integer overflow is undefined behavior in C. "
            "Model variables as Lean integers and show every arithmetic operation stays "
            "within its C type range (int: [-2147483648, 2147483647], etc.)."
        )
    if "reach_error" in p or "unreach-call" in p:
        return (
            "Prove using Lean 4 and Mathlib that the function reach_error() in this "
            "C program is NEVER called during any execution. Show that no execution "
            "path reachable from main() leads to a call of reach_error()."
        )
    if "termination" in p or "ltl(f end)" in p:
        return (
            "Prove using Lean 4 and Mathlib that this C program always terminates for "
            "all valid inputs. Show every loop has a strictly decreasing non-negative "
            "termination measure and every recursive call is on a strictly smaller argument."
        )
    return f"Prove using Lean 4 and Mathlib that the program satisfies: {prop_text.strip()}"

# ==============================================================================
# PROMPTS
# ==============================================================================

SYSTEM = (
    "You are an expert Lean 4 and Mathlib programmer. "
    "Convert C programs and their formal specifications into complete, verified Lean 4 proofs.\n\n"
    "Your final answer MUST be placed inside these delimiters and nowhere else:\n"
    "===BEGIN LEAN PROOF===\n"
    "<your complete Lean 4 source here>\n"
    "===END LEAN PROOF===\n\n"
    "Hard rules:\n"
    "  • The proof must start with `import Mathlib`.\n"
    "  • Do NOT use `sorry` — any proof containing `sorry` is INCOMPLETE and rejected.\n"
    "  • The proof must compile and fully verify in Lean 4."
)


def initial_user(c_code: str, prop_nl: str) -> str:
    return (
        f"### Task\n{prop_nl}\n\n"
        f"### C Code\n```c\n{c_code}\n```\n\n"
        "Write your complete, sorry-free Lean 4 proof between the delimiters:\n"
        "===BEGIN LEAN PROOF===\n"
        "import Mathlib\n"
        "-- your proof here\n"
        "===END LEAN PROOF==="
    )


def repair_user(lean_code: str, result: dict, iteration: int) -> str:
    parts = []

    if result.get("errors"):
        rows = [
            f"  Line {e['pos']['line']}, Col {e['pos']['column']}: {e['data']}"
            for e in result["errors"]
        ]
        parts.append(
            "#### Compilation Errors\n```\n" + "\n".join(rows) + "\n```"
        )

    if result.get("sorries"):
        blocks = []
        for s in result["sorries"]:
            line = s.get("pos", {}).get("line", "?")
            goal = s.get("goal", "(unavailable)")
            blocks.append(
                f"  Line {line} — remaining proof goal:\n"
                f"  ```\n  {goal}\n  ```"
            )
        parts.append(
            "#### Incomplete Proof (`sorry` detected)\n"
            "Close every sorry. Proof goals:\n\n" + "\n".join(blocks) + "\n\n"
            "Useful tactics: `omega` · `ring` · `linarith` · `norm_num` · "
            "`decide` · `simp [*]` · `tauto`"
        )

    if not parts:
        parts.append("#### Proof Incomplete\nNo errors but verification failed.")

    header = (
        f"### Compiler Feedback — Iteration {iteration}/{MAX_ITER}\n\n"
        + "\n\n".join(parts)
    )
    footer = (
        f"\n\n---\n### Your Previous Proof\n```lean\n{lean_code}\n```\n\n"
        "Fix ALL issues. Return the corrected proof:\n"
        "===BEGIN LEAN PROOF===\n"
        "<corrected proof>\n"
        "===END LEAN PROOF==="
    )
    return header + footer


def no_delimiter_msg(iteration: int) -> str:
    return (
        f"### Iteration {iteration} — Missing Delimiters\n"
        "Wrap your entire proof exactly like this:\n"
        "===BEGIN LEAN PROOF===\n"
        "import Mathlib\n"
        "-- proof\n"
        "===END LEAN PROOF==="
    )

# ==============================================================================
# LEAN EXTRACTION
# ==============================================================================

def extract_proof(text: str) -> str:
    m = re.search(
        r"===BEGIN LEAN PROOF===\s*(.*?)\s*===END LEAN PROOF===",
        text, re.DOTALL
    )
    if not m:
        return ""
    content = m.group(1).strip()
    # Strip any markdown fences inside the delimiters
    lines = [l for l in content.splitlines() if not l.strip().startswith("```")]
    content = "\n".join(lines).strip()
    if "import" in content:
        content = content[content.find("import"):]
    return content

# ==============================================================================
# BLEU VALIDATION
# ==============================================================================

def validate_lean_with_bleu(lean_code: str, original_c: str) -> dict:
    """
    1. Ask validation model to regenerate C code from the Lean proof.
    2. Compute BLEU between original C and regenerated C.
    This checks whether the Lean proof actually encodes the C semantics.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    try:
        resp = client.chat.completions.create(
            model=VALIDATION_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Given this Lean 4 formal proof, reconstruct the C program it verifies.\n"
                    "Output ONLY a valid C program, no explanation.\n\n"
                    f"```lean4\n{lean_code}\n```"
                )
            }],
            temperature=0.0,
        )
        regen = resp.choices[0].message.content.strip()

        # Extract C code block if wrapped in markdown
        c_match = re.search(r"```(?:c|cpp)?\s*\n(.*?)```", regen, re.DOTALL)
        if c_match:
            regen = c_match.group(1).strip()

        # BLEU on token level (treats each word/symbol as a token)
        ref_tok  = original_c.split()
        hyp_tok  = regen.split()
        smooth   = SmoothingFunction().method1
        bleu_val = sentence_bleu([ref_tok], hyp_tok, smoothing_function=smooth)

        return {
            "bleu_score"    : round(bleu_val, 2),   # 0-100 scale
            "regenerated_c" : regen,
            "validation_ok" : True,
        }
    except Exception as e:
        return {"bleu_score": -1.0, "regenerated_c": "", "validation_ok": False,
                "validation_error": str(e)}

# ==============================================================================
# CORE REPAIR LOOP  (one model × one temperature × one seed)
# ==============================================================================

def run_single(
    c_code: str,
    prop_nl: str,
    model: str,
    temperature: float,
    seed: int,
) -> dict:
    """
    Single repair-loop run.  Returns a metrics dict.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    t0         = time.time()
    tokens_in  = 0
    tokens_out = 0
    iter_log   = []
    success    = False
    success_it = -1
    final_code = ""
    feedback   = ""

    history = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": initial_user(c_code, prop_nl)},
    ]

    for it in range(1, MAX_ITER + 1):
        if feedback:
            history.append({"role": "user", "content": feedback})
            feedback = ""

        rec = {
            "iteration"   : it,
            "model"       : model,
            "temperature" : temperature,
            "seed"        : seed,
            "extracted"   : False,
            "pass"        : False,
            "complete"    : False,
            "n_errors"    : 0,
            "n_sorries"   : 0,
            "tokens_in"   : 0,
            "tokens_out"  : 0,
            "verify_time" : 0.0,
            "iter_time"   : 0.0,
        }
        it_t0 = time.time()

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=history,
                temperature=temperature,
                seed=seed,
                # no max_tokens — let model decide
            )
            rec["tokens_in"]  = resp.usage.prompt_tokens
            rec["tokens_out"] = resp.usage.completion_tokens
            tokens_in        += resp.usage.prompt_tokens
            tokens_out       += resp.usage.completion_tokens

            raw = resp.choices[0].message.content
            history.append({"role": "assistant", "content": raw})

            lean = extract_proof(raw)
            if not lean:
                feedback       = no_delimiter_msg(it)
                rec["extracted"] = False
                rec["iter_time"] = round(time.time() - it_t0, 2)
                iter_log.append(rec)
                continue

            rec["extracted"] = True

            v = verify_lean4_file(lean, timeout=VERIFY_TIMEOUT)
            rec.update({
                "pass"        : bool(v.get("pass",     False)),
                "complete"    : bool(v.get("complete", False)),
                "n_errors"    : len(v.get("errors",   [])),
                "n_sorries"   : len(v.get("sorries",  [])),
                "verify_time" : round(v.get("verify_time", 0.0), 2),
            })

            if v["pass"] and v["complete"]:
                success    = True
                success_it = it
                final_code = lean
                rec["iter_time"] = round(time.time() - it_t0, 2)
                iter_log.append(rec)
                break

            feedback = repair_user(lean, v, it)

        except Exception as exc:
            rec["error"]   = str(exc)
            feedback = (
                f"### System Error at Iteration {it}\n```\n{exc}\n```\n"
                "Please provide a fresh Lean 4 proof attempt."
            )

        rec["iter_time"] = round(time.time() - it_t0, 2)
        iter_log.append(rec)

    # BLEU validation only when proof is complete
    bleu_data = {}
    if success:
        bleu_data = validate_lean_with_bleu(final_code, c_code)

    return {
        "model"       : model,
        "temperature" : temperature,
        "seed"        : seed,
        "success"     : success,
        "success_it"  : success_it,
        "total_iters" : len(iter_log),
        "tokens_in"   : tokens_in,
        "tokens_out"  : tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "duration_sec": round(time.time() - t0, 2),
        "final_code"  : final_code,
        "iter_log"    : iter_log,
        **bleu_data,
    }

# ==============================================================================
# SAMPLE PROCESSOR
# ==============================================================================

def process_sample(item) -> list[dict]:
    """
    For one sample, run all (model × temperature × seed) combos in parallel.
    Returns a list of result dicts, one per combo.
    """
    idx, row = item
    try:
        with open(row["c_file_abs"]) as f:
            c_code = f.read()
        with open(f"{PROP_BASE}{row['property']}.prp") as f:
            prop_nl = paraphrase_property(f.read())
    except Exception as e:
        tqdm.write(f"  [SKIP] idx={idx}: {e}")
        return []

    combos = [
        (model, temp, seed)
        for model in DIRECT_MODELS
        for temp  in TEMPERATURES
        for seed  in SEEDS
    ]

    results = []
    # Each combo is independent → parallelize within the sample
    with ThreadPoolExecutor(max_workers=min(len(combos), 9)) as ex:
        futs = {
            ex.submit(run_single, c_code, prop_nl, m, t, s): (m, t, s)
            for m, t, s in combos
        }
        for fut in as_completed(futs):
            m, t, s = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {
                    "model": m, "temperature": t, "seed": s,
                    "success": False, "success_it": -1,
                    "total_iters": 0, "tokens_in": 0, "tokens_out": 0,
                    "total_tokens": 0, "duration_sec": 0.0,
                    "final_code": "", "iter_log": [],
                    "bleu_score": -1.0, "error": str(exc),
                }
            # Attach sample metadata
            r["sample_idx"] = idx
            r["c_file"]     = row.get("c_file_abs", "")
            r["property"]   = row.get("property", "")
            r['expected_verdict'] = row.get('expected_verdict', "")
            r["iter_log"]   = json.dumps(r.get("iter_log", []))
            results.append(r)

    return results

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)
    if TEST_MODE:
        df = df.query('expected_verdict==True').sample(TEST_N, random_state=42)
        print(f"[TEST MODE] {TEST_N} samples")

    samples = list(df.iterrows())
    print(
        f"[INFO] {len(samples)} samples × {len(DIRECT_MODELS)} models × "
        f"{len(TEMPERATURES)} temps × {len(SEEDS)} seeds = "
        f"{len(samples)*len(DIRECT_MODELS)*len(TEMPERATURES)*len(SEEDS)} runs"
    )

    all_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_sample, s): s for s in samples}
        for fut in tqdm(as_completed(futs), total=len(samples),
                        desc="Samples", unit="sample"):
            rows = fut.result()
            all_rows.extend(rows)
            # Checkpoint every 5 samples
            if len(all_rows) % (5 * len(DIRECT_MODELS) * len(TEMPERATURES) * len(SEEDS)) == 0:
                pd.DataFrame(all_rows).to_csv(
                    os.path.join(OUTPUT_DIR, "checkpoint.csv"), index=False
                )

    out = os.path.join(OUTPUT_DIR, "final_results.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\n[Done] {(time.time()-t0)/3600:.2f}h | {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()