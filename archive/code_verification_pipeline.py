#!/usr/bin/env python3
"""
Lean proof generation with Qwen self-repair loop.

Replace Apollo with direct Lean 4 compiler feedback:
  Generate → Verify → Feed diagnostics back → Re-generate

The key driver is sorry proof-state information:
  sorries[i]['goal'] tells Qwen exactly what remains to be proven,
  which is far more informative than "compilation failed".
"""

import os
import re
import json
import pandas as pd
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ─── PATHS ────────────────────────────────────────────────────────────────────
APOLLO_PATH = '/workspace/APOLLO'
if APOLLO_PATH not in sys.path:
    sys.path.insert(0, APOLLO_PATH)
os.chdir(APOLLO_PATH)

from prover.lean.verifier import verify_lean4_file   # Apollo no longer used; verifier only

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH        = '/workspace/CS527-Project/results/c_sample.csv'
PROP_BASE       = '/workspace/sv-benchmarks/c/properties/'
OUTPUT_DIR      = "/workspace/CS527-Project/results/results_self_repair"
SEEDS           = [42, 123, 999]
MAX_ITER        = 5
TEMPERATURE     = 0.8
MAX_WORKERS     = 6
VERIFY_TIMEOUT  = 120   # seconds — increase for longer proofs

# Set True to smoke-test on a single random sample before a full run
TEST_MODE       = False
TEST_N          = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
q_client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm-token")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM = (
    "You are an expert Lean 4 and Mathlib programmer. "
    "You convert C programs and their formal specifications into complete, verified Lean 4 proofs.\n\n"
    "Your final answer MUST be placed inside these delimiters and nowhere else:\n"
    "===BEGIN LEAN PROOF===\n"
    "<your complete Lean 4 source here>\n"
    "===END LEAN PROOF===\n\n"
    "Hard rules:\n"
    "  • The proof must start with `import Mathlib`.\n"
    "  • Do NOT use `sorry` — any proof containing `sorry` is considered INCOMPLETE and will be rejected.\n"
    "  • The proof must compile and fully verify in Lean 4."
)


def _initial_user_msg(c_code: str, prop_text: str) -> str:
    return (
        "Translate the C program below into a complete, sorry-free Lean 4 proof "
        "of the given property.\n\n"
        f"### C Code\n```c\n{c_code}\n```\n\n"
        f"### Property\n```\n{prop_text}\n```\n\n"
        "Write your final Lean 4 proof between the delimiters:\n"
        "===BEGIN LEAN PROOF===\n"
        "<import Mathlib and full proof here>\n"
        "===END LEAN PROOF==="
    )


def _build_compiler_feedback(lean_code: str, v: dict, it: int) -> str:
    """
    Construct a structured feedback message from verify_lean4_file() output.

    verify_lean4_file returns a dict with:
        pass     (bool)  – False means compilation errors
        complete (bool)  – False means sorries / unfinished goals remain
        errors   (list)  – [{pos: {line, column}, data: str, severity: str}, ...]
        sorries  (list)  – [{pos: {line, column}, goal: str, endPos: ...}, ...]
        warnings (list)  – [{pos: {line, column}, data: str}, ...]

    The `goal` field inside a sorry entry is gold: it is the exact Lean proof
    state (context + ⊢ target) that Qwen must prove to close that sorry.
    """
    sections: list[str] = []

    # ── Compilation errors (pass = False) ─────────────────────────────────────
    if v.get("errors"):
        rows = []
        for e in v["errors"]:
            line = e.get("pos", {}).get("line",   "?")
            col  = e.get("pos", {}).get("column", "?")
            msg  = e.get("data", repr(e))
            rows.append(f"  Line {line}, Col {col}: {msg}")
        sections.append(
            "#### ❌ Compilation Errors — must be fixed\n"
            "```\n" + "\n".join(rows) + "\n```"
        )

    # ── Sorry / incomplete proof (complete = False) ────────────────────────────
    if v.get("sorries"):
        blocks = []
        for s in v["sorries"]:
            line = s.get("pos", {}).get("line", "?")
            goal = s.get("goal", "(proof state unavailable)")
            blocks.append(
                f"  - **Line {line}** — outstanding proof goal:\n"
                f"    ```\n    {goal}\n    ```\n"
                f"    ↑ This is the exact Lean proof state you must close."
            )
        sections.append(
            "#### ⚠️ Incomplete Proof — `sorry` detected\n"
            "Replace **every** `sorry` with real tactic proofs.\n"
            "The Lean compiler reports these remaining goals:\n\n"
            + "\n".join(blocks)
            + "\n\n**Tactic suggestions** for arithmetic/logical goals:\n"
              "`omega` · `ring` · `linarith` · `norm_num` · `decide` · "
              "`simp [*]` · `tauto` · `exact?` · `apply?`"
        )

    # ── Compiled but not complete, no sorries (rare edge case) ────────────────
    if not sections:
        sections.append(
            "#### ⚠️ Proof Incomplete\n"
            "The file compiled without errors but is still not fully verified. "
            "Ensure every goal is closed and no tactics are left in a pending state."
        )

    remaining = MAX_ITER - it
    header = (
        f"### Compiler Feedback — Iteration {it}/{MAX_ITER} "
        f"({remaining} attempt{'s' if remaining != 1 else ''} remaining)\n\n"
        f"Your proof attempt has the following issues:\n\n"
    )
    footer = (
        f"\n---\n"
        f"### Your Current Proof (for reference)\n"
        f"```lean\n{lean_code}\n```\n\n"
        "Fix **all** issues above and provide a complete, sorry-free proof:\n"
        "===BEGIN LEAN PROOF===\n"
        "<full corrected proof here>\n"
        "===END LEAN PROOF==="
    )
    return header + "\n\n".join(sections) + footer


def _no_delimiter_feedback(it: int) -> str:
    return (
        f"### Iteration {it} — Missing Delimiters\n\n"
        "Your response did not contain the required proof delimiters.\n"
        "You MUST wrap your entire Lean 4 proof exactly like this:\n\n"
        "===BEGIN LEAN PROOF===\n"
        "import Mathlib\n"
        "-- ... your full proof ...\n"
        "===END LEAN PROOF==="
    )

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – LEAN EXTRACTION  (identical to base code)
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – SEED WORKER  (replaces Apollo with compiler feedback loop)
# ══════════════════════════════════════════════════════════════════════════════

def process_single_seed(idx: int, seed: int, c_code: str, p_text: str) -> dict:
    """
    Self-repair loop for one seed:
        [Generate] → [Extract] → [Verify] → (success?) → done
                                          ↘ (failure)  → [Build feedback] → [Generate]

    iter_log captures per-iteration diagnostics so you can later analyse
    how quickly each proof converges and what kinds of errors appear.
    """
    t0           = time.time()
    success      = False
    token_sum    = 0
    feedback_msg = ""
    success_it   = -1
    final_proof  = ""
    iter_log: list[dict] = []

    history = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _initial_user_msg(c_code, p_text)},
    ]

    for it in range(1, MAX_ITER + 1):
        print(it)
        # Append compiler feedback from previous iteration (if any)
        if feedback_msg:
            history.append({"role": "user", "content": feedback_msg})
            feedback_msg = ""

        it_record: dict = {
            "it": it, "extracted": False, "pass": False,
            "complete": False, "n_errors": 0, "n_sorries": 0,
            "verify_time_sec": 0.0,
        }

        try:
            # ── STEP 1: Generate ──────────────────────────────────────────────
            resp = q_client.chat.completions.create(
                model="Qwen/Qwen2.5-32B-Instruct-AWQ",
                messages=history,
                temperature=TEMPERATURE,
                seed=seed,
            )
            token_sum += resp.usage.total_tokens
            raw_ans    = resp.choices[0].message.content
            history.append({"role": "assistant", "content": raw_ans})

            # ── STEP 2: Extract proof from delimiters ─────────────────────────
            lean_code = extract_proof(raw_ans)
            # print(f'first leancode {lean_code} @ {it}')
            if not lean_code:
                feedback_msg = _no_delimiter_feedback(it)
                it_record["extracted"] = False
                iter_log.append(it_record)
                continue

            it_record["extracted"] = True

            # ── STEP 3: Verify with Lean 4 compiler ───────────────────────────
            v = verify_lean4_file(lean_code, timeout=VERIFY_TIMEOUT)
            # print(v)
            it_record.update({
                "pass"            : bool(v.get("pass",     False)),
                "complete"        : bool(v.get("complete", False)),
                "n_errors"        : len(v.get("errors",    [])),
                "n_sorries"       : len(v.get("sorries",   [])),
                "verify_time_sec" : round(v.get("verify_time", 0.0), 2),
            })

            # ── STEP 4: Check for full success ────────────────────────────────
            if v["pass"] and v["complete"]:
                success     = True
                success_it  = it
                final_proof = lean_code
                iter_log.append(it_record)
                break   # ← done for this seed

            # ── STEP 5: Build structured compiler feedback ────────────────────
            feedback_msg = _build_compiler_feedback(lean_code, v, it)

        except Exception as exc:
            it_record["system_error"] = str(exc)
            feedback_msg = (
                f"### System Error at Iteration {it}\n"
                f"```\n{exc}\n```\n"
                "Please provide a fresh Lean 4 proof attempt."
            )

        iter_log.append(it_record)
        # — end of iteration loop —

    return {
        "seed"       : seed,
        "success"    : success,
        "success_it" : success_it,
        "token_sum"  : token_sum,
        "final_code" : final_proof,
        "duration"   : round(time.time() - t0, 2),
        "iter_log"   : iter_log,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – SAMPLE PROCESSOR  (identical structure to base code)
# ══════════════════════════════════════════════════════════════════════════════

def process_sample(item) -> dict | None:
    idx, row = item
    t0 = time.time()

    try:
        with open(row["c_file_abs"]) as f:
            c_code = f.read()
        with open(f"{PROP_BASE}{row['property']}.prp") as f:
            p_text = f.read()
    except Exception as e:
        tqdm.write(f"  [SKIP] idx={idx}: {e}")
        return None

    # Run all 3 seeds in parallel (same as base code)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_map = {
            ex.submit(process_single_seed, idx, s, c_code, p_text): s
            for s in SEEDS
        }
        for fut in as_completed(fut_map):
            try:
                results.append(fut.result())
            except Exception as exc:
                s = fut_map[fut]
                results.append({
                    "seed": s, "success": False, "success_it": -1,
                    "token_sum": 0, "final_code": "", "duration": 0.0,
                    "iter_log": [], "error": str(exc),
                })

    # Assemble output row (same columns as base code + iter_log per seed)
    row_data       = row.to_dict()
    seed_succ_list = []
    tokens_list    = []

    for r in results:
        s = r["seed"]
        row_data[f"s{s}_pass"]       = r["success"]
        row_data[f"s{s}_success_at"] = r["success_it"]
        row_data[f"s{s}_tokens"]     = r["token_sum"]
        row_data[f"s{s}_time_sec"]   = r["duration"]
        row_data[f"s{s}_code"]       = r["final_code"]
        # JSON-encoded per-iteration diagnostics for post-hoc analysis
        row_data[f"s{s}_iter_log"]   = json.dumps(r.get("iter_log", []))
        seed_succ_list.append(r["success"])
        tokens_list.append(r["token_sum"])

    row_data["final_prediction"]       = int(sum(seed_succ_list) >= 2)
    row_data["avg_tokens"]             = round(sum(tokens_list) / len(SEEDS), 1)
    row_data["total_process_time_sec"] = round(time.time() - t0, 2)
    return row_data

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)

    if TEST_MODE:
        df = df.sample(TEST_N, random_state=42)
        print(f"[TEST MODE] Running on {TEST_N} sample(s).")

    samples = list(df.iterrows())
    print(f"[INFO] {len(samples)} samples × {len(SEEDS)} seeds × "
          f"up to {MAX_ITER} iterations each")
    print(f"[INFO] Output directory: {OUTPUT_DIR}\n")

    final_output: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        fut_map = {executor.submit(process_sample, s): s for s in samples}

        for fut in tqdm(as_completed(fut_map), total=len(samples),
                        desc="Samples", unit="sample"):
            res = fut.result()
            if res:
                final_output.append(res)
                if len(final_output) % 3 == 0:
                    pd.DataFrame(final_output).to_csv(
                        os.path.join(OUTPUT_DIR, "checkpoint.csv"), index=False
                    )

    elapsed = (time.time() - t0) / 3600
    print(f"\n[Done] {elapsed:.2f} hrs | {len(final_output)} samples completed")

    out_path = os.path.join(OUTPUT_DIR, "final_results.csv")
    pd.DataFrame(final_output).to_csv(out_path, index=False)
    print(f"[Done] Results → {out_path}")


if __name__ == "__main__":
    main()