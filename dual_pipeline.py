```python
#!/usr/bin/env python3
"""
pipeline_neg_dual.py

Two new experiment modes — use alongside existing pipeline1_direct.py results:

  MODE "negative" : Always try to DISPROVE the specification.
                    Lean proof shows a concrete VIOLATION / COUNTEREXAMPLE.
                    Predicts False if negation proof succeeds, True if it fails.

  MODE "dual"     : Run positive AND negative in parallel (10 iterations each).
                    Priority: negation proven → False
                              positive proven (negation failed) → True
                              both failed → False  (conservative default)

Decision table:
  ┌─────────────────┬────────────────┬────────────┐
  │  neg succeeded  │ pos succeeded  │ prediction │
  ├─────────────────┼────────────────┼────────────┤
  │      True       │    anything    │   False    │  ← trust negative
  │      False      │      True      │   True     │
  │      False      │     False      │   False    │  ← conservative default
  └─────────────────┴────────────────┴────────────┘
"""

import os, re, sys, json, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dotenv import load_dotenv
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

env_path = Path("/workspace/CS527-Project/.env")
load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VALIDATION_MODEL   = "openai/gpt-5-mini"

# Set which modes to run in this execution
# Remove either entry if you only want to run one mode
MODES = ["negative", "dual"]
# negative means we will only generate the result for the negation of the property
# dual is where we onlt predict 

DIRECT_MODELS = [
    "openai/gpt-4.1-mini",
    "google/gemini-3.1-flash-lite-preview",
    "deepseek/deepseek-v4-flash",
]
TEMPERATURES  = [0.2, 0.5, 0.8]
SEEDS         = [42, 123, 999]
MAX_ITER      = 10
VERIFY_TIMEOUT = 120
MAX_WORKERS   = 4

CSV_PATH   = "/workspace/CS527-Project/sample_df.csv"
PROP_BASE  = "/workspace/sv-benchmarks/c/properties/"
OUTPUT_DIR = "/workspace/CS527-Project/results/pipeline_neg_dual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_MODE = True
TEST_N    = 2

# ==============================================================================
# SPEC TYPE DETECTION
# ==============================================================================

def detect_spec_type(prop_text: str) -> str:
    p = prop_text.lower()
    if "overflow" in p:
        return "overflow"
    if "reach_error" in p or "unreach-call" in p:
        return "unreach-call"
    if "termination" in p or "ltl(f end)" in p:
        return "termination"
    return "unknown"

# ==============================================================================
# PROPERTY PARAPHRASING  (positive — unchanged from pipeline1)
# ==============================================================================

def paraphrase_property(prop_text: str) -> str:
    p = prop_text.lower()
    if "overflow" in p:
        return (
            "Prove using Lean 4 and Mathlib that this C program does not cause any "
            "signed integer overflow. Model variables as Lean integers and show every "
            "arithmetic operation stays within the C type range "
            "(int: [-2147483648, 2147483647]; long long: [-2^63, 2^63-1], etc.)."
        )
    if "reach_error" in p or "unreach-call" in p:
        return (
            "Prove using Lean 4 and Mathlib that reach_error() is NEVER called during "
            "any execution starting from main(). Show no reachable execution path "
            "leads to that function."
        )
    if "termination" in p or "ltl(f end)" in p:
        return (
            "Prove using Lean 4 and Mathlib that this C program always terminates. "
            "Show every loop has a strictly decreasing non-negative termination measure "
            "and every recursive call is on a strictly smaller argument."
        )
    return f"Prove using Lean 4 and Mathlib that the program satisfies: {prop_text.strip()}"


def negate_property(prop_text: str, spec_type: str) -> str:
    """Natural language description of the NEGATED property."""
    if spec_type == "overflow":
        return (
            "Disprove the no-overflow property: show this C program DOES cause "
            "signed integer overflow for at least one concrete input."
        )
    if spec_type == "unreach-call":
        return (
            "Disprove unreachability: show reach_error() CAN be called during "
            "some concrete execution of this program."
        )
    if spec_type == "termination":
        return (
            "Disprove termination: show this C program can fail to terminate "
            "(find concrete inputs making the loop run at least 10,000 iterations)."
        )
    return f"Disprove: {prop_text.strip()}"

# ==============================================================================
# LEAN EXTRACTION  (identical to pipeline1)
# ==============================================================================

def extract_proof(text: str) -> str:
    m = re.search(
        r"===BEGIN LEAN PROOF===\s*(.*?)\s*===END LEAN PROOF===",
        text, re.DOTALL
    )
    if not m:
        return ""
    content = m.group(1).strip()
    lines   = [l for l in content.splitlines() if not l.strip().startswith("```")]
    content = "\n".join(lines).strip()
    if "import" in content:
        content = content[content.find("import"):]
    return content


def rule_based_fixes(code: str) -> str:
    code = re.sub(r"\bconstant\b", "def", code)
    code = code.replace("| unit |", "()")
    code = code.replace("loop body", "loopBody")
    code = re.sub(r"termination_by\s*\((\w+),\s*(\w+)\)\s*=>",
                  r"termination_by \1 \2 =>", code)
    code = re.sub(r"(\w+)\s*<\|\s*\((\w+),\s*(\w+)\)", r"\1 \2 \3", code)
    code = re.sub(r"\bT_min\b", "T_MIN", code)
    code = re.sub(r"\bT_max\b", "T_MAX", code)
    return code

# ==============================================================================
# POSITIVE PROMPTS  (unchanged from pipeline1)
# ==============================================================================

POSITIVE_SYSTEM = (
    "You are an expert Lean 4 and Mathlib programmer. "
    "Convert C programs and their formal specifications into complete, verified Lean 4 proofs.\n\n"
    "Your final answer MUST be placed inside these delimiters and nowhere else:\n"
    "===BEGIN LEAN PROOF===\n"
    "<your complete Lean 4 source here>\n"
    "===END LEAN PROOF===\n\n"
    "Hard rules:\n"
    "  • The proof must start with `import Mathlib`.\n"
    "  • Do NOT use `sorry`.\n"
    "  • The proof must compile and fully verify in Lean 4."
)


def positive_initial_user(c_code: str, prop_nl: str) -> str:
    return (
        f"### Task\n{prop_nl}\n\n"
        f"### C Code\n```c\n{c_code}\n```\n\n"
        "Write your complete, sorry-free Lean 4 proof between the delimiters:\n"
        "===BEGIN LEAN PROOF===\n"
        "import Mathlib\n"
        "-- proof here\n"
        "===END LEAN PROOF==="
    )


def positive_repair_user(lean_code: str, result: dict, iteration: int) -> str:
    parts = []
    if result.get("errors"):
        rows = [
            f"  Line {e['pos']['line']}, Col {e['pos']['column']}: {e['data']}"
            for e in result["errors"]
        ]
        parts.append("#### Compilation Errors\n```\n" + "\n".join(rows) + "\n```")
    if result.get("sorries"):
        blocks = [
            f"  Line {s.get('pos',{}).get('line','?')}: {s.get('goal','')}"
            for s in result["sorries"]
        ]
        parts.append(
            "#### Incomplete Proof (`sorry`)\nClose every sorry.\n"
            "Remaining goals:\n" + "\n".join(blocks) + "\n\n"
            "Tactics: `omega` · `ring` · `linarith` · `norm_num` · `decide` · `simp [*]`"
        )
    if not parts:
        parts.append("#### Proof Incomplete\nVerification failed.")

    return (
        f"### Compiler Feedback — Iteration {iteration}/{MAX_ITER}\n\n"
        + "\n\n".join(parts)
        + f"\n\n---\n### Previous Proof\n```lean\n{lean_code}\n```\n\n"
        "Fix ALL issues:\n"
        "===BEGIN LEAN PROOF===\n<corrected proof>\n===END LEAN PROOF==="
    )


def no_delimiter_msg(iteration: int) -> str:
    return (
        f"### Iteration {iteration} — Missing Delimiters\n"
        "Wrap your entire proof exactly like this:\n"
        "===BEGIN LEAN PROOF===\nimport Mathlib\n-- proof\n===END LEAN PROOF==="
    )

# ==============================================================================
# NEGATIVE PROMPTS  (new — spec-type aware)
# ==============================================================================

NEGATIVE_SYSTEM = (
    "You are an expert Lean 4 programmer specialising in finding counterexamples "
    "and DISPROVING program properties.\n\n"
    "Your goal is to show that the given C program VIOLATES the specification "
    "by exhibiting a CONCRETE COUNTEREXAMPLE or VIOLATION PROOF in Lean 4.\n\n"
    "Your final answer MUST be placed inside these delimiters:\n"
    "===BEGIN LEAN PROOF===\n"
    "<your complete Lean 4 source here>\n"
    "===END LEAN PROOF===\n\n"
    "Hard rules:\n"
    "  • The proof must start with `import Mathlib`.\n"
    "  • Do NOT use `sorry`.\n"
    "  • Show a CONCRETE VIOLATION with specific witness values.\n"
    "  • The proof must compile and verify in Lean 4."
)

_NEGATION_APPROACH = {
    "overflow": """\
### What to Prove
Show this C program DOES cause signed integer overflow for at least one input.

### Approach
1. Identify which arithmetic operation could overflow (e.g. addition, subtraction, n++)
2. Find CONCRETE integer values for the input variables that trigger the overflow
3. Show the result exceeds INT_MAX (2147483647) or is below INT_MIN (-2147483648)

### Lean 4 Structure
```lean4
import Mathlib

def INT_MIN : Int := -2147483648
def INT_MAX : Int :=  2147483647

-- Substitute your concrete witness values below
theorem overflow_witness :
    -- example: with input a = X, the computation a + Y overflows
    (WITNESS_VALUE : Int) + (ADDEND : Int) > INT_MAX := by
  norm_num
```
Use norm_num or decide to close the arithmetic goal.
Adjust the theorem to match the ACTUAL arithmetic in the C code above.""",

    "unreach-call": """\
### What to Prove
Show reach_error() CAN be called — find a concrete execution path to it.

### Approach
1. Read the C code and find the guard condition protecting reach_error()
2. Find CONCRETE input values satisfying that guard condition
3. Show those values make the guard true in Lean 4

### Lean 4 Structure
```lean4
import Mathlib

-- Concrete witness: with input x = WITNESS, the guard is satisfied
-- and execution reaches reach_error()
theorem reach_error_reachable :
    -- e.g. if guard is (x < 0), show your witness satisfies it
    (WITNESS_VALUE : Int) < 0 := by
  norm_num
```
Replace WITNESS_VALUE and the condition with the ACTUAL guard from the C code.""",

    "termination": """\
### What to Prove
Show this C program can fail to terminate — find inputs making the loop run ≥ 10000 steps.

### Approach
1. Model the loop body as a Lean 4 function with a fuel/step counter
2. Find CONCRETE initial values for the loop variables
3. Show the loop still has x >= 0 after 10000 iterations with those inputs

### Lean 4 Structure
```lean4
import Mathlib

-- Model one loop iteration
def loopStep (x y : Int) : Int × Int := (x + y, (-2) * y - 1)

-- Apply N iterations
def iterate : Nat → Int × Int → Int × Int
  | 0,     state => state
  | n + 1, state => iterate n (loopStep state.1 state.2)

-- Show concrete initial values keep the loop alive for 10000 steps
-- (i.e. x >= 0 after 10000 iterations applied to witness values)
theorem non_termination_witness :
    (iterate 10000 (INIT_X, INIT_Y)).1 ≥ 0 := by
  decide   -- or norm_num if decide is too slow
```
Replace INIT_X and INIT_Y with concrete integer values from the C code's input range.""",

    "unknown": """\
### What to Prove
Find a concrete input for which the C program violates its specification.
Exhibit the violation as a Lean 4 proof with concrete witness values.""",
}


def negative_initial_user(c_code: str, neg_prop_nl: str, spec_type: str) -> str:
    approach = _NEGATION_APPROACH.get(spec_type, _NEGATION_APPROACH["unknown"])
    return (
        f"### Task\n{neg_prop_nl}\n\n"
        f"### C Code\n```c\n{c_code}\n```\n\n"
        f"{approach}\n\n"
        "Write your complete, sorry-free violation proof between the delimiters:\n"
        "===BEGIN LEAN PROOF===\n"
        "import Mathlib\n"
        "-- concrete violation proof here\n"
        "===END LEAN PROOF==="
    )


def negative_repair_user(lean_code: str, result: dict, iteration: int) -> str:
    """Same structure as positive repair, with a reminder about the negation goal."""
    parts = []
    if result.get("errors"):
        rows = [
            f"  Line {e['pos']['line']}, Col {e['pos']['column']}: {e['data']}"
            for e in result["errors"]
        ]
        parts.append("#### Compilation Errors\n```\n" + "\n".join(rows) + "\n```")
    if result.get("sorries"):
        blocks = [
            f"  Line {s.get('pos',{}).get('line','?')}: {s.get('goal','')}"
            for s in result["sorries"]
        ]
        parts.append(
            "#### Incomplete Proof (`sorry`)\nClose every sorry.\n"
            "Remaining goals:\n" + "\n".join(blocks) + "\n\n"
            "Tactics: `norm_num` · `decide` · `omega` · `native_decide`"
        )
    if not parts:
        parts.append("#### Proof Incomplete\nVerification failed.")

    return (
        f"### Compiler Feedback — Iteration {iteration}/{MAX_ITER}\n\n"
        + "\n\n".join(parts)
        + f"\n\n---\n### Previous Proof\n```lean\n{lean_code}\n```\n\n"
        "**REMINDER**: You are proving a VIOLATION. You need concrete witness values "
        "that demonstrate the property is broken. Try simpler/smaller witness values "
        "if the current proof is too complex.\n\n"
        "Fix ALL issues:\n"
        "===BEGIN LEAN PROOF===\n<corrected violation proof>\n===END LEAN PROOF==="
    )

# ==============================================================================
# BLEU VALIDATION
# ==============================================================================

def validate_lean_with_bleu(lean_code: str, original_c: str, client: OpenAI) -> dict:
    try:
        resp = client.chat.completions.create(
            model=VALIDATION_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Given this Lean 4 proof, reconstruct the original C program it relates to.\n"
                    "Output ONLY valid C code, no explanation.\n\n"
                    f"```lean4\n{lean_code}\n```"
                )
            }],
            temperature=0.0,
        )
        regen  = resp.choices[0].message.content.strip()
        c_m    = re.search(r"```(?:c|cpp)?\s*\n(.*?)```", regen, re.DOTALL)
        regen  = c_m.group(1).strip() if c_m else regen
        smooth = SmoothingFunction().method1
        bleu   = sentence_bleu([original_c.split()], regen.split(), smoothing_function=smooth)
        return {"bleu_score": round(bleu * 100, 2), "regenerated_c": regen, "bleu_ok": True}
    except Exception as exc:
        return {"bleu_score": -1.0, "regenerated_c": "", "bleu_ok": False,
                "bleu_error": str(exc)}

# ==============================================================================
# GENERIC REPAIR LOOP  (shared by positive and negative pipelines)
# ==============================================================================

def _run_repair_loop(
    model: str,
    temperature: float,
    seed: int,
    client: OpenAI,
    system_prompt: str,
    initial_user_msg: str,
    repair_fn,          # callable(lean_code, result, iteration) -> str
    no_delim_fn,        # callable(iteration) -> str
) -> dict:
    """
    Core repair loop used by both pipelines.
    Prompts differ; loop structure is identical.
    """
    t0         = time.time()
    tokens_in  = 0
    tokens_out = 0
    iter_log   = []
    success    = False
    success_it = -1
    final_code = ""
    feedback   = ""

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": initial_user_msg},
    ]

    for it in range(1, MAX_ITER + 1):
        if feedback:
            history.append({"role": "user", "content": feedback})
            feedback = ""

        rec = {
            "iteration"  : it,
            "extracted"  : False,
            "pass"       : False,
            "complete"   : False,
            "n_errors"   : 0,
            "n_sorries"  : 0,
            "tokens_in"  : 0,
            "tokens_out" : 0,
            "verify_time": 0.0,
            "iter_time"  : 0.0,
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
                feedback         = no_delim_fn(it)
                rec["extracted"] = False
                rec["iter_time"] = round(time.time() - it_t0, 2)
                iter_log.append(rec)
                continue

            rec["extracted"] = True
            lean = rule_based_fixes(lean)

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

            feedback = repair_fn(lean, v, it)

        except Exception as exc:
            rec["error"]   = str(exc)
            feedback = (
                f"### System Error at Iteration {it}\n```\n{exc}\n```\n"
                "Please provide a fresh attempt."
            )

        rec["iter_time"] = round(time.time() - it_t0, 2)
        iter_log.append(rec)

    return {
        "success"     : success,
        "success_it"  : success_it,
        "total_iters" : len(iter_log),
        "tokens_in"   : tokens_in,
        "tokens_out"  : tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "duration_sec": round(time.time() - t0, 2),
        "final_code"  : final_code,
        "iter_log"    : iter_log,
    }

# ==============================================================================
# POSITIVE PIPELINE
# ==============================================================================

def run_positive_pipeline(
    c_code: str,
    prop_nl: str,
    model: str,
    temperature: float,
    seed: int,
    client: OpenAI,
) -> dict:
    return _run_repair_loop(
        model=model, temperature=temperature, seed=seed, client=client,
        system_prompt    = POSITIVE_SYSTEM,
        initial_user_msg = positive_initial_user(c_code, prop_nl),
        repair_fn        = positive_repair_user,
        no_delim_fn      = no_delimiter_msg,
    )

# ==============================================================================
# NEGATIVE PIPELINE
# ==============================================================================

def run_negative_pipeline(
    c_code: str,
    prop_nl: str,
    spec_type: str,
    model: str,
    temperature: float,
    seed: int,
    client: OpenAI,
) -> dict:
    neg_prop_nl = negate_property(prop_nl, spec_type)
    return _run_repair_loop(
        model=model, temperature=temperature, seed=seed, client=client,
        system_prompt    = NEGATIVE_SYSTEM,
        initial_user_msg = negative_initial_user(c_code, neg_prop_nl, spec_type),
        repair_fn        = negative_repair_user,
        no_delim_fn      = no_delimiter_msg,
    )

# ==============================================================================
# PREDICTION LOGIC
# ==============================================================================

def determine_prediction(mode: str, pos_result: dict, neg_result: dict) -> bool:
    """
    mode = "negative":
        neg proves violation → False  (confirmed violation)
        neg fails            → True   (couldn't disprove → assume it holds)

    mode = "dual":
        neg proves           → False  (trust negative, even if pos also proved)
        pos proves, neg fails→ True
        both fail            → False  (conservative default)
    """
    if mode == "negative":
        return not neg_result["success"]  # True = "couldn't disprove"

    if mode == "dual":
        if neg_result["success"]:
            return False
        if pos_result["success"]:
            return True
        return False  # both failed → default False

    raise ValueError(f"Unknown mode: {mode}")

# ==============================================================================
# EMPTY RESULT PLACEHOLDER
# ==============================================================================

def _empty_result() -> dict:
    """Placeholder for the unused pipeline in mode='negative'."""
    return {
        "success"     : None,
        "success_it"  : None,
        "total_iters" : None,
        "tokens_in"   : None,
        "tokens_out"  : None,
        "total_tokens": None,
        "duration_sec": None,
        "final_code"  : None,
        "iter_log"    : [],
        "bleu_score"  : None,
        "regenerated_c": None,
        "bleu_ok"     : None,
    }

# ==============================================================================
# CORE DISPATCHER  (one mode × model × temperature × seed)
# ==============================================================================

def run_single(
    c_code: str,
    prop_nl: str,
    spec_type: str,
    mode: str,
    model: str,
    temperature: float,
    seed: int,
) -> dict:
    """
    Dispatches to the correct pipeline(s) based on mode.
    Returns a unified result dict with pos_* and neg_* prefixed columns.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    t0 = time.time()

    if mode == "negative":
        # ── Negative only ──────────────────────────────────────────────────────
        neg = run_negative_pipeline(c_code, prop_nl, spec_type, model, temperature, seed, client)
        pos = _empty_result()
        prediction = determine_prediction("negative", pos, neg)

    elif mode == "dual":
        # ── Both in parallel, full 10 iterations each ──────────────────────────
        with ThreadPoolExecutor(max_workers=2) as ex:
            pos_fut = ex.submit(
                run_positive_pipeline,
                c_code, prop_nl, model, temperature, seed, client
            )
            neg_fut = ex.submit(
                run_negative_pipeline,
                c_code, prop_nl, spec_type, model, temperature, seed, client
            )
            pos = pos_fut.result()
            neg = neg_fut.result()
        prediction = determine_prediction("dual", pos, neg)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ── BLEU validation (run after loop, only when pipeline succeeded) ─────────
    if pos.get("success") and pos.get("final_code"):
        pos.update(validate_lean_with_bleu(pos["final_code"], c_code, client))
    else:
        pos.update({"bleu_score": None, "regenerated_c": None, "bleu_ok": None})

    if neg.get("success") and neg.get("final_code"):
        neg.update(validate_lean_with_bleu(neg["final_code"], c_code, client))
    else:
        neg.update({"bleu_score": None, "regenerated_c": None, "bleu_ok": None})

    return {
        # ── Metadata ─────────────────────────────────────────────────────────
        "mode"        : mode,
        "model"       : model,
        "temperature" : temperature,
        "seed"        : seed,
        "spec_type"   : spec_type,
        "prediction"  : prediction,
        # ── Positive pipeline ─────────────────────────────────────────────────
        "pos_success"      : pos["success"],
        "pos_success_it"   : pos["success_it"],
        "pos_total_iters"  : pos["total_iters"],
        "pos_tokens_in"    : pos["tokens_in"],
        "pos_tokens_out"   : pos["tokens_out"],
        "pos_total_tokens" : pos["total_tokens"],
        "pos_duration_sec" : pos["duration_sec"],
        "pos_final_code"   : pos["final_code"],
        "pos_bleu_score"   : pos.get("bleu_score"),
        "pos_regenerated_c": pos.get("regenerated_c"),
        "pos_iter_log"     : json.dumps(pos["iter_log"]),
        # ── Negative pipeline ─────────────────────────────────────────────────
        "neg_success"      : neg["success"],
        "neg_success_it"   : neg["success_it"],
        "neg_total_iters"  : neg["total_iters"],
        "neg_tokens_in"    : neg["tokens_in"],
        "neg_tokens_out"   : neg["tokens_out"],
        "neg_total_tokens" : neg["total_tokens"],
        "neg_duration_sec" : neg["duration_sec"],
        "neg_final_code"   : neg["final_code"],
        "neg_bleu_score"   : neg.get("bleu_score"),
        "neg_regenerated_c": neg.get("regenerated_c"),
        "neg_iter_log"     : json.dumps(neg["iter_log"]),
        # ── Totals ────────────────────────────────────────────────────────────
        "total_tokens"     : (pos.get("total_tokens") or 0) + (neg.get("total_tokens") or 0),
        "total_duration_sec": round(time.time() - t0, 2),
    }

# ==============================================================================
# SAMPLE PROCESSOR
# ==============================================================================

def process_sample(item) -> list[dict]:
    """
    For one CSV row, run all (mode × model × temperature × seed) combos in parallel.
    Returns one result dict per combo.
    """
    idx, row = item
    try:
        with open(row["c_file_abs"]) as f:
            c_code = f.read()
        with open(f"{PROP_BASE}{row['property']}.prp") as fh:
            prop_raw = fh.read()
    except Exception as exc:
        tqdm.write(f"  [SKIP] idx={idx}: {exc}")
        return []

    prop_nl   = paraphrase_property(prop_raw)
    spec_type = detect_spec_type(prop_raw)

    combos = [
        (mode, model, temp, seed)
        for mode  in MODES
        for model in DIRECT_MODELS
        for temp  in TEMPERATURES
        for seed  in SEEDS
    ]

    results = []
    with ThreadPoolExecutor(max_workers=min(len(combos), 12)) as ex:
        futs = {
            ex.submit(run_single, c_code, prop_nl, spec_type, mo, mdl, t, s): (mo, mdl, t, s)
            for mo, mdl, t, s in combos
        }
        for fut in as_completed(futs):
            mo, mdl, t, s = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {
                    "mode": mo, "model": mdl, "temperature": t, "seed": s,
                    "spec_type": spec_type, "prediction": False,
                    "pos_success": None, "pos_success_it": None,
                    "pos_total_iters": None, "pos_tokens_in": None,
                    "pos_tokens_out": None, "pos_total_tokens": None,
                    "pos_duration_sec": None, "pos_final_code": None,
                    "pos_bleu_score": None, "pos_regenerated_c": None,
                    "pos_iter_log": "[]",
                    "neg_success": None, "neg_success_it": None,
                    "neg_total_iters": None, "neg_tokens_in": None,
                    "neg_tokens_out": None, "neg_total_tokens": None,
                    "neg_duration_sec": None, "neg_final_code": None,
                    "neg_bleu_score": None, "neg_regenerated_c": None,
                    "neg_iter_log": "[]",
                    "total_tokens": None, "total_duration_sec": None,
                    "error": str(exc),
                }

            r["sample_idx"]       = idx
            r["c_file"]           = row.get("c_file_abs", "")
            r["property"]         = row.get("property", "")
            r["expected_verdict"] = row.get("expected_verdict", "")
            results.append(r)

    return results

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)

    if TEST_MODE:
        df = df.query("expected_verdict == True").sample(TEST_N, random_state=42)
        print(f"[TEST MODE] {TEST_N} samples")

    samples   = list(df.iterrows())
    n_combos  = len(MODES) * len(DIRECT_MODELS) * len(TEMPERATURES) * len(SEEDS)
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.csv")

    print(f"[INFO] Modes          : {MODES}")
    print(f"[INFO] Models         : {DIRECT_MODELS}")
    print(f"[INFO] Temperatures   : {TEMPERATURES}")
    print(f"[INFO] Seeds          : {SEEDS}")
    print(f"[INFO] Max iter each  : {MAX_ITER}  "
          f"(dual = up to {MAX_ITER * 2} total per combo)")
    print(
        f"[INFO] {len(samples)} samples × {n_combos} combos = "
        f"{len(samples) * n_combos} total runs"
    )
    print(f"[INFO] Output → {OUTPUT_DIR}\n")

    all_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_sample, s): s for s in samples}
        for fut in tqdm(as_completed(futs), total=len(samples),
                        desc="Samples", unit="sample"):
            rows = fut.result()
            all_rows.extend(rows)
            if len(all_rows) % (5 * n_combos) == 0 and all_rows:
                pd.DataFrame(all_rows).to_csv(ckpt_path, index=False)
                tqdm.write(f"  [ckpt] {len(all_rows)} rows saved")

    out = os.path.join(OUTPUT_DIR, "final_results.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\n[Done] {(time.time()-t0)/3600:.2f}h | {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()
# ```

# ---

# ## Summary of What Was Built

# # ```
# # _run_repair_loop()          ← single generic loop, takes prompt fns as args
# #        │
# #        ├── run_positive_pipeline()   uses POSITIVE_SYSTEM + positive_* prompts
# #        │
# #        └── run_negative_pipeline()   uses NEGATIVE_SYSTEM + negative_* prompts
# #                                      spec-type-aware approach per violation type

# # run_single(mode=...)
# #        │
# #        ├── "negative"  → run_negative only  → determine_prediction → True/False
# #        │
# #        └── "dual"      → ThreadPoolExecutor(2) runs both in parallel
# #                           both run FULL 10 iterations independently
# #                           no cancellation → full logs captured for analysis
# #                           → determine_prediction → priority: neg > pos > False
# # ```

# | Column group | present in `negative` | present in `dual` |
# |---|---|---|
# | `pos_*` | `None` (not run) | ✅ filled |
# | `neg_*` | ✅ filled | ✅ filled |
# | `pos_bleu_score` | `None` | ✅ if pos succeeded |
# | `neg_bleu_score` | ✅ if neg succeeded | ✅ if neg succeeded |
# | `prediction` | `not neg_success` | priority table |