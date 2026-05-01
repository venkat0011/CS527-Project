#!/usr/bin/env python3
"""
pipeline_neg_dual.py

MODEL       : deepseek/deepseek-v4-flash  (via OpenRouter → SiliconFlow)
TEMPERATURE : 0.2
MODES       : negative | dual

Key optimisations over base version:
  • include_reasoning=False  — strips <think> tokens from output (reasoning
                               still happens server-side, quality preserved,
                               tokens_out drops from ~50k to ~2k per call)
  • Provider routing          — forces SiliconFlow (68 tps) not Parasail (20 tps)
  • Streaming + early stop    — connection closed the moment ===END LEAN PROOF===
                               appears; no waiting for trailing text
  • client timeout=120s       — hard limit per HTTP call, prevents silent hangs
  • Full 10-iteration context — no history trimming, all repair rounds in context

MODE "negative" : Prove the spec is VIOLATED.
                  Predicts False if negation proves, True if negation fails.

MODE "dual"     : Run positive AND negative sequentially (no nested pool).
                  Both run to full MAX_ITER — no early exit — full research logs.
                  Priority: neg proves → False
                            pos proves (neg failed) → True
                            both fail → False (conservative default)

Single output CSV with a `mode` column:
    df[df["mode"] == "negative"]
    df[df["mode"] == "dual"]
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

MODEL       = "deepseek/deepseek-v4-flash"
TEMPERATURE = 0.2

VALIDATION_MODEL = "openai/gpt-5-mini"

MODES = ["negative", "dual"]
SEEDS = [42, 123, 999]

MAX_ITER       = 10
VERIFY_TIMEOUT = 120
MAX_WORKERS    = 4

CSV_PATH   = "/workspace/CS527-Project/sample_df.csv"
PROP_BASE  = "/workspace/sv-benchmarks/c/properties/"
OUTPUT_DIR = "/workspace/CS527-Project/results/final/pipeline_neg_dual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_MODE = False
TEST_N    = 2

# ==============================================================================
# DEEPSEEK-SPECIFIC API CONFIGURATION
#
# include_reasoning=False:
#   DeepSeek V4 Flash is a hybrid reasoning model. By default OpenRouter
#   includes the <think>...</think> block in the response — this can be
#   50,000-600,000 tokens per call, making each call take 10-60 minutes.
#   Setting include_reasoning=False strips the think block from the OUTPUT.
#   The model still reasons internally — quality is preserved.
#   tokens_out drops from ~50k → ~2k per call.
#
# Provider order:
#   SiliconFlow = 68 tps  (fastest)
#   NovitaAI   = 63 tps
#   AtlasCloud  = 63 tps
#   Parasail    = 20 tps  ← OpenRouter default can land here → 3min/call
#   AkashML    = 13 tps  ← worst case → 1hr/call
#
# max_tokens=None:
#   Let streaming early-stop handle termination.
#   Setting 32k would truncate reasoning before the proof appears.
# ==============================================================================

DEEPSEEK_EXTRA_BODY = {
    "provider": {
        "order": ["SiliconFlow", "NovitaAI", "AtlasCloud"],
        "allow_fallbacks": True,
    },
    "include_reasoning": False,
}

END_MARKER = "===END LEAN PROOF==="

# ==============================================================================
# SPEC TYPE DETECTION
# ==============================================================================

def detect_spec_type(prop_text: str) -> str:
    p = prop_text.lower()
    if "overflow"    in p:                        return "overflow"
    if "reach_error" in p or "unreach-call" in p: return "unreach-call"
    if "termination" in p or "ltl(f end)"   in p: return "termination"
    return "unknown"

# ==============================================================================
# PROPERTY PARAPHRASING
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


def negate_property(spec_type: str) -> str:
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
            "Disprove termination: show this C program can fail to terminate — "
            "find concrete inputs making the loop run at least 10,000 iterations."
        )
    return "Disprove the specification by finding a concrete counterexample."

# ==============================================================================
# LEAN EXTRACTION + FIXES
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
# POSITIVE PROMPTS
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
        "===BEGIN LEAN PROOF===\nimport Mathlib\n-- proof\n===END LEAN PROOF==="
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
            "Goals:\n" + "\n".join(blocks) + "\n\n"
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

# ==============================================================================
# NEGATIVE PROMPTS
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
1. Identify which arithmetic operation could overflow
2. Find CONCRETE integer values that trigger the overflow
3. Show the result exceeds INT_MAX (2147483647) or is below INT_MIN (-2147483648)

### Lean 4 Structure
```lean4
import Mathlib
def INT_MIN : Int := -2147483648
def INT_MAX : Int :=  2147483647
-- Replace WITNESS_VALUE and ADDEND with actual values from the C code
theorem overflow_witness :
    (WITNESS_VALUE : Int) + (ADDEND : Int) > INT_MAX := by norm_num
```""",

    "unreach-call": """\
### What to Prove
Show reach_error() CAN be called — find a concrete execution path to it.

### Approach
1. Find the guard condition protecting reach_error() in the C code
2. Find CONCRETE input values satisfying that guard
3. Prove those values make the guard true in Lean 4

### Lean 4 Structure
```lean4
import Mathlib
-- Replace WITNESS_VALUE and the condition with the ACTUAL guard from the C code
theorem reach_error_reachable :
    (WITNESS_VALUE : Int) < 0 := by norm_num
```""",

    "termination": """\
### What to Prove
Show this C program can fail to terminate — find inputs making the loop run ≥ 10000 steps.

### Approach
1. Model the loop body as a Lean 4 function
2. Find CONCRETE initial values for the loop variables
3. Show the loop condition stays true after 10000 iterations

### Lean 4 Structure
```lean4
import Mathlib
def loopStep (x y : Int) : Int × Int := (x + y, (-2) * y - 1)
def iterate : Nat → Int × Int → Int × Int
  | 0,     s => s
  | n + 1, s => iterate n (loopStep s.1 s.2)
-- Replace INIT_X and INIT_Y with concrete values
theorem non_termination_witness :
    (iterate 10000 (INIT_X, INIT_Y)).1 ≥ 0 := by decide
```""",

    "unknown": "Find a concrete input violating the specification. Exhibit it in Lean 4.",
}


def negative_initial_user(c_code: str, neg_prop_nl: str, spec_type: str) -> str:
    approach = _NEGATION_APPROACH.get(spec_type, _NEGATION_APPROACH["unknown"])
    return (
        f"### Task\n{neg_prop_nl}\n\n"
        f"### C Code\n```c\n{c_code}\n```\n\n"
        f"{approach}\n\n"
        "Write your complete, sorry-free violation proof between the delimiters:\n"
        "===BEGIN LEAN PROOF===\nimport Mathlib\n-- violation proof\n===END LEAN PROOF==="
    )


def negative_repair_user(lean_code: str, result: dict, iteration: int) -> str:
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
            "Goals:\n" + "\n".join(blocks) + "\n\n"
            "Tactics: `norm_num` · `decide` · `omega` · `native_decide`"
        )
    if not parts:
        parts.append("#### Proof Incomplete\nVerification failed.")
    return (
        f"### Compiler Feedback — Iteration {iteration}/{MAX_ITER}\n\n"
        + "\n\n".join(parts)
        + f"\n\n---\n### Previous Proof\n```lean\n{lean_code}\n```\n\n"
        "**REMINDER**: Prove a VIOLATION — exhibit concrete witness values.\n"
        "Try simpler/smaller witnesses if the proof is getting complex.\n\n"
        "Fix ALL issues:\n"
        "===BEGIN LEAN PROOF===\n<corrected violation proof>\n===END LEAN PROOF==="
    )


def no_delimiter_msg(iteration: int) -> str:
    return (
        f"### Iteration {iteration} — Missing Delimiters\n"
        "Wrap your proof exactly like this:\n"
        "===BEGIN LEAN PROOF===\nimport Mathlib\n-- proof\n===END LEAN PROOF==="
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
        regen = resp.choices[0].message.content.strip()
        c_m   = re.search(r"```(?:c|cpp)?\s*\n(.*?)```", regen, re.DOTALL)
        regen = c_m.group(1).strip() if c_m else regen
        bleu  = sentence_bleu(
            [original_c.split()], regen.split(),
            smoothing_function=SmoothingFunction().method1
        )
        return {"bleu_score": round(bleu * 100, 2), "regenerated_c": regen, "bleu_ok": True}
    except Exception as exc:
        return {"bleu_score": -1.0, "regenerated_c": "", "bleu_ok": False,
                "bleu_error": str(exc)}

# ==============================================================================
# STREAMING API CALL
#
# Streams tokens and closes the connection the moment END_MARKER appears.
# This avoids waiting for any text the model generates after the proof block.
#
# For DeepSeek with include_reasoning=False:
#   - <think> tokens stripped from output (still generated server-side)
#   - Only the proof text is streamed to us
#   - tokens_out ≈ 1k-5k instead of 50k-600k
#   - Per-call time drops from ~60min (Parasail, no routing) to ~30-90s
# ==============================================================================

def _call_api(client: OpenAI, messages: list) -> tuple[str, int, int]:
    """
    Stream the response, closing connection at END_MARKER.
    Returns (full_text_up_to_end_marker, prompt_tokens, completion_tokens).
    """
    content       = ""
    prompt_tokens = 0
    comp_tokens   = 0

    with client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        stream=True,
        # No max_tokens — streaming early-stop handles termination.
        # Setting a hard limit risks cutting off reasoning before the proof appears.
        extra_body=DEEPSEEK_EXTRA_BODY,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            content += delta

            # Capture usage metadata when the server sends it
            # (usually in the final chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                comp_tokens   = chunk.usage.completion_tokens

            # Stop streaming the moment the proof closing delimiter appears
            if END_MARKER in content:
                break

    # Fallback token estimate when the stream ended before the usage chunk arrived
    if comp_tokens == 0:
        comp_tokens = max(1, len(content) // 4)

    return content, prompt_tokens, comp_tokens

# ==============================================================================
# GENERIC REPAIR LOOP  (shared by positive and negative — only prompts differ)
# Full 10-iteration context preserved — no history trimming.
# ==============================================================================

def _run_repair_loop(
    client: OpenAI,
    system_prompt: str,
    initial_user_msg: str,
    repair_fn,       # (lean_code, result, iteration) -> str
    no_delim_fn,     # (iteration) -> str
) -> dict:
    t0         = time.time()
    tokens_in  = tokens_out = 0
    iter_log   = []
    success    = False
    success_it = -1
    final_code = ""
    feedback   = ""

    # Full history — all 10 iterations stay in context (no trimming)
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": initial_user_msg},
    ]

    for it in range(1, MAX_ITER + 1):
        if feedback:
            history.append({"role": "user", "content": feedback})
            feedback = ""

        rec   = {"iteration": it, "extracted": False, "pass": False,
                 "complete": False, "n_errors": 0, "n_sorries": 0,
                 "tokens_in": 0, "tokens_out": 0,
                 "verify_time": 0.0, "iter_time": 0.0}
        it_t0 = time.time()

        try:
            # Streaming call — stops at END_MARKER, routes to SiliconFlow,
            # strips reasoning tokens from output
            raw, tok_in, tok_out = _call_api(client, history)

            rec["tokens_in"]  = tok_in
            rec["tokens_out"] = tok_out
            tokens_in        += tok_in
            tokens_out       += tok_out

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
            rec["error"] = str(exc)
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
# NAMED PIPELINES
# ==============================================================================

def run_positive_pipeline(c_code: str, prop_nl: str, client: OpenAI) -> dict:
    return _run_repair_loop(
        client           = client,
        system_prompt    = POSITIVE_SYSTEM,
        initial_user_msg = positive_initial_user(c_code, prop_nl),
        repair_fn        = positive_repair_user,
        no_delim_fn      = no_delimiter_msg,
    )


def run_negative_pipeline(c_code: str, prop_nl: str, spec_type: str,
                           client: OpenAI) -> dict:
    neg_nl = negate_property(spec_type)
    return _run_repair_loop(
        client           = client,
        system_prompt    = NEGATIVE_SYSTEM,
        initial_user_msg = negative_initial_user(c_code, neg_nl, spec_type),
        repair_fn        = negative_repair_user,
        no_delim_fn      = no_delimiter_msg,
    )

# ==============================================================================
# PREDICTION LOGIC
# ==============================================================================

def determine_prediction(mode: str, pos: dict, neg: dict) -> bool:
    if mode == "negative":
        return not neg["success"]   # True = couldn't disprove
    if mode == "dual":
        if neg["success"]:  return False
        if pos["success"]:  return True
        return False                # both failed → conservative default
    raise ValueError(f"Unknown mode: {mode}")


def _empty_result() -> dict:
    return {k: None for k in [
        "success", "success_it", "total_iters",
        "tokens_in", "tokens_out", "total_tokens",
        "duration_sec", "final_code", "iter_log",
        "bleu_score", "regenerated_c", "bleu_ok",
    ]}

# ==============================================================================
# CORE DISPATCHER  (one mode × one seed)
# ==============================================================================

def run_single(
    c_code: str,
    prop_nl: str,
    spec_type: str,
    mode: str,
    seed: int,
) -> dict:
    # Separate client per call — thread-safe, 120s hard timeout per HTTP request
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    t0 = time.time()

    if mode == "negative":
        neg = run_negative_pipeline(c_code, prop_nl, spec_type, client)
        pos = _empty_result()

    elif mode == "dual":
        # Sequential — avoids nested thread pool deadlock.
        # Positive first, then negative.
        # Outer flat pool handles sample-level parallelism.
        pos = run_positive_pipeline(c_code, prop_nl, client)
        neg = run_negative_pipeline(c_code, prop_nl, spec_type, client)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    prediction = determine_prediction(mode, pos, neg)

    if pos.get("success") and pos.get("final_code"):
        pos.update(validate_lean_with_bleu(pos["final_code"], c_code, client))
    if neg.get("success") and neg.get("final_code"):
        neg.update(validate_lean_with_bleu(neg["final_code"], c_code, client))

    def _jlog(d, k):
        v = d.get(k)
        return json.dumps(v) if isinstance(v, list) else v

    return {
        "mode"              : mode,
        "model"             : MODEL,
        "temperature"       : TEMPERATURE,
        "seed"              : seed,
        "spec_type"         : spec_type,
        "prediction"        : prediction,
        "pos_success"       : pos.get("success"),
        "pos_success_it"    : pos.get("success_it"),
        "pos_total_iters"   : pos.get("total_iters"),
        "pos_tokens_in"     : pos.get("tokens_in"),
        "pos_tokens_out"    : pos.get("tokens_out"),
        "pos_total_tokens"  : pos.get("total_tokens"),
        "pos_duration_sec"  : pos.get("duration_sec"),
        "pos_final_code"    : pos.get("final_code"),
        "pos_bleu_score"    : pos.get("bleu_score"),
        "pos_regenerated_c" : pos.get("regenerated_c"),
        "pos_iter_log"      : json.dumps(_jlog(pos, "iter_log") or []),
        "neg_success"       : neg.get("success"),
        "neg_success_it"    : neg.get("success_it"),
        "neg_total_iters"   : neg.get("total_iters"),
        "neg_tokens_in"     : neg.get("tokens_in"),
        "neg_tokens_out"    : neg.get("tokens_out"),
        "neg_total_tokens"  : neg.get("total_tokens"),
        "neg_duration_sec"  : neg.get("duration_sec"),
        "neg_final_code"    : neg.get("final_code"),
        "neg_bleu_score"    : neg.get("bleu_score"),
        "neg_regenerated_c" : neg.get("regenerated_c"),
        "neg_iter_log"      : json.dumps(_jlog(neg, "iter_log") or []),
        "total_tokens"      : (pos.get("total_tokens") or 0) +
                              (neg.get("total_tokens") or 0),
        "total_duration_sec": round(time.time() - t0, 2),
    }

# ==============================================================================
# FLAT COMBO RUNNER  (no nested thread pools)
# ==============================================================================

def run_one_combo(idx: int, row: pd.Series, mode: str, seed: int) -> dict | None:
    try:
        with open(row["c_file_abs"]) as f:
            c_code = f.read()
        with open(f"{PROP_BASE}{row['property']}.prp") as fh:
            prop_raw = fh.read()
    except Exception as exc:
        tqdm.write(f"  [SKIP] idx={idx} {mode} seed={seed}: {exc}")
        return None

    prop_nl   = paraphrase_property(prop_raw)
    spec_type = detect_spec_type(prop_raw)

    try:
        r = run_single(c_code, prop_nl, spec_type, mode, seed)
    except Exception as exc:
        tqdm.write(f"  [ERROR] idx={idx} {mode} seed={seed}: {exc}")
        r = {
            "mode": mode, "model": MODEL, "temperature": TEMPERATURE,
            "seed": seed, "spec_type": spec_type, "prediction": False,
            **{f"pos_{k}": None for k in [
                "success","success_it","total_iters","tokens_in",
                "tokens_out","total_tokens","duration_sec","final_code",
                "bleu_score","regenerated_c","iter_log"]},
            **{f"neg_{k}": None for k in [
                "success","success_it","total_iters","tokens_in",
                "tokens_out","total_tokens","duration_sec","final_code",
                "bleu_score","regenerated_c","iter_log"]},
            "total_tokens": None, "total_duration_sec": None,
            "error": str(exc),
        }

    r["sample_idx"]       = idx
    r["c_file"]           = row.get("c_file_abs", "")
    r["property"]         = row.get("property", "")
    r["expected_verdict"] = row.get("expected_verdict", "")
    return r

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)

    if TEST_MODE:
        a = df.query("expected_verdict == True").sample(1, random_state=42)
        b = df.query("expected_verdict == False").sample(1, random_state=42)
        df = pd.concat([a,b])
        print(f"[TEST MODE] {TEST_N} samples")

    samples = list(df.iterrows())

    # All combos upfront — flat, no nesting
    all_combos = [
        (idx, row, mode, seed)
        for idx, row in samples
        for mode in MODES
        for seed in SEEDS
    ]
    n_total   = len(all_combos)
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.csv")

    print(f"[INFO]  Model              : {MODEL}")
    print(f"[INFO]  Temperature        : {TEMPERATURE}")
    print(f"[INFO]  Provider routing   : SiliconFlow → NovitaAI → AtlasCloud")
    print(f"[INFO]  include_reasoning  : False  (think tokens stripped from output)")
    print(f"[INFO]  Streaming early stop at: {END_MARKER!r}")
    print(f"[INFO]  History            : full (no trimming, all {MAX_ITER} iters in context)")
    print(f"[INFO]  Modes              : {MODES}")
    print(f"[INFO]  Seeds              : {SEEDS}")
    print(f"[INFO]  Max iter each      : {MAX_ITER}  "
          f"(dual = {MAX_ITER} pos + {MAX_ITER} neg sequential)")
    print(f"[INFO]  Total combos       : {n_total}  "
          f"({len(samples)} samples × {len(MODES)} modes × {len(SEEDS)} seeds)")
    print(f"[INFO]  Workers            : {MAX_WORKERS}  (flat pool, no nested pools)")
    print(f"[INFO]  Output             : {OUTPUT_DIR}\n")

    all_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(run_one_combo, idx, row, mode, seed): (idx, mode, seed)
            for idx, row, mode, seed in all_combos
        }

        for fut in tqdm(as_completed(futs), total=n_total,
                        desc="Combos", unit="combo"):
            idx, mode, seed = futs[fut]
            try:
                r = fut.result(timeout=7200)   # 2hr hard cap per combo (20 iters × dual)
            except TimeoutError:
                tqdm.write(f"  [TIMEOUT] idx={idx} {mode} seed={seed} exceeded 2hr")
                r = None
            except Exception as exc:
                tqdm.write(f"  [FUTURE ERROR] idx={idx} {mode} seed={seed}: {exc}")
                r = None

            if r:
                all_rows.append(r)
                tqdm.write(
                    f"  ✓ idx={idx:4d} | {mode:8s} | seed={seed} | "
                    f"pred={r.get('prediction')} | "
                    f"pos_ok={r.get('pos_success')} | "
                    f"neg_ok={r.get('neg_success')} | "
                    f"tokens={r.get('total_tokens','?')} | "
                    f"time={r.get('total_duration_sec','?')}s"
                )

            if len(all_rows) % 10 == 0 and all_rows:
                pd.DataFrame(all_rows).to_csv(ckpt_path, index=False)

    out = os.path.join(OUTPUT_DIR, "final_results.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    elapsed = (time.time() - t0) / 3600
    print(f"\n[Done] {elapsed:.2f}h | {len(all_rows)} rows → {out}")

    df_out = pd.read_csv(out)
    print("\n── Quick Summary ────────────────────────────────────")
    for m in MODES:
        sub = df_out[df_out["mode"] == m]
        if len(sub) == 0:
            continue
        pct  = sub["prediction"].mean() * 100
        pos_ok = sub["pos_success"].sum() if "pos_success" in sub else "N/A"
        neg_ok = sub["neg_success"].sum()
        print(
            f"  {m:10s}: {len(sub):4d} rows | "
            f"predicted True={pct:.1f}% | "
            f"pos proved={pos_ok} | neg proved={neg_ok}"
        )


if __name__ == "__main__":
    main()