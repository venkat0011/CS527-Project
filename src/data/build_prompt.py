#!/usr/bin/env python3
"""
Prompt template for translating a C program + specification into Lean 4.
The LLM only does translation. Proof solving is handled separately by Apollo.

Usage:
    from prompt_template import build_prompt
    system_prompt, user_prompt = build_prompt(c_code, specification)
"""

SYSTEM_PROMPT = """\
You are an expert in formal verification and the Lean 4 theorem prover.
Your task is to translate a C program and its correctness specification into Lean 4.

You must generate exactly two things:
1. A Lean 4 function `main_lean` that faithfully models the behavior of the C program.
2. A Lean 4 theorem `main_spec` that states the correctness property from the specification.

Rules:
- Output ONLY a valid, self-contained Lean 4 file. No explanation, no markdown.
- Start the file with `import Mathlib`.
- The proof body must be exactly `by sorry`.
- Model C `int` as `Int`. Use `BitVec 32` if bit-level overflow behavior is relevant.
- Model C `unsigned int` as `BitVec 32`.
- Translate `__VERIFIER_nondet_int()` as a universally quantified variable in the theorem.
- Translate `__VERIFIER_assume(cond)` as a hypothesis in the theorem statement.
- Translate `__VERIFIER_error()` as the goal `False`.
""".strip()


def build_prompt(c_code: str, specification: str) -> tuple[str, str]:
    """
    Args:
        c_code:        Full C source code as a string.
        specification: The property specification as a string (contents of the .prp file).

    Returns:
        (system_prompt, user_prompt)
    """
    user_prompt = f"""\
Translate the following C program and its specification into Lean 4.

## Specification
```
{specification.strip()}
```

## C Source Code
```c
{c_code.strip()}
```

Generate a complete Lean 4 file with `import Mathlib`, a function `main_lean`, \
a theorem `main_spec`, and proof body `by sorry`.
Output ONLY the Lean 4 code."""

    return SYSTEM_PROMPT, user_prompt


if __name__ == "__main__":
    example_c = """
#include <assert.h>
void __VERIFIER_error() { return; }
int __VERIFIER_nondet_int();

int main() {
    int x = __VERIFIER_nondet_int();
    if (x > 0) {
        if (x < 0) {
            __VERIFIER_error();
        }
    }
    return 0;
}
"""
    example_spec = "CHECK( init(main()), LTL(G ! call(__VERIFIER_error())) )"

    system_p, user_p = build_prompt(example_c, example_spec)

    print("=== SYSTEM PROMPT ===")
    print(system_p)
    print("\n=== USER PROMPT ===")
    print(user_p)