import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
CSV_PATH = '/workspace/CS527-Project/c_sample.csv'
BASE_OUTPUT_DIR = '/workspace/CS527-Project/'
SEEDS = [42, 123, 999]
TEMPERATURES = [0.2]
PROPERTIES_BASE_PATH = '/workspace/sv-benchmarks/c/properties/'
BATCH_SIZE = 16  # Tune based on your GPU VRAM (lower if OOM)

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ✅ Critical: causal LMs must left-pad so all sequences end at the same position
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()


def build_prompt(c_code: str, property_text: str) -> str:
    """Build the full prompt string (without tokenizing)."""
    messages = [
        {
            "role": "system",
            "content": "You output only Lean 4 code. No English. No explanations. No self-correction."
        },
        {
            "role": "user",
            "content": (
                f"Translate this C code into a Lean 4 proof. Output Lean 4 code only.\n\n"
                f"## C Code\n```c\n{c_code}\n```\n\n"
                f"## Specification\n{property_text}\n\n"
                f"Begin your response with `import Mathlib` and nothing else before it.\n"
                f"Do not write any English text before or after the Lean 4 code block."
            )
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "import Mathlib\n"


def generate_batch(prompts: list[str], seed: int, temp: float) -> list[str]:
    """Tokenize a batch of prompts and run generation in one forward pass."""
    torch.manual_seed(seed)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,          # pad shorter sequences on the LEFT
        truncation=True,
        max_length=3000        # leave headroom for 4000 new tokens
    ).to(model.device)

    input_lengths = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4000,
            temperature=temp,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens for each item
    results = []
    for i, output in enumerate(outputs):
        new_tokens = output[input_lengths:]          # strip the shared prompt prefix
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append("import Mathlib\n" + text)

    return results


# --- Main Execution ---
for temp in TEMPERATURES:
    print(f"\n>>> Starting generation for Temperature: {temp}")


    for s in SEEDS:
        df_temp[f'lean_proof_seed_{s}'] = ""

    # Pre-load all C code and property files once (avoids redundant I/O inside loops)
    prompts_cache: dict[int, str] = {}   # index -> prompt string
    skipped = set()

    for index, row in df_temp.iterrows():
        try:
            with open(row['c_file_abs'], 'r', encoding='utf-8') as f:
                c_code = f.read()
            property_path = f"{PROPERTIES_BASE_PATH}{row['property']}.prp"
            with open(property_path, 'r', encoding='utf-8') as f:
                prop_text = f.read()
            prompts_cache[index] = build_prompt(c_code, prop_text)
        except Exception as e:
            print(f"  [Skip] Row {index} file read error: {e}")
            skipped.add(index)

    valid_indices = [i for i in df_temp.index if i not in skipped]

    # Outer loop: seed  →  Inner loop: batches of rows
    # This way torch.manual_seed(seed) is set once per batch, keeping results reproducible
    for seed in SEEDS:
        print(f"  Seed {seed} — {len(valid_indices)} rows in batches of {BATCH_SIZE}")

        for batch_start in tqdm(range(0, len(valid_indices), BATCH_SIZE), desc=f"Temp {temp} | Seed {seed}"):
            batch_indices = valid_indices[batch_start : batch_start + BATCH_SIZE]
            batch_prompts = [prompts_cache[i] for i in batch_indices]

            try:
                batch_outputs = generate_batch(batch_prompts, seed, temp)
                for idx, output in zip(batch_indices, batch_outputs):
                    df_temp.at[idx, f'lean_proof_seed_{seed}'] = output
            except Exception as e:
                print(f"  [Error] Batch {batch_start}–{batch_start+len(batch_indices)} seed {seed}: {e}")

    output_filename = os.path.join(BASE_OUTPUT_DIR, f'dataset_temp_{str(temp).replace(".", "")}.csv')
    df_temp.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}")

print("\nAll generations complete.")