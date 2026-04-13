import pandas as pd
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import os

# --- Configuration ---
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CSV_PATH = '/workspace/CS527-Project/c_sample.csv'
BASE_OUTPUT_DIR = '/workspace/CS527-Project/'
SEEDS = [42, 123, 999]
TEMPERATURES = [0.2, 0.4, 0.6]
PROPERTIES_BASE_PATH = '/workspace/sv-benchmarks/c/properties/'
BATCH_SIZE = 8

# --- Quantisation Config (4-bit NF4) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# --- Load Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()


def strip_thinking(text: str) -> str:
    """Remove DeepSeek R1's <think>...</think> block, keep only the Lean code."""
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    if "<think>" in text:
        return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text.strip()


def build_prompt(c_code: str, property_text: str) -> str:
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
    torch.manual_seed(seed)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3000
    ).to(model.device)

    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4000,
            temperature=temp,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for output in outputs:
        new_tokens = output[input_length:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        lean_code = strip_thinking(raw_text)          # strip R1 <think> block
        if not lean_code.startswith("import Mathlib"):
            lean_code = "import Mathlib\n" + lean_code
        results.append(lean_code)

    return results


# --- Main Execution ---
for temp in TEMPERATURES:
    print(f"\n>>> Starting generation for Temperature: {temp}")

    df_temp = pd.read_csv(CSV_PATH)

    for s in SEEDS:
        df_temp[f'lean_proof_seed_{s}'] = ""

    prompts_cache: dict[int, str] = {}
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

    for seed in SEEDS:
        print(f"  Seed {seed} — {len(valid_indices)} rows in batches of {BATCH_SIZE}")

        for batch_start in tqdm(
            range(0, len(valid_indices), BATCH_SIZE),
            desc=f"Temp {temp} | Seed {seed}"
        ):
            batch_indices = valid_indices[batch_start : batch_start + BATCH_SIZE]
            batch_prompts = [prompts_cache[i] for i in batch_indices]

            try:
                batch_outputs = generate_batch(batch_prompts, seed, temp)
                for idx, output in zip(batch_indices, batch_outputs):
                    df_temp.at[idx, f'lean_proof_seed_{seed}'] = output
            except Exception as e:
                print(f"  [Error] Batch {batch_start}–{batch_start+len(batch_indices)} seed {seed}: {e}")

        # ✅ Free GPU cache after every seed
        torch.cuda.empty_cache()

    output_filename = os.path.join(
        BASE_OUTPUT_DIR,
        f'deepseek_temp_{str(temp).replace(".", "")}.csv'
    )
    df_temp.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}")

# ✅ Fully release model from VRAM when all generations are done
del model, tokenizer
torch.cuda.empty_cache()
print("\nAll generations complete. VRAM released.")