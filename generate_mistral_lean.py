import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
CSV_PATH = '/workspace/CS527-Project/dataset.csv'
BASE_OUTPUT_DIR = '/workspace/CS527-Project/'
SEEDS = [42, 123, 999]
TEMPERATURES = [0.2, 0.4, 0.6]
PROPERTIES_BASE_PATH = '/workspace/sv-benchmarks/c/properties/'

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

def generate_lean_code(c_code, property_text, seed, temp):
    messages = [
        {
            "role": "system",
            "content": "You output only Lean 4 code. No English. No explanations. No self-correction."
        },
        {
            "role": "user", 
            "content": f"Translate this C code into a Lean 4 proof. Output Lean 4 code only.\n\n## C Code\n```c\n{c_code}\n```\n\n## Specification\n{property_text}\n\nBegin your response with `import Mathlib` and nothing else before it.\nDo not write any English text before or after the Lean 4 code block."
        }
    ]

    # Apply chat template and force prefix
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_prompt = prompt + "import Mathlib\n"

    torch.manual_seed(seed)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=4000,
        temperature=temp,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return "import Mathlib\n" + generated_text

# --- Main Execution ---
for temp in TEMPERATURES:
    print(f"\n>>> Starting generation for Temperature: {temp}")
    
    # Reload a fresh copy of the dataframe for each temperature
    df_temp = pd.read_csv(CSV_PATH)
    
    # Prepare seed columns
    for s in SEEDS:
        df_temp[f'lean_proof_seed_{s}'] = ""

    for index, row in tqdm(df_temp.iterrows(), total=df_temp.shape[0], desc=f"Temp {temp}"):
        try:
            # Read C code
            with open(row['c_file_abs'], 'r', encoding='utf-8') as file:
                c_code_string = file.read()

            # Read Property file
            property_path = f"{PROPERTIES_BASE_PATH}{row['property']}.prp"
            with open(property_path, 'r', encoding='utf-8') as file:
                property_string = file.read()

            # Generate for each seed at this temperature
            for seed in SEEDS:
                lean_output = generate_lean_code(c_code_string, property_string, seed, temp)
                df_temp.at[index, f'lean_proof_seed_{seed}'] = lean_output

        except Exception as e:
            print(f"Error on row {index} at temp {temp}: {e}")

    # Save CSV for this specific temperature
    output_filename = os.path.join(BASE_OUTPUT_DIR, f'dataset_temp_{str(temp).replace(".", "")}.csv')
    df_temp.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}")

print("\nAll generations complete.")