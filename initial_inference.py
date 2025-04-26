import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import re
from rapidfuzz import fuzz
from sympy import sympify, simplify, SympifyError
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load student model and tokenizer before training
model = AutoModelForCausalLM.from_pretrained(
    "./model_setup_output/student_model", trust_remote_code=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "./model_setup_output/student_model", trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Load test set from subfolder
dataset = load_from_disk("./processed_math_dataset/test")

# ✅ Use only 10% of test set (at least 1)
subset_size = max(1, len(dataset) // 100)
dataset = dataset.select(range(subset_size))

# ✅ Track completed samples to allow resume
output_log_path = "initial_inference_results.jsonl"
seen_ids = set()

if os.path.exists(output_log_path):
    with open(output_log_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                seen_ids.add(data["index"])
            except:
                continue

# === Helpers ===
def extract_boxed_answer(text):
    # Try boxed extraction
    match = re.search(r"\\boxed{([^{}]*)}", text)
    if match:
        return match.group(1).strip()

    # Try fallback: return last number-like phrase (very rough fallback)
    match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    if match:
        return match[-1]
    
    return None


def is_symbolically_equivalent(a, b):
    try:
        a_expr = sympify(a.replace("^", "**"))
        b_expr = sympify(b.replace("^", "**"))
        return simplify(a_expr - b_expr) == 0
    except (SympifyError, TypeError):
        return False

# === Evaluation Counters ===
exact_match = 0
fuzzy_match = 0
symbolic_match = 0
skipped = 0
total = 0

print("Evaluating...")

with open(output_log_path, "a") as log_file:
    for i, sample in enumerate(tqdm(dataset)):
        if i in seen_ids:
            continue

        input_ids = torch.tensor([sample["input_ids"]]).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )

        gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
        pred_ans = extract_boxed_answer(gen_text)
        true_ans = extract_boxed_answer(tokenizer.decode(sample["labels"], skip_special_tokens=True))

        result = {
            "index": i,
            "predicted": pred_ans,
            "true": true_ans,
            "raw_output": gen_text
        }

        if pred_ans is not None and true_ans is not None:
            if pred_ans == true_ans:
                exact_match += 1
                result["match_type"] = "exact"
            elif fuzz.ratio(pred_ans, true_ans) >= 95:
                fuzzy_match += 1
                result["match_type"] = "fuzzy"
            elif is_symbolically_equivalent(pred_ans, true_ans):
                symbolic_match += 1
                result["match_type"] = "symbolic"
            else:
                result["match_type"] = "none"
        else:
            result["match_type"] = "invalid"
            skipped += 1

        log_file.write(json.dumps(result) + "\n")
        log_file.flush()
        total += 1

# === Final Results ===
# Final results
combined_correct = exact_match + fuzzy_match + symbolic_match
evaluated = exact_match + fuzzy_match + symbolic_match
skipped = len(dataset) - evaluated

print("\nEvaluation Results:")
print(f"Total examples processed: {len(dataset)}")
print(f"Skipped (missing or invalid answers): {skipped}")
print(f"Evaluated (with valid answers): {evaluated}")
print(f"Exact matches: {exact_match}")
print(f"Fuzzy matches (>=95%): {fuzzy_match}")
print(f"Symbolic matches: {symbolic_match}")

if evaluated > 0:
    print(f"Combined accuracy: {combined_correct / evaluated:.4f}")
else:
    print("⚠️ No valid answers found — accuracy not computed.")

