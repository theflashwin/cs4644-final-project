import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import re
from rapidfuzz import fuzz
from sympy import sympify, simplify, SympifyError

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./final_model", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("./final_model", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load test set
dataset = load_from_disk("./processed_math_dataset")["train"]  # or ["test"]

# Helper to extract \boxed{...} answer
def extract_boxed_answer(text):
    match = re.search(r"\\boxed{([^{}]*)}", text)
    if match:
        return match.group(1).strip()
    return None

# Check if two expressions are symbolically equivalent
def is_symbolically_equivalent(a, b):
    try:
        a_expr = sympify(a.replace("^", "**"))
        b_expr = sympify(b.replace("^", "**"))
        return simplify(a_expr - b_expr) == 0
    except (SympifyError, TypeError):
        return False

# Counters
exact_match = 0
fuzzy_match = 0
symbolic_match = 0
total = 0

print("Evaluating...")

for sample in tqdm(dataset):
    input_ids = torch.tensor([sample["input_ids"]]).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
    pred_ans = extract_boxed_answer(gen_text)
    true_ans = extract_boxed_answer(tokenizer.decode(sample["labels"], skip_special_tokens=True))

    if pred_ans is not None and true_ans is not None:
        if pred_ans == true_ans:
            exact_match += 1
        elif fuzz.ratio(pred_ans, true_ans) >= 95:
            fuzzy_match += 1
        elif is_symbolically_equivalent(pred_ans, true_ans):
            symbolic_match += 1

    total += 1

# Final results
combined_correct = exact_match + fuzzy_match + symbolic_match
print("\nEvaluation Results:")
print(f"Total examples: {total}")
print(f"Exact matches: {exact_match}")
print(f"Fuzzy matches (>=95%): {fuzzy_match}")
print(f"Symbolic matches: {symbolic_match}")
print(f"Combined accuracy: {combined_correct / total:.4f}")