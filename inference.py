import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from rapidfuzz import fuzz
from sympy import sympify, simplify, SympifyError
import re

# ── CONFIG ─────────────────────────────────────────────────────────────
MODEL_DIR        = "./final_model_v2"
DATA_DIR         = "./processed_math_dataset/test"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE  = "inference_checkpoint.json"
OUTPUT_PNG       = "inference_results.png"
# ────────────────────────────────────────────────────────────────────────

print("Loading model and tokenizer…")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading test dataset…")
ds = load_from_disk(DATA_DIR)
total = len(ds)
print(f"→ {total} examples in test set\n")

# ── HELPERS ──────────────────────────────────────────────────────────────
def extract_boxed_answer(text: str) -> str:
    # truncate at “Final Answer:” if present
    m0 = re.search(r"Final\s+Answer[:\s]+", text, re.IGNORECASE)
    snippet = text[m0.end():] if m0 else text

    # first \boxed{…}
    m = re.search(r"\\boxed{([^{}]*)}", snippet)
    if m:
        return m.group(1).strip()

    # number right after “Final Answer:”
    m2 = re.match(r"\s*([+-]?\d*\.?\d+)", snippet)
    if m2:
        return m2.group(1).strip()

    # fallback: last numeric token
    nums = re.findall(r"[+-]?\d*\.?\d+", snippet)
    return nums[-1] if nums else None

def normalize(ans: str) -> str:
    if not ans:
        return ""
    return ans.strip().lstrip("+").rstrip(".,;")

def is_exact(p: str, t: str) -> bool:
    p, t = normalize(p), normalize(t)
    if p == t:
        return True
    # integer comparison
    if re.fullmatch(r"-?\d+", p) and re.fullmatch(r"-?\d+", t):
        return int(p) == int(t)
    # float comparison
    try:
        return abs(float(p) - float(t)) < 1e-3
    except:
        return False

def is_symbolically_equivalent(a: str, b: str) -> bool:
    try:
        A = sympify(a.replace("^", "**"))
        B = sympify(b.replace("^", "**"))
        return simplify(A - B) == 0
    except (SympifyError, TypeError):
        return False
# ────────────────────────────────────────────────────────────────────────

# ── RESUME STATE IF EXISTING ─────────────────────────────────────────────
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        state = json.load(f)
    start_idx = state["last_index"] + 1
    counters = state["counters"]
    print(f"Resuming from index {start_idx} (loaded checkpoint)")
else:
    start_idx = 0
    counters = {"exact":0, "fuzzy":0, "symbolic":0, "none":0, "invalid":0}

# ── INFERENCE LOOP ───────────────────────────────────────────────────────
print("Running inference…")
for i in tqdm(range(start_idx, total), initial=start_idx, total=total, desc="inference"):
    sample = ds[i]
    inp = torch.tensor([sample["input_ids"]]).to(DEVICE)
    mask = (inp != tokenizer.pad_token_id).long()

    with torch.no_grad():
        out = model.generate(
            inp,
            attention_mask=mask,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

    gen  = tokenizer.decode(out[0], skip_special_tokens=True)
    pred = extract_boxed_answer(gen)
    true = extract_boxed_answer(tokenizer.decode(sample["labels"], skip_special_tokens=True))

    if pred is None or true is None:
        mt = "invalid"
    elif is_exact(pred, true):
        mt = "exact"
    elif fuzz.ratio(normalize(pred), normalize(true)) >= 80:
        mt = "fuzzy"
    elif is_symbolically_equivalent(pred, true):
        mt = "symbolic"
    else:
        mt = "none"

    counters[mt] += 1

    # checkpoint after each sample
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_index": i, "counters": counters}, f)

# once done, remove checkpoint
os.remove(CHECKPOINT_FILE)

# ── SUMMARIZE & PLOT ────────────────────────────────────────────────────
df = pd.DataFrame([{"match_type":k, "count":v} for k,v in counters.items()])
df["percentage"] = df["count"] / df["count"].sum() * 100

print("\n=== Inference Results ===")
print(df.to_string(index=False))

plt.figure(figsize=(8,5))
plt.bar(df["match_type"], df["count"])
plt.title("Match-Type Breakdown on Full Test Set")
plt.xlabel("Match Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_PNG)
print(f"\nBar chart saved to {OUTPUT_PNG}")
