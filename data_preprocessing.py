from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login
import os

print("Loading dataset...")
dataset = load_dataset("open-r1/OpenR1-Math-220k", "default")
# print("Available columns:", dataset["train"].column_names)

# Log in to Hugging Face using the token from your environment variable.
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

# Use a text-only model that has a compatible tokenizer.
model_name = "meta-llama/Meta-Llama-3-8B"

print("Loading tokenizer...")
# Using trust_remote_code=True forces the library to load the custom code that Meta provides.
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"],
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token for LLaMA-style models

# Preprocessing function for each problem-solution pair.
def preprocess(example):
    # Use "problem" as the text (instead of "question") since that's what the dataset provides.
    problem_text = example["problem"]
    solution_text = example["solution"]

    # Tokenize the problem and solution using fixed max lengths.
    input_enc = tokenizer(problem_text, truncation=True, padding="max_length", max_length=128)
    output_enc = tokenizer(solution_text, truncation=True, padding="max_length", max_length=256)
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc

print("Tokenizing dataset...")
processed_dataset = dataset["train"].map(preprocess, batched=False)

print("\nExample of processed data:\n")
for idx in range(3):
    print(f"Problem {idx+1}: {tokenizer.decode(processed_dataset[idx]['input_ids'])}")
    print(f"Solution {idx+1}: {tokenizer.decode(processed_dataset[idx]['labels'])}")
    print("=" * 80)

# Save the processed dataset for faster future loading.
processed_dataset.save_to_disk("./processed_math_dataset")

print("Done!")
