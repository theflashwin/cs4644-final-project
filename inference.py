import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading final model...")
# Load the final model checkpoint saved after training.
model = AutoModelForCausalLM.from_pretrained("./final_model", trust_remote_code=True)

# Select device: GPU is preferred.
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Loading tokenizer...")
# Load the corresponding tokenizer (should be the same as during training).
tokenizer = AutoTokenizer.from_pretrained("./final_model", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Example prompt for inference.
prompt = "What is the sum of 15 and 27?"
print("Input prompt:", prompt)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Generating model output...")
# Generate output using beam search for higher quality; adjust max_length as needed.
outputs = model.generate(**inputs, max_length=64, num_beams=5, early_stopping=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Output:")
print(generated_text)
