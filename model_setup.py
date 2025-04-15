import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# Log in to Hugging Face (ensure your token is set in the environment)
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

# Specify the model you want to use (the gated vision-instruct model)
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Configure quantization to reduce memory usage.
# Here we use 4-bit quantization with bfloat16 computations.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model with quantization configuration and remote code trust...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)

# Define LoRA configuration parameters.
# Here r, lora_alpha, and dropout are set for efficient fine-tuning.
# The target_modules list should specify which parts of the model to apply LoRA to.
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Adjust based on the model architecture.
)

# Wrap the model with LoRA, which inserts trainable adapters without modifying all parameters.
model = get_peft_model(model, lora_config)
print("LoRA injected successfully into the model.")

# Load the corresponding tokenizer (using trust_remote_code=True as well)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"],
    trust_remote_code=True
)
# Set the pad token to the eos token for consistency with LLaMA style.
tokenizer.pad_token = tokenizer.eos_token

# Run a quick test forward pass with a dummy input
dummy_text = "What is 2+2?"
dummy_input = tokenizer(dummy_text, return_tensors="pt").to("cuda")
print("Running a dummy forward pass...")
with torch.no_grad():
    outputs = model(**dummy_input)
print("Forward pass successful. Model outputs obtained.")

# Save the prepared model checkpoint for future training steps
save_path = "./model_setup_output"
model.save_pretrained(save_path)
print(f"Model saved to {save_path}.")
