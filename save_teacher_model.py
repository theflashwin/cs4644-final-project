import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# Login using your HF token from env
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

# Quantization for memory savings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

teacher_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Loading teacher model...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)

teacher_model.eval()
teacher_model.to("cpu")  # Prevent GPU OOM during save

print("Loading tokenizer...")
teacher_tokenizer = AutoTokenizer.from_pretrained(
    teacher_model_name,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"],
    trust_remote_code=True
)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Save safely
save_path = "./model_setup_output/teacher_model"
os.makedirs(save_path, exist_ok=True)

print("Saving teacher model...")
teacher_model.save_pretrained(save_path, safe_serialization=False)
teacher_tokenizer.save_pretrained(save_path)
print("✅ Teacher model saved correctly.")
