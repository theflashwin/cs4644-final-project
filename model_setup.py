import os
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

def log_ram(tag=""):
    ram = psutil.virtual_memory()
    print(f"[{tag}] RAM usage: {ram.used // (1024**3)} GB / {ram.total // (1024**3)} GB")

# Log in to Hugging Face (ensure your token is set in the environment)
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

# Configure quantization to reduce memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# --- STUDENT MODEL SETUP ---
model_name = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading student model with quantization configuration and remote code trust...")
student_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)

# Inject LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
student_model = get_peft_model(student_model, lora_config)
print("LoRA injected successfully into the model.")

# Load tokenizer for student model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"],
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# --- TEACHER MODEL SETUP ---
teacher_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
print("Loading teacher model with quantization config...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)
teacher_model.eval()

print("Loading teacher tokenizer...")
teacher_tokenizer = AutoTokenizer.from_pretrained(
    teacher_model_name,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"],
    trust_remote_code=True,
)
teacher_tokenizer.pad_token = tokenizer.eos_token

# --- SAVE MODELS LOCALLY ---
save_student_path = "./model_setup_output/student_model"
save_teacher_path = "./model_setup_output/teacher_model"

os.makedirs(save_student_path, exist_ok=True)
os.makedirs(save_teacher_path, exist_ok=True)

print(f"Saving student model to {save_student_path}...")
student_model.save_pretrained(save_student_path)
tokenizer.save_pretrained(save_student_path)

print("Moving teacher model to CPU before saving...")
teacher_model.to("cpu")

log_ram("Before saving teacher")
try:
    teacher_model.save_pretrained(save_teacher_path)
    teacher_tokenizer.save_pretrained(save_teacher_path)
    print("✅ Teacher model saved successfully.")
except Exception as e:
    print("❌ Saving teacher model failed:")
    print(e)

log_ram("After saving teacher")
print("Model setup complete.")
