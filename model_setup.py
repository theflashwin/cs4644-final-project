import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# Log in to Hugging Face (ensure your token is set in the environment)
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

# Configure quantization to reduce memory usage.
# Here we use 4-bit quantization with bfloat16 computations.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# --- STUDENT MODEL SETUP ---
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# load student tokenizer
print("Loading student model with quantization configuration and remote code trust...")
student_model = AutoModelForCausalLM.from_pretrained(
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
student_model = get_peft_model(student_model, lora_config)
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

# --- TEACHER MODEL SETUP ---
teacher_model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
print("Loading teacher model with quantization config")
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    torch_dtype = torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)

teacher_model.eval() # freeze weights

print("Loading Teacher Tokenizer...")
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

print(f"Saving teacher model to {save_teacher_path}...")
teacher_model.save_pretrained(save_teacher_path)
teacher_tokenizer.save_pretrained(save_teacher_path)

print("Model setup complete.")