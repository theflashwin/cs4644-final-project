import os
import json
import math
import torch
import shutil
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import PeftModel
from ContextDistillation import ContextDistillationTrainer

# ──────────────────────────────────────────────────────────────────────────────
STUDENT_ADAPTER_DIR = "./model_setup_output/student_model"
BASE_STUDENT_HF     = "meta-llama/Llama-3.2-3B-Instruct"
TEACHER_HF          = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_DIR            = "./processed_math_dataset/train"
OUTPUT_DIR          = "./train_output_v2"
FINAL_MODEL_DIR     = "./final_model_v2"
# ──────────────────────────────────────────────────────────────────────────────

print("1/6) Loading dataset…")
train_dataset = load_from_disk(DATA_DIR)
ds_len = len(train_dataset)

print("2/6) Loading student (4‑bit + LoRA) onto GPU…")
bnb_student = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
base_student = AutoModelForCausalLM.from_pretrained(
    BASE_STUDENT_HF,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_student
)
student_model = PeftModel.from_pretrained(
    base_student,
    STUDENT_ADAPTER_DIR,
    device_map="auto"
)

# unfreeze only LoRA parameters
for name, param in student_model.named_parameters():
    param.requires_grad = "lora" in name.lower()

total = sum(p.numel() for p in student_model.parameters())
trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
print(f"🔍 Student params: {total:,} total, {trainable:,} trainable")

print("3/6) Loading teacher (4‑bit) onto GPU…")
bnb_teacher = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_HF,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_teacher
)
teacher_model.eval()

print("4/6) Preparing tokenizer & Trainer…")
tokenizer = AutoTokenizer.from_pretrained(BASE_STUDENT_HF, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    fp16=True,
    report_to="none",
    save_total_limit=2
)

# calculate total steps
effective_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
steps_per_epoch = math.ceil(ds_len / effective_bs)
total_steps = steps_per_epoch * training_args.num_train_epochs

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = ContextDistillationTrainer(
    teacher_model=teacher_model,
    temp=2.0,
    alpha=0.5,
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("5/6) Starting training…")
resume_ckpt = None
resume_step = 0
if os.path.isdir(OUTPUT_DIR):
    ckpts = sorted(
        [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1])
    )
    if ckpts:
        resume_ckpt = os.path.join(OUTPUT_DIR, ckpts[-1])
        # read trainer_state.json to get global_step
        state_file = os.path.join(resume_ckpt, "trainer_state.json")
        if os.path.isfile(state_file):
            state = json.load(open(state_file, "r"))
            resume_step = state.get("global_step", 0)
        remaining = total_steps - resume_step
        print(f"   ↻ Resuming from {resume_ckpt} (step {resume_step}); {remaining} steps remaining out of {total_steps}")
    else:
        print(f"   No checkpoints found in {OUTPUT_DIR}, starting from scratch")

trainer.train(resume_from_checkpoint=resume_ckpt)
print("✓ Training complete.")

print(f"6/6) Saving final student model to {FINAL_MODEL_DIR}")
trainer.save_model(FINAL_MODEL_DIR)
print("All done!")
