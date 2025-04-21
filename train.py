import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel  # for sanity-checking trainable parameters (optional)
from ContextDistillation import ContextDistillationTrainer

# 1. Load the preprocessed dataset (saved from data_preprocessing.py)
print("Loading processed dataset from disk...")
dataset = load_from_disk("./processed_math_dataset")["train"]

# 2. --- LOAD MODELS ---
print("Loading student model...")
student_model = AutoModelForCausalLM.from_pretrained(
    "./model_setup_output/student_model",
    trust_remote_code=True,
    device_map="auto"
)

# Load the teacher model from local setup (frozen)
print("Loading teacher model...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    "./model_setup_output/teacher_model",
    trust_remote_code=True,
    device_map="auto"
)
teacher_model.eval()

# Optionally, print the number of trainable parameters (with LoRA, this should be a fraction of the full model)
total_params = sum(p.numel() for p in student_model.parameters()) + sum(p.numel() for p in teacher_model.parameters())
trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 3. --- LOAD THE TOKENIZERS ---
print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(
#     studen,
#     trust_remote_code=True
# )
# tokenizer.pad_token = tokenizer.eos_token

tokenizer = AutoTokenizer.from_pretrained(
    "./model_setup_output/student_model",
    trust_remote_code=True,
)

tokenizer.pad_token = tokenizer.eos_token

# 4. Set up training arguments.
training_args = TrainingArguments(
    output_dir="./train_output",             # directory for saving checkpoints
    per_device_train_batch_size=2,             # batch size per device (adjust based on GPU memory)
    gradient_accumulation_steps=8,             # to effectively use a larger batch size
    num_train_epochs=3,                        # number of training epochs
    learning_rate=1e-4,
    logging_steps=50,                          # log every 50 steps
    save_steps=500,                            # save checkpoint every 500 steps
    fp16=True,                                 # enable mixed precision training (if supported)
    report_to="none",                          # disable logging to external services for simplicity
)

# 5. Create a data collator for causal language modeling.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Create the Trainer.
trainer = ContextDistillationTrainer(
    teacher_model=teacher_model,
    temp=2.0,
    alpha=0.5,
    model=student_model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 7. Start training.
print("Starting training...")
trainer.train()

print("Training complete.")

# 8. Save the final model checkpoint.
final_save_path = "./final_model"
trainer.save_model(final_save_path)
print(f"Final model saved to {final_save_path}.")
