from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.optim import AdamW

from datasets import load_dataset

from huggingface_hub import login
import os

# Login to HuggingFace
login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])

model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ["HUGGINGFACE_HUB_TOKEN"],
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    model_name,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
)

model.to("cuda")


class MathDistillationDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question, solution = self.data[index]
        
        inputs = self.processor(question, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        outputs = self.processor(solution, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        
        return inputs.input_ids.squeeze(), outputs.input_ids.squeeze()


dataset = load_dataset("open-r1/OpenR1-Math-220k", "default")

sampled_data = [(sample["question"], sample["solution"]) for sample in dataset["train"]]

dataset = MathDistillationDataset(sampled_data, processor)
dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-5)
cross_entropy_loss = CrossEntropyLoss()
kl_divergence = KLDivLoss(reduction="batchmean")

model.train()
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids, output_ids = batch
        input_ids, output_ids = input_ids.to("cuda"), output_ids.to("cuda")

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=output_ids)
        scores = outputs.logits

        model_probs = torch.nn.functional.softmax(scores, dim=-1)
        teacher_probs = torch.nn.functional.softmax(scores.detach(), dim=-1)

        kl_loss = kl_divergence(model_probs, teacher_probs)
        ce_loss = outputs.loss

        loss = ce_loss + kl_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

model.save_pretrained("math-distillation-model")
processor.save_pretrained("math-distillation-model")

model.eval()

query = "What is the sum of 2 and 3?"
inputs = processor(query, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=256)
print(processor.decode(outputs[0], skip_special_tokens=True))

print("Done")