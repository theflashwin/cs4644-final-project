from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers import AdamW

from datasets import load_dataset
class MathDistilliationDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        question, solution = self.data[index]

        input = self.tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length = 128).input_ids.squeeze()
        outputs = self.tokenizer(solution, return_tensors="pt", padding="max_length", truncation=True, max_length = 256).input_ids.squeeze()

        return input, outputs

model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to("cuda")

dataset = load_dataset("open-r1/OpenR1-Math-220k", "default")

optimizer = AdamW(model.parameters(), lr=1e-5)
cross_entropy_loss = CrossEntropyLoss()
kl_divergence = KLDivLoss(reduction="batchmean")

sampled_data = []
for sample in dataset["train"]:
    question = sample["question"]
    solution = sample["solution"]

    sampled_data.append((question, solution))

dataset = MathDistilliationDataset(sampled_data, tokenizer)
dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

model.train()
epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for batch in dataloader:

        input, output = batch
        input, output = input.to("cuda"), output.to("cuda")
        optimizer.zero_grad()

        outputs = model(input, labels=output)
        scores = outputs.logits

        model_probs = torch.nn.functional.softmax(scores, dim=-1)
        teacher_probs = torch.nn.functional.softmax(scores.detach(), dim=-1)

        kl_loss = kl_divergence(model_probs, teacher_probs)
        ce_loss = outputs.loss

        loss = ce_loss + kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

model.save_pretrained("math-distillation-model")
tokenizer.save_pretrained("math-distillation-model")

model.eval()
input = "What is the sum of 2 and 3?"
input = tokenizer(input, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input, max_length=256)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(output)
print("Done")
