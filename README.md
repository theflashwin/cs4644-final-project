# Context Distillation for Mathematical Reasoning in LLMs

## Project Overview

This project explores the application of Context Distillation (CD) and Low-Rank Adaptation (LoRA) techniques to improve the mathematical reasoning capabilities of smaller large language models (LLMs). Our goal is to train a smaller student model to mimic the in-context learning behavior of a larger teacher model, without requiring massive computational resources during inference. We specifically target mathematical problem-solving tasks using the OpenR1-Math-220k dataset as our benchmark.

Through this approach, we aim to bridge the performance gap between large and small models, providing a scalable and resource-efficient framework for advanced math reasoning in real-world applications.

---

## Reproducing the Experiments

Follow the steps below to set up the environment and reproduce the experiments, especially if running on the Georgia Tech PACE cluster.

### 1. Connect to PACE

First, connect to the PACE cluster via SSH.  
Make sure you have access to a GPU node (e.g., with an A100 or V100 GPU).

```bash
ssh your_username@your_pace_address
Request a GPU allocation (example for a debug allocation):
pace-allocations --account your_allocation_account
srun --partition=gpu-a100 --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=02:00:00 --pty bash
```
Adjust resources if needed for longer runs.

### 2. Clone the Repository
Once connected to your GPU node:

```bash
git clone https://github.com/theflashwin/cs4644-final-project.git
cd cs4644-final-project
```
### 3. Set Up Environment
Create and activate a Python environment (recommended using Conda):

```bash
conda create -n math-distillation python=3.10
conda activate math-distillation
```
Install the required packages:

```bash

pip install -r requirements.txt
```

Also load the appropriate CUDA modules on PACE:

```bash
module load gcc/9.3.0
module load cuda/11.4
```
### 4. Prepare the Dataset
The processed dataset should be placed under:

```bash
./processed_math_dataset/
```
If starting from raw data, you can preprocess using:

```bash
python data_preprocessing.py
```
This will tokenize the problems, normalize formats, and prepare the input-output pairs.

### 5. Run Baseline Inference (Optional Sanity Check)
Before training, you can verify the model and dataset setup by running baseline inference:

```bash
python inference.py
```
This script loads a quantized teacher model and runs inference on a small subset of problems to establish baseline performance.

### 6. Train the Student Model (Context Distillation)
To perform context distillation training:

``` bash
python train.py
```
This script sets up the teacher-student framework and applies the custom composite loss (combination of KL Divergence and Cross-Entropy).
Intermediate checkpoints will be saved during training to allow recovery in case of session interruptions.

7. Evaluate the Final Student Model
After training, you can evaluate the distilled student model's performance across the full test set:

```bash
python inference.py
```
This step compares the student's outputs against ground truth answers using exact match, fuzzy match, and symbolic equivalence metrics.

### Notes
Make sure you request enough wall time and memory when training on PACE, especially for full dataset runs.

Checkpointing is enabled to allow progress recovery across multiple GPU allocations.

Inference and evaluation use batch size 1 for compatibility with 4-bit quantized models and constrained GPU memory.

If running locally instead of on PACE, adjust resource settings and install CUDA-compatible PyTorch accordingly.

Repository Structure
```php

cs4644-final-project/
│
├── data_preprocessing.py     # Preprocesses raw math dataset
├── inference.py              # Baseline teacher model inference
├── train.py                  # Context distillation training script
├── inference.py       # Student model evaluation script
├── processed_math_dataset/   # Tokenized and formatted dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── models/                   # (Optional) Checkpoint storage directory
```
Citation
If you use or reference this work, please cite:

Dan Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset", 2021.
