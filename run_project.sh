#!/bin/bash

# Load the anaconda module (if required on your system)
module load anaconda3

# Activate the conda environment
conda activate dl-final-project || source activate dl-final-project

# Source your bashrc to load any environment variables (if needed)
source ~/.bashrc

# Export your Hugging Face token
export HUGGINGFACE_HUB_TOKEN=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

# Set PyTorch CUDA memory configuration to help reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change directory to your project directory
cd /storage/ice1/2/7/svijayasankar3/train/cs4644-final-project

# Run the data preprocessing script
echo "Running data_preprocessing.py..."
python data_preprocessing.py || { echo "Data preprocessing failed!"; exit 1; }

# Run the model setup script (loads model with quantization and LoRA)
echo "Running model_setup.py..."
python model_setup.py || { echo "Model setup failed!"; exit 1; }

echo "Initial inference run..."
python initial_inference.py || { echo "Initial interence run failed"; exit 1; }

# Check if final model already exists to avoid retraining
if [ -d "./final_model" ] && [ "$(ls -A ./final_model)" ]; then
  echo "Found existing final_model directory; skipping training."
else
  echo "Running train.py..."
  python train.py || { echo "Training failed!"; exit 1; }
fi

# Run the inference script to test the final model output
echo "Running inference.py..."
python inference.py || { echo "Inference failed!"; exit 1; }

echo "All steps completed successfully."
