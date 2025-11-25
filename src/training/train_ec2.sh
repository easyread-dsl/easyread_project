#!/bin/bash
# Training script for AWS EC2 with multi-GPU support using accelerate
# Usage: bash train_ec2.sh

# Exit on error
set -e

# Base directories (adjust PROJECT_ROOT to your EC2 path)
PROJECT_ROOT="/mnt/data2/easyread_project"  # Change this to your EC2 project path
BASE_OUTPUT_DIR="$PROJECT_ROOT/results/lora_output_arsaac_diverse"

# Create results directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/results"

# Find next run number automatically
RUN_NUM=1
while [ -d "${BASE_OUTPUT_DIR}_run${RUN_NUM}" ]; do
    ((RUN_NUM++))
done
OUTPUT_DIR="${BASE_OUTPUT_DIR}_run${RUN_NUM}"

# Print chosen run folder
echo "========================================"
echo "Starting multi-GPU training on AWS EC2"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --list-gpus || echo "Warning: nvidia-smi not found"
echo "========================================"

# Go to training directory to keep paths consistent
cd "$PROJECT_ROOT/src/training" || exit 1

# Launch with accelerate (automatically uses all available GPUs)
accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    train_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --data_dir="/mnt/data2/training_data_arsaac" \
    --output_dir="$OUTPUT_DIR" \
    --resolution=256 \
    --train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=50 \
    --learning_rate=1e-4 \
    --lora_rank=16 \
    --lora_alpha=24 \
    --mixed_precision="fp16" \
    --seed=42 \
    --logging_steps=50 \
    --save_steps=500 \
    --dataloader_num_workers=4 \
    --wandb_entity="dsl-25" \
    --wandb_project="dsl" \
    --wandb_run_name="arsaac_ec2_run${RUN_NUM}"

echo "========================================"
echo "Training complete for run ${RUN_NUM}!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"
