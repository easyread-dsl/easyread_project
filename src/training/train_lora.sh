#!/bin/bash
#SBATCH --job-name=train_lora
#SBATCH --account=dslab_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err

# Training configuration script for LoRA training
#
# To resume from a checkpoint, add the --resume_from_checkpoint argument:
# --resume_from_checkpoint="../../lora_output/checkpoint-4000" \

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Load bashrc
source ~/.bashrc

# Activate conda environment
conda activate diffusers

# Then run training
echo "Starting training..."
python train_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --data_dir="../../data/training_data" \
    --output_dir="../../lora_output" \
    --resolution=256 \
    --train_batch_size=16 \
    --gradient_accumulation_steps=16 \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --lora_rank=16 \
    --lora_alpha=16 \
    --mixed_precision="fp16" \
    --seed=42 \
    --logging_steps=50 \
    --save_steps=500 \
    --dataloader_num_workers=2 \
    --wandb_entity="dsl-25" \
    --wandb_project="dsl" \
    --wandb_run_name="long run"

echo "Training complete!"
