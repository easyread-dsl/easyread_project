#!/bin/bash 
#SBATCH --job-name=train_lora_arsaac
#SBATCH --account=dslab_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Load bashrc (in case it defines conda, modules, etc.)
source ~/.bashrc

# Activate conda environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate diffusers

# Base output directory
PROJECT_ROOT="/work/courses/dslab/team4/easyread_project"
BASE_OUTPUT_DIR="$PROJECT_ROOT/results/lora_output_arsaac_long"

# Find next run number automatically
RUN_NUM=1
while [ -d "${BASE_OUTPUT_DIR}_run${RUN_NUM}" ]; do
    ((RUN_NUM++))
done
OUTPUT_DIR="${BASE_OUTPUT_DIR}_run${RUN_NUM}"

# Print chosen run folder
echo "Starting training in: $OUTPUT_DIR"

# Go to training directory to keep paths consistent
cd "$PROJECT_ROOT/src/training" || exit 1

# Run training
python train_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --data_dir="$PROJECT_ROOT/data/training_data_arsaac" \
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
    --dataloader_num_workers=2 \
    --wandb_entity="dsl-25" \
    --wandb_project="dsl" \
    --wandb_run_name="arsaac_long_run${RUN_NUM}"

echo "Training complete for run ${RUN_NUM}!"
