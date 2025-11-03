#!/bin/bash
#SBATCH --job-name=regularize_data
#SBATCH --partition=inf-train
#SBATCH --account=dslab
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --output=regularize_data_%j.out
#SBATCH --error=regularize_data_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
python -u prepare_dataset.py
