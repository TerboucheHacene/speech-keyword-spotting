#!/bin/bash
#SBATCH --partition=low 
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-gpu=8  # Cores proportional to GPUs
#SBATCH --mem-per-gpu=32000M       # Memory proportional to GPUs
#SBATCH --output=artifacts/out/%N-%j.out



poetry run python scripts/train.py \
    --batch_size 1024 \
    --max_epochs 40 \
    --learning_rate 0.001 \