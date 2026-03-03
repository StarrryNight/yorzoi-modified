#!/bin/bash
#SBATCH --job-name=yorzoi_train
#SBATCH --account=def-cdeboer
#SBATCH --time=24:00:00         
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --array=0
#SBATCH --mem=32GB
#SBATCH --output=results/test_outputs/%A_%a.out
#SBATCH --error=results/training_log/%A_%a.err
#SBATCH --profile=none

python3 train_yeast.py