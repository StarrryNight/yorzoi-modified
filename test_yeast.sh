#!/bin/bash
#SBATCH --job-name=yorzoi_test
#SBATCH --account=rrg-cdeboer
#SBATCH --time=00:15:00         
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --output=results/test_outputs/%A_%a.out
#SBATCH --error=results/training_log/%A_%a.err


python test_yeast.py