#!/bin/bash
#SBATCH --job-name=distill_reasoning
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/distill_%j.out
#SBATCH --account=eecs498f25s017_class
#SBATCH --qos=class

echo "===== Starting GSM8K Rationale Generation Job ====="
date
hostname

module load python/3.10.4

# Install Python packages if needed (ignored if already installed)
pip install --user openai tqdm

# Set your API key
export OPENAI_API_KEY="KEY_GOES_HERE"

mkdir -p logs

python generate_full_rationales.py

echo "===== Job Finished ====="
date
