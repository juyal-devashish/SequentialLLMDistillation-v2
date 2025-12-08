#!/bin/bash
#SBATCH --job-name=kpod_medium
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/kpod_full_%j.out
#SBATCH --account=eecs498f25s017_class
#SBATCH --qos=class

echo "===== Starting KPOD Medium Training Job ====="
date
hostname

module load python/3.10.4

# OPTIONAL: activate venv if using one
# source /home/$USER/venvs/distill/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TORCH_BF16=1                   # Enable BF16 if supported
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs

# Run your main medium-scale training script
srun python train_full_scale.py

echo "===== Job Finished ====="
date
