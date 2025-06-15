#!/bin/bash
#SBATCH -p aisdlc_gpu-rtx2080       # partition (hav access to htis one)
#SBATCH --gres=gpu:1                # request 1 GPU
#SBATCH --mem=8000                  # 8GB memory
#SBATCH -t 0-07:00                  # time limit: 7 hours
#SBATCH -c 1                        # 2 CPU cores
#SBATCH -o log/train.%j.out         # STDOUT log (job ID in filename)
#SBATCH -e log/train.%j.err         # STDERR log (job ID in filename)
#SBATCH -J reward-train             # Job name

echo "Working dir: $PWD"
echo "Started at $(date)"
echo "Running on $(hostname)"
echo "Loading conda env..."

# Activate your conda environment
source ~/.bashrc
conda activate py310env

export WANDB_API_KEY=28c48f9795c35a55ee1bf7a82a05994bf808294a

# Run your Python script
echo "Running training script..."
CUDA_LAUNCH_BLOCKING=1 python main.py

echo "Finished at $(date)"