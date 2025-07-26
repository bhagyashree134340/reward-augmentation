#!/bin/bash
#SBATCH -p aisdlc_gpu-rtx2080       
#SBATCH --gres=gpu:1                
#SBATCH --mem=8000                  
#SBATCH -t 0-07:00                 
#SBATCH -c 1                        
#SBATCH -o log/train.%j.out         
#SBATCH -e log/train.%j.err         
#SBATCH -J reward-train            

echo "Working dir: $PWD"
echo "Started at $(date)"
echo "Running on $(hostname)"
echo "Loading conda env..."

# Activate your conda environment
source ~/.bashrc
conda activate py310env

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export WANDB_API_KEY=28c48f9795c35a55ee1bf7a82a05994bf808294a

# Run your Python script
echo "Running training script..."
CUDA_LAUNCH_BLOCKING=1 python main.py

echo "Finished at $(date)"