#!/bin/bash
#SBATCH -p aisdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=2000
#SBATCH -t 0-00:10
#SBATCH -c 1
#SBATCH -o log/evaluate.%j.out
#SBATCH -e log/evaluate.%j.err
#SBATCH -J reward-evaluate

echo "Working dir: $PWD"
echo "Started at $(date)"
echo "Running on $(hostname)"
echo "Loading conda env..."

source ~/miniconda3/bin/activate
conda activate py310env

export WANDB_API_KEY=28c48f9795c35a55ee1bf7a82a05994bf808294a

start=`date +%s`

echo "Running evaluation script..."
export PYTHONPATH=$PWD
python evaluate_dqn.py

end=`date +%s`
runtime=$((end-start))
echo "Finished in ${runtime}s"