#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 0-20:00
#SBATCH -c 1
#SBATCH -o log/cfn.%j.out
#SBATCH -e log/cfn.%j.err
#SBATCH -J cfn-dqn

if [ -z "$1" ]; then
    echo "ERROR: No run message provided."
    echo "Usage: sbatch cfn.sh \"your message here\""
    exit 1
fi

RUN_MESSAGE="$1"

echo "========================================"
echo "RUN MESSAGE: $RUN_MESSAGE"
echo "========================================"
echo "Working dir: $PWD"
echo "Started at $(date)"
echo "Running on $(hostname)"


echo "Loading conda env..."
source ~/miniconda3/bin/activate
conda activate rl310


export WANDB_API_KEY=28c48f9795c35a55ee1bf7a82a05994bf808294a
export WANDB_RUN_GROUP="$RUN_MESSAGE"

start=$(date +%s)


echo "Running training script..."
cd /work/dlclarge2/raneb-project/reward-augmentation || exit 1
export PYTHONPATH=$PWD

python agents/dqn_cfn.py

end=$(date +%s)
runtime=$((end-start))

echo "========================================"
echo "RUN MESSAGE: $RUN_MESSAGE"
echo "Finished at $(date)"
echo "Runtime (s): $runtime"
echo "========================================"
