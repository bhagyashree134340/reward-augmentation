#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080    
#SBATCH --gres=gpu:1                             
#SBATCH -t 0-20:00                  
#SBATCH -c 1                        
#SBATCH -o log/mcfn.%j.out         # STDOUT log (job ID in filename)
#SBATCH -e log/mcfn.%j.err         # STDERR log (job ID in filename)
#SBATCH -J mcfn-dqn           # Job name

echo "Working dir: $PWD"
echo "Started at $(date)"
echo "Running on $(hostname)"
echo "Loading conda env..."

# Activate your conda environment
source ~/miniconda3/bin/activate
conda activate py310env

export WANDB_API_KEY=28c48f9795c35a55ee1bf7a82a05994bf808294a

start=`date +%s`


# Run your Python script
echo "Running training script..."
cd /work/dlclarge2/raneb-project/reward-augmentation
export PYTHONPATH=$PWD
python agents/dqn_cfn_mrl.py

end=`date +%s`
runtime=$((end-start))

echo "Finished at $runtime"