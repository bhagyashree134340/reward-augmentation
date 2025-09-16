#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080    
#SBATCH --gres=gpu:1                             
#SBATCH -t 0-20:00                  
#SBATCH -c 1                        
#SBATCH -o log/mrl.%j.out         # STDOUT log (job ID in filename)
#SBATCH -e log/mrl.%j.err         # STDERR log (job ID in filename)
#SBATCH -J mrl-dqn           # Job name

echo "Working dir: $PWD"
echo "Started at $(date)"
echo "Running on $(hostname)"
echo "Loading conda env..."

# Activate your conda environment
source ~/miniconda3/bin/activate
conda activate py310env

export WANDB_API_KEY=c2dddd2dd918a67e72290a2c5ab4247fe12be92b

start=`date +%s`


# Run your Python script
echo "Running training script..."
cd /work/dlclarge2/raneb-project/reward-augmentation
export PYTHONPATH=$PWD
python agents/mrl-dqn.py

end=`date +%s`
runtime=$((end-start))

echo "Finished at $runtime"