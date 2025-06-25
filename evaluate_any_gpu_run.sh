#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080     
#SBATCH --gres=gpu:1                
#SBATCH --mem=2000              
#SBATCH -t 0-00:10                  
#SBATCH -c 1                        
#SBATCH -o log/evaluate.%j.out         # STDOUT log (job ID in filename)
#SBATCH -e log/evaluate.%j.err         # STDERR log (job ID in filename)
#SBATCH -J reward-evaluate           # Job name

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
python evaluate_any.py

end=`date +%s`
runtime=$((end-start))

echo "Finished at $runtime"