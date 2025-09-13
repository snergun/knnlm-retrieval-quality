#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --job-name=check_faiss
#SBATCH --output=log/check_faiss%j.out
#SBATCH --account=bcyi-delta-gpu

# Load module
conda deactivate
module purge
module reset
module load anaconda3_gpu

# Activate the environment
source activate faiss-gpu

python -u scripts/check_faiss.py
