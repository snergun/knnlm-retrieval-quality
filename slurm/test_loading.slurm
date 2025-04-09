#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --job-name=test_loading
#SBATCH --output=log/test_loading_%j.out

# Load modules and activate environment
unset PYTHONPATH
unset PYTHONHOME
module --force purge
module reset
module load anaconda3_gpu

python test_loading.py