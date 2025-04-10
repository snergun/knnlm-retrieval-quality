#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --job-name=simple_knnlm
#SBATCH --output=log/simple_knnlm_%j.out

# Load modules and activate environment
unset PYTHONPATH
unset PYTHONHOME
module --force purge
module reset
module load anaconda3_gpu

cd /work/hdd/bcyi/$USER/knnlm-retrieval-quality

python simple_knn_eval_with_bins.py