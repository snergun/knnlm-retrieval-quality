#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --account=bcyi-delta-gpu
#SBATCH --time=00:20:00
#SBATCH --job-name=test_gpu_transfer
#SBATCH --output=log/test_gpu_transfer%j.out
#SBATCH --error=log/test_gpu_transfer%j.err
module purge
module reset
module load anaconda3_gpu
source activate faiss-gpu

python my_scripts/test_gpu_transfer.py \
    --faiss_index /work/nvme/bcyi/eergun/datastore/train/knn_ip.index \
    --dimension 1024 \
    --ncentroids 4096 \
    --code_size 64 \
    --probe 8 \
    --metric ip \
    --n_test_queries 1000 \
    --k 10