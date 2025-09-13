#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --account=bcyi-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=knnlm_build_faiss
#SBATCH --output=log/knnlm_build_faiss_%j.out
#SBATCH --error=log/knnlm_build_faiss_%j.err

# Load modules and activate environment
module purge
module reset
module load anaconda3_gpu
source activate knnlm-retr

# Step 1: Save keys and values to datastore
DSTORE_DIR=/work/nvme/bcyi/eergun/datastore/train
DSTORE_SIZE=103225485  # Number of tokens in the training set

python -u build_dstore.py \
    --dstore_mmap $DSTORE_DIR/dstore \
    --dstore_size $DSTORE_SIZE \
    --dimension 1024 \
    --faiss_index $DSTORE_DIR/knn_ip.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 \
    --metric ip \
    --dstore_fp16 \