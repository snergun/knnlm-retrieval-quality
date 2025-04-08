#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=knnlm_build_faiss
#SBATCH --output=log/knnlm_build_faiss_%j.out

# Load modules and activate environment
module load anaconda3_gpu
source activate knnlm

# Step 1: Save keys and values to datastore
CHECKPOINT=adaptive_lm_wiki103.v2/model.pt
DSTORE_DIR=datastore/wikitext-103/train
DSTORE_SIZE=103225485  # Number of tokens in the training set

python build_dstore.py \
    --dstore_mmap $DSTORE_DIR/dstore \
    --dstore_size $DSTORE_SIZE \
    --dimension 1024 \
    --faiss_index $DSTORE_DIR/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 