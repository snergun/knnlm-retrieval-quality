#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --job-name=knnlm_build_datastore
#SBATCH --output=log/knnlm_build_datastore_%j.out

# Load modules and activate environment
module load anaconda3_gpu
source activate knnlm

cd /work/hdd/bcyi/$USER/knnlm-retrieval-quality

# Create directories for datastores
mkdir -p datastore/wikitext-103/train
mkdir -p datastore/wikitext-103/valid

# Step 1: Save keys and values to datastore
CHECKPOINT=adaptive_lm_wiki103.v2/model.pt
DSTORE_DIR=datastore/wikitext-103/train
DSTORE_SIZE=103225485  # Number of tokens in the training set

python eval_lm.py data-bin/wikitext-103 \
    --path $CHECKPOINT \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap $DSTORE_DIR/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size $DSTORE_SIZE --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16
