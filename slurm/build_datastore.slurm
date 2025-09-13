#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --job-name=build_datastore
#SBATCH --error=log/knnlm_build_datastore_%j.err
#SBATCH --output=log/knnlm_build_datastore_%j.out

# Load modules and activate environment
module purge
module reset
module load anaconda3_gpu
source activate knnlm-retr

DSTORE_DIR=/work/nvme/bcyi/eergun/datastore/train
# Create directories for datastores
mkdir -p $DSTORE_DIR

# Step 1: Save keys and values to datastore
CHECKPOINT=adaptive_lm_wiki103.v2/model.pt

# DSTORE_SIZE=103225485  # Number of tokens in the training set

# python eval_lm.py data-bin/wikitext-103 \
#     --path $CHECKPOINT \
#     --sample-break-mode none --max-tokens 3072 \
#     --softmax-batch 1024 --gen-subset train \
#     --context-window 1536 --tokens-per-sample 1536 \
#     --dstore-mmap $DSTORE_DIR/dstore --knn-keytype 'decoder_out' \
#     --dstore-size $DSTORE_SIZE --model-overrides "{'knn_keytype': 'decoder_out'}" \
#     --save-knnlm-dstore --fp16 --dstore-fp16

DSTORE_SIZE=217646  # Number of tokens in the validation set

DSTORE_DIR=/work/nvme/bcyi/eergun/datastore/decoder_out/valid
# Create directories for datastores
mkdir -p $DSTORE_DIR

python eval_lm.py data-bin/wikitext-103 \
    --path $CHECKPOINT \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --context-window 2560 --tokens-per-sample 512 \
    --dstore-mmap $DSTORE_DIR/dstore --knn-keytype 'decoder_out' \
    --dstore-size $DSTORE_SIZE --model-overrides "{'knn_keytype': 'decoder_out'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

DSTORE_SIZE=245569  # Number of tokens in the test set
DSTORE_DIR=/work/nvme/bcyi/eergun/datastore/decoder_out/test
# Create directories for datastores
mkdir -p $DSTORE_DIR

python eval_lm.py data-bin/wikitext-103 \
    --path $CHECKPOINT \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset test \
    --context-window 2560 --tokens-per-sample 512 \
    --dstore-mmap $DSTORE_DIR/dstore --knn-keytype 'decoder_out' \
    --dstore-size $DSTORE_SIZE --model-overrides "{'knn_keytype': 'decoder_out'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16