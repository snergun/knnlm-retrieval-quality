#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --job-name=knnlm_eval_valid
#SBATCH --output=log/knnlm_eval_valid_%j.out

# Load modules and activate environment
module load anaconda3_gpu
source activate knnlm

cd /work/hdd/bcyi/$USER/knnlm-retrieval-quality
# Create directory for validation datastore
mkdir -p datastore/wikitext-103/valid
mkdir -p datastore/wikitext-103/valid.cache

# Step 0: First build the datastore for validation (if not already done)
CHECKPOINT=adaptive_lm_wiki103.v2/model.pt
DSTORE_DIR=datastore/wikitext-103/valid
DSTORE_SIZE=217646  # Number of tokens in the validation set

# Check if validation datastore already exists
if [ -f "$DSTORE_DIR/dstore_keys.npy" ] && [ -f "$DSTORE_DIR/dstore_vals.npy" ] && [ -f "$DSTORE_DIR/dstore_prob.npy" ]; then
    echo "Validation datastore already exists. Skipping datastore creation step."
else
    echo "Creating validation datastore..."
    
    # Create directory if it doesn't exist
    mkdir -p $DSTORE_DIR

    python eval_lm.py data-bin/wikitext-103 \
        --path $CHECKPOINT \
        --sample-break-mode none --max-tokens 3072 \
        --softmax-batch 1024 --gen-subset valid \
        --context-window 1536 --tokens-per-sample 1536 \
        --dstore-mmap $DSTORE_DIR/dstore --knn-keytype 'last_ffn_input' \
        --dstore-size $DSTORE_SIZE --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
        --save-knnlm-dstore --fp16 --dstore-fp16
fi

# Step 1: Save the nearest neighbors (cache the retrieval)
python rq/fast_evaluate.py \
    --preset wiki_valid \
    --save-knns

# Step 2: Calculate exact distances for the retrieved neighbors
python rq/fast_evaluate.py \
    --preset wiki_valid \
    --save-exact \
    --load-pct 20

# Step 3: Compute perplexity using exact distances
python rq/fast_evaluate.py \
    --preset wiki_valid \
    --exact
