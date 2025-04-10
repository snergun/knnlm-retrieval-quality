#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --job-name=create_cache
#SBATCH --output=log/create_cache_%j.out

# Load modules and activate environment
module purge
module reset
module load anaconda3_gpu
source activate knnlm

cd /work/hdd/bcyi/$USER/knnlm-retrieval-quality

# Set the mode (valid or test)
SPLIT=test  # Default to 'valid' if not set

# Create directory for validation datastore
mkdir -p datastore/wikitext-103/$SPLIT
mkdir -p datastore/wikitext-103/$SPLIT.cache

# Step 0: First build the datastore for split (if not already done)
CHECKPOINT=adaptive_lm_wiki103.v2/model.pt
DSTORE_DIR=datastore/wikitext-103/$SPLIT
# Set DSTORE_SIZE based on SCRIPT_MODE
if [ "$SPLIT" == "valid" ]; then
    DSTORE_SIZE=217646
elif [ "$SPLIT" == "test" ]; then
    DSTORE_SIZE=245569
else
    echo "Invalid SCRIPT_MODE: $SPLIT. Must be 'valid' or 'test'."
    exit 1
fi


echo "Creating $SPLIT datastore..."

# Create directory if it doesn't exist
mkdir -p $DSTORE_DIR

python eval_lm.py data-bin/wikitext-103 \
    --path $CHECKPOINT \
    --sample-break-mode complete --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset $SPLIT \
    --context-window 2560 \
    --dstore-mmap $DSTORE_DIR/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size $DSTORE_SIZE --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16


# Step 1: Save the nearest neighbors (cache the retrieval)
python rq/fast_evaluate.py \
    --preset wiki_$SPLIT \
    --save-knns

# Step 2: Calculate exact distances for the retrieved neighbors
python rq/fast_evaluate.py \
    --preset wiki_$SPLIT \
    --save-exact \
    --load-pct 20
    