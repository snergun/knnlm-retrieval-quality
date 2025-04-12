#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --job-name=orig_eval
#SBATCH --output=log/orig_eval_%j.out

# Load modules and activate environment
unset PYTHONPATH
unset PYTHONHOME
module --force purge
module reset
module load anaconda3_gpu
source activate knnlm

# Create temp directories if they don't exist
mkdir -p /tmp/train
mkdir -p /tmp/bin

# Copy necessary files to local storage for faster access
echo "Copying datastore files to /tmp..."
if [ ! -f /tmp/train/dstore_keys.npy ]; then
  cp datastore/wikitext-103/train/dstore_keys.npy /tmp/train/
fi
if [ ! -f /tmp/train/dstore_vals.npy ]; then
  cp datastore/wikitext-103/train/dstore_vals.npy /tmp/train/
fi
if [ ! -f /tmp/train/knn.index ]; then
  cp datastore/wikitext-103/train/knn.index /tmp/train/
  cp datastore/wikitext-103/train/knn.index.trained /tmp/train/
fi

# Copy binary files (data-bin) if needed
# This is optional but could speed things up
if [ ! -d /tmp/bin/wikitext-103 ]; then
  mkdir -p /tmp/bin/wikitext-103
  cp -r data-bin/wikitext-103/* /tmp/bin/wikitext-103/
fi

echo "Starting evaluation..."

# Run evaluation with paths pointing to /tmp
python eval_lm.py /tmp/bin/wikitext-103 \
    --path adaptive_lm_wiki103.v2/model.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename /tmp/train/dstore \
    --indexfile /tmp/train/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 103225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 \
    --no-load-keys --knn-sim-func "do_not_recomp_l2" \