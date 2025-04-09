#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --job-name=eval_knn_lm
#SBATCH --output=log/eval_knn_lm_%j.out

# Load modules and activate environment
unset PYTHONPATH
unset PYTHONHOME
module --force purge
module reset
module load anaconda3_gpu
source activate knnlm

python eval_lm.py data-bin/wikitext-103 \
    --path adaptive_lm_wiki103.v2/model.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename datastore/wikitext-103/train/dstore \
    --indexfile datastore/wikitext-103/train/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 103225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16