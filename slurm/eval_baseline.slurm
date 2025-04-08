#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_baseline_val
#SBATCH --output=log/eval_baseline_%j.out

# Load modules and activate environment
module purge
module reset
module load anaconda3_gpu
source activate knnlm

cd /work/hdd/bcyi/$USER/knnlm-retrieval-quality
CHECKPOINTPATH=checkpoints/wt103_checkpoint_best.pt
echo "Evaluating baseline model on validation set..."
echo "Model path: $CHECKPOINTPATH" 
python eval_lm.py data-bin/wikitext-103 \
    --path $CHECKPOINTPATH \
    --max-tokens 3072 \
    --context-window 2560 \
    --gen-subset valid \
    --sample-break-mode complete \
    --softmax-batch 1024 \
