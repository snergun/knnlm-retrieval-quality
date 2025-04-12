#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G  # Increased memory
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=6:00:00  # Increased time allowance
#SBATCH --job-name=exact_dist
#SBATCH --output=log/exact_dist_%j.out

# Load modules and activate environment
module purge
module reset
module load anaconda3_gpu
source activate knnlm

cd /work/hdd/bcyi/$USER/knnlm-retrieval-quality

# Set the mode (valid or test)
SPLIT=valid

# Make sure the tmp cache directory exists
mkdir -p /tmp/$SPLIT.cache

# Only run the exact distance calculation step
python rq/fast_evaluate.py \
    --preset wiki_$SPLIT \
    --save-exact \
    --load-pct 10  # Use a smaller percentage to reduce memory usage