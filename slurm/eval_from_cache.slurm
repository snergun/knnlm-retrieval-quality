#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --job-name=knnlm_eval_valid
#SBATCH --output=log/knnlm_eval_valid_%j.out

# Load modules and activate environment
unset PYTHONPATH
unset PYTHONHOME
module --force purge
module reset
module load anaconda3_gpu
## Comment the next line to evaluate from cache using anaconda3_gpu (much faster)
# source activate knnlm

cd /work/hdd/bcyi/$USER/knnlm-retrieval-quality

python rq/fast_evaluate.py \
    --preset wiki_valid \
    --exact \
    --from_cache \
    --validation_split \
    --external_lm_prob /u/eergun/pos_lm/misc/ADT_validation_results.pt \
