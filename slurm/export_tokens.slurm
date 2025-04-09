#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --account=bcyi-delta-cpu
#SBATCH --time=2:00:00
#SBATCH --job-name=export_tokens
#SBATCH --output=log/export_tokens_%j.out
#SBATCH --error=log/export_tokens_%j.err

# Load modules and activate environment
module purge 
module reset
module load anaconda3_gpu
source activate knnlm

python export_tokens.py \
    --data-dir data-bin/wikitext-103 \
    --split test \
    --output-dir data/wikitext-103-preprocessed \
    --format torch