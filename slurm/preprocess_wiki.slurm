#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --time=1:00:00
#SBATCH --job-name=preprocess_wiki
#SBATCH --output=log/preprocess_wiki_%j.out
#SBATCH --error=log/preprocess_wiki_%j.err
#SBATCH --account=bcyi-delta-cpu

# Load modules and activate environment
module purge
module reset
module load anaconda3_cpu
source activate knnlm
# Create directories
mkdir -p data/wikitext-103
#Preprocess the dataset
python preprocess.py --only-source \
    --trainpref data/wikitext-103/wiki.train.tokens \
    --validpref data/wikitext-103/wiki.valid.tokens \
    --testpref data/wikitext-103/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 16