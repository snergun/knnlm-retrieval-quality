#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --job-name=knnlm_data_prep
#SBATCH --output=log/knnlm_data_prep_%j.out
#SBATCH --error=log/knnlm_data_prep_%j.err
#SBATCH --account=bcyi-delta-gpu

# Load modules and activate environment
module load anaconda3_gpu
source activate knnlm

# Create directories
mkdir -p data/wikitext-103

if [ -f "data/wikitext-103/wiki.train.tokens" ] && [ -f "data/wikitext-103/wiki.valid.tokens" ] && [ -f "data/wikitext-103/wiki.test.tokens" ]; then
    echo "Dataset files already exist. Skipping download step."
else
    echo "Dataset files not found. Downloading wikitext-103..."
    
    # Download using Hugging Face datasets
    pip install datasets

    # Download and save wikitext-103
    python -c "
from datasets import load_dataset
import os

# Create directory
os.makedirs('data/wikitext-103', exist_ok=True)

# Load and save the dataset
dataset = load_dataset('wikitext', 'wikitext-103-v1')

# Save train, validation, and test sets
with open('data/wikitext-103/wiki.train.tokens', 'w', encoding='utf-8') as f:
    for item in dataset['train']:
        f.write(item['text'] + '\n')

with open('data/wikitext-103/wiki.valid.tokens', 'w', encoding='utf-8') as f:
    for item in dataset['validation']:
        f.write(item['text'] + '\n')

with open('data/wikitext-103/wiki.test.tokens', 'w', encoding='utf-8') as f:
    for item in dataset['test']:
        f.write(item['text'] + '\n')

print('Wikitext-103 dataset has been successfully saved.')
"
fi
# Preprocess the data
python preprocess.py --only-source \
    --trainpref data/wikitext-103/wiki.train.tokens \
    --validpref data/wikitext-103/wiki.valid.tokens \
    --testpref data/wikitext-103/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20