#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --time=2:00:00
#SBATCH --job-name=download_wiki
#SBATCH --output=log/download_wiki_%j.out
#SBATCH --error=log/download_wiki_%j.err
#SBATCH --account=bcyi-delta-cpu

# Load modules and activate environment
# We can use the default anaconda3_gpu module for downloading the dataset (no fairseq needed)
module purge
module reset
module load anaconda3_gpu
# Create directories
mkdir -p data/wikitext-103

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
        if len(item['text']) == 0:
            f.write('\n')
            continue
        f.write(item['text'])

with open('data/wikitext-103/wiki.valid.tokens', 'w', encoding='utf-8') as f:
    for item in dataset['validation']:
        if len(item['text']) == 0:
            f.write('\n')
            continue
        f.write(item['text'])

with open('data/wikitext-103/wiki.test.tokens', 'w', encoding='utf-8') as f:
    for item in dataset['test']:
        if len(item['text']) == 0:
            f.write('\n')
            continue
        f.write(item['text'])

print('Wikitext-103 dataset has been successfully saved.')
"