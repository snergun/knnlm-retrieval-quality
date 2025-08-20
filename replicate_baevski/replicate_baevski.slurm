#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=replicate
#SBATCH --output=replicate_baevski/log/%j.out

# Load modules and activate environment
conda deactivate
unset PYTHONPATH
unset PYTHONHOME
module --force purge
module reset
module load anaconda3_gpu
## Comment the next line to evaluate from cache using anaconda3_gpu (much faster)
source activate knnlm

python train.py --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/transformer_wikitext-103 \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --lr 1.0 --max-lr 1.01 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --optimizer nag --min-lr 1e-08 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --fp16\