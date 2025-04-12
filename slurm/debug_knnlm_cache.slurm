#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --account=bcyi-delta-cpu
#SBATCH --time=2:00:00
#SBATCH --job-name=debug_knnlm_cache
#SBATCH --output=log/debug_knnlm_cache_%j.out

# Load modules and activate environment
module --force purge
module reset
module load anaconda3_gpu
source activate knnlm


python debug_knnlm_cache.py