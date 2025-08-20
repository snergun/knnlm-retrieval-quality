#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --account=bcyi-delta-cpu
#SBATCH --time=1:00:00
#SBATCH --job-name=save_cache_to_hf
#SBATCH --output=log/save_cache_to_hf%j.out
#SBATCH --error=log/save_cache_to_hf%j.err

# Load modules and activate environment
module load anaconda3_cpu
python my_scripts/save_cache_to_HF.py