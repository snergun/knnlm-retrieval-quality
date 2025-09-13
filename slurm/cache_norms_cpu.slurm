#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=cache_norms
#SBATCH --output=log/cache_norms_%j.out
#SBATCH --account=bcyi-delta-cpu

module purge
module reset
module load anaconda3_cpu

python scripts/cache_neighbor_norms.py

