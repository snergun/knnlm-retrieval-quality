#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=copy_datastore
#SBATCH --output=log/copy_datastore_%j.out
#SBATCH --partition=cpu
#SBATCH --account=bcyi-delta-cpu
rsync -ah --progress datastore/ /work/nvme/bcyi/eergun/datastore