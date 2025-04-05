#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=check_cuda
#SBATCH --output=log/check_cuda_%j.out
#SBATCH --error=log/check_cuda_%j.err
#SBATCH --account=bcyi-delta-gpu

# Load module
module load anaconda3_gpu

# Activate the environment
source activate knnlm

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import faiss; print('FAISS installed successfully')"
