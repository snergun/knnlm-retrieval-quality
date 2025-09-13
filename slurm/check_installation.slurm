#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=check_cuda
#SBATCH --output=log/check_cuda_%j.out
#SBATCH --account=bcyi-delta-gpu

# Load module
module load anaconda3_gpu

# Activate the environment
source activate knnlm-retr

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import faiss; print('FAISS installed successfully'); print('GPUs:', faiss.get_num_gpus()); res = faiss.StandardGpuResources(); print('OK')"\
