#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64 
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=knnlm_env_setup
#SBATCH --output=log/knnlm_setup_%j.out
#SBATCH --error=log/knnlm_setup_%j.err
#SBATCH --account=bcyi-delta-gpu

# Load module
module load anaconda3_gpu

# Activate the environment
source activate knnlm

# Install main dependencies
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

# Clone the repository if needed (or you can upload it)
cd /work/hdd/bcyi/$USER # Or your preferred location
if [ ! -d "knnlm-retrieval-quality" ]; then
    git clone https://github.com/snergun/knnlm-retrieval-quality.git
fi
cd knnlm-retrieval-quality

# Install the package in editable mode
pip install --editable .
pip install faiss-cpu

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import faiss; print('FAISS installed successfully')"

echo "Environment setup complete. You can now use 'source activate knnlm' in future jobs."