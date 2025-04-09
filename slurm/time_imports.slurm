#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --job-name=time_imports_unset
#SBATCH --output=log/import_timing_%j.out
#SBATCH --error=log/import_timing_%j.err
#SBATCH --account=bcyi-delta-gpu

# Load modules and activate environment
echo "Unsetting PYTHONPATH and PYTHONHOME"
unset PYTHONPATH
unset PYTHONHOME
module --force purge
module reset

module load anaconda3_gpu
source activate knnlm

# Create log directory if it doesn't exist
mkdir -p log

# Go to the correct directory
cd /work/hdd/bcyi/$USER/knnlm-retrieval-quality

# Run Python script to time imports using heredoc
cat > import_timing_test.py << 'EOL'
import time
import sys

def log_with_timestamp(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

log_with_timestamp("Script starting")

# Time base imports
start_time = time.time()
log_with_timestamp("Importing numpy...")
import numpy as np
log_with_timestamp(f"numpy imported in {time.time() - start_time:.2f}s")

start_time = time.time()
log_with_timestamp("Importing torch...")
import torch
log_with_timestamp(f"torch imported in {time.time() - start_time:.2f}s")

# Print CUDA info
if torch.cuda.is_available():
    log_with_timestamp(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    log_with_timestamp(f"Current device: {torch.cuda.current_device()}")
    log_with_timestamp(f"Device name: {torch.cuda.get_device_name()}")
else:
    log_with_timestamp("CUDA is NOT available")

# Time knnlm-specific imports
start_time = time.time()
log_with_timestamp("Importing tqdm...")
from tqdm import tqdm
log_with_timestamp(f"tqdm imported in {time.time() - start_time:.2f}s")

start_time = time.time()
log_with_timestamp("Importing faiss...")
import faiss
log_with_timestamp(f"faiss imported in {time.time() - start_time:.2f}s")

start_time = time.time()
log_with_timestamp("Importing data_structures...")
from rq.data_structures import Dataset, Dstore
log_with_timestamp(f"data_structures imported in {time.time() - start_time:.2f}s")

start_time = time.time()
log_with_timestamp("Importing vocab...")
from rq.vocab import Dictionary
log_with_timestamp(f"vocab imported in {time.time() - start_time:.2f}s")

start_time = time.time()
log_with_timestamp("Importing knnlm_func...")
import rq.knnlm_func
log_with_timestamp(f"knnlm_func imported in {time.time() - start_time:.2f}s")

log_with_timestamp("All imports completed")

# Optional - time basic torch operations
log_with_timestamp("Testing basic torch operations...")
start_time = time.time()
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
torch.cuda.synchronize()
log_with_timestamp(f"Basic CUDA tensor operations completed in {time.time() - start_time:.2f}s")

log_with_timestamp("Script completed")
EOL

# Execute the script
python import_timing_test.py

# Clean up
rm import_timing_test.py