#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=cuda_test
#SBATCH --output=log/cuda_test_%j.out
#SBATCH --error=log/cuda_test_%j.err
#SBATCH --account=bcyi-delta-gpu

# Ensure clean environment
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

# Create the test script
cat > cuda_timing_test.py << 'EOL'
import time
import torch
import gc

def log_with_timestamp(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

def run_cuda_test(size=1000, iter_count=3, test_name="Standard matrix multiplication"):
    log_with_timestamp(f"Starting {test_name} test - Matrix size: {size}x{size}, Iterations: {iter_count}")
    
    # Initial warmup - not timed
    log_with_timestamp("Running warmup operation...")
    x_warmup = torch.randn(10, 10).cuda()
    y_warmup = torch.randn(10, 10).cuda()
    z_warmup = torch.matmul(x_warmup, y_warmup)
    torch.cuda.synchronize()
    del x_warmup, y_warmup, z_warmup
    gc.collect()
    torch.cuda.empty_cache()
    log_with_timestamp("Warmup complete")
    
    # Test operations
    total_time = 0
    
    for i in range(iter_count):
        # Allocate tensors
        allocation_start = time.time()
        x = torch.randn(size, size).cuda()
        y = torch.randn(size, size).cuda()
        torch.cuda.synchronize()
        allocation_time = time.time() - allocation_start
        log_with_timestamp(f"Iteration {i+1}: Tensor allocation took {allocation_time:.4f}s")
        
        # Run computation
        compute_start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        compute_time = time.time() - compute_start
        log_with_timestamp(f"Iteration {i+1}: Computation took {compute_time:.4f}s")
        
        # Total time for this iteration
        iteration_time = allocation_time + compute_time
        total_time += iteration_time
        log_with_timestamp(f"Iteration {i+1}: Total time: {iteration_time:.4f}s")
        
        # Clean up to avoid memory accumulation
        del x, y, z
        gc.collect()
        torch.cuda.empty_cache()
        
        # Small delay between iterations
        time.sleep(1)
    
    avg_time = total_time / iter_count
    log_with_timestamp(f"Test complete. Average time per iteration: {avg_time:.4f}s")
    return avg_time

# Initialize CUDA
log_with_timestamp("Script starting")
log_with_timestamp("Importing torch...")
import_start = time.time()
import torch
import_time = time.time() - import_start
log_with_timestamp(f"torch imported in {import_time:.2f}s")

if not torch.cuda.is_available():
    log_with_timestamp("CUDA is NOT available. Exiting.")
    exit(1)

log_with_timestamp(f"CUDA is available. Device count: {torch.cuda.device_count()}")
log_with_timestamp(f"Current device: {torch.cuda.current_device()}")
log_with_timestamp(f"Device name: {torch.cuda.get_device_name()}")
log_with_timestamp(f"CUDA version: {torch.version.cuda}")

# Get GPU memory information
log_with_timestamp("GPU Memory Information:")
log_with_timestamp(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
log_with_timestamp(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
log_with_timestamp(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# First test with medium matrices
run_cuda_test(size=1000, iter_count=5, test_name="Initial test")

# Second test with larger matrices
run_cuda_test(size=2000, iter_count=3, test_name="Larger matrix test")

# Third test with smaller matrices but more iterations
run_cuda_test(size=500, iter_count=10, test_name="Multiple iterations test")

log_with_timestamp("Testing memory transfer speed...")
for size_mb in [10, 100, 500]:
    size = int((size_mb * 1024 * 1024) / 4)  # Convert MB to float32 elements
    dim = int(size ** 0.5)  # Square matrix dimension
    
    log_with_timestamp(f"Transfer test: {size_mb}MB...")
    
    # CPU to GPU transfer
    start_time = time.time()
    x_cpu = torch.randn(dim, dim)
    x_gpu = x_cpu.cuda()
    torch.cuda.synchronize()
    cpu_to_gpu_time = time.time() - start_time
    log_with_timestamp(f"CPU to GPU transfer: {cpu_to_gpu_time:.4f}s ({size_mb/cpu_to_gpu_time:.2f} MB/s)")
    
    # GPU to CPU transfer
    start_time = time.time()
    x_cpu_2 = x_gpu.cpu()
    gpu_to_cpu_time = time.time() - start_time
    log_with_timestamp(f"GPU to CPU transfer: {gpu_to_cpu_time:.4f}s ({size_mb/gpu_to_cpu_time:.2f} MB/s)")
    
    # Clean up
    del x_cpu, x_gpu, x_cpu_2
    gc.collect()
    torch.cuda.empty_cache()

log_with_timestamp("All tests completed")
EOL

# Execute the script
python cuda_timing_test.py

# Clean up
rm cuda_timing_test.py