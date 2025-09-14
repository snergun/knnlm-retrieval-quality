import argparse
import os
import numpy as np
import faiss
import time
import ctypes

parser = argparse.ArgumentParser()
parser.add_argument('--faiss_index', type=str, required=True, help='path to the trained faiss index')
parser.add_argument('--valid_mmap', type=str, help='memmap for validation keys')
parser.add_argument('--test_mmap', type=str, help='memmap for test keys')
parser.add_argument('--valid_size', type=int, help='number of validation items')
parser.add_argument('--test_size', type=int, help='number of test items')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids (must match trained index)')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors (must match trained index)')
parser.add_argument('--dstore_fp16', default=False, action='store_true')
parser.add_argument('--k', type=int, default=1024, help='number of nearest neighbors to retrieve')
parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size for processing queries')
parser.add_argument('--metric', type=str, default='l2', help='distance metric used', choices=['l2', 'ip', 'cos'])
parser.add_argument('--output_dir', type=str, default='.', help='directory to save cached results')


args = parser.parse_args()
print("Arguments:")
print(args)

# Memory mapping optimization
def optimize_memmap(memmap_array):
    """Apply memory advice for sequential access"""
    madvise = ctypes.CDLL("libc.so.6").madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    assert madvise(memmap_array.ctypes.data, memmap_array.size * memmap_array.dtype.itemsize, 1) == 0, "MADVISE FAILED"

def load_memmap_keys(mmap_path, size, dimension, fp16=False):
    """Load memory mapped keys"""
    dtype = np.float16 if fp16 else np.float32
    keys = np.memmap(mmap_path + '_keys.npy', dtype=dtype, mode='r', shape=(size, dimension))
    optimize_memmap(keys)
    return keys

def search_and_save_knns(index, query_keys, output_prefix, k, batch_size, metric):
    """Search for KNNs and save results"""
    n_queries = query_keys.shape[0]
    
    # Pre-allocate arrays for results
    all_distances = np.zeros((n_queries, k), dtype=np.float32)
    all_indices = np.zeros((n_queries, k), dtype=np.int64)
    
    print(f"Processing {n_queries} queries in batches of {batch_size}")
    start_time = time.time()
    
    for i in range(0, n_queries, batch_size):
        batch_start = time.time()
        end_idx = min(i + batch_size, n_queries)
        batch_queries = query_keys[i:end_idx].copy().astype(np.float32)
        
        # Normalize for cosine similarity
        if metric == "cos":
            faiss.normalize_L2(batch_queries)
        
        # Search
        distances, indices = index.search(batch_queries, k)
        
        # Store results
        all_distances[i:end_idx] = distances
        all_indices[i:end_idx] = indices
        
        if (i // batch_size + 1) % 4 == 0:
            batch_time = time.time() - batch_start
            avg_time = (time.time() - start_time) / (i // batch_size + 1)
            remaining_batches = (n_queries - end_idx) // batch_size
            eta = remaining_batches * avg_time
            print(f"Processed batch {i//batch_size + 1}/{(n_queries-1)//batch_size + 1} "
                  f"({end_idx}/{n_queries} queries) in {batch_time:.2f}s, ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Total search time: {total_time:.2f}s ({n_queries/total_time:.1f} queries/sec)")
    
    # Save results

    knn_file = os.path.join(output_prefix, f"knns_{metric}.npy")
    print(f"Saving KNN indices to {knn_file}")
    np.save(knn_file, all_indices)
    
    dist_file = os.path.join(output_prefix, f"dist_{metric}.npy")
    print(f"Saving distances to {dist_file}")
    np.save(dist_file, all_distances)
    
    return all_distances, all_indices

# Load the trained FAISS index
print("Loading FAISS index...")
start = time.time()
cpu_index = faiss.read_index(args.faiss_index)
print(f"Loading index took {time.time() - start:.2f}s")
print(f"Index contains {cpu_index.ntotal} vectors")

# Set probe parameter
cpu_index.nprobe = args.probe
print(f"Set nprobe to {args.probe}")

# Try to move index to GPU if available
ngpus = faiss.get_num_gpus()
print("Number of GPUs detected by Faiss:", ngpus)
if ngpus > 0:
    print("Moving index to GPU...")
    start = time.time()
    res = faiss.StandardGpuResources()
    co = faiss.GpuIndexIVFPQConfig()
    co.device = 0
    co.useFloat16LookupTables = True
    gpu_index = faiss.GpuIndexIVFPQ(
        res, args.dimension,
        args.ncentroids, args.code_size, 8,  # 8 = nbits
        faiss.METRIC_L2 if args.metric == 'l2' else faiss.METRIC_INNER_PRODUCT,
        co
    )

    print("Copying trained parameters from CPU to GPU...")
    gpu_index.copyFrom(cpu_index)
    gpu_index.nprobe = args.probe
    
    transfer_time = time.time() - start
    print(f"GPU transfer completed in {transfer_time:.4f}s")
    index = gpu_index
else:
    index = cpu_index
    print("No GPU available, using CPU index")

print("Index type:", type(index))

# Process validation set if provided
if args.valid_mmap and args.valid_size:
    print("\nProcessing validation set...")
    valid_keys = load_memmap_keys(args.valid_mmap, args.valid_size, args.dimension, args.dstore_fp16)
    output_prefix = os.path.join(args.output_dir, "valid.cache")
    
    valid_distances, valid_knns = search_and_save_knns(
        index, valid_keys, output_prefix, args.k, args.batch_size, args.metric
    )
    print(f"Validation set processed: {valid_knns.shape}")

# Process test set if provided
if args.test_mmap and args.test_size:
    print("\nProcessing test set...")
    test_keys = load_memmap_keys(args.test_mmap, args.test_size, args.dimension, args.dstore_fp16)
    output_prefix = os.path.join(args.output_dir, "test.cache")
    
    test_distances, test_knns = search_and_save_knns(
        index, test_keys, output_prefix, args.k, args.batch_size, args.metric
    )
    print(f"Test set processed: {test_knns.shape}")

print("\nKNN caching completed!")

# Print some statistics
if args.valid_mmap and args.valid_size:
    print(f"\nValidation set statistics:")
    print(f"  Shape: {valid_knns.shape}")
    print(f"  Mean distance: {valid_distances.mean():.4f}")
    print(f"  Min distance: {valid_distances.min():.4f}")
    print(f"  Max distance: {valid_distances.max():.4f}")

if args.test_mmap and args.test_size:
    print(f"\nTest set statistics:")
    print(f"  Shape: {test_knns.shape}")
    print(f"  Mean distance: {test_distances.mean():.4f}")
    print(f"  Min distance: {test_distances.min():.4f}")
    print(f"  Max distance: {test_distances.max():.4f}")