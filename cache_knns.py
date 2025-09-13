import argparse
import numpy as np
import faiss
import time

parser = argparse.ArgumentParser()
parser.add_argument('--faiss_index', type=str, required=True, help='path to the trained faiss index')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids (must match trained index)')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors (must match trained index)')
parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
parser.add_argument('--metric', type=str, default='l2', help='distance metric used', choices=['l2', 'ip', 'cos'])
parser.add_argument('--n_test_queries', type=int, default=100, help='number of test queries to generate')
parser.add_argument('--k', type=int, default=10, help='number of nearest neighbors for test')

args = parser.parse_args()
print("Test Arguments:")
print(args)

def create_test_queries(n_queries, dimension, metric):
    """Create random test queries"""
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    if metric == "cos":
        faiss.normalize_L2(queries)
    return queries

def test_search_performance(index, queries, k, name):
    """Test search performance and return results"""
    print(f"\n--- Testing {name} ---")
    print(f"Index type: {type(index)}")
    print(f"Index ntotal: {index.ntotal}")
    print(f"Index nprobe: {index.nprobe}")
    
    # Warm up
    index.search(queries[:10], k)
    
    # Actual test
    start_time = time.time()
    distances, indices = index.search(queries, k)
    search_time = time.time() - start_time
    
    print(f"Search time: {search_time:.4f}s ({len(queries)/search_time:.1f} queries/sec)")
    print(f"Result shape: {distances.shape}")
    print(f"Sample distances (first query): {distances[0][:5]}")
    print(f"Sample indices (first query): {indices[0][:5]}")
    
    return distances, indices, search_time

# Load CPU index
print("Loading CPU FAISS index...")
start = time.time()
cpu_index = faiss.read_index(args.faiss_index)
print(f"Loading took {time.time() - start:.4f}s")
print(f"CPU index contains {cpu_index.ntotal} vectors")

# Set probe parameter
cpu_index.nprobe = args.probe

# Create test queries
print(f"\nGenerating {args.n_test_queries} random test queries...")
test_queries = create_test_queries(args.n_test_queries, args.dimension, args.metric)

# Test CPU performance
cpu_distances, cpu_indices, cpu_time = test_search_performance(cpu_index, test_queries, args.k, "CPU Index")

# Check GPU availability
ngpus = faiss.get_num_gpus()
print(f"\nNumber of GPUs detected: {ngpus}")

if ngpus == 0:
    print("No GPU available, test completed.")
    exit(0)

# Try to create and copy to GPU index
try:
    print("\n=== Attempting GPU Transfer ===")
    start = time.time()
    
    # Initialize GPU index with proper configuration
    res = faiss.StandardGpuResources()
    co = faiss.GpuIndexIVFPQConfig()
    co.device = 0
    co.useFloat16LookupTables = True
    
    print("Creating GPU index with configuration:")
    print(f"  Dimension: {args.dimension}")
    print(f"  Centroids: {args.ncentroids}")
    print(f"  Code size: {args.code_size}")
    print(f"  Metric: {args.metric}")
    print(f"  useFloat16LookupTables: {co.useFloat16LookupTables}")
    
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
    
    # Test GPU performance
    gpu_distances, gpu_indices, gpu_time = test_search_performance(gpu_index, test_queries, args.k, "GPU Index")
    
    # Compare results
    print(f"\n=== Performance Comparison ===")
    print(f"CPU time: {cpu_time:.4f}s ({args.n_test_queries/cpu_time:.1f} queries/sec)")
    print(f"GPU time: {gpu_time:.4f}s ({args.n_test_queries/gpu_time:.1f} queries/sec)")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    # Verify results are similar (allowing for small numerical differences)
    distance_diff = np.abs(cpu_distances - gpu_distances).max()
    indices_match = np.mean(cpu_indices == gpu_indices)
    
    print(f"\n=== Result Verification ===")
    print(f"Max distance difference: {distance_diff:.6f}")
    print(f"Index match rate: {indices_match:.3f}")
    
    if distance_diff < 1e-4 and indices_match > 0.95:
        print("✅ GPU transfer successful! Results match CPU within tolerance.")
    else:
        print("⚠️  Large differences detected between CPU and GPU results.")
        print("This might indicate an issue with the GPU transfer.")
    
except Exception as e:
    print(f"❌ GPU transfer failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")