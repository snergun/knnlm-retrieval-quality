import numpy as np
import faiss

def check_faiss_gpu():
    ngpus = faiss.get_num_gpus()
    print("Number of GPUs detected by FAISS:", ngpus)
    
    if ngpus == 0:
        print("No GPU detected, FAISS will use CPU.")
        return
    
    # Create a small dataset
    d = 128  # dimension
    nb = 10000  # database size
    nq = 5      # number of queries
    np.random.seed(123)
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')
    
    # CPU index first
    cpu_index = faiss.IndexFlatL2(d)
    cpu_index.add(xb)
    
    # Move index to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    # Search
    D, I = gpu_index.search(xq, 5)  # top-5 nearest neighbors
    print("Search results (indices):", I)
    print("Search results (distances):", D)
    
    print("FAISS GPU test completed successfully!")

if __name__ == "__main__":
    check_faiss_gpu()
