import argparse
import os
import numpy as np
import faiss
import time
import ctypes

parser = argparse.ArgumentParser()
parser.add_argument('--write-interval', type=int, default=1000000)
parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore_fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
parser.add_argument('--num_keys_to_add_at_a_time', default=1000000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
parser.add_argument('--starting_point', type=int, help='index to start adding keys at')
parser.add_argument('--metric', type=str, default='l2', help='distance metric of choice, l2, ip or cos', choices=['l2', 'ip', 'cos'])
args = parser.parse_args()
print("Arguments:")
print(args)

if args.dstore_fp16:
    keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int32, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

# if not os.path.exists(args.faiss_index+f".trained"):
# Initialize faiss index
# quantizer = faiss.IndexFlatL2(args.dimension)
# index = faiss.IndexIVFPQ(quantizer, args.dimension,
#     args.ncentroids, args.code_size, 8)
# metric = faiss.METRIC_L2 if args.metric == 'l2' else faiss.METRIC_INNER_PRODUCT
# quantizer = faiss.IndexFlatL2(args.dimension) if args.metric == 'l2' else faiss.IndexFlatIP(args.dimension)
# index = faiss.IndexIVFPQ(quantizer, args.dimension,
#                         args.ncentroids, args.code_size, 8, metric)
# index.nprobe = args.probe
# Try moving to GPU
ngpus = faiss.get_num_gpus()
print("Number of GPUs detected by Faiss:", ngpus)
if ngpus > 0:
    print("Moving index to GPU")
    res = faiss.StandardGpuResources()
    co = faiss.GpuIndexIVFPQConfig()
    co.device = 0
    co.useFloat16LookupTables = True   # saves shared memory
    index = faiss.GpuIndexIVFPQ(
        res, args.dimension,
        args.ncentroids, args.code_size, 8,  # 8 = nbits
        faiss.METRIC_L2 if args.metric == 'l2' else faiss.METRIC_INNER_PRODUCT, # use inner product for cos
        co
    )
else:
    raise Exception("FAISS GPU not found")

index.nprobe = args.probe
print("Index type:", type(index))
print('Training Index')
np.random.seed(args.seed)
start = time.time()
random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
# ensure sequential reading
random_sample.sort()
# Faiss does not handle adding keys in fp16 as of writing this.
print("Reading index", flush=True)
x = keys[random_sample].astype(np.float32)
if args.metric == "cos":
    faiss.normalize_L2(x)
print(f'reading indexing took {time.time() - start} seconds')
print("Training now", flush=True)
start = time.time()
index.train(x)
print('Training took {} s'.format(time.time() - start))
print('Writing index after training')
start = time.time()
print("Converting to CPU")
cpu_index = faiss.index_gpu_to_cpu(index)
print("Converting to CPU took {} s".format(time.time() - start))
print("Writing index")
start = time.time()
faiss.write_index(cpu_index, args.faiss_index+f".trained")
print('Writing index took {} s'.format(time.time()-start))

print('Adding Keys')
# index = faiss.read_index(args.faiss_index+f".trained")
# ngpus = faiss.get_num_gpus()
# print("Number of GPUs detected by Faiss:", ngpus)
# if ngpus > 0:
#     print("Moving index to GPU")
#     res = faiss.StandardGpuResources()
#     co = faiss.GpuClonerOptions()
#     co.device = 0
#     co.useFloat16LookupTables = True   # saves shared memory
#     co.useFloat16 = True
#     index = faiss.index_cpu_to_gpu(res, 0, index, co)
#     print("Index type:", type(index))
start_pt = args.starting_point
while start_pt < args.dstore_size:
    end = min(args.dstore_size, start_pt+args.num_keys_to_add_at_a_time)
    print(f"Adding keys {start_pt} to {end}")
    start_time = time.time()
    to_add = keys[start_pt:end].copy().astype(np.float32)
    if args.metric == "cos":
        faiss.normalize_L2(to_add)
    print('Reading keys took {} s'.format(time.time() - start_time))
    index.add_with_ids(to_add, np.arange(start_pt, end))
    start_pt += args.num_keys_to_add_at_a_time

    if (start_pt % args.write_interval) == 0:
        print('Added %d tokens so far' % start_pt)
        start = time.time()
        cpu_index = faiss.index_gpu_to_cpu(index)
        print("Converting to CPU took {} s".format(time.time() - start))
        start = time.time()
        faiss.write_index(cpu_index, args.faiss_index)
        print('Writing index took {} s'.format(time.time()-start))

print("Adding total %d keys" % start)
print('Adding took {} s'.format(time.time() - start_time))
print('Writing Index')
start = time.time()
cpu_index = faiss.index_gpu_to_cpu(index)
print("Converting to CPU took {} s".format(time.time() - start))
start = time.time() 
faiss.write_index(cpu_index, args.faiss_index)
print('Writing index took {} s'.format(time.time()-start))
