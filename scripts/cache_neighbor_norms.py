import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap

dstore_keys = np.memmap("datastore/wikitext-103/train/dstore_keys.npy", dtype=np.float32, mode='r', shape=(103225485, 1024))
val_knns = np.memmap("datastore/wikitext-103/valid.cache/dstore_cache_knns.npy", dtype=np.int32, mode='r', shape=(217646, 1024))
neighbor_norms = open_memmap("datastore/wikitext-103/train/norms.npy", dtype=np.float32, mode='w+', shape=(103225485, ))
for i, key in tqdm(enumerate(dstore_keys), total=len(dstore_keys), miniters=int(len(dstore_keys)/100)):
    key = np.array(key)
    # Vectorized norm computation
    norm = np.sum(np.square(key), axis=0)  # L2 norm squared per row
    neighbor_norms[i] = norm
neighbor_norms.flush()