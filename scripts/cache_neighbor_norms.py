import numpy as np
from tqdm import tqdm
import torch 

@torch.no_grad()
def get_bin_counts(data, min_val=None, max_val=None, num_bins=10):
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1).to("cuda")
    bin_indices = torch.bucketize(data, bin_edges)
    counts = torch.bincount(bin_indices, minlength=len(bin_edges)+1)
    return counts

dstore_keys = np.memmap("datastore/wikitext-103/train/dstore_keys.npy", dtype=np.float32, mode='r', shape=(103225485, 1024))
val_knns = np.memmap("datastore/wikitext-103/valid.cache/dstore_cache_knns.npy", dtype=np.int32, mode='r', shape=(217646, 1024))
device = "cuda"
min_val = -20
max_val = 20
n_bins = 100
batch_size = 1000

bin_counts = torch.zeros(n_bins+2 , dtype=torch.float32)
for i in range(0,len(val_knns), batch_size):
    knns = val_knns[i:i+batch_size] # B x K
    neighbor_keys = torch.from_numpy(dstore_keys[knns].copy()).to(device) # B x K x D
    norms = torch.square(neighbor_keys).sum(dim=-1) # B x K
    norms = norms - norms.mean(dim=1, keepdim=True)
    np.save(f"datastore/wikitext-103/valid.cache/neighbor_norms_{i}.npy", norms)