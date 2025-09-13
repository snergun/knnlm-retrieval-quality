import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from numpy.lib.format import open_memmap
os.makedirs("plots", exist_ok=True)
val_knns = np.memmap("datastore/wikitext-103/valid.cache/dstore_cache_knns.npy", dtype=np.int32, mode='r', shape=(217646, 1024))
norms = np.load("datastore/wikitext-103/train/norms.npy", mmap_mode= "r")
dists = np.memmap("datastore/wikitext-103/valid.cache/dstore_cache_exact_dists.npy", dtype=np.float32, mode='r', shape=(217646, 1024))
# Plot histogram of all norms
plt.figure(figsize=(10, 6))
plt.hist(norms, bins=100, color='blue', alpha=0.7)
plt.title('Context Embedding Norms')
plt.xlabel('L2 Norm Squared')
plt.ylabel('Frequency')
plt.grid()
plt.savefig('plots/train_ctx_norms.png')

def calculate_KL(dist, norms):
    dist_prob = np.exp(dist)
    dist_prob /= np.sum(dist_prob, axis=1, keepdims=True)
    ip = (norms + dist)
    ip = ip - np.max(ip, axis=1, keepdims=True)
    ip_prob = np.exp(ip)
    ip_prob /= np.sum(ip_prob, axis=1, keepdims=True)
    kl_div = np.sum(dist_prob * (np.log(np.clip(dist_prob, a_min = 1e-10, a_max=None)) - np.log(np.clip(ip_prob, a_min = 1e-10, a_max=None))), axis=1)
    return kl_div, ip

min_val = -15
max_val = 15
n_bins = 100
bins = np.linspace(min_val, max_val, n_bins)
bin_counts = np.zeros(n_bins + 1, dtype = np.int64)
kl_divergence = np.zeros(len(val_knns), dtype = np.float32)
# Plot histogram of demeaned context norms for each set of knns
batch_size = 1000
for start in tqdm(range(0, len(val_knns), batch_size)):
    end = min(start + batch_size, len(val_knns))
    knns = np.array(val_knns[start:end])
    knn_norms = np.array(norms[knns])
    demeaned_knn_norms = knn_norms - np.mean(knn_norms, axis=1, keepdims=True)
    bin_indices = np.digitize(demeaned_knn_norms.flatten(), bins)
    bin_counts += np.bincount(bin_indices, minlength=len(bin_counts)) 
    # Calculate KL Divergence between L2 and IP dist over neighbors
    d = np.array(dists[start:end])
    batch_kl_div, batch_ip = calculate_KL(d, knn_norms)
    kl_divergence[start:end] = batch_kl_div

np.save("plots/val_kl_l2_ip.npy", kl_divergence)

# Plot histogram of demeaned norms
plt.figure(figsize=(10, 6))
plt.bar(bins, bin_counts[:-1], width=(max_val - min_val) / n_bins, color='green', alpha=0.7)
plt.title('Demeaned Norms of kNN contexts')
plt.xlabel('Demeaned L2 Norm Squared')
plt.ylabel('Frequency')
plt.grid()
plt.savefig('plots/val_knn_norms.png')

