import numpy as np
import os
import time
import torch
from pathlib import Path

def log_progress(msg):
    """Print a message with timestamp"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_from_cache(path, dtype, shape):
    """Load memmap file from specified path"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)
    return data

def calculate_basic_stats(array, name):
    """Calculate and print basic statistics about an array"""
    log_progress(f"{name} statistics:")
    log_progress(f"  - Shape: {array.shape}")
    log_progress(f"  - Type: {array.dtype}")
    log_progress(f"  - Min: {array.min()}")
    log_progress(f"  - Max: {array.max()}")
    log_progress(f"  - Mean: {array.mean()}")
    if array.ndim > 1:
        log_progress(f"  - Row sums: min={array.sum(axis=1).min()}, max={array.sum(axis=1).max()}")
    return array.min(), array.max(), array.mean()

def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    """Safe implementation of the probability interpolation"""
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    # Safe log computation
    coeffs[0] = np.log(1 - coeff) if coeff < 1 else -1e13
    coeffs[1] = np.log(coeff) if coeff > 0 else -1e13
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
    return curr_prob

def eval_ppl(p):
    """Calculate perplexity from log probabilities"""
    return 2**(-p.mean()/np.log(2))

def check_specific_examples(dataset_path, cache_path, sample_indices=[0, 1, 2]):
    """Examine specific examples"""
    log_progress("Setting up paths...")
    # Setup paths
    dstore_size = 217646  # For validation set

    # Load basic dataset elements
    log_progress("Loading dataset elements...")
    target = load_from_cache(f"{dataset_path}/dstore_vals.npy", np.int32, (dstore_size, 1))
    query = load_from_cache(f"{dataset_path}/dstore_keys.npy", np.float32, (dstore_size, 1024))
    prob = load_from_cache(f"{dataset_path}/dstore_prob.npy", np.float32, (dstore_size, 1))
    
    # Load cache elements
    log_progress("Loading cache elements...")
    dists = load_from_cache(f"{cache_path}/dstore_cache_dists.npy", np.float32, (dstore_size, 1024))
    knns = load_from_cache(f"{cache_path}/dstore_cache_knns.npy", np.int32, (dstore_size, 1024))
    
    # Check if exact distances file exists
    exact_path = f"{cache_path}/dstore_cache_exact_dists.npy"
    if os.path.exists(exact_path):
        log_progress("Loading exact distances...")
        exact_dists = load_from_cache(exact_path, np.float32, (dstore_size, 1024))
        calculate_basic_stats(exact_dists, "Exact distances")
        
        # Compare with approximate distances
        log_progress("Comparing approximate vs exact distances:")
        diff = dists - exact_dists
        calculate_basic_stats(diff, "Difference between approximate and exact distances")
    else:
        log_progress("Exact distances file not found.")

    # Basic statistics
    calculate_basic_stats(prob, "LM probabilities")
    calculate_basic_stats(dists, "Distances")
    calculate_basic_stats(knns, "Neighbor indices")
    
    # Check top-k normalization for a few examples
    log_progress("\nChecking probability calculations for sample examples:")
    for idx in sample_indices:
        if idx >= dstore_size:
            continue
            
        log_progress(f"\nExample {idx}:")
        sample_prob = prob[idx]
        sample_dists = torch.from_numpy(exact_dists[idx]).float()
        
        # Normalize into a probability distribution
        sample_dist = torch.log_softmax(sample_dists, dim=0)
        index_mask = torch.eq(torch.from_numpy(dstore.vals[knns]).to(device).long().squeeze(-1), torch.from_numpy(target).to(device).long()).float()

        log_progress(f"  - LM log probability: {sample_prob.item()}")
        log_progress(f"  - Top 5 kNN distances: {sample_dists[:5].numpy()}")
        log_progress(f"  - Top 5 kNN probabilities: {sample_dist[:5].numpy()}")
        log_progress(f"  - kNN probs sum: {sample_dist.sum().item()}")
        
        # Check perplexity calculation
        lm_prob = torch.from_numpy(sample_prob).float()
        
        # Create a simple kNN probability (just use the normalized distances)
        knn_prob = torch.log(sample_dist[0:1]).float()
        
        # Try different interpolation coefficients
        for coeff in [0.1, 0.25, 0.5, 0.75]:
            combined = combine_knn_and_vocab_probs(knn_prob, lm_prob, coeff)
            log_progress(f"  - Interpolation with λ={coeff}: LM={lm_prob.item()}, kNN={knn_prob.item()}, Combined={combined.item()}")

if __name__ == "__main__":
    # Set paths - adjust these to your configuration
    dataset_path = "datastore/wikitext-103/valid"
    cache_path = "datastore/wikitext-103/valid.cache"
    
    # Run the checks
    log_progress("Starting diagnostic checks...")
    check_specific_examples(dataset_path, cache_path)
    log_progress("Diagnostic checks completed.")