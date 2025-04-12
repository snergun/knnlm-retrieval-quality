import shutil
import os
import time
import torch
import numpy as np
from tqdm import tqdm

def log_progress(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def copy_files_to_tmp():
    """Copy necessary files to /tmp for faster access"""
    src_dirs = [
        "datastore/wikitext-103/train",
        "datastore/wikitext-103/valid",
        "datastore/wikitext-103/valid.cache"
    ]
    tmp_dirs = [
        "/tmp/wikitext-103/train",
        "/tmp/wikitext-103/valid",
        "/tmp/wikitext-103/valid.cache"
    ]
    
    # Create tmp directories
    for tmp_dir in tmp_dirs:
        os.makedirs(tmp_dir, exist_ok=True)
    
    # Copy essential files
    needed_files = [
        ("datastore/wikitext-103/valid/dstore_vals.npy", "/tmp/wikitext-103/valid/dstore_vals.npy"),
        ("datastore/wikitext-103/valid/dstore_prob.npy", "/tmp/wikitext-103/valid/dstore_prob.npy"),
        ("datastore/wikitext-103/valid.cache/dstore_cache_knns.npy", "/tmp/wikitext-103/valid.cache/dstore_cache_knns.npy"),
        ("datastore/wikitext-103/valid.cache/dstore_cache_exact_dists.npy", "/tmp/wikitext-103/valid.cache/dstore_cache_exact_dists.npy"),
        ("datastore/wikitext-103/train/dstore_vals.npy", "/tmp/wikitext-103/train/dstore_vals.npy")
    ]
    
    for src, dst in needed_files:
        if not os.path.exists(dst):
            log_progress(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
        else:
            log_progress(f"File already exists: {dst}")
    
    return tmp_dirs

def eval_ppl(p):
    return 2**(-p.mean()/np.log(2))

def main():
    log_progress("Starting optimized kNN-LM evaluation")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_progress(f"Using device: {device}")
    
    # Move files to tmp
    log_progress("Preparing local storage")
    copy_files_to_tmp()
    
    # Use tmp directories
    dstore_dir = "/tmp/wikitext-103/train"
    eval_dstore = "/tmp/wikitext-103/valid"
    eval_cache = "/tmp/wikitext-103/valid.cache"
    
    # Load data
    log_progress("Loading evaluation data")
    start_time = time.time()
    target = np.memmap(f'{eval_dstore}/dstore_vals.npy', dtype=np.int64, mode='r', shape=(217646, 1))
    lm_prob = np.memmap(f'{eval_dstore}/dstore_prob.npy', dtype=np.float32, mode='r', shape=(217646, 1))
    knns = np.memmap(f'{eval_cache}/dstore_cache_knns.npy', dtype=np.int32, mode='r', shape=(217646, 1024))
    
    dists_path = f'{eval_cache}/dstore_cache_exact_dists.npy'
    if os.path.exists(dists_path):
        dists = np.memmap(dists_path, dtype=np.float32, mode='r', shape=(217646, 1024))
        log_progress("Using exact distances")
    else:
        dists = np.memmap(f'{eval_cache}/dstore_cache_dists.npy', dtype=np.float32, mode='r', shape=(217646, 1024))
        log_progress("Using approximate distances")
    print("Min distances:")
    print(dists.max(axis=1))
    print("Mean of min distances:")
    print(dists.min(axis=1).mean())

    log_progress(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # Check dtype and values of lm_prob
    log_progress(f"LM prob dtype: {lm_prob.dtype}, min: {lm_prob.min()}, max: {lm_prob.max()}")
    
    # Original values might be log probabilities but need conversion
    # Convert from float16 to float32 to avoid precision issues
    lm_prob_tensor = torch.tensor(lm_prob, dtype=torch.float32)
    
    # Verify if these values are in correct range for log probs [-inf, 0]
    # If not, they may need to be converted from a different format
    if lm_prob_tensor.max() > 0 or lm_prob_tensor.min() > -0.1:
        log_progress("WARNING: LM probabilities might not be in correct log space!")
        log_progress("Converting probabilities to proper log space...")
        
    
    target_tensor = torch.tensor(target, dtype=torch.int64)
    
    # Load train dataset values (target lookup)
    log_progress("Loading target values datastore")
    vals = np.memmap(f'{dstore_dir}/dstore_vals.npy', dtype=np.int64, mode='r', shape=(103225485, 1))
    
    # Computing kNN probabilities
    log_progress("Computing kNN probabilities")
    batch_size = 4096
    knn_probs = []
    #For Debugging
    knn_matching_keys = [] # Count how many neighbors have values matching the target
    knn_min_dist = [] #Keep track of the min distance
    # Use a smaller k for faster processing if needed
    k = min(1024, knns.shape[1])  # Use all available neighbors, up to 1024
    
    # Process in batches
    start_time = time.time()
    for start_idx in tqdm(range(0, target.shape[0], batch_size)):
        end_idx = min(start_idx + batch_size, target.shape[0])
        actual_batch_size = end_idx - start_idx
        
        # Get current batch targets and probs
        batch_target = target_tensor[start_idx:end_idx].to(device)
        batch_dists = torch.tensor(dists[start_idx:end_idx, :k], dtype=torch.float32).to(device)

        knn_min_dist += batch_dists.max(dim=1).values.cpu().tolist()

        probs = torch.log_softmax(batch_dists, dim=-1)
        
        # Get batch KNN indices
        batch_knns = knns[start_idx:end_idx, :k]
        
        # Process the batch efficiently
        batch_log_prob = torch.zeros(actual_batch_size, device=device)
        
        for i in range(actual_batch_size):
            # Get targets for KNN indices
            knn_indices = batch_knns[i]
            knn_targets = np.take(vals, knn_indices, axis=0).flatten()
            
            # Create and apply mask
            mask = (knn_targets == target[start_idx + i, 0]).astype(np.float32)
            mask_tensor = torch.tensor(mask, device=device)

            knn_matching_keys.append(mask_tensor.sum().item())

            mask_tensor[mask_tensor == 0] = -10000
            mask_tensor[mask_tensor == 1] = 0
            
            # Calculate log probability
            batch_log_prob[i] = torch.logsumexp(probs[i] + mask_tensor, dim=0)
        
        knn_probs.append(batch_log_prob.cpu())
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    print(f"Mean Min kNN distances: {-np.mean(knn_min_dist)}")
    print(f"Median Min kNN distances: {-np.median(knn_min_dist)}")

    print(f"Mean matching keys: {np.mean(knn_matching_keys)}")
    print(f"Median matching keys: {np.median(knn_matching_keys)}")

    log_progress("kNN probabilities computed")
    processing_time = time.time() - start_time
    log_progress(f"kNN probabilities calculated in {processing_time:.2f} seconds")
    
    # Combine results
    knn_prob = torch.cat(knn_probs).view(-1, 1)
    print(f"KNN perplexity {(-knn_prob.mean()).exp().item()}")
    # Calculate baseline perplexity
    base_ppl = eval_ppl(lm_prob_tensor.numpy())
    log_progress(f"Base LM perplexity: {base_ppl:.3f}")
    
    coeff_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    best_ppl = float('inf')
    best_coeff = None
    
    # Evaluate different coefficients
    for coeff in coeff_list:
        # Interpolate
        if coeff <= 0:
            combined_prob = lm_prob_tensor
        elif coeff >= 1:
            combined_prob = knn_prob
        else:
            combine_probs = torch.stack([lm_prob_tensor, knn_prob], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = torch.log(torch.tensor(1 - coeff))
            coeffs[1] = torch.log(torch.tensor(coeff))
            combined_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
        
        ppl = eval_ppl(combined_prob.numpy())
        log_progress(f"Coefficient {coeff:.2f}: Perplexity {ppl:.3f}")
        
        if ppl < best_ppl:
            best_ppl = ppl
            best_coeff = coeff
    
    log_progress(f"Best interpolation coefficient: {best_coeff:.2f}")
    log_progress(f"Best perplexity: {best_ppl:.3f}")
    log_progress(f"Improvement over baseline: {(base_ppl - best_ppl) / base_ppl * 100:.2f}%")

if __name__ == "__main__":
    main()
