import torch
import time
import os
import numpy as np
from tqdm import tqdm
import shutil

def log_progress(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def copy_to_tmp(src_path, shape, dtype):
    """Copy a file to /tmp for faster access"""
    filename = os.path.basename(src_path)
    dir_path = os.path.dirname(src_path)
    tmp_dir = os.path.join('/tmp', os.path.basename(dir_path))
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, filename)
    
    if not os.path.exists(tmp_path):
        log_progress(f"Copying {src_path} to {tmp_path}")
        shutil.copy2(src_path, tmp_path)
    else:
        log_progress(f"File already exists: {tmp_path}")
    
    return np.memmap(tmp_path, dtype=dtype, mode='r', shape=shape)

def eval_ppl(p):
    """Calculate perplexity from log probabilities"""
    return 2**(-p.mean()/np.log(2))

def find_bins(measure, number_of_bins):
    """Assign each entry to a bin based on statistics of the measure"""
    bins = np.full(measure.shape, -1)
    pct_size = 100 / number_of_bins
    for i in range(number_of_bins):
        if i == number_of_bins - 1:
            pct_start = i * pct_size
            pct_end = 100
            pct_mask = np.logical_and(measure >= np.percentile(measure, pct_start), measure <= np.percentile(measure, pct_end))
        else:
            pct_start = i * pct_size
            pct_end = pct_start + pct_size
            pct_mask = np.logical_and(measure >= np.percentile(measure, pct_start), measure < np.percentile(measure, pct_end))
        bins[pct_mask] = i
    return bins

def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    """Combine kNN and vocabulary probabilities with interpolation coefficient"""
    if coeff <= 0.0:
        return vocab_p
    if coeff >= 1.0:
        return knn_p
        
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = torch.log(torch.tensor(1 - coeff))
    coeffs[1] = torch.log(torch.tensor(coeff))
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
    return curr_prob

def main():
    log_progress("Starting optimized kNN-LM evaluation")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_progress(f"Using device: {device}")
    
    # Define paths
    orig_dstore_dir = "datastore/wikitext-103/train"
    orig_eval_dstore = "datastore/wikitext-103/valid"
    orig_eval_cache = "datastore/wikitext-103/valid.cache"
    
    # Copy files to tmp
    log_progress("Copying files to /tmp")
    
    # Create tmp directories
    tmp_dstore_dir = "/tmp/wikitext-103/train"
    tmp_eval_dstore = "/tmp/wikitext-103/valid"
    tmp_eval_cache = "/tmp/wikitext-103/valid.cache"
    
    os.makedirs(tmp_dstore_dir, exist_ok=True)
    os.makedirs(tmp_eval_dstore, exist_ok=True)
    os.makedirs(tmp_eval_cache, exist_ok=True)
    
    # Copy files
    target = copy_to_tmp(f'{orig_eval_dstore}/dstore_vals.npy', dtype=np.int32, shape=(217646, 1))
    lm_prob = copy_to_tmp(f'{orig_eval_dstore}/dstore_prob.npy', dtype=np.float32, shape=(217646, 1))
    knns = copy_to_tmp(f'{orig_eval_cache}/dstore_cache_knns.npy', dtype=np.int32, shape=(217646, 1024))
    
    dists_path = f'{orig_eval_cache}/dstore_cache_exact_dists.npy'
    if os.path.exists(dists_path):
        dists = copy_to_tmp(dists_path, dtype=np.float32, shape=(217646, 1024))
        log_progress("Using exact distances")
    else:
        dists = copy_to_tmp(f'{orig_eval_cache}/dstore_cache_dists.npy', dtype=np.float32, shape=(217646, 1024))
        log_progress("Using approximate distances")
    
    # Load datastore values
    vals = copy_to_tmp(f'{orig_dstore_dir}/dstore_vals.npy', dtype=np.int32, shape=(103225485, 1))
    log_progress("All files copied to /tmp")
    
    # Check prob values
    log_progress(f"LM prob dtype: {lm_prob.dtype}, min: {lm_prob.min()}, max: {lm_prob.max()}")
    
    # Convert to float32 for better precision
    lm_prob_tensor = torch.tensor(lm_prob, dtype=torch.float32)
    
    # Scale probabilities if they're in an unusual range
    if lm_prob_tensor.max() > 100 or lm_prob_tensor.min() < -100:
        log_progress("Normalizing probabilities")
        scale_factor = 512.0  # Based on your observation
        lm_prob_tensor = lm_prob_tensor / scale_factor
    
    # Calculate kNN probabilities
    log_progress("Computing kNN probabilities")
    batch_size = 1024
    knn_probs = []
    
    # Process in batches
    for start_idx in tqdm(range(0, target.shape[0], batch_size)):
        end_idx = min(start_idx + batch_size, target.shape[0])
        actual_batch_size = end_idx - start_idx
        
        # Get current batch
        batch_target = torch.tensor(target[start_idx:end_idx], dtype=torch.int64).to(device)
        batch_dists = torch.tensor(dists[start_idx:end_idx], dtype=torch.float32).to(device)
        probs = torch.log_softmax(batch_dists, dim=-1)
        
        # Process neighbors efficiently
        batch_log_prob = torch.zeros(actual_batch_size, device=device)
        
        for i in range(actual_batch_size):
            knn_indices = knns[start_idx + i]
            knn_targets = np.take(vals, knn_indices, axis=0).flatten()
            
            mask = (knn_targets == target[start_idx + i, 0]).astype(np.float32)
            mask_tensor = torch.tensor(mask, device=device)
            mask_tensor[mask_tensor == 0] = -10000
            mask_tensor[mask_tensor == 1] = 0
            
            batch_log_prob[i] = torch.logsumexp(probs[i] + mask_tensor, dim=0)
        
        knn_probs.append(batch_log_prob.cpu())
        torch.cuda.empty_cache()
    
    # Combine results
    knn_prob = torch.cat(knn_probs).view(-1, 1)
    log_progress("kNN probabilities calculated")
    
    # Standard perplexity evaluation
    base_ppl = eval_ppl(lm_prob_tensor.numpy())
    log_progress(f"Base LM perplexity: {base_ppl:.3f}")
    
    # Evaluate standard interpolation
    coeff_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    best_ppl = float('inf')
    best_coeff = None
    
    for coeff in coeff_list:
        combined_prob = combine_knn_and_vocab_probs(knn_prob, lm_prob_tensor, coeff)
        ppl = eval_ppl(combined_prob.numpy())
        log_progress(f"Static coefficient {coeff:.2f}: Perplexity {ppl:.3f}")
        
        if ppl < best_ppl:
            best_ppl = ppl
            best_coeff = coeff
    
    log_progress(f"Best static coefficient: {best_coeff:.2f}, perplexity: {best_ppl:.3f}")
    
    # Adaptive coefficients based on binning
    log_progress("Computing adaptive coefficients")
    
    # Calculate min distance for each query
    min_dist = (-1 * dists).min(axis=1)
    
    # Try different bin counts
    for number_of_bins in [8, 16, 32, 64]:
        log_progress(f"Testing with {number_of_bins} bins")
        
        # Assign bins based on distance
        bins = find_bins(min_dist, number_of_bins)
        
        # Determine best coefficient for each bin
        bin_coeffs = []
        bin_prob = torch.zeros_like(lm_prob_tensor)
        
        for i in range(number_of_bins):
            mask = bins == i
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            
            if not mask_tensor.any():
                bin_coeffs.append(0.0)
                continue
                
            this_knn_prob = knn_prob[mask_tensor]
            this_lm_prob = lm_prob_tensor[mask_tensor]
            
            best_bin_ppl = float('inf')
            best_bin_coeff = None
            
            for coeff in coeff_list:
                this_prob = combine_knn_and_vocab_probs(this_knn_prob, this_lm_prob, coeff)
                this_ppl = eval_ppl(this_prob.numpy())
                
                if this_ppl < best_bin_ppl:
                    best_bin_ppl = this_ppl
                    best_bin_coeff = coeff
            
            bin_coeffs.append(best_bin_coeff)
            bin_prob[mask_tensor] = combine_knn_and_vocab_probs(this_knn_prob, this_lm_prob, best_bin_coeff)
        
        # Calculate overall perplexity with adaptive coefficients
        adaptive_ppl = eval_ppl(bin_prob.numpy())
        log_progress(f"Adaptive perplexity with {number_of_bins} bins: {adaptive_ppl:.3f}")
        log_progress(f"Bin coefficients: {bin_coeffs}")
        
        # Calculate improvement over static interpolation
        improvement = (best_ppl - adaptive_ppl) / best_ppl * 100
        log_progress(f"Improvement over static: {improvement:.2f}%")
    
    log_progress("Evaluation complete")

if __name__ == "__main__":
    main()