from tqdm import tqdm
import numpy as np
import torch
import time 

import shutil

from utils import log_progress, copy_to_tmp

def eval_ppl(p):
    return 2**(-p.mean()/np.log(2))


def get_knn_prob(dstore, target, dists, knns):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    log_progress(f"Using {device}")
    
    d = torch.from_numpy(dists).to(device).float()
    probs = torch.log_softmax(d, -1)
    log_progress(f"kNN probs calculated")
    index_mask = torch.eq(torch.from_numpy(dstore.vals[knns]).to(device).long().squeeze(-1), torch.from_numpy(target).to(device).long()).float()
    index_mask[index_mask == 0] = -10000 # for stability
    index_mask[index_mask == 1] = 0

    log_prob = torch.logsumexp(probs + index_mask, dim=-1).cpu()

    return log_prob


def run_eval_ppl(context):

    # Local variables.
    dstore = context['dstore']
    keys = dstore.keys
    vals = dstore.vals
    dataset = context['dataset']
    query = dataset.query
    target = dataset.target
    knns = dataset.knns
    dists = context['dists']

    # LM perplexity.
    log_progress("get kNN probabilities")
    knn_prob = get_knn_prob(dstore, target, dists, knns).view(-1, 1)
    log_progress("kNN probabilities calculated")
    lm_prob = torch.from_numpy(dataset.prob).float()
    ppl = eval_ppl(lm_prob)
    # kNN-LM perplexity with validation split
    if args.validation_split:
        log_progress("Using split validation approach for coefficient tuning")
        
        # Split the data in half
        total_samples = len(target)
        split_point = total_samples // 2
        
        # Dev0 - first half for boundary definition and coefficient tuning
        dev0_knn_prob = knn_prob[:split_point]
        dev0_lm_prob = lm_prob[:split_point]
        dev0_dists = dists[:split_point]
        
        # Dev1 - second half for bucket number selection
        dev1_knn_prob = knn_prob[split_point:]
        dev1_lm_prob = lm_prob[split_point:]
        dev1_dists = dists[split_point:]
        
        # Find optimal coefficients for different bucket numbers on Dev0
        bucket_options = [1, 2, 4, 8, 16, 32, 64, 128]  # As used in the paper
        best_ppl = float('inf')
        best_b = None
        best_coeffs = None
        
        for b in bucket_options:
            log_progress(f"Testing with {b} buckets")
            # Find bins and coefficients using Dev0
            dev0_min_dist = (-1 * dev0_dists).min(-1)
            dev0_bins = find_bins(dev0_min_dist, b)
            coeff_list = (np.arange(0, 100) / 100).tolist()
            dev0_bin_coeffs = []
            
            # Find optimal coefficients for each bin using Dev0
            for i in range(b):
                mask = dev0_bins == i
                if not mask.any():  # Skip empty bins
                    dev0_bin_coeffs.append(0.25)  # Default value
                    continue
                    
                this_knn_prob = dev0_knn_prob[mask]
                this_lm_prob = dev0_lm_prob[mask]
                best_bin_ppl, best_coeff = np.inf, None
                
                for coeff in coeff_list:
                    this_prob = combine_knn_and_vocab_probs(this_knn_prob, this_lm_prob, coeff)
                    this_ppl = eval_ppl(this_prob)
                    if this_ppl < best_bin_ppl:
                        best_bin_ppl, best_coeff = this_ppl, coeff
                
                dev0_bin_coeffs.append(best_coeff)
            
            # Apply these coefficients to Dev1 and measure perplexity
            dev1_min_dist = (-1 * dev1_dists).min(-1)
            dev1_bins = find_bins(dev1_min_dist, b)
            dev1_bin_prob = torch.full(dev1_lm_prob.shape, 1, dtype=torch.float)
            
            for i in range(b):
                mask = dev1_bins == i
                if not mask.any():
                    continue
                    
                this_knn_prob = dev1_knn_prob[mask]
                this_lm_prob = dev1_lm_prob[mask]
                dev1_bin_prob[mask] = combine_knn_and_vocab_probs(
                    this_knn_prob, this_lm_prob, dev0_bin_coeffs[i])
            
            dev1_ppl = eval_ppl(dev1_bin_prob)
            log_progress(f"Number of buckets: {b}, Dev1 PPL: {dev1_ppl:.4f}")
            
            if dev1_ppl < best_ppl:
                best_ppl = dev1_ppl
                best_b = b
                best_coeffs = dev0_bin_coeffs
        
        # Re-compute with full validation set using optimal bucket number
        log_progress(f"Selected optimal number of buckets: {best_b}")
        min_dist = (-1 * dists).min(-1)
        bins = find_bins(min_dist, best_b)
        coeff_list = (np.arange(0, 100) / 100).tolist()
        bin_prob, bin_coeffs = dynamic_combine_knn_and_vocab_probs(knn_prob, lm_prob, bins, coeff_list)
        bin_ppl = eval_ppl(bin_prob)
        
        print(f'Original PPL = {ppl:.4f}')
        print(f'Best validation PPL = {best_ppl:.4f} with {best_b} buckets')
        print(f'Final PPL = {bin_ppl:.4f} with {best_b} buckets')
        print(f'Final coefficients: {bin_coeffs}')
    else:
        # kNN-LM perplexity.
        log_progress("Evaluate coefficients using entire dataset")
        coeff_list = (np.arange(0, 100) / 100).tolist()
        new_ppl_list = []
        for coeff in tqdm(coeff_list, desc='coeff'):
            def fn():
                new_prob = dstore.combine_knn_and_vocab_probs(knn_prob, lm_prob, coeff)
                return eval_ppl(new_prob)
            new_ppl_list.append(fn())

        # Print a window around the best perplexity.
        topk = 5
        for ix in sorted(np.argsort(new_ppl_list)[:topk]):
            new_ppl = new_ppl_list[ix]
            coeff = coeff_list[ix]
            print(f'ppl = {ppl:.3f}, new_ppl = {new_ppl:.3f} ({coeff})')

        # Assign bins to each entry based on vector distance.
        number_of_bins = 64
        min_dist = (-1 * dists).min(-1)
        bins = find_bins(min_dist, number_of_bins)
        coeff_list = (np.arange(0, 100) / 100).tolist()
        bin_prob, bin_coeffs = dynamic_combine_knn_and_vocab_probs(knn_prob, lm_prob, bins, coeff_list)
        bin_ppl = eval_ppl(bin_prob)
        print(f'bin_ppl = {bin_ppl:.3f}, coeffs [{number_of_bins}] = {bin_coeffs}')


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
    return curr_prob


def dynamic_combine_knn_and_vocab_probs(knn_prob, lm_prob, bins, coeff_list):
    bin_prob = torch.full(lm_prob.shape, 1, dtype=torch.float)
    bin_coeffs = []
    number_of_bins = bins.max().item() + 1
    for i in range(number_of_bins):
        mask = bins == i
        this_knn_prob = knn_prob[mask]
        this_lm_prob = lm_prob[mask]
        best_ppl, best_coeff = np.inf, None
        for coeff in coeff_list:
            this_prob = combine_knn_and_vocab_probs(this_knn_prob, this_lm_prob, coeff)
            this_ppl = eval_ppl(this_prob)
            if this_ppl < best_ppl:
                best_ppl, best_coeff = this_ppl, coeff
        assert best_coeff is not None
        bin_coeffs.append(best_coeff)
        bin_prob[mask] = combine_knn_and_vocab_probs(this_knn_prob, this_lm_prob, best_coeff)
    assert (bin_prob < 1).all().item()
    return bin_prob, bin_coeffs


def find_bins(measure, number_of_bins):
    # Assign each entry to a bin based on statistics of the measure.
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
    assert np.all(bins > -1).item()
    return bins

