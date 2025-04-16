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
def find_optimal_coefficients(knn_prob, lm_prob, lm_modified_prob=None, num_bins=64):
    """
    Find optimal interpolation coefficients for each bin using the first half of dataset.
    
    Args:
        knn_prob: kNN probabilities
        lm_prob: Language model probabilities
        lm_modified_prob: Modified language model probabilities (optional)
        num_bins: Number of bins to divide data into
    
    Returns:
        bin_coeffs: List of optimal coefficients for each bin
        bin_ppl: Perplexity achieved with optimal coefficients
    """
    # Calculate minimum distances for binning
    dists = (-1 * dists).min(-1)  # This needs to be passed in or calculated within function
    bins = find_bins(dists, num_bins)
    
    # Initialize results
    bin_prob = torch.full(lm_prob.shape, 1, dtype=torch.float)
    bin_coeffs = []
    
    # Process each bin
    for i in range(num_bins):
        mask = bins == i
        if not mask.any():  # Skip empty bins
            if lm_modified_prob is not None:
                bin_coeffs.append((0.33, 0.33, 0.34))  # Default balanced weights for 3-way
            else:
                bin_coeffs.append(0.5)  # Default balanced weight for 2-way
            continue
            
        this_knn_prob = knn_prob[mask]
        this_lm_prob = lm_prob[mask]
        
        if lm_modified_prob is not None:
            # 3-way interpolation
            this_lm_modified_prob = lm_modified_prob[mask]
            best_bin_ppl, best_lambdas = np.inf, None
            
            # Define grid search space for coefficients
            lambda_step = 0.1
            lambda_values = np.arange(0, 1+lambda_step, lambda_step)
            
            # Grid search over lambda combinations
            for lambda1 in lambda_values:
                for lambda2 in lambda_values:
                    if lambda1 + lambda2 > 1:
                        continue  # Skip invalid combinations
                    
                    lambda3 = 1 - lambda1 - lambda2
                    
                    # Calculate combined probability
                    this_prob = combine_three_probs(
                        this_lm_prob, this_knn_prob, this_lm_modified_prob, 
                        lambda1, lambda2, lambda3
                    )
                    
                    # Evaluate perplexity
                    this_ppl = eval_ppl(this_prob)
                    if this_ppl < best_bin_ppl:
                        best_bin_ppl = this_ppl
                        best_lambdas = (lambda1, lambda2, lambda3)
                        
            bin_coeffs.append(best_lambdas)
            # Apply optimal coefficients to this bin
            bin_prob[mask] = combine_three_probs(
                this_lm_prob, this_knn_prob, this_lm_modified_prob,
                *best_lambdas
            )
        else:
            # 2-way interpolation
            coeff_list = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05).tolist()]
            best_bin_ppl, best_coeff = np.inf, None
            
            # Grid search over coefficients
            for coeff in coeff_list:
                this_prob = combine_knn_and_vocab_probs(this_knn_prob, this_lm_prob, coeff)
                this_ppl = eval_ppl(this_prob)
                if this_ppl < best_bin_ppl:
                    best_bin_ppl, best_coeff = this_ppl, coeff
            
            bin_coeffs.append(best_coeff)
            # Apply optimal coefficient to this bin
            bin_prob[mask] = combine_knn_and_vocab_probs(
                this_knn_prob, this_lm_prob, best_coeff
            )
    
    # Calculate overall perplexity with optimal coefficients
    bin_ppl = eval_ppl(bin_prob)
    
    return bin_coeffs, bin_ppl

def run_eval_ppl(context, validation_split=False, use_external):
    # Local variables.
    dstore = context['dstore']
    keys = dstore.keys
    vals = dstore.vals
    dataset = context['dataset']
    query = dataset.query
    target = dataset.target
    knns = dataset.knns
    dists = context['dists']
    #External LM Probabilities
    ext_lm_prob = context.get('ext_lm_prob')
    ext_lm_modified_prob = context.get('ext_lm_modified_prob')
    ext_weight = context.get('ext_weight')
    # LM perplexity.
    log_progress("get kNN probabilities")
    knn_prob = get_knn_prob(dstore, target, dists, knns).view(-1, 1)
    log_progress("kNN probabilities calculated")
    lm_prob = torch.from_numpy(dataset.prob).float()

    #Fairseq evaluates prob for the first token in dataset which we skip
    if ext_lm_prob is not None:
        log_progress("Using external LM probabilities")
        lm_prob[1:] = ext_lm_prob.unsqueeze(1) 
    if ext_lm_modified_prob is not None:
        lm_modified_prob = lm_prob.clone()
        lm_modified_prob[1:] = ext_lm_modified_prob.unsqueeze(1)

    ppl = eval_ppl(lm_prob)

    # Dev - Full Validation set
    dev_probs = (lm_prob, knn_prob)
    dev_probs += (lm_modified_prob,) if lm_modified_prob is not None else ()
    dev_min_dist = (-1 * dists).min(-1)

    # Set Up Parameter Space
    lambda_step = 0.1
    grid = np.arange(0, 1+lambda_step, lambda_step)
    # For 3-way interpolation, we need to search over pairs of coefficients
    if lm_modified_prob is not None:
        lambda_values = [(lambda1, lambda2, 1-lambda1-lambda2) for lambda1 in grid for lambda2 in grid if lambda1 + lambda2 <= 1]
    else:
        lambda_values = [(lambda1, 1-lambda1) for lambda1 in grid]

    # Find optimal coefficients for different bucket numbers on Dev0
    bucket_options = [1, 2, 4, 8, 16, 32, 64, 128]  # As used in the paper
    best_ppl = float('inf')
    best_b = None
    best_coeffs = None

    # kNN-LM perplexity with validation split
    if validation_split:
        log_progress("Using split validation approach for coefficient tuning")
        # Split the data in half
        total_samples = len(target)
        split_point = total_samples // 2
        # Dev0 - first half for boundary definition and coefficient tuning
        dev0_probs = (lm_prob[:split_point], knn_prob[:split_point])
        dev0_probs += (lm_modified_prob[:split_point],) if lm_modified_prob is not None else ()
        dev0_dists = dists[:split_point]
        dev0_min_dist = (-1 * dev0_dists).min(-1)
        # Dev1 - second half for bucket number selection
        dev1_probs = (lm_prob[split_point:], knn_prob[split_point:])
        dev1_probs += (lm_modified_prob[split_point:],) if lm_modified_prob is not None else ()
        dev1_dists = dists[split_point:]
        dev1_min_dist = (-1 * dev1_dists).min(-1)

        for b in bucket_options:
            log_progress(f"Testing with {b} buckets")
            # Find bins and coefficients using Dev0
            dev0_bins = find_bins(dev0_min_dist, b)
            dev0_bin_coeffs = []
            # Find optimal coefficients for each bin using Dev0
            for i in range(b):
                mask = dev0_bins == i
                if not mask.any():  # Skip empty bins
                    if len(dev0_probs) == 3:
                        dev0_bin_coeffs.append((0.33, 0.33, 0.34))  # Default equal values for 3-way
                    else:
                        dev0_bin_coeffs.append((0.5,0.5))  # Default value for 2-way
                    continue
                
                bin_probs = (temp[mask] for temp in dev0_probs)
                best_bin_ppl, best_coeffs_tuple = np.inf, None
                for coeffs in lambda_values:
                    combined_bin_probs = combine_probs(bin_probs, coeffs)
                    this_ppl = eval_ppl(combined_bin_probs)
                        if this_ppl < best_bin_ppl:
                            best_bin_ppl = this_ppl
                            best_coeffs_tuple = coeffs  
                dev0_bin_coeffs.append(best_coeffs_tuple)
                
            # Calculate dev0 perplexity with optimal coefficients
            dev0_combined_probs = dynamic_combine(dev0_probs, dev0_bins, dev0_bin_coeffs)
            dev0_ppl = eval_ppl(dev0_combined_probs)
            log_progress(f"Number of buckets: {b}, Dev0 PPL: {dev0_ppl:.4f}")

            # Apply these coefficients to Dev1 and measure perplexity
            dev1_bins = find_bins(dev1_min_dist, b)
            dev1_combined_probs = dynamic_combine(dev1_probs, dev1_bins, dev0_bin_coeffs)
            dev1_ppl = eval_ppl(dev1_combined_probs)
            log_progress(f"Number of buckets: {b}, Dev1 PPL: {dev1_ppl:.4f}")
            
            if dev1_ppl < best_ppl:
                best_ppl = dev1_ppl
                best_b = b
                best_coeffs = dev0_bin_coeffs
    else:
        for b in bucket_options:
            # Find optimal coefficients for each bin full Dev set
            log_progress(f"Testing with {b} buckets")
            dev_bins = find_bins(dev_min_dist, b)
            dev_bin_coeffs = []
            for i in range(b):
                mask = dev0_bins == i
                if not mask.any():  # Skip empty bins
                    if len(dev_probs) == 3:
                        dev_bin_coeffs.append((0.33, 0.33, 0.34))  # Default equal values for 3-way
                    else:
                        dev_bin_coeffs.append((0.5,0.5))  # Default value for 2-way
                    continue
    
                bin_probs = (temp[mask] for temp in dev_probs)
                best_bin_ppl, best_coeffs_tuple = np.inf, None
                for coeffs in lambda_values:
                    combined_bin_probs = combine_probs(bin_probs, coeffs)
                    this_ppl = eval_ppl(combined_bin_probs)
                        if this_ppl < best_bin_ppl:
                            best_bin_ppl = this_ppl
                            best_coeffs_tuple = coeffs  
                dev_bin_coeffs.append(best_coeffs_tuple)

                dev_combined_probs = dynamic_combine(dev_probs, dev_bins, dev_bin_coeffs)
                dev_ppl = eval_ppl(dev_combined_probs)
                log_progress(f"Number of buckets: {b}, Dev PPL: {dev_ppl:.4f}")
                
                if dev_ppl < best_ppl:
                    best_ppl = dev_ppl
                    best_b = b
                    best_coeffs = dev_bin_coeffs

    log_progress(f"Selected optimal number of buckets: {best_b}")
    log_progress(f"Selected optimal coefficients: {best_coeffs}")
    # Apply these coefficients to full Dev and measure perplexity
    dev_bins = find_bins(dev_min_dist, best_b)
    dev_combined_probs = dynamic_combine(dev_probs, dev_bins, best_coeffs)
    dev_ppl = eval_ppl(dev_combined_probs)
    log_progress(f"Number of buckets: {b}, Full Dev PPL: {dev_ppl:.4f}")        
    print(f'Original PPL = {ppl:.4f}')
    print(f'Final PPL = {dev_ppl:.4f} with {best_b} buckets')
    print(f'Final coefficients: {best_coeffs}')

def dynamic_combine(probs, bins, coeffs):
    """
    Args:
        probs: List of probabilities to combine
        bins: Bin assignments for each token 
        coeffs: List of coefficients for each bin
    Returns:
        output: Combined probabilities
    """
    for i in range(len(probs)-1):
        assert len(probs[0]) == len(probs[i+1]), f"Lengths of probs[0] and probs[{i+1}] do not match"

    num_bins = bins.max().item() + 1
    # Initialize output probabilities
    output = torch.full(probs[0].shape, 1, dtype=torch.float)
    # Apply coefficients to each bin
    for i in range(num_bins):
        mask = bins == i
        if not mask.any():
            continue
        # Get coefficients and probabilities for this bin
        bin_probs = (prob[mask] for prob in probs)
        bin_coeffs = coeffs[i]
        output[mask] = combine_probs(bin_probs, bin_coeffs)
    return output
    
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

def combine_probs(probs, coeffs):
    """
    Combine three probability distributions with log-domain interpolation
    probs: list of N probabilities to combine, shape (N, T)
    coeffs: list of N coefficients for each probability, shape (N,)
    """
    assert len(probs) == len(coeffs), "Number of probabilities and coefficients must match"
    stacked_probs = torch.stack(probs, dim=0)
    stacked_coeffs = torch.ones_like(combine_probs)

    for coeff in coeffs:
        stacked_coeffs[i] = np.log(coeff) if coeff > 0 else -1e13

    return torch.logsumexp(combine_probs + coeffs, dim=0)

def dynamic_combine_three_probs(lm_prob, knn_prob, modified_lm_prob, bins, lambda_values):
    """
    Find optimal lambda weights for each bin when combining three probability distributions
    
    This function assumes modified_lm_prob is not None
    """
    bin_prob = torch.full(lm_prob.shape, 1, dtype=torch.float)
    bin_coeffs = []
    number_of_bins = bins.max().item() + 1
    
    for i in range(number_of_bins):
        mask = bins == i
        this_lm_prob = lm_prob[mask]
        this_knn_prob = knn_prob[mask]
        this_modified_lm_prob = modified_lm_prob[mask]
        best_ppl, best_lambdas = np.inf, None
        
        # Grid search over lambda combinations
        for lambda1 in lambda_values:
            for lambda2 in lambda_values:
                if lambda1 + lambda2 > 1:
                    continue  # Skip invalid combinations
                
                lambda3 = 1 - lambda1 - lambda2
                this_prob = combine_three_probs(
                    this_lm_prob, this_knn_prob, this_modified_lm_prob,
                    lambda1, lambda2, lambda3
                )
                this_ppl = eval_ppl(this_prob)
                
                if this_ppl < best_ppl:
                    best_ppl = this_ppl
                    best_lambdas = (lambda1, lambda2, lambda3)
        
        assert best_lambdas is not None
        bin_coeffs.append(best_lambdas)
        bin_prob[mask] = combine_three_probs(
            this_lm_prob, this_knn_prob, this_modified_lm_prob,
            *best_lambdas
        )
    
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

