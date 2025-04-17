from data_structures import Dataset, Dstore
from knnlm_func import get_knn_prob, eval_ppl
import torch
import os
from utils import log_progress
def load_data(args):
    """
    Load and prepare data for training the combiner model
    """
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_progress(f"Using device: {device}")
    
    # Load dataset
    log_progress("Loading dataset...")
    dataset = Dataset(args)
    log_progress("Dataset loaded")
    
    # Load dstore for KNN calculations
    log_progress("Loading datastore...")
    dstore = Dstore(args)
    log_progress("Datastore loaded")
    
    # Load cache for KNN neighbors
    log_progress("Loading cache...")
    dataset.load_cache()
    log_progress("Cache loaded")
    
    # Load exact distances if available
    if os.path.exists(os.path.join(args.eval_dstore_cache, 'dstore_cache_exact_dists.npy')):
        log_progress("Loading exact distances...")
        dataset.load_exact_dists()
        dists = dataset.exact_dists
        log_progress("Exact distances loaded")
    else:
        log_progress("Using approximate distances")
        dists = -1 * dataset.dists
    
    # Get target tokens and query embeddings
    query = dataset.query
    target = dataset.target
    knns = dataset.knns
    
    # Get LM probabilities
    lm_prob = torch.from_numpy(dataset.prob).float()
    
    # Get KNN probabilities
    log_progress("Calculating KNN probabilities...")
    knn_prob = get_knn_prob(dstore, target, dists, knns).view(-1, 1)
    log_progress("KNN probabilities calculated")
    
    # Get POS probabilities
    if args.pos_prob:
        log_progress(f"Loading POS probabilities from {args.pos_prob}")
        external_results = torch.load(args.pos_prob)
        pos_prob = external_results['pos_probs']
    # Load external probabilities if provided
    if args.external_lm_prob:
        log_progress(f"Loading external LM probabilities from {args.external_lm_prob}")
        external_results = torch.load(args.external_lm_prob)
        ext_lm_prob = external_results['word_logits']
        ext_lm_modified_prob = external_results['modified_logits']
        
        # Update LM probs from external source (skip first token)
        lm_prob[1:] = ext_lm_prob.unsqueeze(1)
        
        # Get modified LM probabilities that incorporate POS predictions
        lm_modified_prob = lm_prob.clone()
        lm_modified_prob[1:] = ext_lm_modified_prob.unsqueeze(1)
    else:
        log_progress("No external LM probabilities provided, using only LM and KNN")
        lm_modified_prob = None
    
    # Prepare features (query embeddings or KNN distances)
    features = pos_prob

    
    # Prepare probabilities
    probs = [lm_prob.view(-1), knn_prob.view(-1)]
    if lm_modified_prob is not None:
        probs.append(lm_modified_prob.view(-1))
    
    # Prepare targets
    targets = torch.from_numpy(target).view(-1).long()
    
    # Check sizes
    log_progress(f"Features shape: {features.shape}")
    log_progress(f"Number of probability distributions: {len(probs)}")
    log_progress(f"Targets shape: {targets.shape}")
    
    # Calculate baseline perplexities
    baseline_lm_ppl = eval_ppl(lm_prob)
    baseline_knn_ppl = eval_ppl(knn_prob)
    log_progress(f"Baseline LM PPL: {baseline_lm_ppl:.4f}")
    log_progress(f"Baseline KNN PPL: {baseline_knn_ppl:.4f}")
    
    if lm_modified_prob is not None:
        baseline_modified_ppl = eval_ppl(lm_modified_prob)
        log_progress(f"Baseline Modified LM PPL: {baseline_modified_ppl:.4f}")
    
    return features, probs, targets, device