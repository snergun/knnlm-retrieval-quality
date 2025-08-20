import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import Models
from models import ProbCombiner, PerplexityLoss

# Import utils
from utils import set_presets
from utils import log_progress, copy_to_tmp

# Import Data Utils
from data_utils import load_data

# Import Dataset
from data_structures import ProbDataset

def argument_parser():
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--preset', default='wiki_test', type=str,
                        help='Use a preset configuration for different datasets.')
                        
    # Model paths
    parser.add_argument('--model_path', default=None, type=str,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--results_path', default=None, type=str,
                        help='Path to save evaluation results')
    
    # knnLM parameters
    parser.add_argument('--from_cache', action='store_true',
                        help='Load datastore from cache')
                        
    # Evaluation parameters
    parser.add_argument('--batch_size', default=64, type=int)
    
    # Wandb options
    parser.add_argument('--use_wandb', action='store_true', 
                        help='Enable Weights & Biases logging')
    parser.add_argument('--task', default='test_combiner', type=str,
                        help='Task info for wandb')
    return parser

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on the provided data loader
    """
    model.eval()
    total_loss = 0.0
    all_weights = []
    
    with torch.no_grad():
        for features, probs, targets in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            features = features.to(device)
            probs = [p.to(device) for p in probs]
            targets = targets.to(device)
            
            # Forward pass
            weights = model(features)
            all_weights.append(weights.cpu())
            
            # Calculate loss
            loss = criterion(weights, probs, targets)
            total_loss += loss.item() * len(targets)
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / len(data_loader.dataset)
    perplexity = np.exp(avg_loss)
    
    # Calculate average weights
    all_weights = torch.cat(all_weights, dim=0)
    avg_weights = all_weights.mean(dim=0)
    
    return perplexity, avg_weights, all_weights

def eval_combined_probs(probs, weights, device):
    """
    Calculate perplexity using the combined probabilities
    """
    # Create combined probabilities using the weights
    stacked_probs = torch.stack([p.to(device) for p in probs], dim=1)
    log_weights = torch.log(weights.to(device))
    combined_log_probs = torch.logsumexp(log_weights + stacked_probs, dim=1)
    
    # Calculate perplexity
    perplexity = np.exp(-combined_log_probs.mean().item())
    
    return perplexity

def main():
    # Parse arguments
    args = argument_parser().parse_args()
    set_presets(args)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize Wandb for logging

    # Load data
    log_progress("Loading data...")
    features, probs, targets, device = load_data(args)
    
    # Create dataset and dataloader
    test_dataset = ProbDataset(features, probs, targets)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load the model checkpoint
    log_progress(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Check if model config is stored in checkpoint
    if 'model_config' in checkpoint:
        log_progress("Using model configuration from checkpoint")
        model_config = checkpoint['model_config']
        model = ProbCombiner(**model_config)
    else:
        # Fallback to inferring parameters
        log_progress("Model configuration not found in checkpoint, inferring parameters")
        feature_dim = features.shape[1]
        num_probs = len(probs)
        hidden_dim = checkpoint.get('hidden_dim', 100)  # Default to 100 if not specified
    if args.use_wandb:
        wandb.init(
            project="POS_LM",
            entity="ucsd-alon",
            name=f"{time.strftime('%m%d_%H%M%S')}",
            config= {"args": vars(args),
                    "task": args.task,
                    "model_config": model_config
                    },
            settings=wandb.Settings(code_dir="rq/"), # save source code in current directory
        )
        wandb.run.log_code()
    # Initialize model
    model = ProbCombiner(feature_dim=feature_dim, hidden_dim=hidden_dim, num_probs=num_probs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Define loss function
    criterion = PerplexityLoss()
    
    # Evaluate model
    log_progress("Evaluating model...")
    perplexity, avg_weights, all_weights = evaluate_model(model, test_loader, criterion, device)
    
    # Calculate baseline perplexities
    baseline_perplexities = []
    for i, prob in enumerate(probs):
        baseline_ppl = np.exp(-prob.mean().item())
        baseline_perplexities.append(baseline_ppl)
        log_progress(f"Baseline {i} PPL: {baseline_ppl:.4f}")
    
    # Calculate perplexity with fixed average weights
    fixed_weights = avg_weights.unsqueeze(0).expand(len(targets), -1)
    fixed_weight_ppl = eval_combined_probs(probs, fixed_weights, device)
    
    # Print results
    log_progress(f"Test Perplexity: {perplexity:.4f}")
    log_progress(f"Fixed Weight Perplexity: {fixed_weight_ppl:.4f}")
    log_progress(f"Average Weights: {avg_weights}")
    
    # Save results
    results_path = args.results_path
    if results_path is None:
        results_path = args.model_path.replace('.pt', '_test_results.pt')
    
    results = {
        'perplexity': perplexity,
        'fixed_weight_perplexity': fixed_weight_ppl,
        'avg_weights': avg_weights.numpy(),
        'all_weights': all_weights.numpy(),
        'baseline_perplexities': baseline_perplexities
    }
    
    log_progress(f"Saving results to {results_path}")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    torch.save(results, results_path)

if __name__ == "__main__":
    main()