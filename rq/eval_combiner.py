import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import time
from tqdm import tqdm

#Import Models
from models import ProbCombiner, PerplexityLoss

#Import utils
from utils import set_presets, setup_wandb
from utils import log_progress, copy_to_tmp

#Import Data Utils
from data_utils import load_data
#Import Dataset
from data_structures import ProbDataset

def argument_parser():
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--preset', default='wiki_test', type=str,
                        help='Use a preset configuration for different datasets.')

    # knnLM parameters
    parser.add_argument('--from_cache', action='store_true',
                        help='Load datastore from cache')
                        
    # Wandb options
    parser.add_argument('--use_wandb', action='store_true', 
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='POS_LM', type=str,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default="eergun", type=str,
                        help='Weights & Biases entity (username or team name)')
    parser.add_argument('--wandb_name', default=None, type=str,
                        help='Weights & Biases run name')
    return parser

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the trained model on test data
    """
    model.eval()
    test_loss = 0.0
    all_weights = []
    
    with torch.no_grad():
        for features, probs, targets in test_loader:
            # Move data to device
            features = features.to(device)
            probs = [p.to(device) for p in probs]
            targets = targets.to(device)
            
            # Forward pass
            weights = model(features)
            all_weights.append(weights.cpu())
            
            # Calculate loss
            loss = criterion(weights, probs, targets)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    all_weights = torch.cat(all_weights, dim=0)
    avg_weights = all_weights.mean(dim=0)
    
    log_progress(f"Test Loss: {avg_test_loss:.6f}")
    log_progress(f"Average weights: {avg_weights}")
    
    # Convert negative log likelihood to perplexity
    perplexity = np.exp(avg_test_loss)
    log_progress(f"Test Perplexity: {perplexity:.4f}")
    
    return perplexity, avg_weights

def main():
    # Parse arguments
    args = argument_parser().parse_args()
    set_presets(args)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    #Initialize Wandb for logging
    setup_wandb(args)

    # Load data
    features, probs, targets, device = load_data(args)
    
    # Create dataset
    test_dataset = ProbDataset(features, probs, targets)
    
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    feature_dim = features.shape[1]
    num_probs = len(probs)
    model = ProbCombiner(feature_dim=feature_dim, hidden_dim=args.hidden_dim, num_probs=num_probs)
    model.to(device)
    
    # Define loss function and optimizer
    criterion = PerplexityLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Print model summary
    log_progress(f"Model architecture: {model}")
    log_progress(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, args)
    
    # Evaluate on the entire dataset
    full_loader = DataLoader(dataset, batch_size=args.batch_size)
    perplexity, avg_weights = evaluate_model(model, full_loader, criterion, device)
    
    log_progress(f"Final model perplexity: {perplexity:.4f}")
    log_progress(f"Average probability weights: {avg_weights}")
    
    # Save final model statistics
    results = {
        'perplexity': perplexity,
        'avg_weights': avg_weights.numpy(),
    }
    
    results_path = args.save_path.replace('.pt', '_results.pt')
    torch.save(results, results_path)
    log_progress(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()