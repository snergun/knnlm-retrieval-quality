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
    parser.add_argument('--preset', default='wiki_valid', type=str,
                        help='Use a preset configuration for different datasets.')
    parser.add_argument('--external_lm_prob', default=None, type=str,
                        help='Path to external LM probabilities')
    parser.add_argument('--pos_prob', default=None, type=str,
                        help='Path to POS predictions')
    
    # Model parameters
    parser.add_argument('--hidden_dim', default=100, type=int,
                        help='Hidden dimension size for the neural network')
    # knnLM parameters
    parser.add_argument('--from_cache', action='store_true',
                        help='Load datastore from cache')
    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--save_path', default='models/prob_combiner.pt', type=str)
    parser.add_argument('--validation_split', default=0.1, type=float,
                        help='Percentage of data to use for validation')
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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, args):
    """
    Train the neural network model
    """
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (features, probs, targets) in enumerate(train_loader):
            # Move data to device
            features = features.to(device)
            probs = [p.to(device) for p in probs]
            targets = targets.to(device)
            
            # Forward pass
            weights = model(features)
            
            # Calculate loss
            loss = criterion(weights, probs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx+1) % 50 == 0:
                log_progress(f"Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        training_time = time.time() - start_time
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, probs, targets in val_loader:
                # Move data to device
                features = features.to(device)
                probs = [p.to(device) for p in probs]
                targets = targets.to(device)
                
                # Forward pass
                weights = model(features)
                
                # Calculate loss
                loss = criterion(weights, probs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Print epoch results
        log_progress(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
                        
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': best_val_loss,
                'num_probs': len(probs[0]),  # Save the number of probability distributions
            }, args.save_path)
            
    log_progress(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    return model


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
    dataset = ProbDataset(features, probs, targets)
    
    # Split into train and validation sets
    val_size = int(len(dataset) * args.validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
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