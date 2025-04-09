import os
import torch
import numpy as np
import json
import argparse
from fairseq.data.indexed_dataset import MMapIndexedDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Export fairseq tokens as a simple list')
    parser.add_argument('--data-dir', type=str, default='data-bin/wikitext-103',
                        help='Directory containing fairseq preprocessed data')
    parser.add_argument('--split', type=str, default='valid',
                        help='Data split to export (train, valid, test)')
    parser.add_argument('--output-dir', type=str, default='custom_data',
                        help='Directory to save exported data')
    parser.add_argument('--format', type=str, choices=['numpy', 'torch', 'json'], default='torch',
                        help='Output format (numpy, torch, or json)')
    return parser.parse_args()

def load_fairseq_data(data_dir, split):
    """Load fairseq preprocessed data"""
    prefix = os.path.join(data_dir, split)
    
    # Check if dataset exists
    if not os.path.exists(f"{prefix}.idx"):
        raise FileNotFoundError(f"Could not find dataset at {prefix}.idx")
    
    # Load the dataset
    dataset = MMapIndexedDataset(prefix)
    return dataset

def export_tokens(dataset, args):
    """Export tokens from fairseq dataset as a simple list"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract all tokens from the dataset
    all_tokens = []
    
    print(f"Dataset has {len(dataset)} examples")
    
    for i in range(len(dataset)):
        tokens = dataset[i].tolist()
        all_tokens.extend(tokens)
    
    print(f"Extracted {len(all_tokens)} tokens")
    
    # Save in the requested format
    output_path = os.path.join(args.output_dir, f'{args.split}_tokens')
    
    if args.format == 'numpy':
        np.save(f"{output_path}.npy", np.array(all_tokens, dtype=np.int64))
        print(f"Saved tokens to {output_path}.npy")
        
    elif args.format == 'torch':
        torch.save(torch.tensor(all_tokens, dtype=torch.long), f"{output_path}.pt")
        print(f"Saved tokens to {output_path}.pt")
        
    elif args.format == 'json':
        with open(f"{output_path}.json", 'w') as f:
            json.dump(all_tokens, f)
        print(f"Saved tokens to {output_path}.json")
    
    # Also save a small sample for verification
    sample = all_tokens[:100]
    with open(os.path.join(args.output_dir, f'{args.split}_sample.json'), 'w') as f:
        json.dump(sample, f, indent=2)
    
    print(f"Saved sample of first 100 tokens to {args.output_dir}/{args.split}_sample.json")
    
    return len(all_tokens)

def main():
    args = parse_args()
    
    print(f"Loading fairseq data from {args.data_dir}")
    dataset = load_fairseq_data(args.data_dir, args.split)
    
    print(f"Exporting tokens to {args.output_dir} in {args.format} format")
    token_count = export_tokens(dataset, args)
    
    print(f"Successfully exported {token_count} tokens")

if __name__ == "__main__":
    main()