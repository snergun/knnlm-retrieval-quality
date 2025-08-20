import wandb
import subprocess
import time
import os
import sys

# Initialize wandb
wandb.login()

# Define sweep configuration
sweep_config = {
    'program': 'rq/train_combiner.py',
    'method': 'grid',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'hidden_dim': {'values': [4, 8, 16, 32]},
        'batch_size': {'values': [512, 1024, 2048]},
        'epochs': {'value': 50},
        'preset': {'value': 'wiki_valid'},
        'from_cache': {'value': True},
        'use_wandb': {'value': True}
    }
}

# Initialize the sweep
print("Initializing sweep...")
sweep_id = wandb.sweep(sweep_config, project="POS_LM", entity="ucsd-alon")
print(f"Sweep ID: {sweep_id}")

# Function to run a single training with specific parameters
def train_run(hidden_dim, batch_size):
    # Create a unique checkpoint directory
    timestamp = time.strftime('%m%d_%H%M%S')
    checkpoint_dir = f"checkpoints/h{hidden_dim}_b{batch_size}_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "rq/train_combiner.py",
        "--preset", "wiki_valid",
        "--from_cache",
        "--use_wandb",
        "--hidden_dim", str(hidden_dim),
        "--batch_size", str(batch_size),
        "--epochs", "50",
        "--checkpoint_dir", checkpoint_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.run(cmd)
    return process.returncode, checkpoint_dir

# Grid search over parameters
hidden_dims = [4, 8, 16, 32]
batch_sizes = [512, 1024, 2048]

results = []
for hidden_dim in hidden_dims:
    for batch_size in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Training with hidden_dim={hidden_dim}, batch_size={batch_size}")
        print(f"{'='*50}\n")
        
        return_code, checkpoint_dir = train_run(hidden_dim, batch_size)
        
        results.append({
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
            'return_code': return_code,
            'checkpoint_dir': checkpoint_dir
        })
        
        # Run evaluation immediately after training
        eval_cmd = [
            "python", "rq/eval_combiner.py",
            "--preset", "wiki_test",
            "--from_cache",
            "--use_wandb",
            "--model_path", f"{checkpoint_dir}/best_model.pt"
        ]
        
        print(f"Evaluating model: {' '.join(eval_cmd)}")
        eval_process = subprocess.run(eval_cmd)
        
        # Add a delay between runs
        time.sleep(10)

# Print summary of all runs
print("\n\nSummary of all runs:")
print("=" * 80)
for result in results:
    status = "SUCCESS" if result['return_code'] == 0 else "FAILED"
    print(f"hidden_dim={result['hidden_dim']}, batch_size={result['batch_size']}: {status}, saved to {result['checkpoint_dir']}")
