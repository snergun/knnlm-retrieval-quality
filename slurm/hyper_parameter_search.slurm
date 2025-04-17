#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=00:02:00
#SBATCH --job-name=combiner_grid_search
#SBATCH --output=log/combiner_grid_search_%j.out

# Load modules and activate environment
unset PYTHONPATH
unset PYTHONHOME
module --force purge
module reset
module load anaconda3_gpu
## Comment the next line to run from cache using anaconda3_gpu (much faster)
# source activate knnlm

cd /work/hdd/bcyi/$USER/knnlm-retrieval-quality

# Define parameter combinations
HIDDEN_DIMS=(4 8 16 32 64)
BATCH_SIZES=(64 512 1024 2048)

# Get a unique job identifier
JOB_ID=${SLURM_JOB_ID}
START_TIME=$(date +%m%d_%H%M%S)

echo "Starting grid search job ${JOB_ID} at ${START_TIME}"

# Loop through all parameter combinations
for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    # Create a unique checkpoint directory for this parameter combination
    # Using job ID, parameter values, and a timestamp within the combo run
    TIMESTAMP=$(date +%m%d_%H%M%S)
    CHECKPOINT_DIR="checkpoints/job_${JOB_ID}_h${HIDDEN_DIM}_b${BATCH_SIZE}_${TIMESTAMP}"
    mkdir -p ${CHECKPOINT_DIR}
    
    echo "======================================"
    echo "Using checkpoint directory: ${CHECKPOINT_DIR}"
    echo "Running with hidden_dim=${HIDDEN_DIM}, batch_size=${BATCH_SIZE}"
    echo "======================================"
    
    # Train the model with the current parameter combination
    echo "Starting training..."
    python rq/train_combiner.py \
        --preset wiki_valid \
        --from_cache \
        --batch_size ${BATCH_SIZE} \
        --epochs 100 \
        --hidden_dim ${HIDDEN_DIM} \
        --use_wandb \
        --run_name "job_${JOB_ID}_h${HIDDEN_DIM}_b${BATCH_SIZE}" \
        --checkpoint_dir ${CHECKPOINT_DIR}
    
    # Evaluate the trained model
    echo "Starting evaluation..."
    python rq/eval_combiner.py \
        --preset wiki_test \
        --from_cache \
        --use_wandb \
        --model_path "${CHECKPOINT_DIR}/best_model.pt"
    
    echo "Completed run for hidden_dim=${HIDDEN_DIM}, batch_size=${BATCH_SIZE}"
    echo ""
  done
done

echo "All parameter combinations completed successfully!"