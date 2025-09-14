#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --account=bcyi-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --job-name=cache_knns
#SBATCH --output=log/cache_knns_%j.out
#SBATCH --error=log/cache_knns_%j.err
module purge
module reset
module load anaconda3_gpu
source activate faiss-gpu
# Set the metric type here (change this to: ip, l2, or cos)
for METRIC in ip l2; do
    for KNN_KEYTYPE in last_ffn_input decoder_out; do
        if [ "$KNN_KEYTYPE" == "last_ffn_input" ]; then
            BASE_PATH="/work/nvme/bcyi/eergun/datastore"
        elif [ "$KNN_KEYTYPE" == "decoder_out" ]; then
            BASE_PATH="/work/nvme/bcyi/eergun/datastore/decoder_out"
        else
        # Base paths
        TRAIN_PATH="${BASE_PATH}/train"
        INDEX_PATH="${TRAIN_PATH}/knn_${METRIC}.index"

        echo "Using metric: ${METRIC}"
        echo "Index path: ${INDEX_PATH}"

        # Check if index file exists
        if [ ! -f "${INDEX_PATH}" ]; then
            echo "Error: Index file ${INDEX_PATH} does not exist!"
            exit 1
        fi

        # Create output directory with metric name
        OUTPUT_DIR=${BASE_PATH}
        mkdir -p ${OUTPUT_DIR}/valid.cache
        mkdir -p ${OUTPUT_DIR}/test.cache

        echo "Output directory: ${OUTPUT_DIR}"
        echo "Starting KNN caching for ${METRIC} metric..."

        python -u cache_knns.py \
            --faiss_index "${INDEX_PATH}" \
            --valid_mmap "${BASE_PATH}/valid/dstore" \
            --test_mmap "${BASE_PATH}/test/dstore" \
            --valid_size 217646 \
            --test_size 245569 \
            --dimension 1024 \
            --ncentroids 4096 \
            --code_size 64 \
            --batch_size 10000 \
            --k 1024 \
            --probe 8 \
            --metric ${METRIC} \
            --output_dir ${OUTPUT_DIR} \

        echo "KNN caching completed for ${METRIC} metric!"
    done
done