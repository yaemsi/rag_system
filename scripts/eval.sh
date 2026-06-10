#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export OLLAMA_GPU_OVERHEAD=1000000000  # Reserves ~1GB
#export OLLAMA_FLASH_ATTENTION=false


# Parameters
export DATA_DIR="./data/gz"
export INDEX_DIR="./data/index"
export SPLIT="valid"
export BATCH_SIZE=16


python main.py \
    --data_dir $DATA_DIR \
    --index_dir $INDEX_DIR \
    --split $SPLIT \
    --eval_batch_size $BATCH_SIZE \
    --verbose \
