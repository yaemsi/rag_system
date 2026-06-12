#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export OLLAMA_GPU_OVERHEAD=1000000000  # Reserves ~1GB
#export OLLAMA_FLASH_ATTENTION=false


# Parameters
export DATA_DIR="./data/gz"
export INDEX_DIR="./data/index"
export SPLIT="train"
export BATCH_SIZE=2
export EMBED_MODEL="mxbai-embed-large"
export CHUNK_SIZE=400 #800
export CHUNK_OVERLAP=80 #150
export MIN_SPLIT_LEN=450 #850
export MAX_TOKENS=8192
export NUM_CTX=16384  #4096 #32768 #65536
export GEN_MODEL="qwen2.5:14b" #"mistral-nemo" #"qwen2.5:7b"
export OUTPUT_DIR="./output"


python main.py \
    --data_dir $DATA_DIR \
    --index_dir $INDEX_DIR \
    --split $SPLIT \
    --eval_batch_size $BATCH_SIZE \
    --ret_eval \
    --verbose \
    --max_tokens $MAX_TOKENS \
    --num_ctx $NUM_CTX \
    --generation_model $GEN_MODEL \
    --output_dir $OUTPUT_DIR \

#--rebuild \
#--embed_model $EMBED_MODEL \
#--chunk_size $CHUNK_SIZE \
#--chunk_overlap $CHUNK_OVERLAP \
#--min_split_len $MIN_SPLIT_LEN \
#--rebuild \
#--max_tokens $MAX_TOKENS \
#--num_ctx $NUM_CTX \
