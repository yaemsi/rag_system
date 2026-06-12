#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export OLLAMA_GPU_OVERHEAD=1000000000  # Reserves ~1GB
#export OLLAMA_FLASH_ATTENTION=1
#export GGML_CUDA_DISABLE_GRAPHS=1
#export OLLAMA_NUM_PARALLEL=1


# Parameters
export DATA_DIR="./data/gz"
export INDEX_DIR="./data/index"
export SPLIT="train_a"
export BATCH_SIZE=16
export EMBED_MODEL="nomic-embed-text" # "nomic-embed-text" "mxbai-embed-large"
export CHUNK_SIZE=800    
export CHUNK_OVERLAP=150 
export MIN_SPLIT_LEN=850 
export MAX_TOKENS=8192   
export NUM_CTX=8192      
export GEN_MODEL="qwen2.5:14b" #"mistral-nemo" "qwen2.5:7b" "qwen2.5:14b"
export RERANKER_MODEL="dengcao/Qwen3-Reranker-4B:Q5_K_M" 
export TOP_K_READER=5
export INF_DELAY=2.0  # Seconds to sleep after each LLM call (prevents thermal crashes on sustained workloads, e.g. RTX 5090 under full train eval)
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
    --top_k_reader $TOP_K_READER \
    --output_dir $OUTPUT_DIR \
    --embed_model $EMBED_MODEL \
    --chunk_size $CHUNK_SIZE \
    --chunk_overlap $CHUNK_OVERLAP \
    --min_split_len $MIN_SPLIT_LEN \
    --max_tokens $MAX_TOKENS \
    --num_ctx $NUM_CTX \
    --inference_delay $INF_DELAY \

# --rebuild \
# --use_reranker \
# --rerank_model $RERANKER_MODEL \
# --top_k_reader $TOP_K_READER

