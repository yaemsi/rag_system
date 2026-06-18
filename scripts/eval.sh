#!/bin/bash
# eval.sh — Launch vLLM servers and run evaluation
#
# Usage:
#   bash scripts/eval.sh            # run with defaults
#   bash scripts/eval.sh --rebuild  # force re-embedding
#
# vLLM servers must be running before executing this script.
# Start them with:
#   vllm serve nomic-ai/nomic-embed-text-v1 \
#       --port 8000 --task embed \
#       --trust-remote-code
#
#   vllm serve mistralai/Mistral-Nemo-Instruct-2407 \
#       --port 8001 \
#       --max-model-len 4096
#
#   # Optional reranker:
#   vllm serve Qwen/Qwen3-Reranker-4B \
#       --port 8002 \
#       --max-model-len 4096

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TRITON_CACHE_DIR="./.triton_cache"

# ── Data / index / output paths ──────────────────────────────────────────────
export DATA_DIR="./data/gz"
export INDEX_DIR="./data/index"
export OUTPUT_DIR="./output"
export SPLIT="valid"

# ── Evaluation settings ───────────────────────────────────────────────────────
export BATCH_SIZE=16

# ── Embedding server ──────────────────────────────────────────────────────────
export EMBED_MODEL="nomic-ai/nomic-embed-text-v1"
export EMBED_BASE_URL="http://localhost:8000/v1"
export EMBED_DIM=768

# ── Chunker ───────────────────────────────────────────────────────────────────
export CHUNK_SIZE=800
export CHUNK_OVERLAP=150
export MIN_SPLIT_LEN=850

# ── Generation server ─────────────────────────────────────────────────────────
#export GEN_MODEL="casperhansen/mistral-nemo-instruct-2407-awq"
#export GEN_MODEL="mistralai/mistral-nemo-instruct-2407"
export GEN_MODEL="Qwen/Qwen3-8B-AWQ"
export GEN_BASE_URL="http://localhost:8001/v1"
export MAX_TOKENS=2048
export TEMPERATURE=0.0
export INFERENCE_DELAY=0.0

# ── Reranker server (disabled by default) ────────────────────────────────────
export USE_RERANKER=false
export RERANK_MODEL="Qwen/Qwen3-Reranker-4B"
export RERANK_BASE_URL="http://localhost:8002/v1"
export TOP_K_READER=5

python main.py \
    --data_dir          $DATA_DIR \
    --index_dir         $INDEX_DIR \
    --output_dir        $OUTPUT_DIR \
    --split             $SPLIT \
    --eval_batch_size   $BATCH_SIZE \
    --ret_eval \
    --verbose \
    --embed_model       $EMBED_MODEL \
    --embed_base_url    $EMBED_BASE_URL \
    --embed_dim         $EMBED_DIM \
    --chunk_size        $CHUNK_SIZE \
    --chunk_overlap     $CHUNK_OVERLAP \
    --min_split_len     $MIN_SPLIT_LEN \
    --generation_model  $GEN_MODEL \
    --generation_base_url $GEN_BASE_URL \
    --max_tokens        $MAX_TOKENS \
    --temperature       $TEMPERATURE \
    --inference_delay   $INFERENCE_DELAY \
    --use_reranker      $USE_RERANKER \
    --rerank_model      $RERANK_MODEL \
    --rerank_base_url   $RERANK_BASE_URL \
    --top_k_reader      $TOP_K_READER \
    --rebuild \

# --use_reranker      $USE_RERANKER \
# --rerank_model      $RERANK_MODEL \
# --rerank_base_url   $RERANK_BASE_URL \
# --top_k_reader      $TOP_K_READER \
# --rebuild \
