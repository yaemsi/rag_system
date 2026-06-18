#!/bin/bash

# Start embedding server and wait until it's ready
uv run vllm serve nomic-ai/nomic-embed-text-v1 \
    --port 8000 \
    --runner pooling \
    --trust-remote-code \
    --gpu-memory-utilization 0.10 \
    --enforce-eager

echo "Waiting for embedding server..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 2; done
echo "Embedding server ready."

# Start generation server and wait until it's ready
vllm serve casperhansen/mistral-nemo-instruct-2407-awq \
    --port 8001 \
    --runner generate \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.70 \
    --enforce-eager

echo "Waiting for generation server..."
until curl -s http://localhost:8001/health > /dev/null 2>&1; do sleep 2; done
echo "Generation server ready."

# Start reranker server and wait until it's ready
uv run vllm serve Qwen/Qwen3-Reranker-4B \
    --port 8002 \
    --max-model-len 4096 \
    --runner generate \
    --gpu-memory-utilization 0.20 \
    --enforce-eager

echo "Waiting for reranker server..."
until curl -s http://localhost:8002/health > /dev/null 2>&1; do sleep 2; done
echo "All servers ready."