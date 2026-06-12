# Mini-RAG — Question Answering System over a Document Corpus

A Retrieval-Augmented Generation (RAG) system that answers natural-language
queries grounded in a document corpus using local models via Ollama.

---

## Pre-requisites

- **Python** ≥ 3.13
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **[Ollama](https://ollama.com/)** — local LLM inference runtime
- **CUDA-capable GPU** recommended (tested on NVIDIA RTX 5090, 24 GB VRAM)

---

## Installation

### 1. Install uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 2. Clone and install dependencies

```bash
git clone <repo-url>
cd coveo-challenge

# Create virtual environment and install all dependencies
uv sync

# Activate the environment
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 3. Pull Ollama models

```bash
# Embedding model (required)
ollama pull nomic-embed-text       # 768-dim, 8k context — default
# ollama pull mxbai-embed-large    # 1024-dim, alternative

# Generation model (pick one)
ollama pull mistral-nemo           # 12B — recommended default
# ollama pull qwen2.5:14b          # 14B — better quality, higher VRAM
# ollama pull qwen2.5:7b           # 7B — lighter option

# Reranker model (optional)
# ollama pull dengcao/Qwen3-Reranker-4B:Q5_K_M
```

---

## Data Setup

Place the dataset files under `data/gz/`:

```
data/
└── gz/
    ├── corpus.jsonl.gz   # Document corpus (3,630 docs)
    ├── train.jsonl.gz    # Training queries + gold answers (164)
    ├── valid.jsonl.gz    # Validation queries + gold answers (49)
    └── bonus.jsonl.gz    # Bonus queries, no gold answers (7)
```

---

## Run

The recommended way to run is via the provided shell script:

```bash
bash scripts/eval.sh
```

Or directly with `main.py`:

```bash
# First run — builds and saves the FAISS index (~10–20 min)
python main.py --split valid

# Subsequent runs — loads cached index (fast)
python main.py --split valid

# Force full index rebuild
python main.py --split valid --rebuild

# Evaluate on training set (use sub-splits to avoid thermal issues)
python main.py --split train_1   # 41 questions
python main.py --split train_2
python main.py --split train_3
python main.py --split train_4

# Run on bonus queries (no gold answers — prints responses only)
python main.py --split bonus

# Enable retrieval evaluation + verbose per-query diagnosis
python main.py --split valid --ret_eval --verbose

# Enable reranker
python main.py --split valid --use_reranker \
    --rerank_model dengcao/Qwen3-Reranker-4B:Q5_K_M

# Add a cool-down delay between LLM calls (prevents GPU thermal crashes
# on sustained workloads, e.g. RTX 5090 under full train eval)
python main.py --split train_1 --inference_delay 2.0
```

Results are saved to `output/results-{split}.json` after each run.
Retrieval results are saved to `output/results-{split}-retrieval.json`.

---

## Project Structure

```
coveo-challenge/
├── main.py               # Entry point
├── pyproject.toml        # Dependencies (managed by uv)
├── uv.lock               # Locked dependency versions
├── scripts/
│   └── eval.sh           # Configurable evaluation script
├── data/
│   ├── gz/               # Raw data files (.jsonl.gz)
│   └── index/            # Auto-created: FAISS index + BM25 cache
├── output/               # Auto-created: evaluation results (JSON)
└── mini_rag/
    ├── arguments.py      # All hyperparameters as HF-style dataclass groups
    ├── chunker.py        # Heading-based + sliding-window chunking
    ├── corpus.py         # Document model, data loaders, inferred_suffix
    ├── evaluation.py     # EvaluationLoop + RetrievalEvaluationLoop
    ├── index_store.py    # FAISS + BM25 index persistence
    ├── metrics.py        # QA metrics (Token F1, Coverage, EM, Refusal Rate)
    │                     # Retrieval metrics (Recall@k, Precision@k, MRR)
    ├── qna_system.py     # Abstract QnASystem base class
    ├── rag_system.py     # RAGQnASystem — wires all components together
    ├── reader.py         # LLM reader (Ollama chat API)
    ├── reranker.py       # Cross-encoder reranker (Qwen3-Reranker via Ollama)
    └── retriever.py      # Hybrid dense (FAISS) + sparse (BM25) + RRF fusion
```

---

## Pipeline Architecture

```
Query
  │
  ▼
Metadata pre-filter       ← product prefix/suffix/version matching
  │                          (synthetic product docs only)
  ▼
Hybrid Retrieval          ← top_k_final=20 candidates
  ├── Dense: nomic-embed-text + FAISS (cosine similarity)
  └── Sparse: BM25Okapi
        └── RRF fusion (k=60)
  │
  ▼
Reranker [optional]       ← Qwen3-Reranker via Ollama chat API
  │                          top_k_final=20 → top_k_reader=5
  ▼
LLM Reader                ← mistral-nemo / qwen2.5:14b via Ollama
  └── Grounded answer or explicit refusal
  │
  ▼
output/results-{split}.json
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--split` | `train` | Dataset split: `train`, `train_1`–`4`, `valid`, `bonus` |
| `--generation_model` | `mistral-nemo` | Ollama generation model |
| `--embed_model` | `nomic-embed-text` | Ollama embedding model |
| `--chunk_size` | `800` | Chunk size in characters |
| `--chunk_overlap` | `150` | Overlap between consecutive chunks |
| `--top_k_final` | `20` | Candidates retrieved before reranking |
| `--top_k_reader` | `5` | Chunks passed to the LLM reader |
| `--max_tokens` | `None` | Max generation tokens (model default if unset) |
| `--num_ctx` | `None` | Ollama context window (model default if unset) |
| `--inference_delay` | `0.0` | Sleep (s) between LLM calls — prevents thermal crashes |
| `--ret_eval` | `False` | Run retrieval evaluation (Recall@k, MRR, etc.) |
| `--verbose` | `False` | Print per-question scores and retrieval diagnosis |
| `--use_reranker` | `False` | Enable cross-encoder reranking |
| `--rebuild` | `False` | Force re-embedding of the corpus |
| `--output_dir` | `./output` | Directory for JSON result files |

---

## Evaluation Metrics

### QA Metrics (end-to-end)

| Metric | Description |
|---|---|
| `token_f1` | Token-level precision/recall F1 — primary QA metric |
| `answer_coverage` | Token recall of gold answer in prediction |
| `exact_match` | Normalised string equality — always 0 (answers are paraphrased) |
| `refusal_rate` | Fraction of questions the system refused to answer |

### Retrieval Metrics (independent of LLM)

| Metric | Description |
|---|---|
| `recall@k` | Fraction of queries where gold doc is in top-k |
| `precision@k` | (1/k) if gold doc in top-k, else 0 |
| `mrr` | Mean Reciprocal Rank of the gold document |

Retrieval failures are classified as `OK`, `LATE`, `MISS`, or `FILTER_KILL`
for targeted diagnosis.

---

## Notes

- **GPU thermal crashes (RTX 5090):** a known hardware issue under sustained
  CUDA workloads. Use `--inference_delay 2.0` and/or cap GPU power with
  `sudo nvidia-smi -pl 450` before running long evaluations.
- **Index caching:** the FAISS index is saved to `data/index/` after the first
  build. Subsequent runs skip embedding and load in seconds. Use `--rebuild`
  only when changing the embedding model or chunk size.
- **Sub-splits:** `train_1` through `train_4` (41 questions each) are provided
  to work around the thermal crash issue on the full 164-question train set.
