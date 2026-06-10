# **Technical Challenge: Build a Question Answering System Over a Document Corpus**

## **Pre-requisites**

## **Installation**



## Setup

```bash
pip install -e ".[dev]"

# Pull Ollama models (one-time)
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

## Run

```bash
# First run: builds the FAISS index (~10–20 min, embeds all chunks)
python main.py --split train

# Subsequent runs use the cached index (fast)
python main.py --split valid
python main.py --split bonus

# Force full rebuild
python main.py --rebuild
```

## Project Structure

```
coveo_challenge/
├── main.py                  # Entry point
├── pyproject.toml
├── data/                    # corpus.jsonl.gz, train/valid/bonus.jsonl.gz
├── index/                   # FAISS index + BM25 cache
└── coveo_challenge/
    ├── corpus.py            # Document model + data loaders
    ├── chunker.py           # Document → chunks (heading-based + sliding window)
    ├── retriever.py         # Hybrid dense (FAISS) + sparse (BM25) + RRF fusion
    ├── reader.py            # Ollama LLM reader (qwen2.5:14b)
    ├── index_store.py       # Save / load FAISS index to disk
    ├── rag_system.py        # RAGQnASystem (implements QnASystem interface)
    ├── evaluation.py        # EvaluationLoop + 4 metrics
    └── qna_system.py        # Abstract QnASystem base class
```

## Architecture

```
Query
  │
  ▼
Metadata pre-filter ──► narrows to matching product docs (synthetic only)
  │
  ▼
Hybrid Retrieval
  ├── Dense: nomic-embed-text + FAISS (cosine similarity)
  └── Sparse: BM25Okapi
        └── RRF fusion → top-5 chunks
  │
  ▼
Confidence check (score threshold → low-confidence flag)
  │
  ▼
LLM Reader (qwen2.5:14b via Ollama)
  └── Grounded answer or refusal
```

## Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | Normalised string equality |
| `token_f1` | Token-level precision/recall F1 (SQuAD-style) |
| `answer_coverage` | Token recall — how much of the gold answer is covered |
| `refusal_rate` | Fraction of questions refused (diagnostic) |