"""
retriever.py — Hybrid retriever: dense (FAISS + nomic-embed-text) + sparse (BM25).
 
Architecture
------------
1. Each document chunk is embedded via Ollama nomic-embed-text.
2. Embeddings are stored in a FAISS flat L2 index (exact search, no quantisation).
3. BM25 index is built in parallel over the same chunks.
4. At query time:
   a. Metadata pre-filter (product_prefix / suffix / version) narrows candidate
      doc IDs for synthetic product queries, avoiding irrelevant corpus noise.
   b. Dense + sparse scores are fused with Reciprocal Rank Fusion (RRF).
   c. Top-k unique documents are returned (de-duplicated by doc_id).
"""
from __future__ import annotations

from argparse import Namespace
 
import re
from dataclasses import dataclass
 
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import ollama
 
from mini_rag.corpus import Document
  
 
@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_text: str
    chunk_index: int
    score: float
 
 
def _embed(model_id: str, texts: list[str]) -> np.ndarray:
    """Embed a list of texts via Ollama; return (N, DIM) float32 array."""
    vecs = []
    for text in texts:
        resp = ollama.embeddings(model=model_id, prompt=text)
        vecs.append(resp["embedding"])
    return np.array(vecs, dtype=np.float32)
 
 
def _tokenize(text: str) -> list[str]:
    """Lightweight tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())
 
 
def _rrf_fuse(
    dense_ranked: list[int],
    sparse_ranked: list[int],
    k: int,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion of two ranked lists of chunk indices.
    Returns list of (chunk_idx, rrf_score) sorted descending.
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
 
 
class Retriever:
    """
    Builds and queries a hybrid dense+sparse index over document chunks.
    """
    def __init__(self, params: Namespace) -> None:
        self._params = params                 # generic parameters
        self._chunks: list[str] = []          # flat list of all chunk texts
        self._chunk_doc_ids: list[str] = []   # parallel: which doc each chunk belongs to
        self._docs: dict[str, Document] = {}  # doc_id -> Document (for metadata)
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._bm25: BM25Okapi | None = None
 
    # ── Build ────────────────────────────────────────────────────────────────
 
    def build(self, docs: list[Document], show_progress: bool = True) -> None:
        """
        Index all documents. Embeds every chunk via Ollama.
        Expensive on first run — results should be cached externally.
        """
        from tqdm import tqdm
 
        self._docs = {doc.id: doc for doc in docs}
 
        # Flatten chunks
        for doc in docs:
            for chunk in doc.chunks:
                self._chunks.append(chunk)
                self._chunk_doc_ids.append(doc.id)
 
        n = len(self._chunks)
        print(f"[Retriever] Indexing {n} chunks from {len(docs)} documents …")
 
        # ── Dense index ───────────────────────────────────────────────────
        batch_size = self._params.embedding_batch_size
        all_vecs: list[np.ndarray] = []
        iterator = range(0, n, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding chunks")
 
        for start in iterator:
            batch = self._chunks[start : start + batch_size]
            vecs = _embed(self._params.embed_model, batch)
            all_vecs.append(vecs)
 
        matrix = np.vstack(all_vecs).astype(np.float32)
        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(matrix)
 
        self._faiss_index = faiss.IndexFlatIP(matrix.shape[1])
        self._faiss_index.add(matrix)
 
        # ── Sparse index ──────────────────────────────────────────────────
        tokenized = [_tokenize(c) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
 
        print(f"[Retriever] Index ready.")
 
    # ── Query ────────────────────────────────────────────────────────────────
 
    def retrieve(
        self,
        query: str,
        candidate_doc_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-k relevant chunks using hybrid RRF search.
 
        Parameters
        ----------
        query : str
        candidate_doc_ids : set[str] | None
            If provided, restrict retrieval to these doc IDs (metadata pre-filter).
        """
        assert self._faiss_index is not None, "Call build() first."
 
        # ── Optional metadata pre-filter ─────────────────────────────────
        if candidate_doc_ids is not None:
            valid_indices = [
                i for i, did in enumerate(self._chunk_doc_ids)
                if did in candidate_doc_ids
            ]
            if not valid_indices:
                # Filter too aggressive — fall back to full index
                valid_indices = list(range(len(self._chunks)))
        else:
            valid_indices = list(range(len(self._chunks)))
 
        # ── Dense retrieval ───────────────────────────────────────────────
        q_vec = _embed(self._params.embed_model, [query]).astype(np.float32)
        faiss.normalize_L2(q_vec)
 
        # Search full index, then filter to valid indices
        top_k_dense = self._params.top_k_dense
        n_search = min(top_k_dense * 5, len(self._chunks))
        _, raw_indices = self._faiss_index.search(q_vec, n_search)
        dense_ranked = [int(i) for i in raw_indices[0] if int(i) in set(valid_indices)][:top_k_dense]
 
        # ── Sparse retrieval ──────────────────────────────────────────────
        q_tokens = _tokenize(query)
        bm25_scores = self._bm25.get_scores(q_tokens)
        # Zero out scores for chunks outside the filter
        mask = np.zeros(len(self._chunks), dtype=np.float32)
        for i in valid_indices:
            mask[i] = 1.0
        bm25_scores = bm25_scores * mask
        top_k_sparse = self._params.top_k_sparse
        sparse_ranked = list(np.argsort(bm25_scores)[::-1][:top_k_sparse])
 
        # ── RRF fusion ────────────────────────────────────────────────────
        fused = _rrf_fuse(dense_ranked, sparse_ranked, self._params.rrf_k)
 
        # De-duplicate by doc_id, keeping highest-scoring chunk per doc
        seen_docs: set[str] = set()
        results: list[RetrievedChunk] = []
        top_k = self._params.top_k_final
        for chunk_idx, score in fused:
            doc_id = self._chunk_doc_ids[chunk_idx]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            results.append(RetrievedChunk(
                doc_id=doc_id,
                chunk_text=self._chunks[chunk_idx],
                chunk_index=chunk_idx,
                score=score,
            ))
            if len(results) >= top_k:
                break
 
        return results
 
    # ── Metadata pre-filter helper ───────────────────────────────────────────
 
    def filter_by_metadata(self, query: str) -> set[str] | None:
        """
        Heuristic: if the query mentions a known product name / version,
        return the set of matching doc IDs to pre-filter retrieval.
        Returns None if no strong metadata signal is found.
        """
        q_lower = query.lower()
        matches: set[str] = set()
 
        for doc in self._docs.values():
            if not doc.is_synthetic:
                continue
            signals = [
                doc.product_prefix,
                doc.product_suffix,
                doc.product_version,
                doc.product_name,
            ]
            for sig in signals:
                if sig and sig.lower() in q_lower:
                    matches.add(doc.id)
                    break
 
        return matches if matches else None
 