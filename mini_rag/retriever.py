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
 
from dataclasses import dataclass

import numpy as np

import re
import faiss
import ollama

from loguru import logger
from rank_bm25 import BM25Okapi
 
from mini_rag.corpus import Document
from tqdm import tqdm
 
@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_text: str
    chunk_index: int
    score: float
 
# Hard context limits per model (in characters, conservative estimate)
# mxbai-embed-large: 512 tokens ≈ 1800 chars
# nomic-embed-text:  8192 tokens ≈ no practical limit for our chunks
_MODEL_CHAR_LIMITS: dict[str, int] = {
    "mxbai-embed-large": 1800,
    "nomic-embed-text":  32000,
}

def _embed(model_id: str, texts: list[str]) -> np.ndarray:
    """Embed a list of texts via Ollama; return (N, DIM) float32 array."""
    vecs = []
    char_limit = _MODEL_CHAR_LIMITS.get(model_id, 1800)
    for text in texts:
        if len(text) > char_limit:
            text = text[:char_limit]
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
 
        self._docs = {doc.id: doc for doc in docs}
 
        # Flatten chunks
        for doc in docs:
            for chunk in doc.chunks:
                self._chunks.append(chunk)
                self._chunk_doc_ids.append(doc.id)
 
        n = len(self._chunks)
        logger.info(f"[Retriever] Indexing {n} chunks from {len(docs)} documents …")
 
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
 
        logger.info(f"[Retriever] Index ready.")
 
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
        Returns None if no strong metadata signal is found — in which case
        retrieval runs over the full index.
 
        Scoring per doc (higher = more specific match):
          +3  exact combined name match (e.g. "Solara 40C" as phrase in query)
          +2  suffix matches (most discriminative for product variants)
          +2  version matches (normalises both "v2.1.15" and "2.1.15")
          +1  prefix matches (broad family signal)
 
        Fallback tiers (applied in order, return first non-empty):
          1. score >= 4  (prefix + suffix + version — very specific, always trust)
          2. score >= 3  (suffix + version, or full combined match)
          3. score >= 2  (suffix alone, or version alone)
          4. score >= 1  (prefix alone — only if result set is large enough,
                          otherwise fall back to full index to avoid FILTER_KILL
                          on small wrong sets)
        """

        # Minimum token length to prevent short common words (e.g. "qubit", "unify")
        # from firing the metadata filter on Coveo docs where they appear as API names.
        _MIN_PREFIX_TOKEN_LEN = 6
    
        # If the filter returns a small candidate set (≤ this) and we're only matching
        # on prefix alone (score=1), fall back to full index — a tight wrong set is
        # worse than no filter at all.
        _MAX_PREFIX_ONLY_SIZE = 6

        q_lower = query.lower()
        q_tokens = set(re.findall(r"\w+", q_lower))
        scores_map: dict[str, int] = {}
 
        for doc in self._docs.values():
            if not doc.is_synthetic:
                continue
 
            score = 0
            prefix  = doc.product_prefix
            suffix  = doc.product_suffix
            version = doc.product_version
 
            # Full combined name (e.g. "Solara 40C")
            if prefix and suffix:
                combined = f"{prefix.lower()} {suffix.lower()}"
                if combined in q_lower:
                    score += 3
 
            # Individual fields
            if prefix and prefix.lower() in q_lower:
                score += 1
            if suffix and suffix.lower() in q_lower:
                score += 2
            if version:
                # Normalise both "v1.2.3" and "1.2.3" forms
                v_clean = version.lower().lstrip("v")
                if version.lower() in q_lower or v_clean in q_lower:
                    score += 2
 
            # Token-level prefix match — enforce minimum token length to prevent
            # short common words ("qubit", "unify") from matching Coveo API queries
            if score == 0 and prefix:
                for tok in re.findall(r"\w+", prefix.lower()):
                    if len(tok) >= _MIN_PREFIX_TOKEN_LEN and tok in q_tokens:
                        score = 1
                        break
 
            if score > 0:
                scores_map[doc.id] = score
 
        if not scores_map:
            return None
 
        # Return most specific match tier with at least 1 result.
        # For tier 1 (prefix-only, score=1): only apply the filter if the
        # candidate set is large enough — a small tight wrong set is worse
        # than no filter (it causes FILTER_KILL with no fallback).
        for threshold in (4, 3, 2):
            tier = {did for did, s in scores_map.items() if s >= threshold}
            if tier:
                # Version-mismatch expansion: if the query contains a version string
                # but no doc in this tier matches it, expand to the full prefix+suffix
                # family to avoid locking into wrong-version docs.
                # E.g. "Yore ZQ8 version 1.1.10" → suffix matches (score=2) but
                # version 1.1.10 is absent → expand to all Yore ZQ8 docs.
                version_in_query = bool(re.search(r"\b\d+\.\d+\.\d+\b", query.lower()))
                if version_in_query and threshold == 2:
                    # Check if any doc in this tier actually matched the version
                    version_matched = {
                        did for did, s in scores_map.items()
                        if s >= 4  # score>=4 requires version match
                    }
                    if not version_matched:
                        # No version match — expand to full prefix family to avoid
                        # locking into wrong-version docs
                        expanded = {did for did, s in scores_map.items() if s >= 1}
                        if len(expanded) > len(tier):
                            return expanded
                return tier
 
        # Tier 4: prefix-only match — only trust if set is large enough
        # (large set = prefix is genuinely discriminative; small set = likely wrong family)
        prefix_tier = {did for did, s in scores_map.items() if s >= 1}
        if len(prefix_tier) > _MAX_PREFIX_ONLY_SIZE:
            return prefix_tier
 
        # Prefix-only with small set → fall back to full index
        return None
 
 