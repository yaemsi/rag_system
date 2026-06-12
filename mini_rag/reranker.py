"""
reranker.py — Cross-encoder reranker using Qwen3-Reranker via Ollama chat API.

Ollama does not expose a dedicated /rerank endpoint. Instead, Qwen3-Reranker
models are used through the standard chat API with a structured prompt that
asks the model to judge relevance. The model outputs a yes/no judgment and
we use the log-probability of the "yes" token as the relevance score.

If log-probabilities are unavailable we fall back to parsing the text output
directly ("yes" → 1.0, "no" → 0.0).

Recommended model: dengcao/Qwen3-Reranker-0.6B:Q8_0  (~0.7GB, fast)
                   dengcao/Qwen3-Reranker-4B:Q5_K_M   (~3GB,  better quality)

Pull with:
    ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
    ollama pull dengcao/Qwen3-Reranker-4B:Q5_K_M
"""
from __future__ import annotations
import math
import re
from argparse import Namespace
from dataclasses import dataclass

import ollama

from mini_rag.retriever import RetrievedChunk

# Qwen3-Reranker prompt template (from official Qwen3 docs)
_RERANK_SYSTEM = (
    "Judge whether the following Document is relevant to the Query. "
    'Output "yes" if it is relevant, "no" if it is not relevant. '
    "Only output a single word."
)

_RERANK_USER_TEMPLATE = "Query: {query}\n\nDocument: {document}"


def _score_chunk(model: str, query: str, document: str) -> float:
    """
    Score a single (query, document) pair using Qwen3-Reranker.

    Returns a float in [0, 1]: higher = more relevant.
    Tries log-probabilities first; falls back to text parsing.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _RERANK_SYSTEM},
            {"role": "user",   "content": _RERANK_USER_TEMPLATE.format(
                query=query, document=document
            )},
        ],
        options={
            "temperature": 0.0,
            "num_predict": 4,   # we only need "yes" or "no"
        },
    )

    text = response["message"]["content"].strip().lower()

    # Try log-probability extraction if available
    logprobs = (
        response.get("message", {}).get("logprobs") or
        response.get("logprobs")
    )
    if logprobs:
        # Sum log-probs of "yes" tokens in the first generated token
        try:
            first_token_lps = logprobs[0] if isinstance(logprobs, list) else logprobs
            for entry in (first_token_lps if isinstance(first_token_lps, list) else [first_token_lps]):
                token = (entry.get("token") or "").strip().lower()
                if token in ("yes", "▁yes"):
                    lp = entry.get("logprob", -10)
                    return math.exp(lp)   # convert log-prob → prob
        except Exception:
            pass

    # Fallback: parse text directly
    if re.search(r"\byes\b", text):
        return 1.0
    if re.search(r"\bno\b", text):
        return 0.0
    # Ambiguous output — assign neutral score
    return 0.5


class Reranker:
    """
    Cross-encoder reranker. Scores each (query, chunk) pair using
    Qwen3-Reranker via the Ollama chat API and returns the top-k
    most relevant chunks.

    When disabled (params.use_reranker=False), passes chunks through
    unchanged — the pipeline behaves exactly as before, just truncated
    to top_k_reader by RRF score.
    """

    def __init__(self, params: Namespace) -> None:
        self._params = params

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks and return top_k_reader for the LLM reader.

        Parameters
        ----------
        query  : the user question
        chunks : candidates from the hybrid retriever (typically top-20)

        Returns
        -------
        Top top_k_reader chunks ordered by rerank score (descending).
        """
        top_k = self._params.top_k_reader

        if not self._params.use_reranker or not chunks:
            return chunks[:top_k]

        model = self._params.rerank_model

        # Score each chunk — cross-encoder scores every (query, chunk) pair
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            score = _score_chunk(model, query, chunk.chunk_text)
            scored.append((score, chunk))

        # Sort by rerank score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top_k with rerank score stored in the chunk's score field
        result: list[RetrievedChunk] = []
        for score, chunk in scored[:top_k]:
            result.append(RetrievedChunk(
                doc_id=chunk.doc_id,
                chunk_text=chunk.chunk_text,
                chunk_index=chunk.chunk_index,
                score=score,
            ))

        return result
