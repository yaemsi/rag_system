"""
reranker.py — Cross-encoder reranker using vLLM completions with logprobs.

vLLM exposes token log-probabilities via its OpenAI-compatible completions
endpoint (logprobs parameter). This gives continuous relevance scores in (0, 1)
rather than the binary yes/no fallback required under Ollama — which was the
root cause of the reranker regression in the Ollama experiments.

How it works
------------
For each (query, chunk) pair, the reranker sends a relevance judgment prompt
and requests logprobs=True for the first generated token. It then extracts the
log-probability of "yes" tokens from the top-logprobs distribution and converts
to a probability via exp(logprob). This gives a well-calibrated continuous
relevance score rather than a hard 0/1 boundary.

Model
-----
Any instruction-tuned model works. Qwen3-Reranker is purpose-built for this
task. Start the vLLM server with:
    vllm serve Qwen/Qwen3-Reranker-4B --port 8002
"""
from __future__ import annotations

import math
import re
from argparse import Namespace
from functools import lru_cache

from openai import OpenAI

from mini_rag.retriever import RetrievedChunk

_RERANK_PROMPT_TEMPLATE = """\
Judge whether the following Document is relevant to the Query.
Output "yes" if it is relevant, "no" if it is not relevant.
Only output a single word.

Query: {query}

Document: {document}"""

# Token surface forms that map to "yes" relevance
_YES_TOKENS = {"yes", " yes", "Yes", " Yes", "\u2581yes", "\u2581Yes"}


@lru_cache(maxsize=1)
def _get_rerank_client(base_url: str) -> OpenAI:
    """Return a cached OpenAI client pointed at the vLLM reranker server."""
    return OpenAI(base_url=base_url, api_key="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")


def _score_chunk(model: str, query: str, document: str, base_url: str) -> float:
    """
    Score a single (query, document) pair.

    Returns a float in [0, 1]: higher = more relevant.
    Uses vLLM logprobs for continuous scoring — no binary fallback needed.
    """
    client = _get_rerank_client(base_url)
    prompt = _RERANK_PROMPT_TEMPLATE.format(query=query, document=document)

    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=20,   # top-20 token log-probs for the first generated token
    )

    top_logprobs = response.choices[0].logprobs.top_logprobs[0]
    # Find the highest log-prob among yes-token surface forms
    lp = max((top_logprobs.get(t, -100.0) for t in _YES_TOKENS), default=-100.0)
    return math.exp(lp)


class Reranker:
    """
    Cross-encoder reranker using vLLM logprobs for continuous relevance scoring.

    When disabled (params.use_reranker=False), passes chunks through unchanged —
    the pipeline behaves exactly as before, truncated to top_k_reader by RRF score.
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
        print(f"############ Reranking {len(chunks)} chunks for query: {query}")
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            score = _score_chunk(
                self._params.rerank_model,
                query,
                chunk.chunk_text,
                self._params.rerank_base_url,
            )
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            RetrievedChunk(
                doc_id=chunk.doc_id,
                chunk_text=chunk.chunk_text,
                chunk_index=chunk.chunk_index,
                score=score,
            )
            for score, chunk in scored[:top_k]
        ]
