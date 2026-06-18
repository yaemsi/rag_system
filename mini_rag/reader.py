"""
reader.py — LLM-based reader that generates answers grounded in retrieved chunks.

Uses a vLLM generation server via the OpenAI-compatible chat completions API.

vLLM advantage over Ollama
---------------------------
- Explicit max_tokens and temperature without Ollama-specific option keys.
- No num_ctx required — vLLM manages the KV cache internally based on the
  model's configured max_model_len, preventing the silent context truncation
  that caused qwen2.5:14b's high refusal rate under Ollama.
"""
from __future__ import annotations

from argparse import Namespace
from functools import lru_cache
import time

from openai import OpenAI

from mini_rag.retriever import RetrievedChunk

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Your answers must be grounded \
exclusively in the provided context passages. Do not use any outside knowledge.

Rules:
- Be EXHAUSTIVE and SPECIFIC. Include ALL relevant details from the context: \
  feature names, version numbers, prices, plan names, configuration steps, \
  parameter names, exact values. Do not summarise vaguely when specifics exist.
- For questions about features, list every feature mentioned in the context.
- For questions about pricing, include every plan name, price, and add-on cost.
- For questions about release notes, list every fix, addition, and change mentioned.
- Only refuse to answer if the context passages are entirely unrelated to the \
  question and contain absolutely no useful information — not even tangentially. \
  When in doubt, answer with whatever relevant information is in the context.
- If you truly cannot answer, respond with exactly: \
  "I cannot answer this question from the available documents."
- Never fabricate facts, names, versions, or numbers not present in the context.
- Cite the source document ID at the end of your answer in brackets, e.g. [doc: 12345].
"""

_PROMPT_TEMPLATE = """\
Question: {question}

Context passages:
{context}

Answer (grounded in the context above):"""

_UNANSWERABLE_SIGNAL = "I cannot answer this question from the available documents."


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Passage {i} | doc_id: {chunk.doc_id}]\n{chunk.chunk_text}")
    return "\n\n---\n\n".join(parts)


@lru_cache(maxsize=1)
def _get_gen_client(base_url: str) -> OpenAI:
    """Return a cached OpenAI client pointed at the vLLM generation server."""
    return OpenAI(base_url=base_url, api_key="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")


def _call_vllm(params: Namespace, system: str, user: str) -> str:
    client = _get_gen_client(params.generation_base_url)
    response = client.chat.completions.create(
        model=params.generation_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=params.max_tokens or 2048,
        temperature=params.temperature,
    )
    return response.choices[0].message.content.strip()


class Reader:
    """
    Generates a grounded answer given a question and retrieved chunks.
    """
    def __init__(self, params: Namespace) -> None:
        self._params = params

    def answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        low_confidence: bool = False,
    ) -> tuple[str | None, list[str]]:
        """
        Generate an answer.

        Returns
        -------
        (answer_text, doc_ids)
            answer_text is None when the system decides not to answer.
            doc_ids lists the source document IDs used.

        Note: low_confidence flag is accepted for API compatibility but no
        longer adds a cautious-mode instruction — empirically this caused
        over-refusal without improving grounding quality.
        """
        if not chunks:
            return None, []

        context = _format_context(chunks)
        prompt = _PROMPT_TEMPLATE.format(question=question, context=context)
        raw = _call_vllm(self._params, _SYSTEM_PROMPT, prompt)

        # Cool-down between inference calls — helps prevent thermal crashes
        # on sustained workloads (e.g. RTX 5090 under long eval runs)
        if self._params.inference_delay > 0:
            time.sleep(self._params.inference_delay)

        if _UNANSWERABLE_SIGNAL.lower() in raw.lower():
            return None, []

        doc_ids = [c.doc_id for c in chunks]
        return raw, doc_ids
