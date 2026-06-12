"""
reader.py — LLM-based reader that generates answers grounded in retrieved chunks.
 
Uses Ollama (configurable model via ReaderArguments).
Handles three answer modes:
  1. Standard — single doc, factual lookup
  2. Multi-doc — aggregation / comparison across several docs
  3. Unanswerable — no relevant evidence found
"""
from __future__ import annotations
 
from argparse import Namespace
import time
 
import ollama
 
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
 
 
def _call_ollama(params: Namespace, system: str, user: str) -> str:
    response = ollama.chat(
        model=params.generation_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={
            "max_tokens": params.max_tokens,
            "num_ctx": params.num_ctx,
            "temperature": params.temperature,
        },
    )
    return response["message"]["content"].strip()
 
 
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
        raw = _call_ollama(self._params, _SYSTEM_PROMPT, prompt)

        # Cool-down between inference calls — helps prevent thermal crashes
        # on sustained workloads (e.g. RTX 5090 under long eval runs)
        if self._params.inference_delay > 0:
            time.sleep(self._params.inference_delay)
 
        if _UNANSWERABLE_SIGNAL.lower() in raw.lower():
            return None, []
 
        doc_ids = [c.doc_id for c in chunks]
        return raw, doc_ids
 