"""
reader.py — LLM-based reader that generates answers grounded in retrieved chunks.
 
Uses Ollama with qwen2.5:14b (configurable).
Handles three answer modes:
  1. Standard — single doc, factual lookup
  2. Multi-doc — aggregation / comparison across several docs
  3. Unanswerable — no relevant evidence found
"""
from __future__ import annotations

from argparse import Namespace
 
import ollama
 
from mini_rag.retriever import RetrievedChunk
 
GENERATION_MODEL = "qwen2.5:14b"
GENERATION_MODEL = "mistral-nemo" # cheaper, but less accurate for grounded QA
 
# Confidence threshold: if the best retrieval score is below this,
# we treat the query as potentially unanswerable and tell the LLM so.
#LOW_CONFIDENCE_THRESHOLD = 0.02
LOW_CONFIDENCE_THRESHOLD = 0.005
 
_SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Your answers must be grounded \
exclusively in the provided context passages. Do not use any outside knowledge.
 
Rules:
- Answer using only facts present in the context. Be specific and complete — \
  include all relevant details such as feature names, version numbers, prices, \
  and configuration steps that appear in the context.
- Only refuse to answer if the context passages are entirely unrelated to the \
  question and contain no useful information whatsoever. If the context is \
  partially relevant, use what is there and answer to the best of your ability.
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
    model_id = params.generation_model
    response = ollama.chat(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={
            "max_tokens": params.max_tokens,
            "num_ctx": params.num_ctx,
            "temperature": params.temperature
            },   
    )
    return response["message"]["content"].strip()
 
 
class Reader:
    """
    Generates a grounded answer given a question and retrieved chunks.
    """
    def __init__(self, params: Namespace) -> None:
        self._params = params                 # generic parameters
 
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
        """
        if not chunks:
            return None, []
 
        context = _format_context(chunks)
        prompt = _PROMPT_TEMPLATE.format(question=question, context=context)
 
        # If retrieval confidence is low, prime the model to be more cautious
        system = _SYSTEM_PROMPT
        if low_confidence:
            system += (
                "\nNote: The retrieved passages may not be relevant to this question. "
                "Be especially critical — if the passages do not clearly address the "
                "question, say you cannot answer."
            )
 
        raw = _call_ollama(self._params, system, prompt)
 
        # Detect explicit refusal
        if _UNANSWERABLE_SIGNAL.lower() in raw.lower():
            return None, []
 
        doc_ids = [c.doc_id for c in chunks]
        return raw, doc_ids