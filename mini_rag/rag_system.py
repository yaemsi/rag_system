"""
rag_system.py — Concrete QnASystem implementation using RAG.
 
Pipeline per question:
  1. Metadata pre-filter (synthetic product docs only)
  2. Hybrid retrieval (dense + BM25, RRF fusion)
  3. Confidence check (score threshold)
  4. LLM reader generates grounded answer
"""
from __future__ import annotations

from argparse import Namespace
 
from pathlib import Path
 
from mini_rag.chunker import attach_chunks
from mini_rag.corpus import Document, load_corpus
from mini_rag.index_store import load_index, save_index
from mini_rag.qna_system import GroundedAnswer, QnASystem
from mini_rag.reader import Reader
from mini_rag.retriever import Retriever
 
 
class RAGQnASystem(QnASystem):
    """
    Retrieval-Augmented Generation QA system.
 
    Usage
    -----
    # Build from scratch (slow — embeds all chunks)
    system = RAGQnASystem.from_corpus(docs, index_dir="index/")
 
    # Load from pre-built index (fast)
    system = RAGQnASystem.load(index_dir="index/")
 
    # Query
    answers = system.get_answers(["What is X?", "How do I configure Y?"])
    """
 
    def __init__(self, retriever: Retriever, reader: Reader) -> None:
        self._retriever = retriever
        self._reader = reader
 
    # ── Constructors ─────────────────────────────────────────────────────────
 
    @classmethod
    def from_corpus(
        cls,
        chunk_params: Namespace,
        ret_params: Namespace,
        read_params: Namespace,
        docs: list[Document],
        index_dir: str | Path | None = None,
    ) -> "RAGQnASystem":
        """
        Build the index from a list of Document objects.
        If index_dir is given, persist the index there for future reuse.
        """
        attach_chunks(chunk_params, docs)
        retriever = Retriever(ret_params)
        retriever.build(docs)
        if index_dir is not None:
            save_index(retriever, index_dir)
        return cls(retriever=retriever, reader=Reader(read_params))
 
    @classmethod
    def load(
        cls,
        ret_params: Namespace,
        read_params: Namespace, 
        index_dir: str | Path
        ) -> "RAGQnASystem":
        """Load a pre-built index from disk."""
        retriever = load_index(ret_params, index_dir)
        return cls(retriever=retriever, reader=Reader(read_params))
 
    # ── QnASystem interface ───────────────────────────────────────────────────
 
    def get_answers(self, questions: list[str]) -> list[GroundedAnswer | None]:
        return [self._answer_one(q) for q in questions]
 
    # ── Internal ─────────────────────────────────────────────────────────────
 
    def _answer_one(self, question: str) -> GroundedAnswer | None:
        # Step 1: metadata pre-filter
        candidate_ids = self._retriever.filter_by_metadata(question)
 
        # Step 2: retrieve
        chunks = self._retriever.retrieve(
            query=question,
            candidate_doc_ids=candidate_ids,
        )
 
        if not chunks:
            return None
 
        # Step 3: confidence check
        best_score = chunks[0].score
        low_confidence = best_score < self._reader._params.low_confidence_threshold
 
        # Step 4: generate
        answer_text, doc_ids = self._reader.answer(
            question=question,
            chunks=chunks,
            low_confidence=low_confidence,
        )
 
        if answer_text is None:
            return None
 
        return GroundedAnswer(text=answer_text, doc_ids=doc_ids)
 
 
# ── Convenience factory matching the skeleton's expected pattern ──────────────
 
def build_system(
    chunk_params: Namespace,
    ret_params: Namespace,
    read_params: Namespace,
    corpus_path: str | Path,
    index_dir: str | Path = "index/",
    force_rebuild: bool = False,
) -> RAGQnASystem:
    """
    Build or load the QA system.
 
    Parameters
    ----------
    corpus_path : path to corpus.jsonl.gz
    index_dir   : directory to save/load the FAISS index
    force_rebuild : if True, always re-embed even if index exists
    """
    index_dir = Path(index_dir)
    index_exists = (index_dir / "faiss.index").exists()
 
    if index_exists and not force_rebuild:
        print(f"[build_system] Loading existing index from {index_dir}")
        return RAGQnASystem.load(ret_params, read_params, index_dir)
 
    print(f"[build_system] Building new index from {corpus_path} …")
    docs = load_corpus(corpus_path)
    return RAGQnASystem.from_corpus(chunk_params, ret_params, read_params, docs, index_dir=index_dir)
 