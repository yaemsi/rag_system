
"""
chunker.py — Split documents into overlapping text chunks.
 
Strategy:
  - Synthetic product docs (5–7k chars, templated sections):
    split on markdown headings (## ...) to preserve semantic units.
  - Coveo docs and other prose:
    fixed-size overlapping windows (default 800 chars, 150 overlap).
  - Docs under the chunk size threshold are returned as-is.
"""
from __future__ import annotations

from argparse import Namespace
import re
 
from mini_rag.corpus import Document
 
# Coveo docs: window size / overlap in characters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
 
# Docs shorter than this are never split
MIN_SPLIT_LEN = CHUNK_SIZE + 50
 
 
def _split_by_headings(params: Namespace, text: str) -> list[str]:
    """Split on markdown ## headings; merge very short sections with the next."""
    chunk_size = params._get_args("chunk_size", default=CHUNK_SIZE)
    parts = re.split(r"(?m)^(?=##\s)", text)
    chunks: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # if a section is still huge, further split it with sliding window
        if len(part) > chunk_size * 2:
            chunks.extend(_sliding_window(params, part))
        else:
            chunks.append(part)
    return chunks if chunks else [text]
 
 
def _sliding_window(params: Namespace, text: str) -> list[str]:
    """Character-level sliding window chunker."""
    
    size = params._get_args("chunk_size", default=CHUNK_SIZE)
    overlap = params._get_args("chunk_overlap", default=CHUNK_OVERLAP)

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks
 
 
def chunk_document(params: Namespace, doc: Document) -> list[str]:
    """
    Return a list of text chunks for a document.
    Short documents are returned as a single chunk.
    """
    min_split_len = params._get_args("min_split_len", default=MIN_SPLIT_LEN)
    text = doc.text.strip()
 
    if len(text) <= min_split_len:
        return [text]
 
    if doc.is_synthetic:
        # Structured product docs: split on section headings
        return _split_by_headings(params, text)
 
    # Sliding window
    return _sliding_window(params, text)
 
 
def attach_chunks(params: Namespace, docs: list[Document]) -> None:
    """Attach chunks in-place to each document in the list."""
    for doc in docs:
        doc.chunks = chunk_document(params, doc)
