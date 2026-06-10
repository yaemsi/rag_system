"""
index_store.py — Persist and reload the Retriever index to/from disk.
 
Saves:
  - FAISS index binary (faiss)
  - chunk texts + doc_id mapping (numpy/json)
  - BM25 tokenized corpus (pickle)
  - Document metadata (json)
"""
 
from __future__ import annotations
from argparse import Namespace
 
import json
import pickle
from pathlib import Path
 
import faiss
 
from mini_rag.corpus import Document
from mini_rag.retriever import Retriever
 
 
def save_index(retriever: Retriever, index_dir: str | Path) -> None:
    """Persist a built Retriever to disk."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
 
    # FAISS index
    faiss.write_index(retriever._faiss_index, str(index_dir / "faiss.index"))
 
    # Chunks + doc_id mapping
    with open(index_dir / "chunks.json", "w") as f:
        json.dump({
            "chunks": retriever._chunks,
            "chunk_doc_ids": retriever._chunk_doc_ids,
        }, f)
 
    # BM25 tokenized corpus (pickle)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(retriever._bm25, f)
 
    # Document metadata
    docs_data = {
        doc_id: {
            "id": doc.id,
            "product_prefix": doc.product_prefix,
            "product_suffix": doc.product_suffix,
            "product_version": doc.product_version,
        }
        for doc_id, doc in retriever._docs.items()
    }
    with open(index_dir / "docs_meta.json", "w") as f:
        json.dump(docs_data, f)
 
    print(f"[IndexStore] Saved index to {index_dir}")
 
 
def load_index(params: Namespace, index_dir: str | Path) -> Retriever:
    """Load a persisted Retriever from disk."""
    index_dir = Path(index_dir)
 
    retriever = Retriever(params)
 
    # FAISS index
    retriever._faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
 
    # Chunks + doc_id mapping
    with open(index_dir / "chunks.json") as f:
        data = json.load(f)
    retriever._chunks = data["chunks"]
    retriever._chunk_doc_ids = data["chunk_doc_ids"]
 
    # BM25
    with open(index_dir / "bm25.pkl", "rb") as f:
        retriever._bm25 = pickle.load(f)
 
    # Document metadata (lightweight — no text stored)
    with open(index_dir / "docs_meta.json") as f:
        docs_data = json.load(f)
    retriever._docs = {
        doc_id: Document(
            id=meta["id"],
            text="",  # text not stored in index; not needed post-build
            product_prefix=meta["product_prefix"],
            product_suffix=meta["product_suffix"],
            product_version=meta["product_version"],
        )
        for doc_id, meta in docs_data.items()
    }
 
    print(f"[IndexStore] Loaded index from {index_dir} "
          f"({len(retriever._chunks)} chunks, {len(retriever._docs)} docs)")
    return retriever
 