
"""
corpus.py — Document model and corpus loading utilities.
"""
 
from __future__ import annotations
 
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
 
 
@dataclass
class Document:
    """A single document from the corpus."""
 
    id: str                          # always stored as str for uniform handling
    text: str
    product_prefix: str | None = None
    product_suffix: str | None = None
    product_version: str | None = None
 
    # derived at load time
    chunks: list[str] = field(default_factory=list, repr=False)
 
    @property
    def is_synthetic(self) -> bool:
        """True for synthetic product docs (integer-origin IDs)."""
        try:
            int(self.id)
            return True
        except ValueError:
            return False
 
    @property
    def product_name(self) -> str | None:
        """Best-effort product display name from metadata."""
        parts = [self.product_prefix, self.product_suffix, self.product_version]
        filtered = [p for p in parts if p]
        return " ".join(filtered) if filtered else None
 
 
def load_corpus(path: str | Path) -> list[Document]:
    """Load corpus.jsonl.gz and return a list of Document objects."""
    docs: list[Document] = []
    with gzip.open(path) as f:
        for line in f:
            raw = json.loads(line)
            docs.append(Document(
                id=str(raw["id"]),
                text=raw["text"],
                product_prefix=raw.get("product_prefix"),
                product_suffix=raw.get("product_suffix"),
                product_version=raw.get("product_version"),
            ))
    return docs
 
 
def load_queries(path: str | Path) -> list[dict]:
    """Load a queries jsonl.gz file (train / valid / bonus)."""
    rows: list[dict] = []
    with gzip.open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows
 