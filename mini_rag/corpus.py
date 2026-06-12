"""
corpus.py — Document model and corpus loading utilities.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# Stopwords excluded when inferring suffix from text
_SUFFIX_STOPWORDS = {"SDK", "API", "SLA", "FAQ", "UI", "ID", "QA", "IP", "THE", "FOR"}


def _infer_suffix_from_text(prefix: str | None, text: str) -> str | None:
    """
    Extract the product suffix identifier from the document text header.

    Many synthetic docs have no product_suffix in metadata but encode it
    in the document title, e.g. "# Qoria SGI v1.2.17". This function
    extracts "SGI" so the metadata filter can match it.

    Strategy:
      1. If prefix is known, look for "Prefix SUFFIX" pattern in the first
         300 chars (alphanumeric token of 2–5 chars, including pure numeric
         like "790").
      2. Fallback: scan the first heading line for a short uppercase identifier.
    """
    header = text[:300]

    if prefix:
        # Match prefix followed by an alphanumeric suffix token (e.g. "Qoria SGI", "Xylia 790")
        pattern = rf"{re.escape(prefix)}\s+([A-Z0-9][A-Z0-9]{{1,4}})\b"
        m = re.search(pattern, header, re.IGNORECASE)
        if m:
            candidate = m.group(1).upper()
            if candidate not in _SUFFIX_STOPWORDS:
                return candidate

    # Fallback: first heading line (e.g. "# Helios D7P v1.0.5")
    first_line = header.split("\n")[0]
    tokens = re.findall(r"\b([A-Z][A-Z0-9]{1,4})\b", first_line)
    for tok in tokens:
        if tok not in _SUFFIX_STOPWORDS and len(tok) >= 2:
            return tok.upper()

    return None


@dataclass
class Document:
    """A single document from the corpus."""

    id: str                           # always stored as str for uniform handling
    text: str
    product_prefix: str | None = None
    product_suffix: str | None = None
    product_version: str | None = None

    # Inferred from text at load time when product_suffix is absent
    inferred_suffix: str | None = field(default=None, repr=False)

    # Derived at index time
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
    def effective_suffix(self) -> str | None:
        """Returns product_suffix if set, otherwise the text-inferred suffix."""
        return self.product_suffix or self.inferred_suffix

    @property
    def product_name(self) -> str | None:
        """Best-effort product display name from metadata."""
        parts = [self.product_prefix, self.effective_suffix, self.product_version]
        filtered = [p for p in parts if p]
        return " ".join(filtered) if filtered else None


def load_corpus(path: str | Path) -> list[Document]:
    """Load corpus.jsonl.gz and return a list of Document objects."""
    docs: list[Document] = []
    with gzip.open(path) as f:
        for line in f:
            raw = json.loads(line)
            prefix = raw.get("product_prefix")
            suffix = raw.get("product_suffix")
            text   = raw["text"]
            docs.append(Document(
                id=str(raw["id"]),
                text=text,
                product_prefix=prefix,
                product_suffix=suffix,
                product_version=raw.get("product_version"),
                # Infer suffix from text only when metadata field is absent
                inferred_suffix=_infer_suffix_from_text(prefix, text) if not suffix else None,
            ))
    return docs


def load_queries(path: str | Path) -> list[dict]:
    """Load a queries jsonl.gz file (train / valid / bonus)."""
    rows: list[dict] = []
    with gzip.open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows
