from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List
import uuid


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    doc_title: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def split_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no external NLP dependency)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital/digit.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    doc_title: str = "",
    chunk_size: int = 400,
    overlap: int = 60,
    metadata: dict | None = None,
) -> List[Chunk]:
    """
    Chunk by sentence, packing sentences into ~chunk_size-word windows with a
    sliding overlap of ~overlap words between consecutive chunks.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current_words: List[str] = []
    idx = 0

    def flush(words: List[str]):
        nonlocal idx
        if not words:
            return
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=" ".join(words),
                doc_id=doc_id,
                doc_title=doc_title,
                chunk_index=idx,
                metadata=dict(metadata or {}),
            )
        )
        idx += 1

    for sentence in sentences:
        words = sentence.split()
        if len(current_words) + len(words) > chunk_size and current_words:
            flush(current_words)
            # carry the tail of the previous chunk forward as overlap
            current_words = current_words[-overlap:] if overlap > 0 else []
        current_words.extend(words)

    flush(current_words)
    return chunks


def chunk_documents(
    documents: List[dict],
    chunk_size: int = 400,
    overlap: int = 60,
) -> List[Chunk]:
    """
    documents: list of {"id": str, "title": str, "text": str, "metadata": dict}
    Returns a flat list of Chunk objects across all documents.
    """
    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_text(
                text=doc["text"],
                doc_id=doc["id"],
                doc_title=doc.get("title", doc["id"]),
                chunk_size=chunk_size,
                overlap=overlap,
                metadata=doc.get("metadata", {}),
            )
        )
    return all_chunks
