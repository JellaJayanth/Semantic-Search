from __future__ import annotations

from typing import List

import numpy as np

from chunking import Chunk


class NumpyVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype="float32")
        self.chunks: List[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        vectors = vectors.astype("float32")
        self.vectors = np.vstack([self.vectors, vectors]) if self.vectors.size else vectors
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[dict]:
        if len(self.chunks) == 0:
            return []
        query_vector = query_vector.astype("float32")
        # cosine similarity == dot product since both sides are L2-normalized
        scores = self.vectors @ query_vector
        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]  # sort just the top-k, not all N

        results = []
        for idx in top_idx:
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "text": chunk.text,
                    "score": float(scores[idx]),
                    "metadata": chunk.metadata,
                }
            )
        return results

    def count(self) -> int:
        return len(self.chunks)
