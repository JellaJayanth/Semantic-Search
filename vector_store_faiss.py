from __future__ import annotations

import json
import os
import pickle
from typing import List, Optional

import numpy as np

from chunking import Chunk


class FaissVectorStore:
    def __init__(self, dim: int):
        try:
            import faiss
        except ImportError as e:
            raise ImportError("Run: pip install faiss-cpu") from e
        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # exact cosine sim via normalized dot product
        self.chunks: List[Chunk] = []  # position i corresponds to FAISS vector id i

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[dict]:
        query_vector = query_vector.astype("float32").reshape(1, -1)
        scores, ids = self.index.search(query_vector, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "text": chunk.text,
                    "score": float(score),  # cosine similarity, higher = more relevant
                    "metadata": chunk.metadata,
                }
            )
        return results

    def count(self) -> int:
        return self.index.ntotal

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self._faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(dir_path, "meta.json"), "w") as f:
            json.dump({"dim": self.dim}, f)

    @classmethod
    def load(cls, dir_path: str) -> "FaissVectorStore":
        import faiss

        with open(os.path.join(dir_path, "meta.json")) as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "chunks.pkl"), "rb") as f:
            store.chunks = pickle.load(f)
        return store
