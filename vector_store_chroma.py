from __future__ import annotations

from typing import List, Optional

import numpy as np

from chunking import Chunk


class ChromaVectorStore:
    def __init__(self, collection_name: str = "documents", persist_dir: Optional[str] = None):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("Run: pip install chromadb") from e

        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.EphemeralClient()  # in-memory, non-persistent

        # cosine similarity space; we already pass L2-normalized embeddings
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc_id": c.doc_id, "doc_title": c.doc_title, "chunk_index": c.chunk_index, **c.metadata}
                for c in chunks
            ],
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5, where: Optional[dict] = None) -> List[dict]:
        res = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,  # optional metadata filter, e.g. {"doc_id": "policy_2024"}
        )
        results = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]  # cosine distance = 1 - cosine similarity
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                {
                    "chunk_id": cid,
                    "doc_id": meta.get("doc_id"),
                    "doc_title": meta.get("doc_title"),
                    "text": text,
                    "score": 1.0 - dist,  # convert back to similarity, higher = more relevant
                    "metadata": meta,
                }
            )
        return results

    def count(self) -> int:
        return self.collection.count()
