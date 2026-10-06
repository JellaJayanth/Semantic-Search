from __future__ import annotations

from typing import List, Optional

from chunking import Chunk, chunk_documents
from embedder import get_embedder


class DocumentSearchEngine:
    def __init__(
        self,
        embedder_backend: str = "auto",
        vector_backend: str = "faiss",
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        embedder_kwargs: Optional[dict] = None,
        vector_store_kwargs: Optional[dict] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_backend = vector_backend

        self.embedder = get_embedder(embedder_backend, **(embedder_kwargs or {}))

        self._vector_store_kwargs = vector_store_kwargs or {}
        self.store = None  # created lazily once we know the embedding dim (tfidf fits on first batch)

    def _ensure_store(self):
        if self.store is not None:
            return
        dim = self.embedder.dim
        if self.vector_backend == "faiss":
            from vector_store_faiss import FaissVectorStore

            self.store = FaissVectorStore(dim=dim)
        elif self.vector_backend == "chroma":
            from vector_store_chroma import ChromaVectorStore

            self.store = ChromaVectorStore(**self._vector_store_kwargs)
        elif self.vector_backend == "numpy":
            from vector_store_numpy import NumpyVectorStore

            self.store = NumpyVectorStore(dim=dim)
        else:
            raise ValueError(f"Unknown vector_backend: {self.vector_backend}")

    def ingest(self, documents: List[dict]) -> int:
        """
        documents: [{"id": str, "title": str, "text": str, "metadata": dict?}, ...]
        Returns the number of chunks indexed.
        """
        chunks: List[Chunk] = chunk_documents(
            documents, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )
        if not chunks:
            return 0

        texts = [c.text for c in chunks]

        # TF-IDF fallback needs a fit pass over the corpus before its dim is meaningful.
        if hasattr(self.embedder, "fit") and not getattr(self.embedder, "_fitted", True):
            self.embedder.fit(texts)

        vectors = self.embedder.encode(texts)
        self._ensure_store()
        self.store.add(vectors, chunks)
        return len(chunks)

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[dict]:
        if self.store is None or self.store.count() == 0:
            return []
        query_vector = self.embedder.encode([query])[0]
        return self.store.search(query_vector, top_k=top_k, **kwargs)

    def save(self, dir_path: str) -> None:
        if self.vector_backend != "faiss":
            raise NotImplementedError("save()/load() helpers shown for FAISS; Chroma persists via persist_dir.")
        self.store.save(dir_path)

    def load(self, dir_path: str) -> None:
        from vector_store_faiss import FaissVectorStore

        self.store = FaissVectorStore.load(dir_path)
