from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    """Common interface every embedding backend must implement."""

    dim: int

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalized embeddings."""
        raise NotImplementedError

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        """L2-normalize rows so cosine similarity == dot product."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vecs / norms


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Recommended default backend for real deployments.

    Runs locally (no API key, no per-call cost), produces strong semantic
    embeddings. Requires: pip install sentence-transformers

    Good general-purpose models:
      - "all-MiniLM-L6-v2"      (384-dim, fast, great default)
      - "all-mpnet-base-v2"     (768-dim, higher quality, slower)
      - "BAAI/bge-small-en-v1.5" (384-dim, strong retrieval performance)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run:\n"
                "    pip install sentence-transformers\n"
            ) from e
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # already L2-normalized
        )
        return vecs.astype("float32")


class OpenAIEmbedder(BaseEmbedder):
    """
    Calls OpenAI's embeddings API. Requires: pip install openai, an API key,
    and network access. Useful if you'd rather not run a local model.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Run: pip install openai") from e
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.dim = 1536 if "small" in model_name else 3072

    def encode(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype="float32")
        return self._normalize(vecs)


class TfidfEmbedder(BaseEmbedder):
    """
    Dependency-free fallback backend (pure Python + numpy).

    Not a deep-learning semantic embedder -- it's a TF-IDF bag-of-words
    vectorizer -- but it implements the exact same interface, so the rest of
    the pipeline (chunking, vector store, retrieval) is fully exercised and
    testable without internet access or heavyweight ML dependencies. Swap in
    SentenceTransformerEmbedder for real semantic search in production.
    """

    _token_re = re.compile(r"[A-Za-z0-9]+")

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in self._token_re.findall(text)]

    def fit(self, corpus: List[str]) -> None:
        """Build vocabulary + IDF table from a corpus (call once before encode)."""
        doc_freq: Counter = Counter()
        for doc in corpus:
            for tok in set(self._tokenize(doc)):
                doc_freq[tok] += 1

        # Keep the most common `dim` tokens as the vocabulary (hashing-free, deterministic).
        most_common = [tok for tok, _ in doc_freq.most_common(self.dim)]
        self.vocab = {tok: i for i, tok in enumerate(most_common)}

        n_docs = max(len(corpus), 1)
        idf = np.ones(self.dim, dtype="float32")
        for tok, idx in self.vocab.items():
            idf[idx] = math.log((n_docs + 1) / (doc_freq[tok] + 1)) + 1.0
        self.idf = idf
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            # Auto-fit on first use if the caller never called fit() explicitly.
            self.fit(texts)

        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue
            counts = Counter(tokens)
            total = len(tokens)
            for tok, cnt in counts.items():
                idx = self.vocab.get(tok)
                if idx is not None:
                    tf = cnt / total
                    vecs[row, idx] = tf * self.idf[idx]
        return self._normalize(vecs)


def get_embedder(backend: str = "auto", **kwargs) -> BaseEmbedder:
    """
    Factory function. backend in {"auto", "sentence-transformers", "openai", "tfidf"}.
    "auto" tries sentence-transformers first and falls back to tfidf if the
    package (or its model download) isn't available -- handy for demos that
    must run both online and offline.
    """
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(**kwargs)
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    if backend == "tfidf":
        return TfidfEmbedder(**kwargs)
    if backend == "auto":
        try:
            return SentenceTransformerEmbedder(**kwargs)
        except Exception:
            return TfidfEmbedder(**{k: v for k, v in kwargs.items() if k == "dim"})
    raise ValueError(f"Unknown backend: {backend}")
