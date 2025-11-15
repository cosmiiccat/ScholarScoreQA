# kspan/embedder.py
"""
Embedding interface with a dummy and a sentence-transformer implementation.
Replace or extend this class with SciBERT / Instructor / Jina backends as needed.
"""

from typing import List, Sequence
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # optional dependency


class BaseEmbedder:
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class DummyEmbedder(BaseEmbedder):
    """Deterministic random embeddings for quick testing."""
    def __init__(self, dim: int = 768, seed: int = 42):
        self.dim = dim
        np.random.seed(seed)
        self._rng = np.random.RandomState(seed)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # deterministic pseudo-random vectors (not meaningful for real retrieval)
        return [self._rng.randn(self.dim).tolist() for _ in texts]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper around sentence-transformers."""
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. pip install sentence-transformers")
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        embs = self.model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        return embs.tolist()


def get_embedder(backend: str = "dummy", **kwargs) -> BaseEmbedder:
    backend = backend.lower()
    if backend == "dummy":
        return DummyEmbedder(**kwargs)
    if backend in ("sbert", "sentence-transformer", "sentence_transformer", "sentence-transformers"):
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder backend: {backend}")
