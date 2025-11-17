# context_reducer/cross_reranker.py

"""
CrossEncoder Reranking
----------------------

Uses SentenceTransformers CrossEncoder for semantic reranking.
"""

from typing import List, Tuple
from sentence_transformers import CrossEncoder


class CrossReranker:
    """Dense reranker using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, spans: List[str]) -> List[float]:
        """
        Compute semantic scores using CrossEncoder.

        Args:
            query (str): The question/query.
            spans (List[str]): Text spans.

        Returns:
            List[float]: Semantic similarity scores.
        """
        pairs: List[Tuple[str, str]] = [(query, span) for span in spans]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]
