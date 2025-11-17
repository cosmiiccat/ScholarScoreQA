# context_reducer/bm25_retriever.py

"""
BM25 Retriever
--------------

Implements sparse retrieval using BM25 Okapi.
"""

from typing import List
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Sparse retriever using BM25."""

    @staticmethod
    def compute_bm25(spans: List[str], query: str) -> List[float]:
        """
        Compute BM25 scores for each span.

        Args:
            spans (List[str]): List of text spans.
            query (str): Query/question.

        Returns:
            List[float]: BM25 scores.
        """
        tokenized = [s.split() for s in spans]
        bm25 = BM25Okapi(tokenized)
        return bm25.get_scores(query.split())
