# context_reducer/reducer.py

"""
Hybrid Context Reduction
------------------------

Combines:
- Sentence splitting
- BM25 sparse retrieval
- CrossEncoder semantic reranking
- Top-K span selection
"""

from typing import List, Dict

from .span_splitter import SpanSplitter
from .retriever import BM25Retriever
from .cross_reranker import CrossReranker
from .utils import sort_by_key


class ContextReducer:
    """
    Orchestration class for hybrid retrieval + reranking context reduction.
    """

    def __init__(
        self,
        bm25_top_n: int = 50,
        final_k: int = 10,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Args:
            bm25_top_n (int): Number of spans from BM25 to rerank.
            final_k (int): Number of final spans to keep.
            cross_encoder_model (str): CrossEncoder model name.
        """
        self.bm25_top_n = bm25_top_n
        self.final_k = final_k

        self.splitter = SpanSplitter()
        self.reranker = CrossReranker(cross_encoder_model)

    # ----------------------------------------------
    def reduce(self, full_text: str, question: str) -> str:
        """
        Reduce long text into a question-relevant condensed context.

        Args:
            full_text (str): Long text/document.
            question (str): The query.

        Returns:
            str: Condensed relevant spans.
        """

        # ---- Step 1: Sentence Splitting ----
        spans = self.splitter.split_sentences(full_text)
        if not spans:
            return ""

        # ---- Step 2: BM25 Retrieval ----
        bm25_scores = BM25Retriever.compute_bm25(spans, question)

        bm25_candidates: List[Dict] = [
            {"text": spans[i], "bm25_score": float(bm25_scores[i])}
            for i in range(len(spans))
        ]

        # Take top-N by BM25
        top_bm25 = sort_by_key(bm25_candidates, "bm25_score")[: self.bm25_top_n]

        # ---- Step 3: CrossEncoder Reranking ----
        cross_scores = self.reranker.rerank(
            question, [c["text"] for c in top_bm25]
        )

        for i, score in enumerate(cross_scores):
            top_bm25[i]["cross_score"] = score

        # ---- Step 4: Select top-K spans ----
        final_spans = sort_by_key(top_bm25, "cross_score")[: self.final_k]
        merged = " ".join([c["text"] for c in final_spans])

        return merged
