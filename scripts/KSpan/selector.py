# kspan/selector.py
"""
K-Span selection algorithm implementation.

Algorithm:
1. For each document:
   a. For each paragraph:
      i. Chunk paragraph into spans (Chunks(P, η, λ))
     ii. Embed each span (SP <- Embed(C))
    iii. Compute similarity between query embedding and span embeddings
     iv. Keep TopK similar spans per paragraph or per document (configurable)
2. Merge all candidate spans across docs and return global Top-K
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import logging

from kspan.typing import Span, Embedding
from kspan.chunker import chunks_from_paragraph
from kspan.embedder import BaseEmbedder, get_embedder

LOGGER = logging.getLogger(__name__)


class KSpanSelector:
    def __init__(self, embedder: BaseEmbedder, span_length_words: int = 120, span_overlap_sentences: int = 2, top_k: int = 10, batch_size: int = 32):
        self.embedder = embedder
        self.span_length_words = span_length_words
        self.span_overlap_sentences = span_overlap_sentences
        self.top_k = top_k
        self.batch_size = batch_size

    def _embed_texts_batched(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in batches to avoid OOM on large lists."""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_emb = self.embedder.embed(batch)
            embeddings.extend(batch_emb)
        return embeddings

    def select_from_doc(self, doc_id: str, paragraphs: List[str]) -> List[Span]:
        """
        Process a single document and return candidate spans with similarity scores.
        Returns a list of spans with added 'score' and meta fields.
        """
        query_embed = None  # set by top-level select which knows the query
        # We'll produce a candidate heap per doc
        doc_candidates: List[Tuple[float, Span]] = []

        # First, create all spans and keep reference mapping so we can embed in bulk
        spans: List[Span] = []
        for para_id, para in enumerate(paragraphs):
            para_spans = chunks_from_paragraph(para, self.span_length_words, self.span_overlap_sentences)
            for span_id, s in enumerate(para_spans):
                span_obj: Span = {
                    "doc_id": doc_id,
                    "para_id": para_id,
                    "span_id": span_id,
                    "text": s["text"],
                    "sent_start": s["sent_start"],
                    "sent_end": s["sent_end"],
                    "start_word": s["start_word"],
                    "end_word": s["end_word"],
                }
                spans.append(span_obj)

        return spans  # actual scoring is done in `select` where we have query embedding

    def select(self, query: str, docs: Dict[str, List[str]]) -> List[Dict]:
        """
        Top-level selection:
        - docs: dict mapping doc_id -> list_of_paragraphs
        - returns selected top-k spans (global across documents) sorted by score desc
        """
        # Step 1: embed query
        q_emb = self.embedder.embed([query])[0]
        # Step 2: collect spans and texts
        all_spans: List[Span] = []
        for doc_id, paras in docs.items():
            doc_spans = self.select_from_doc(doc_id, paras)
            all_spans.extend(doc_spans)

        if len(all_spans) == 0:
            return []

        texts = [s["text"] for s in all_spans]
        span_embs = self._embed_texts_batched(texts)

        # compute cosine similarities in a memory-efficient way
        # convert to numpy arrays
        q_vec = np.asarray(q_emb).reshape(1, -1)
        embs = np.asarray(span_embs)

        # If shapes mismatch (dummy embedder may have different dim) handle gracefully
        try:
            sims = cosine_similarity(embs, q_vec).reshape(-1)
        except Exception as e:
            LOGGER.warning("Cosine similarity failed: %s. Falling back to dot-product / normalization", e)
            # safe fallback
            embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
            q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            sims = (embs_norm @ q_norm.T).reshape(-1)

        # Collect top-K indices
        topk_idx = np.argsort(-sims)[: self.top_k]
        topk_spans = []
        for idx in topk_idx:
            s = all_spans[idx].copy()
            s["score"] = float(sims[idx])
            topk_spans.append(s)

        # Sort final spans by score descending
        topk_spans = sorted(topk_spans, key=lambda x: x["score"], reverse=True)
        return topk_spans
