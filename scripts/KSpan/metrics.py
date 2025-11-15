# kspan/metrics.py
"""
Containment Percentage (CP) and Context Compression Percentage (CCP)
Tokenization uses a tokenizer object if provided; otherwise whitespace tokenization.
"""

from typing import List
from collections import Counter
import logging

LOGGER = logging.getLogger(__name__)


def _tokenize_with_tokenizer(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _whitespace_tokenize(text: str) -> List[str]:
    return text.split()


class SpanMetrics:
    def __init__(self, tokenizer=None):
        """
        tokenizer: optional HuggingFace tokenizer
        """
        self.tokenizer = tokenizer

    def _tokens(self, text: str):
        if self.tokenizer:
            return _tokenize_with_tokenizer(self.tokenizer, text)
        return _whitespace_tokenize(text)

    def containment_percentage(self, gold_evidence: str, selected_spans: str) -> float:
        """
        CP(E, C) = |E ∩ C| / |E| * 100
        uses multiset intersection (counts)
        """
        E = self._tokens(gold_evidence)
        C = self._tokens(selected_spans)

        if len(E) == 0:
            LOGGER.warning("Gold evidence empty; returning CP = 0.0")
            return 0.0

        if isinstance(E[0], int):  # token IDs
            inter = sum((Counter(E) & Counter(C)).values())
            denom = len(E)
        else:
            inter = sum((Counter(E) & Counter(C)).values())
            denom = len(E)

        cp = (inter / denom) * 100.0
        return round(cp, 4)

    def context_compression_percentage(self, full_doc: str, selected_spans: str) -> float:
        """
        CCP(D, C) = (1 - |C| / |D|) * 100
        """
        D = self._tokens(full_doc)
        C = self._tokens(selected_spans)

        if len(D) == 0:
            LOGGER.warning("Full document empty; returning CCP = 0.0")
            return 0.0

        ccp = (1 - (len(C) / len(D))) * 100.0
        return round(ccp, 4)
