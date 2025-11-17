# context_reducer/span_splitter.py

"""
Span Splitter Module
--------------------

Handles segmentation of long text into spans (sentences, paragraphs, etc.)
"""

from typing import List
import nltk

# Ensure NLTK tokenizer available
nltk.download("punkt", quiet=True)


class SpanSplitter:
    """Utility for splitting long text into spans."""

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into cleaned sentence spans.
        """
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
