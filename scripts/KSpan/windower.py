# kspan/chunker.py
"""
Chunking module:
- sentence_split_and_chunk: split paragraphs into windows of words with overlap in sentences
- token-based chunker is optionally provided by users via tokenizer in utils
"""

from typing import List, Tuple, Iterable
import logging
import math
import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

LOGGER = logging.getLogger(__name__)


def sentence_split(paragraph: str) -> List[str]:
    """Split paragraph into sentences using NLTK."""
    return sent_tokenize(paragraph)


def chunks_from_paragraph(paragraph: str, span_length_words: int, span_overlap_sentences: int) -> List[dict]:
    """
    Create overlapping spans from a paragraph.

    - span_length_words: approximate number of words per span (η)
    - span_overlap_sentences: number of sentences overlap between consecutive spans (λ)
    Returns list of dicts: { "text": ..., "start_word": i, "end_word": j, "sent_start": s_idx, "sent_end": e_idx }
    """
    sents = sentence_split(paragraph)
    if not sents:
        return []

    # Precompute tokenized sentences and their word counts
    sent_tokens = [word_tokenize(s) for s in sents]
    sent_word_counts = [len(toks) for toks in sent_tokens]

    spans = []
    n_sents = len(sents)
    sent_idx = 0

    # Build spans by moving a sliding window of sentences such that the total words in the window ≈ span_length_words
    while sent_idx < n_sents:
        # expand window until word count >= span_length_words or reached end
        end_idx = sent_idx
        curr_words = 0
        while end_idx < n_sents and curr_words < span_length_words:
            curr_words += sent_word_counts[end_idx]
            end_idx += 1

        # Compose text from sentences [sent_idx, end_idx)
        text = " ".join(sents[sent_idx:end_idx])
        start_word = sum(sent_word_counts[:sent_idx])
        end_word = start_word + sum(sent_word_counts[sent_idx:end_idx])

        spans.append({
            "text": text,
            "sent_start": sent_idx,
            "sent_end": end_idx - 1,
            "start_word": start_word,
            "end_word": end_word
        })

        # Advance window by (window_size_in_sentences - overlap_sentences) but ensure at least 1
        step = max(1, (end_idx - sent_idx) - span_overlap_sentences)
        sent_idx += step

    LOGGER.debug("Paragraph chunked into %d spans", len(spans))
    return spans
