# kspan/config.py
from dataclasses import dataclass

@dataclass
class KSpanConfig:
    # chunking
    span_length_words: int = 120    # η (optimal ~120 as in the paper)
    span_overlap_sentences: int = 2 # λ (1-3 sentences recommended)

    # retrieval
    top_k: int = 10                 # K (paper recommends 10)
    embedding_batch_size: int = 32

    # embedding backend: 'dummy' (random), 'sentence-transformer', 'instructor', 'jina'
    embedder_backend: str = "dummy"

    # tokenizer settings (if needed for CP/CCP)
    tokenizer_name: str = "t5-base"
