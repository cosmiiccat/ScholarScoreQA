# K-Span Select (research-grade implementation)

This repository contains a modular implementation of the K-Span Select algorithm described in the paper:
"K-Span Select and Multi-Dimensional Judging for Reliable Scholarly Question Answering".

Features:
- Token/sentence-aware chunking with overlap (η, λ)
- Pluggable embedding backends (dummy, sentence-transformer)
- Top-K global span selection using cosine similarity
- CP / CCP metric computation utilities
- CLI script `scripts/run_kspan.py`

## Quickstart

1. Install dependencies (optional):
```bash
pip install sentence-transformers scikit-learn transformers nltk
