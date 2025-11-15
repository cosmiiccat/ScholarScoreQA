#!/usr/bin/env python3
"""
CLI to run K-Span Select on a set of documents.

Example:
    python scripts/run_kspan.py --doc paths.txt --query "What is seed lexicon?" --out results.json
Where paths.txt contains one document path per line (plain text)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from kspan.config import KSpanConfig
from kspan.embedder import get_embedder
from kspan.selector import KSpanSelector
from kspan.utils import read_document_paragraphs, spans_to_context, dump_spans_json
from kspan.metrics import SpanMetrics
from transformers import T5Tokenizer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("kspan_cli")


def load_docs_from_list(file_with_paths: str) -> Dict[str, List[str]]:
    docs: Dict[str, List[str]] = {}
    with open(file_with_paths, "r", encoding="utf-8") as fh:
        for line in fh:
            path = line.strip()
            if not path:
                continue
            doc_id = Path(path).stem
            paragraphs = read_document_paragraphs(path)
            docs[doc_id] = paragraphs
    return docs


def main():
    parser = argparse.ArgumentParser(description="K-Span Select runner")
    parser.add_argument("--docs", required=True, help="File with newline-separated paths to plaintext docs")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--out", required=True, help="Output JSON file for selected spans")
    parser.add_argument("--backend", default="dummy", help="Embedder backend: dummy | sentence-transformer")
    parser.add_argument("--topk", type=int, default=10, help="Top-K spans to output")
    parser.add_argument("--span_len", type=int, default=120, help="Span length in words (η)")
    parser.add_argument("--overlap", type=int, default=2, help="Span overlap in sentences (λ)")
    args = parser.parse_args()

    cfg = KSpanConfig(span_length_words=args.span_len, span_overlap_sentences=args.overlap, top_k=args.topk, embedder_backend=args.backend)
    LOGGER.info("Config: %s", cfg)

    # Load embedder
    embedder = get_embedder(cfg.embedder_backend)  # for SBERT pass model_name param via kwargs if needed
    selector = KSpanSelector(embedder, span_length_words=cfg.span_length_words, span_overlap_sentences=cfg.span_overlap_sentences, top_k=cfg.top_k)

    docs = load_docs_from_list(args.docs)
    LOGGER.info("Loaded %d docs", len(docs))

    top_spans = selector.select(args.query, docs)
    LOGGER.info("Selected %d spans", len(top_spans))

    dump_spans_json(top_spans, args.out)
    LOGGER.info("Saved spans to %s", args.out)

    # Compute CP / CCP example (requires a gold_evidence sample); here we just show usage:
    # If you have gold evidence and original doc text:
    # metrics = SpanMetrics(tokenizer=T5Tokenizer.from_pretrained('t5-base'))
    # cp = metrics.containment_percentage(gold_evidence, spans_to_context(top_spans))
    # ccp = metrics.context_compression_percentage(full_doc_text, spans_to_context(top_spans))
    # LOGGER.info("CP: %.4f | CCP: %.4f", cp, ccp)


if __name__ == "__main__":
    main()
