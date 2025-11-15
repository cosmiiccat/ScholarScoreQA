# kspan/utils.py
from typing import List, Dict
import json
import logging

LOGGER = logging.getLogger(__name__)


def read_document_paragraphs(path: str) -> List[str]:
    """
    Simple utility: read a plaintext document and split by blank lines into paragraphs.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    paragraphs = [p.strip().replace("\n", " ") for p in text.split("\n\n") if p.strip()]
    LOGGER.info("Loaded document '%s' with %d paragraphs", path, len(paragraphs))
    return paragraphs


def spans_to_context(spans: List[Dict]) -> str:
    """
    Merge selected spans into a single condensed context string
    (preserves ordering by doc->para->span where possible).
    """
    # sort by doc_id, para_id, span_id for reproducible order
    sorted_spans = sorted(spans, key=lambda s: (s.get("doc_id", ""), s.get("para_id", -1), s.get("span_id", -1)))
    texts = [s["text"] for s in sorted_spans]
    return "\n\n".join(texts)


def dump_spans_json(spans, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spans, fh, ensure_ascii=False, indent=2)
