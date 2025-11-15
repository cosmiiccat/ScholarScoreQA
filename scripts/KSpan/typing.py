# kspan/typing.py
from typing import List, Tuple, Dict

Span = Dict[str, object]   # { "doc_id": str, "para_id": int, "span_id": int, "text": str, "start_word": int, "end_word": int }
Embedding = List[float]
