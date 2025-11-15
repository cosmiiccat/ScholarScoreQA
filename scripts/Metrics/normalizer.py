# evaluation/normalizer.py

import re
import string

class TextNormalizer:
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text by:
        - lowercasing
        - removing articles
        - removing punctuation
        - trimming whitespace
        """
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        return " ".join(text.split())
