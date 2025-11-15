# metrics/preprocessing.py

import re
import string

class TextPreprocessor:
    """
    Shared normalization utilities for token-based metrics.
    """

    @staticmethod
    def normalize(text: str, remove_punct=True, lower=True) -> str:
        if lower:
            text = text.lower()
        if remove_punct:
            text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text
