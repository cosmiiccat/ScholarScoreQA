# utils/text_utils.py

import re

def clean(text: str):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
