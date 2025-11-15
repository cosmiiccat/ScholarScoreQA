# paperqa_core/utils.py

import os

def set_openai_key(key: str):
    """
    Set the OpenAI API key for PaperQA.
    """
    os.environ["OPENAI_API_KEY"] = key
