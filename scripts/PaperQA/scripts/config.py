# paperqa_core/config.py

import os

class PaperQAConfig:
    """
    Configuration for PaperQA system.
    """

    # You can load this from environment variables too.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    DEFAULT_QUESTION = "What is the seed lexicon?"
