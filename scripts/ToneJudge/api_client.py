# tone_judge/api_client.py

"""
Wrapper for OpenAI API (or any LLM API).
"""

import os
from typing import List, Dict
from openai import OpenAI


class APIClient:
    """Lightweight wrapper around OpenAI API."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.client = OpenAI(api_key=api_key)

    def chat(self, model: str, messages: List[Dict], **kwargs):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
