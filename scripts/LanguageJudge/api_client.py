# language_judge/api_client.py

import os
from openai import OpenAI


class APIClient:
    """Wrapper for OpenAI API."""

    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.client = OpenAI(api_key=key)

    def chat(self, model: str, messages, **kwargs):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
