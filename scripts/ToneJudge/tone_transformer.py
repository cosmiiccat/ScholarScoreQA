# tone_judge/tone_transformer.py

"""
Tone Transformer - rewrites an answer in a different tone using LLM.
"""

from typing import Dict
from .tone_prompt_builder import TonePromptBuilder
from .api_client import APIClient
from .config import OPENAI_MODEL, VALID_TONES


class ToneTransformer:
    def __init__(self):
        self.client = APIClient()

    def transform(self, answer: str, tone: str) -> str:
        """Transform the answer into a given tone."""
        prompt = TonePromptBuilder.build_prompt(answer, tone)

        resp = self.client.chat(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You rewrite text in specific tones."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        return resp.choices[0].message.content.strip()

    def transform_all(self, answer: str) -> Dict[str, str]:
        return {tone: self.transform(answer, tone) for tone in VALID_TONES}
