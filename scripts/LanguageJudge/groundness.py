# language_judge/scorers/groundedness_scorer.py

from .api_client import APIClient
from .config import OPENAI_MODEL


class GroundednessScorer:
    """Ensures answer is grounded in the provided context only."""

    def __init__(self):
        self.client = APIClient()

    def score(self, context: str, question: str, answer: str) -> float:
        prompt = f"""
Score groundedness (0-1): Answer must rely ONLY on given CONTEXT.

CONTEXT: {context}
ANSWER: {answer}

Return a single number between 0 and 1.
"""
        response = self.client.chat(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 0.0
