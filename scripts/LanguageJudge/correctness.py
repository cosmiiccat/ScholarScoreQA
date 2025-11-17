# language_judge/scorers/correctness_scorer.py

from typing import Dict
from .api_client import APIClient
from .config import OPENAI_MODEL


class CorrectnessScorer:
    """Correctness evaluator using LLM."""

    def __init__(self):
        self.client = APIClient()

    def score(self, context: str, question: str, answer: str) -> float:
        prompt = f"""
Rate correctness (0-1). Consider factual accuracy.

CONTEXT: {context}
QUESTION: {question}
ANSWER: {answer}

Return only a number.
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
