# language_judge/language_judge.py

import json
from typing import Dict

from .prompt_builder import PromptBuilder
from .api_client import APIClient
from .config import OPENAI_MODEL
from .correctness import CorrectnessScorer
from .groundness import GroundednessScorer
from .fluency import FluencyScorer
from .format import FormatScorer
from .aggregator import ScoreAggregator


class LanguageJudge:
    """
    Full Language Judge module as described in the paper.
    """

    def __init__(self):
        self.client = APIClient()
        self.correctness = CorrectnessScorer()
        self.groundedness = GroundednessScorer()
        self.fluency = FluencyScorer()
        self.format = FormatScorer()

    def evaluate(self, context: str, question: str, answer: str) -> Dict:
        """Evaluates all 4 dimensions + final LJ score."""

        C = self.correctness.score(context, question, answer)
        G = self.groundedness.score(context, question, answer)
        F = self.fluency.score(answer)
        S = self.format.score(answer)

        final = ScoreAggregator.aggregate(C, G, F, S)

        return {
            "scores": {
                "correctness": C,
                "groundedness": G,
                "fluency": F,
                "format": S,
            },
            "final_language_judge_score": final
        }
