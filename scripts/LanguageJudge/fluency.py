# language_judge/scorers/fluency_scorer.py

from textstat import flesch_reading_ease


class FluencyScorer:
    """
    Fluency/Coherence scoring.

    Uses readability-based heuristic normalized to [0,1].
    """

    @staticmethod
    def score(answer: str) -> float:
        score = flesch_reading_ease(answer)  
        return max(0.0, min(score / 100.0, 1.0))
