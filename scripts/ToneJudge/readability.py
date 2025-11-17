# tone_judge/readability.py

"""
Provides readability features for Linguistic Clarity scoring.
"""

import textstat


class Readability:
    @staticmethod
    def clarity_score(text: str) -> float:
        """
        Normalize reading ease to [0,1].
        Higher = clearer.
        """
        score = textstat.flesch_reading_ease(text)
        return max(0.0, min(score / 100.0, 1.0))
