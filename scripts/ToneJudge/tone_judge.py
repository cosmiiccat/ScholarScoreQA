# tone_judge/tone_judge.py

"""
Tone Judge:
Computes:
- Tone Fidelity (st)
- Content Preservation (sp)
- Linguistic Clarity (sc)
Then: TJ(a) = αt * st + αp * sp + αc * sc
"""

from typing import Dict
from .tone_scorer import ToneScorer
from .tone_transformer import ToneTransformer
from .config import (
    TONE_WEIGHT_TFIDELITY,
    TONE_WEIGHT_CPRESERVE,
    TONE_WEIGHT_CLARITY,
)


class ToneJudge:
    def __init__(self):
        self.transformer = ToneTransformer()
        self.scorer = ToneScorer()

    def evaluate(self, answer: str, tone: str) -> Dict[str, float]:
        """Generate transformed answer and compute all three scores."""
        transformed = self.transformer.transform(answer, tone)

        st = self.scorer.score_tone_fidelity(transformed, tone)
        sp = self.scorer.score_content_preservation(answer, transformed)
        sc = self.scorer.score_linguistic_clarity(transformed)

        final_score = (
            TONE_WEIGHT_TFIDELITY * st
            + TONE_WEIGHT_CPRESERVE * sp
            + TONE_WEIGHT_CLARITY * sc
        )

        return {
            "transformed": transformed,
            "tone_fidelity": st,
            "content_preservation": sp,
            "linguistic_clarity": sc,
            "final_tone_score": final_score,
        }
