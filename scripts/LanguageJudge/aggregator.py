# language_judge/aggregator.py

from .config import (
    WEIGHT_CORRECTNESS,
    WEIGHT_GROUNDEDNESS,
    WEIGHT_FLUENCY,
    WEIGHT_FORMAT,
)


class ScoreAggregator:
    """Weighted aggregation for final LJ score."""

    @staticmethod
    def aggregate(c: float, g: float, f: float, s: float) -> float:
        return (
            WEIGHT_CORRECTNESS * c +
            WEIGHT_GROUNDEDNESS * g +
            WEIGHT_FLUENCY * f +
            WEIGHT_FORMAT * s
        )
