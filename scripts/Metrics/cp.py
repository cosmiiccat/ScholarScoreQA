# metrics/cp.py

from typing import List, Optional
from .base import BaseMetric
from .preprocessing import TextPreprocessor
from .utils import token_intersection


class ContainmentPercentage(BaseMetric):
    """
    Containment Percentage (CP)

    Measures how much of the gold-standard answer evidence (E)
    is preserved inside the retrieved spans (C).

        CP(E, C) = |E ∩ C| / |E| * 100

    Higher CP → better grounding, less evidence loss.
    """

    def metric_name(self) -> str:
        return "Containment Percentage (CP)"

    def compute(
        self,
        gold_evidence: str,
        retrieved_spans: str,
        normalize: bool = True
    ) -> float:

        try:
            if normalize:
                gold_evidence = TextPreprocessor.normalize(gold_evidence)
                retrieved_spans = TextPreprocessor.normalize(retrieved_spans)

            E = self.tokenize(gold_evidence)
            C = self.tokenize(retrieved_spans)

            if len(E) == 0:
                self.logger.warning("Gold evidence is empty — CP defined as 0.")
                return 0.0

            inter = token_intersection(E, C)
            cp = (inter / len(E)) * 100

            return round(cp, 4)

        except Exception as e:
            self.logger.error(f"Error computing CP: {e}")
            raise
