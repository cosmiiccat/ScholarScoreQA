# evaluation/metrics/bleu_metric.py

from .metrics_base import BaseMetric
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BleuMetric(BaseMetric):
    def __init__(self):
        self.smooth = SmoothingFunction().method2

    def compute(self, reference: str, prediction: str) -> float:
        return sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=self.smooth,
        )
