# evaluation/metrics/bertscore_metric.py

from .metrics_base import BaseMetric
from bert_score import score

class BertScoreMetric(BaseMetric):
    def compute(self, reference: str, prediction: str) -> float:
        P, R, F1 = score([prediction], [reference], lang="en", verbose=False)
        return F1.mean().item()
