# evaluation/metrics/tokenf1_metric.py

from collections import Counter
from .metrics_base import BaseMetric
from .normalizer import TextNormalizer

class TokenF1Metric(BaseMetric):
    @staticmethod
    def _compute_f1(pred_tokens, gt_tokens):
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return (2 * precision * recall) / (precision + recall)

    def compute(self, reference: str, prediction: str) -> float:
        pred_tokens = TextNormalizer.normalize(prediction).split()
        gt_tokens = TextNormalizer.normalize(reference).split()
        return self._compute_f1(pred_tokens, gt_tokens)
