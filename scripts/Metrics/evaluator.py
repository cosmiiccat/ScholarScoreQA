# evaluation/evaluator.py

from .tokenf1 import TokenF1Metric
from .rouge import RougeMetric
from .bleu import BleuMetric
from .meteor import MeteorMetric
from .bertscore import BertScoreMetric

class MetricsEvaluator:
    """
    Aggregates all metrics and computes them together.
    """

    def __init__(self):
        self.metrics = {
            "BLEU": BleuMetric(),
            "METEOR": MeteorMetric(),
            "BERTScore": BertScoreMetric(),
            "Token F1": TokenF1Metric(),
            "ROUGE": RougeMetric(),
        }

    def evaluate(self, reference: str, prediction: str) -> dict:
        final_scores = {}

        for name, metric in self.metrics.items():
            result = metric.compute(reference, prediction)

            # ROUGE returns dict; others return float
            if isinstance(result, dict):
                final_scores.update(result)
            else:
                final_scores[name] = result

        return final_scores

    def display(self, scores: dict):
        print("=== Evaluation Metrics ===")
        for name, value in scores.items():
            print(f"{name}: {value:.4f}")
