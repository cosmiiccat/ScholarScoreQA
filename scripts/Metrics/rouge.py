# evaluation/metrics/rouge_metric.py

from .metrics_base import BaseMetric
from rouge_score import rouge_scorer

class RougeMetric(BaseMetric):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute(self, reference: str, prediction: str) -> dict:
        scores = self.scorer.score(reference, prediction)
        return {
            "ROUGE-1": scores["rouge1"].fmeasure,
            "ROUGE-2": scores["rouge2"].fmeasure,
            "ROUGE-L": scores["rougeL"].fmeasure,
        }
