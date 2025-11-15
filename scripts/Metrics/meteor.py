# evaluation/metrics/meteor_metric.py

import nltk
from .metrics_base import BaseMetric
from nltk.translate.meteor_score import single_meteor_score

class MeteorMetric(BaseMetric):
    def __init__(self):
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)

    def compute(self, reference: str, prediction: str) -> float:
        ref_toks = nltk.word_tokenize(reference)
        pred_toks = nltk.word_tokenize(prediction)
        return single_meteor_score(ref_toks, pred_toks)
