# evaluation/metrics_base.py

class BaseMetric:
    """
    Base class for defining evaluation metrics.
    Each metric must implement compute(reference, prediction).
    """

    def compute(self, reference: str, prediction: str) -> float:
        raise NotImplementedError("Metric must implement compute()")
