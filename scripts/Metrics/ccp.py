# metrics/ccp.py

from .base import BaseMetric
from .preprocessing import TextPreprocessor


class ContextCompressionPercentage(BaseMetric):
    """
    Context Compression Percentage (CCP)

    Measures reduction in token count after span selection.

        CCP(D, C) = (1 - |C| / |D|) * 100

    Higher CCP → more compression.
    """

    def metric_name(self) -> str:
        return "Context Compression Percentage (CCP)"

    def compute(
        self,
        full_document: str,
        retrieved_spans: str,
        normalize: bool = True
    ) -> float:

        try:
            if normalize:
                full_document = TextPreprocessor.normalize(full_document)
                retrieved_spans = TextPreprocessor.normalize(retrieved_spans)

            D = self.tokenize(full_document)
            C = self.tokenize(retrieved_spans)

            if len(D) == 0:
                self.logger.warning("Full document is empty — CCP defined as 0.")
                return 0.0

            ccp = (1 - (len(C) / len(D))) * 100
            return round(ccp, 4)

        except Exception as e:
            self.logger.error(f"Error computing CCP: {e}")
            raise
