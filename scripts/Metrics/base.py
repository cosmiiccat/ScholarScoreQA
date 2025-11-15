# metrics/base.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.

    Each metric must implement:
        - compute()
        - metric_name()
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        """Compute the metric value."""
        raise NotImplementedError

    @abstractmethod
    def metric_name(self) -> str:
        """Return the metric display name."""
        raise NotImplementedError

    def tokenize(self, text: str) -> List[int]:
        """Tokenize using HF tokenizer if provided; fallback to whitespace."""
        if self.tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return text.split()

    def info(self) -> Dict[str, Any]:
        """Metric card, used in research reporting."""
        return {
            "name": self.metric_name(),
            "module": self.__class__.__name__,
            "supports_tokenizers": bool(self.tokenizer is not None),
            "description": self.__doc__,
        }
