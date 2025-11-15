# metrics/utils.py

import random
from typing import List
from collections import Counter


def token_intersection(a: List[int], b: List[int]) -> int:
    """Multiset intersection size."""
    return sum((Counter(a) & Counter(b)).values())


def bootstrap(values: List[float], n_samples: int = 2000, ci: float = 0.95):
    """
    Compute bootstrap confidence interval for metrics.
    """
    if len(values) == 0:
        return 0.0, (0.0, 0.0)

    samples = []
    N = len(values)
    for _ in range(n_samples):
        sample = [values[random.randint(0, N - 1)] for _ in range(N)]
        samples.append(sum(sample) / N)

    alpha = (1 - ci) / 2
    lower = sorted(samples)[int(alpha * n_samples)]
    upper = sorted(samples)[int((1 - alpha) * n_samples)]
    return sum(values) / N, (lower, upper)
