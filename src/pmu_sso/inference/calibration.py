from __future__ import annotations
from typing import Iterable
import numpy as np

def conformal_order_statistic(scores: np.ndarray, target_miscoverage: float) -> tuple[float, int]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a nonempty finite one-dimensional array")
    if not 0 < target_miscoverage < 1:
        raise ValueError("target_miscoverage must lie in (0, 1)")
    n = values.size
    k = int(np.ceil((n + 1) * (1 - target_miscoverage)))
    radius = float("inf") if k > n else float(np.partition(values, k - 1)[k - 1])
    return radius, k
