"""Wilson interval excerpt from the supplied holdout evaluator.

This is a single-proportion descriptive interval, not a paired-comparison test.
The original function body is preserved. Callers must supply 0 <= correct <= total.
"""
import math


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        / denominator
    )
    return center - margin, center + margin
