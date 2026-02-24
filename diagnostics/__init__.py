"""
Diagnostics for evaluating physics-informed diffusion models.

Each module computes a specific metric aligned to the project hypotheses:

* :mod:`semigroup`    — Operator consistency (H1, H2)
* :mod:`conservation` — Conservation error, mean vs. sample-wise (H3)
* :mod:`horizon`      — Error growth with horizon / sparsity (H1, H2)
* :mod:`wasserstein`  — Wasserstein distance for transport blurring (H1)
"""

from diagnostics.semigroup import semigroup_consistency
from diagnostics.conservation import conservation_error
from diagnostics.horizon import error_vs_horizon, error_vs_sparsity
from diagnostics.wasserstein import sliced_wasserstein, per_feature_wasserstein

__all__ = [
    "semigroup_consistency",
    "conservation_error",
    "error_vs_horizon",
    "error_vs_sparsity",
    "sliced_wasserstein",
    "per_feature_wasserstein",
]
