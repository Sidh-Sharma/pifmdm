"""
Error-growth-with-horizon / sparsity diagnostic.

Evaluates how imputation / forecasting error scales as either:

* **Prediction horizon** increases (more future steps to predict), or
* **Sparsity** increases (fewer observed points).

Linear error growth suggests the model captures the dynamics;
super-linear / exponential growth indicates failure.

Usage::

    from diagnostics.horizon import error_vs_horizon, error_vs_sparsity

    horizon_curve = error_vs_horizon(
        adapter, data, horizons=[16, 32, 64, 128], n_samples=10,
    )
    sparsity_curve = error_vs_sparsity(
        adapter, data, miss_ratios=[0.1, 0.3, 0.5, 0.7, 0.9], n_samples=10,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from adapters.base import BaseAdapter, UnifiedBatch, UnifiedOutput


# ---------------------------------------------------------------------------
# Error-vs-horizon
# ---------------------------------------------------------------------------

def error_vs_horizon(
    adapter: BaseAdapter,
    data: torch.Tensor,
    *,
    horizons: Sequence[int],
    context_len: int | None = None,
    n_samples: int = 10,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Compute RMSE / MAE at increasing prediction horizons.

    Parameters
    ----------
    adapter : BaseAdapter
        Trained adapter.
    data : torch.Tensor
        ``(N, L, K)`` — full data.
    horizons : sequence of int
        List of prediction lengths to evaluate (e.g. ``[16, 32, 64]``).
    context_len : int, optional
        Number of observed timesteps before the horizon.
        Defaults to ``L - max(horizons)`` (i.e. the maximum context that
        lets the longest horizon fit).
    n_samples : int
        Stochastic samples per prediction.
    device : str or torch.device

    Returns
    -------
    dict with keys:
        ``horizons`` — list of int
        ``rmse``     — list of float, one per horizon
        ``mae``      — list of float, one per horizon
    """
    if isinstance(device, str):
        device = torch.device(device)

    N, L, K = data.shape
    max_h = max(horizons)
    if context_len is None:
        context_len = L - max_h
    assert context_len > 0, "context_len must be positive"

    rmses: List[float] = []
    maes: List[float] = []

    for h in horizons:
        total = context_len + h
        assert total <= L, f"context({context_len}) + horizon({h}) > L({L})"
        window = data[:, :total, :].to(device)

        obs_mask = torch.zeros(N, total, K, device=device)
        obs_mask[:, :context_len, :] = 1.0
        tgt_mask = torch.zeros(N, total, K, device=device)
        tgt_mask[:, context_len:, :] = 1.0
        tp = torch.arange(total, device=device).float() / max(total - 1, 1)
        tp = tp.unsqueeze(0).expand(N, -1)

        batch = UnifiedBatch(
            observed_data=window,
            observed_mask=obs_mask,
            target_mask=tgt_mask,
            timepoints=tp,
        )
        out: UnifiedOutput = adapter.predict(batch, n_samples=n_samples)
        pred = out.samples.median(dim=1).values   # (N, total, K)
        gt = window

        diff = (pred[:, context_len:, :] - gt[:, context_len:, :])
        n_eval = diff.numel()
        rmses.append(((diff ** 2).sum() / n_eval).sqrt().item())
        maes.append((diff.abs().sum() / n_eval).item())

    return {"horizons": list(horizons), "rmse": rmses, "mae": maes}


# ---------------------------------------------------------------------------
# Error-vs-sparsity
# ---------------------------------------------------------------------------

def error_vs_sparsity(
    adapter: BaseAdapter,
    data: torch.Tensor,
    *,
    miss_ratios: Sequence[float],
    n_samples: int = 10,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute RMSE / MAE at increasing levels of data missingness.

    Parameters
    ----------
    adapter : BaseAdapter
        Trained adapter.
    data : torch.Tensor
        ``(N, L, K)`` — full data.
    miss_ratios : sequence of float
        Fraction of points to mask as missing (e.g. ``[0.1, 0.3, 0.5, 0.7]``).
    n_samples : int
        Stochastic samples per prediction.
    device : str or torch.device
    seed : int
        RNG seed for reproducible masking.

    Returns
    -------
    dict with keys:
        ``miss_ratios`` — list of float
        ``rmse``        — list of float
        ``mae``         — list of float
    """
    if isinstance(device, str):
        device = torch.device(device)

    N, L, K = data.shape
    rmses: List[float] = []
    maes: List[float] = []

    for ratio in miss_ratios:
        gen = torch.Generator().manual_seed(seed)
        rand_mask = torch.rand(N, L, K, generator=gen)
        observed_mask = (rand_mask >= ratio).float().to(device)
        target_mask = (1.0 - observed_mask).to(device)

        window = data.to(device)
        tp = torch.arange(L, device=device).float() / max(L - 1, 1)
        tp = tp.unsqueeze(0).expand(N, -1)

        batch = UnifiedBatch(
            observed_data=window * observed_mask,  # zero out missing
            observed_mask=observed_mask,
            target_mask=target_mask,
            timepoints=tp,
        )
        out: UnifiedOutput = adapter.predict(batch, n_samples=n_samples)
        pred = out.samples.median(dim=1).values  # (N, L, K)

        diff = (pred - window) * target_mask
        n_eval = target_mask.sum().clamp(min=1).item()
        rmses.append(((diff ** 2).sum() / n_eval).sqrt().item())
        maes.append((diff.abs().sum() / n_eval).item())

    return {"miss_ratios": list(miss_ratios), "rmse": rmses, "mae": maes}
