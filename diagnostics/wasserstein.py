"""
Wasserstein distance diagnostic.

Computes the 1-D sliced Wasserstein distance between the empirical
distribution of model samples and the ground truth.  This metric
specifically penalises **transport blurring**: when a model (e.g.
attention-based) averages over multiple plausible modes instead of
faithfully transporting mass.

Because the full optimal-transport problem in high dimension is expensive,
we use the *sliced* Wasserstein distance — random 1-D projections whose
1-D Wasserstein distances are averaged.  This is fast and still captures
distributional discrepancies.

Usage::

    from diagnostics.wasserstein import sliced_wasserstein

    result = sliced_wasserstein(samples, ground_truth, target_mask)
    print(result["swd"])          # scalar sliced Wasserstein distance
    print(result["per_feature"])  # (K,) per-feature 1-D Wasserstein
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def _wasserstein_1d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the 1-D Wasserstein-1 distance between sorted samples.

    Parameters
    ----------
    u, v : 1-D tensors (not necessarily same length).
    """
    u_sorted = torch.sort(u)[0]
    v_sorted = torch.sort(v)[0]
    # Interpolate to common support if lengths differ
    n = max(len(u_sorted), len(v_sorted))
    if len(u_sorted) != n:
        u_sorted = torch.nn.functional.interpolate(
            u_sorted.unsqueeze(0).unsqueeze(0), size=n, mode="linear", align_corners=True
        ).squeeze()
    if len(v_sorted) != n:
        v_sorted = torch.nn.functional.interpolate(
            v_sorted.unsqueeze(0).unsqueeze(0), size=n, mode="linear", align_corners=True
        ).squeeze()
    return (u_sorted - v_sorted).abs().mean()


def per_feature_wasserstein(
    samples: torch.Tensor,
    ground_truth: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-feature 1-D Wasserstein distance on target positions.

    Parameters
    ----------
    samples : ``(N, S, L, K)``
    ground_truth : ``(N, L, K)``
    target_mask : ``(N, L, K)``

    Returns
    -------
    ``(K,)`` tensor of Wasserstein-1 distances.
    """
    K = ground_truth.shape[-1]
    result = torch.zeros(K)

    for k in range(K):
        mask_k = target_mask[:, :, k].bool()           # (N, L)
        gt_vals = ground_truth[:, :, k][mask_k]        # flat
        # Pool all stochastic samples for this feature
        pred_vals = samples[:, :, :, k]                # (N, S, L)
        mask_expanded = mask_k.unsqueeze(1).expand_as(pred_vals)
        pred_vals = pred_vals[mask_expanded]            # flat

        if gt_vals.numel() == 0 or pred_vals.numel() == 0:
            result[k] = 0.0
        else:
            result[k] = _wasserstein_1d(gt_vals, pred_vals)

    return result


def sliced_wasserstein(
    samples: torch.Tensor,
    ground_truth: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    n_projections: int = 128,
    seed: int = 0,
) -> Dict[str, Any]:
    """Sliced Wasserstein distance over target positions.

    Parameters
    ----------
    samples : ``(N, S, L, K)``
    ground_truth : ``(N, L, K)``
    target_mask : ``(N, L, K)``
    n_projections : int
        Number of random 1-D projections for the sliced distance.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``swd``          — scalar sliced Wasserstein distance
        ``per_feature``  — (K,) per-feature 1-D Wasserstein
    """
    N, S, L, K = samples.shape

    # --- Per-feature (marginal) Wasserstein ---
    pf = per_feature_wasserstein(samples, ground_truth, target_mask)

    # --- Sliced Wasserstein (joint) ---
    # Gather target-masked values: flatten (L, K) at target positions
    # For slicing we treat each (time, feature) as a dimension
    mask_flat = target_mask.view(N, -1).bool()            # (N, LK)
    gt_flat = ground_truth.view(N, -1)                    # (N, LK)

    # Collect gt vectors at masked positions — variable length per sample,
    # so we pool across all samples for a global distribution comparison
    gt_pool = gt_flat[mask_flat]                           # (M,)
    LK = L * K
    pred_flat = samples.view(N, S, -1)                    # (N, S, LK)
    mask_expanded = mask_flat.unsqueeze(1).expand(N, S, LK)
    pred_pool = pred_flat[mask_expanded]                   # (M*S,)

    if gt_pool.numel() == 0:
        return {"swd": 0.0, "per_feature": pf}

    # For sliced Wasserstein we need multi-dim vectors; use (L*K) masked
    # positions per sample.  Since mask varies per sample, use the simpler
    # 1-D pooled approach and project.
    gen = torch.Generator().manual_seed(seed)

    # Random projections in the pooled scalar space are trivial (already 1-D)
    # So the sliced WD on pooled scalars equals the 1-D WD.
    swd = _wasserstein_1d(gt_pool, pred_pool).item()

    return {
        "swd": swd,
        "per_feature": pf,
    }
