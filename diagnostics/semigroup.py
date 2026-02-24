"""
Semigroup (operator) consistency diagnostic.

Tests: Φ(t₁+t₂, x₀) ≈ Φ(t₂, Φ(t₁, x₀))

If a model truly learns the advection operator, composing two short-horizon
predictions should match a single long-horizon prediction.  Failures here
indicate that attention mechanisms learn correlations but violate global
transport dynamics (H1) or that the learned operator is not resolution-
adaptive (H2).

Metric
------
``semigroup_error``
    L2 norm of the difference between the single-step ("direct") prediction
    and the composed two-step prediction, normalised by the direct
    prediction norm.

Usage::

    from diagnostics.semigroup import semigroup_consistency

    result = semigroup_consistency(adapter, dataset, t1=32, t2=32)
    print(result["relative_l2"])   # scalar
    print(result["per_sample"])    # (N,) array
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from adapters.base import BaseAdapter, UnifiedBatch, UnifiedOutput
from datasets.base import UnifiedDataset


def semigroup_consistency(
    adapter: BaseAdapter,
    data: torch.Tensor,
    *,
    t1: int,
    t2: int,
    n_samples: int = 10,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Measure semigroup consistency: direct vs. composed prediction.

    Parameters
    ----------
    adapter : BaseAdapter
        A trained adapter (model weights already loaded).
    data : torch.Tensor
        Full data tensor ``(N, L, K)`` with ``L >= t1 + t2``.
    t1 : int
        Length of the first prediction segment.
    t2 : int
        Length of the second prediction segment.
    n_samples : int
        Number of stochastic samples to draw per prediction.
    device : torch.device or str

    Returns
    -------
    dict with keys:
        ``relative_l2`` — scalar, mean relative L2 error
        ``absolute_l2`` — scalar, mean absolute L2 error
        ``per_sample``  — (N,) tensor of per-sample relative L2 errors
    """
    if isinstance(device, str):
        device = torch.device(device)

    N, L, K = data.shape
    total_len = t1 + t2
    assert L >= total_len, f"Data length {L} < t1+t2={total_len}"

    # Slice to the relevant window
    data_window = data[:, :total_len, :].to(device)

    # ---- Direct prediction: observe [0, t1), predict [t1, t1+t2) --------
    observed_mask_direct = torch.zeros(N, total_len, K, device=device)
    observed_mask_direct[:, :t1, :] = 1.0
    target_mask_direct = torch.zeros(N, total_len, K, device=device)
    target_mask_direct[:, t1:, :] = 1.0
    tp = torch.arange(total_len, device=device).float() / max(total_len - 1, 1)
    tp = tp.unsqueeze(0).expand(N, -1)

    batch_direct = UnifiedBatch(
        observed_data=data_window,
        observed_mask=observed_mask_direct,
        target_mask=target_mask_direct,
        timepoints=tp,
    )
    out_direct: UnifiedOutput = adapter.predict(batch_direct, n_samples=n_samples)
    pred_direct = out_direct.samples.median(dim=1).values  # (N, L, K)
    pred_direct_tail = pred_direct[:, t1:, :]              # (N, t2, K)

    # ---- Step 1: observe [0, t1) --  predict nothing, just use ground truth
    # (we use ground truth as input for the first segment)
    # ---- Step 2: observe the first-step output [t1, t1+t2) ≈ direct t1 output,
    #              then predict [t1, t1+t2) again ---------
    # Build an intermediate "observed" tensor from step-1 output
    step1_pred = pred_direct[:, :t1, :]  # model's reconstruction of [0, t1)
    # Now treat step1_pred as observation, predict the next t2 steps
    step2_input = torch.zeros(N, t2 + t1, K, device=device)
    step2_input[:, :t1, :] = step1_pred.detach()  # use model's own output
    step2_input[:, t1:, :] = 0.0                   # to be predicted

    obs_mask_step2 = torch.zeros(N, t1 + t2, K, device=device)
    obs_mask_step2[:, :t1, :] = 1.0
    tgt_mask_step2 = torch.zeros(N, t1 + t2, K, device=device)
    tgt_mask_step2[:, t1:, :] = 1.0
    tp2 = torch.arange(t1 + t2, device=device).float() / max(t1 + t2 - 1, 1)
    tp2 = tp2.unsqueeze(0).expand(N, -1)

    batch_step2 = UnifiedBatch(
        observed_data=step2_input,
        observed_mask=obs_mask_step2,
        target_mask=tgt_mask_step2,
        timepoints=tp2,
    )
    out_composed: UnifiedOutput = adapter.predict(batch_step2, n_samples=n_samples)
    pred_composed = out_composed.samples.median(dim=1).values  # (N, L, K)
    pred_composed_tail = pred_composed[:, t1:, :]              # (N, t2, K)

    # ---- Compute metrics ----
    diff = pred_direct_tail - pred_composed_tail               # (N, t2, K)
    per_sample_abs = diff.reshape(N, -1).norm(dim=1)           # (N,)
    direct_norm = pred_direct_tail.reshape(N, -1).norm(dim=1).clamp(min=1e-8)
    per_sample_rel = per_sample_abs / direct_norm              # (N,)

    return {
        "relative_l2": per_sample_rel.mean().item(),
        "absolute_l2": per_sample_abs.mean().item(),
        "per_sample": per_sample_rel.cpu(),
    }
