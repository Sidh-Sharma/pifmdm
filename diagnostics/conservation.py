"""
Conservation error diagnostic.

For linear advection with periodic boundaries the total *mass*
(integral / sum over the spatial domain) is exactly conserved at every
timestep.  This diagnostic compares:

* **Mean conservation error** — the absolute difference between the
  spatially-summed ground truth and the *mean* of spatially-summed
  predictions across samples.  This measures whether the physics is
  satisfied *in expectation*.
* **Sample-wise conservation error** — the mean over individual
  samples of the absolute conservation deviation.  This measures
  whether physics is satisfied *per sample*.

A large gap between the two indicates that soft physics penalties in the
loss satisfy conservation only on average, while individual samples
violate it (H3).

Usage::

    from diagnostics.conservation import conservation_error

    result = conservation_error(predictions, ground_truth, target_mask)
    print(result["mean_error"])        # physics-in-expectation
    print(result["samplewise_error"])  # physics-per-sample
    print(result["gap"])               # difference (H3 signal)
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def conservation_error(
    samples: torch.Tensor,
    ground_truth: torch.Tensor,
    target_mask: torch.Tensor,
) -> Dict[str, Any]:
    """Compute conservation (mass) error.

    Parameters
    ----------
    samples : torch.Tensor
        ``(N, S, L, K)`` — *S* stochastic samples.
    ground_truth : torch.Tensor
        ``(N, L, K)`` — ground truth.
    target_mask : torch.Tensor
        ``(N, L, K)`` — 1 where model predicts, 0 elsewhere.

    Returns
    -------
    dict with keys:
        ``mean_error``       — conservation error of the sample mean
        ``samplewise_error`` — mean per-sample conservation error
        ``gap``              — samplewise − mean (H3 signal)
        ``per_timestep_mean`` — (L,) mean conservation error at each step
        ``per_timestep_sw``   — (L,) sample-wise conservation error at each step
    """
    # N, S, L, K = samples.shape
    N, S, L, K = samples.shape

    # Ground truth total mass at each timestep: sum over spatial (K)
    gt_mass = ground_truth.sum(dim=-1)              # (N, L)

    # Predicted mass per sample at each timestep
    pred_mass = samples.sum(dim=-1)                 # (N, S, L)

    # --- Mean conservation error (physics in expectation) ---
    mean_pred_mass = pred_mass.mean(dim=1)          # (N, L)
    mean_cons_error = (mean_pred_mass - gt_mass).abs()  # (N, L)

    # --- Sample-wise conservation error (physics per sample) ---
    # |predicted_mass_s - gt_mass| for each sample s, then mean over s
    sw_cons_error = (pred_mass - gt_mass.unsqueeze(1)).abs()  # (N, S, L)
    sw_cons_mean = sw_cons_error.mean(dim=1)                  # (N, L)

    # Aggregate over samples and timesteps
    mean_err = mean_cons_error.mean().item()
    sw_err = sw_cons_mean.mean().item()

    return {
        "mean_error": mean_err,
        "samplewise_error": sw_err,
        "gap": sw_err - mean_err,
        "per_timestep_mean": mean_cons_error.mean(dim=0).cpu(),   # (L,)
        "per_timestep_sw": sw_cons_mean.mean(dim=0).cpu(),        # (L,)
    }
