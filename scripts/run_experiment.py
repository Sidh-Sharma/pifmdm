"""
Unified experiment runner for all diffusion models.

Pipeline (no generic training loop — each adapter delegates to its
model's own native training code):

    1. Load data → UnifiedDataset
    2. Split → train / val / test
    3. adapter.prepare_native_loaders()   — wrap into model-specific format
    4. adapter.native_train()             — run native training (GPU-aware)
    5. adapter.native_evaluate()          — run native sampling + metrics
    6. _run_diagnostics()                 — use adapter.predict() for diagnostics

Usage::

    python scripts/run_experiment.py \\
        --model csdi \\
        --config configs/experiment/my_experiment.yaml \\
        --device auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from adapters import create_adapter, UnifiedBatch, UnifiedOutput
from datasets.base import UnifiedDataset


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(config: Dict[str, Any]) -> UnifiedDataset:
    """Build a ``UnifiedDataset`` from the ``data`` section of the config.

    Supported ``data.type`` values:

    * ``"advection"`` — calls ``datasets.synthetic.linear_advection``
      with selectable spatial/temporal stride, window size, etc.
    * ``None`` / any other — falls back to raw ``.npy`` or random data.
    """
    data_cfg = config.get("data", {})
    dtype = data_cfg.get("type", "").lower()
    miss_ratio = data_cfg.get("miss_ratio", 0.1)
    mask_strategy = data_cfg.get("mask_strategy", "random")

    if dtype == "advection":
        from datasets.synthetic.linear_advection import make_advection_dataset

        ds, stats = make_advection_dataset(
            path=data_cfg.get("path", "datasets/synthetic/advection_data_2d.npy"),
            window_size=data_cfg.get("window_size", 64),
            window_stride=data_cfg.get("window_stride", 1),
            spatial_stride=data_cfg.get("spatial_stride", 1),
            temporal_stride=data_cfg.get("temporal_stride", 1),
            normalise=data_cfg.get("normalise", True),
            miss_ratio=miss_ratio,
            mask_strategy=mask_strategy,
        )
        print(f"Advection dataset: {len(ds)} samples, "
              f"L={stats['window_size']}, K={stats['nx_sub']*stats['ny_sub']} "
              f"(spatial {stats['nx_sub']}×{stats['ny_sub']}, "
              f"stride={stats['spatial_stride']})")
        return ds

    data_path = data_cfg.get("path")
    if data_path and os.path.exists(data_path):
        raw = torch.from_numpy(np.load(data_path)).float()
    else:
        N = data_cfg.get("n_samples", 256)
        L = data_cfg.get("seq_len", 96)
        K = data_cfg.get("target_dim", config.get("target_dim", 7))
        raw = torch.randn(N, L, K)

    return UnifiedDataset(raw, miss_ratio=miss_ratio, mask_strategy=mask_strategy)


def _load_config(path: str) -> dict:
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def _auto_device() -> torch.device:
    """Pick the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main experiment pipeline
# ---------------------------------------------------------------------------

def run_experiment(
    model_name: str,
    config: Dict[str, Any],
    device: str | torch.device = "auto",
):
    """End-to-end pipeline: data → native train → native eval → diagnostics."""
    if isinstance(device, str):
        device = _auto_device() if device == "auto" else torch.device(device)
    print(f"Device: {device}")

    # 1. Load dataset
    ds = _load_dataset(config)
    sample0 = ds[0]
    L = sample0["observed_data"].shape[0]
    K = sample0["observed_data"].shape[1]

    # 2. Split
    n_train = int(0.7 * len(ds))
    n_val = int(0.15 * len(ds))
    n_test = len(ds) - n_train - n_val
    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [n_train, n_val, n_test])

    # 3. Create adapter
    config.setdefault("target_dim", K)
    config.setdefault("seq_len", L)
    adapter = create_adapter(model_name, config=config, device=device)

    # 4. Prepare model-specific data loaders
    loaders = adapter.prepare_native_loaders(train_ds, val_ds, test_ds)

    # 5. Native training (delegates to model's own training code)
    train_cfg = config.get("training", {})
    save_dir = train_cfg.get("save_dir", f"results/checkpoints/{model_name}")
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} (native loop)")
    print(f"{'='*60}")
    adapter.native_train(loaders, save_dir=save_dir)

    # 6. Native evaluation (delegates to model's own eval code)
    eval_cfg = config.get("evaluation", {})
    metrics_dir = eval_cfg.get("save_dir", f"results/metrics/{model_name}")
    n_samples = eval_cfg.get("n_samples", 10)
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()} (native eval, {n_samples} samples)")
    print(f"{'='*60}")
    metrics = adapter.native_evaluate(loaders, n_samples=n_samples, save_dir=metrics_dir)

    # 7. Diagnostics (use adapter.predict() on unified batches)
    diag_cfg = config.get("diagnostics", {})
    if diag_cfg:
        _run_diagnostics(adapter, ds, diag_cfg, n_samples=n_samples,
                         device=device, save_dir=metrics_dir)

    return metrics


def _run_diagnostics(
    adapter,
    ds: UnifiedDataset,
    diag_cfg: Dict[str, Any],
    *,
    n_samples: int = 10,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "results/metrics",
):
    """Run configured diagnostics and save results."""
    os.makedirs(save_dir, exist_ok=True)
    results: Dict[str, Any] = {}

    # Grab a small test subset for diagnostics (first 64 samples or all)
    n_diag = min(64, len(ds))
    diag_data = torch.stack([ds[i]["observed_data"] for i in range(n_diag)])  # (N, L, K)

    # Semigroup 
    sg_cfg = diag_cfg.get("semigroup")
    if sg_cfg:
        from diagnostics.semigroup import semigroup_consistency

        t1 = sg_cfg.get("t1", 16)
        t2 = sg_cfg.get("t2", 16)
        print(f"\n== Semigroup consistency (t1={t1}, t2={t2}) ==")
        sg = semigroup_consistency(adapter, diag_data, t1=t1, t2=t2,
                                   n_samples=n_samples, device=device)
        print(f"  Relative L2: {sg['relative_l2']:.6f}")
        print(f"  Absolute L2: {sg['absolute_l2']:.6f}")
        results["semigroup"] = {
            "relative_l2": sg["relative_l2"],
            "absolute_l2": sg["absolute_l2"],
        }

    if diag_cfg.get("conservation"):
        from diagnostics.conservation import conservation_error

        print("\n== Conservation error ==")
        # Run model on diag_data batch
        L = diag_data.shape[1]
        obs_mask = torch.ones_like(diag_data)
        tgt_mask = torch.zeros_like(diag_data)
        # Mask last 25% as target
        cut = L // 4
        obs_mask[:, -cut:, :] = 0.0
        tgt_mask[:, -cut:, :] = 1.0
        tp = torch.arange(L).float() / max(L - 1, 1)
        tp = tp.unsqueeze(0).expand(diag_data.shape[0], -1)

        batch = UnifiedBatch(
            observed_data=diag_data * obs_mask,
            observed_mask=obs_mask,
            target_mask=tgt_mask,
            timepoints=tp,
        )
        out = adapter.predict(batch, n_samples=n_samples)
        cons = conservation_error(out.samples.cpu(), diag_data, tgt_mask)
        print(f"  Mean conservation error:       {cons['mean_error']:.6f}")
        print(f"  Sample-wise conservation error: {cons['samplewise_error']:.6f}")
        print(f"  Gap (H3 signal):               {cons['gap']:.6f}")
        results["conservation"] = {
            "mean_error": cons["mean_error"],
            "samplewise_error": cons["samplewise_error"],
            "gap": cons["gap"],
        }


    hz_cfg = diag_cfg.get("horizon")
    if hz_cfg:
        from diagnostics.horizon import error_vs_horizon

        horizons = hz_cfg.get("horizons", [8, 16, 32])
        ctx = hz_cfg.get("context_len")
        print(f"\n== Error vs horizon ({horizons}) ==")
        hzr = error_vs_horizon(adapter, diag_data, horizons=horizons,
                               context_len=ctx, n_samples=n_samples, device=device)
        for h, r, m in zip(hzr["horizons"], hzr["rmse"], hzr["mae"]):
            print(f"  h={h:3d}  RMSE={r:.6f}  MAE={m:.6f}")
        results["horizon"] = hzr



    sp_cfg = diag_cfg.get("sparsity")
    if sp_cfg:
        from diagnostics.horizon import error_vs_sparsity

        ratios = sp_cfg.get("miss_ratios", [0.1, 0.3, 0.5, 0.7])
        print(f"\n== Error vs sparsity ({ratios}) ==")
        spr = error_vs_sparsity(adapter, diag_data, miss_ratios=ratios,
                                n_samples=n_samples, device=device)
        for r, rmse, mae in zip(spr["miss_ratios"], spr["rmse"], spr["mae"]):
            print(f"  miss={r:.1f}  RMSE={rmse:.6f}  MAE={mae:.6f}")
        results["sparsity"] = spr

    
    ws_cfg = diag_cfg.get("wasserstein")
    if ws_cfg or diag_cfg.get("wasserstein") is not None:
        from diagnostics.wasserstein import sliced_wasserstein

        print("\n== Wasserstein distance ==")
        # Reuse the conservation prediction if available, otherwise run fresh
        L = diag_data.shape[1]
        obs_mask = torch.ones_like(diag_data)
        tgt_mask = torch.zeros_like(diag_data)
        cut = L // 4
        obs_mask[:, -cut:, :] = 0.0
        tgt_mask[:, -cut:, :] = 1.0
        tp = torch.arange(L).float() / max(L - 1, 1)
        tp = tp.unsqueeze(0).expand(diag_data.shape[0], -1)

        batch = UnifiedBatch(
            observed_data=diag_data * obs_mask,
            observed_mask=obs_mask,
            target_mask=tgt_mask,
            timepoints=tp,
        )
        out = adapter.predict(batch, n_samples=n_samples)
        n_proj = ws_cfg.get("n_projections", 128) if isinstance(ws_cfg, dict) else 128
        ws = sliced_wasserstein(out.samples.cpu(), diag_data, tgt_mask,
                                n_projections=n_proj)
        print(f"  Sliced Wasserstein: {ws['swd']:.6f}")
        results["wasserstein"] = {"swd": ws["swd"]}

    # Save
    with open(os.path.join(save_dir, "diagnostics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDiagnostics saved to {save_dir}/diagnostics.json")


def main():
    parser = argparse.ArgumentParser(description="pifmdm experiment runner")
    parser.add_argument("--model", required=True, choices=["csdi", "cfmi", "pidm", "tmdm"])
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--device", default="auto",
                        help="'auto' picks cuda/mps/cpu, or specify explicitly")
    args = parser.parse_args()

    config = _load_config(args.config)
    run_experiment(model_name=args.model, config=config, device=args.device)


if __name__ == "__main__":
    main()
