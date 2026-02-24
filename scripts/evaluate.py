"""
Standalone evaluation script — uses each model's native evaluation.

Usage::

    python scripts/evaluate.py \\
        --model csdi \\
        --config configs/experiment/my_experiment.yaml \\
        --checkpoint results/checkpoints/csdi/best_model.pt \\
        --n_samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from adapters import create_adapter
from scripts.run_experiment import _load_dataset, _auto_device


def main():
    parser = argparse.ArgumentParser(description="pifmdm evaluation")
    parser.add_argument("--model", required=True, choices=["csdi", "cfmi", "pidm", "tmdm"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--run_diagnostics", action="store_true",
                        help="Run diagnostics defined in the config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f) if args.config.endswith((".yml", ".yaml")) else json.load(f)

    device = _auto_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    # Load dataset
    ds = _load_dataset(config)
    sample0 = ds[0]
    L = sample0["observed_data"].shape[0]
    K = sample0["observed_data"].shape[1]

    # Use the same 70/15/15 split as run_experiment
    n_train = int(0.7 * len(ds))
    n_val = int(0.15 * len(ds))
    n_test = len(ds) - n_train - n_val
    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [n_train, n_val, n_test])

    # Create adapter and load checkpoint
    config.setdefault("target_dim", K)
    config.setdefault("seq_len", L)
    adapter = create_adapter(args.model, config=config, device=device)
    adapter.load(args.checkpoint)

    # Prepare native loaders and evaluate
    loaders = adapter.prepare_native_loaders(train_ds, val_ds, test_ds)
    save_dir = args.save_dir or f"results/metrics/{args.model}"
    metrics = adapter.native_evaluate(loaders, n_samples=args.n_samples, save_dir=save_dir)
    print(json.dumps(metrics, indent=2))

    # Diagnostics
    if args.run_diagnostics:
        diag_cfg = config.get("diagnostics", {})
        if diag_cfg:
            from scripts.run_experiment import _run_diagnostics
            _run_diagnostics(adapter, ds, diag_cfg, n_samples=args.n_samples,
                             device=device, save_dir=save_dir)


if __name__ == "__main__":
    main()
