"""
CSDI adapter — thin pipeline wrapper around the CSDI codebase.

Data flow:
    UnifiedDataset  → prepare_native_loaders → CSDI-format DataLoader
    native_train    → calls models/csdi/utils.train()
    native_evaluate → calls models/csdi/utils.evaluate()
    predict         → single-batch imputation for diagnostics

CSDI internally works with shape (B, K, L) (features-first); the unified
format is (B, L, K) (time-first).  This adapter handles the permutation.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Make the CSDI package importable
_CSDI_ROOT = str(Path(__file__).resolve().parent.parent / "models" / "csdi")
if _CSDI_ROOT not in sys.path:
    sys.path.insert(0, _CSDI_ROOT)

if TYPE_CHECKING:
    from main_model import CSDI_base

from adapters.base import BaseAdapter, UnifiedBatch, UnifiedOutput


# ---------------------------------------------------------------------------
# Default diffusion config (matches config/base.yaml values)
# ---------------------------------------------------------------------------
_DEFAULT_CSDI_CONFIG: Dict[str, Any] = {
    "diffusion": {
        "layers": 4,
        "channels": 64,
        "nheads": 8,
        "diffusion_embedding_dim": 128,
        "beta_start": 0.0001,
        "beta_end": 0.5,
        "num_steps": 50,
        "schedule": "quad",
        "is_linear": False,
    },
    "model": {
        "is_unconditional": False,
        "timeemb": 128,
        "featureemb": 16,
        "target_strategy": "random",
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = base.copy()
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Bridge dataset: wraps a UnifiedDataset split into CSDI-format dicts.
# ---------------------------------------------------------------------------

class _CSDIBridgeDataset(Dataset):
    """Wraps items from a UnifiedDataset (or Subset) into CSDI dicts."""

    def __init__(self, unified_ds: Dataset):
        self.ds = unified_ds

    def __len__(self):
        return len(self.ds)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        item = self.ds[idx]
        obs = item["observed_data"]    # (L, K)
        mask = item["observed_mask"]   # (L, K)
        tgt = item["target_mask"]      # (L, K)
        tp = item["timepoints"]        # (L,)
        gt_mask = (mask - tgt).clamp(min=0.0)
        return {
            "observed_data": obs,
            "observed_mask": mask,
            "gt_mask": gt_mask,
            "timepoints": tp,
        }


def _csdi_collate(batch: list) -> Dict[str, torch.Tensor]:
    return {
        "observed_data": torch.stack([s["observed_data"] for s in batch]),
        "observed_mask": torch.stack([s["observed_mask"] for s in batch]),
        "gt_mask": torch.stack([s["gt_mask"] for s in batch]),
        "timepoints": torch.stack([s["timepoints"] for s in batch]),
    }


class CSDIAdapter(BaseAdapter):
    """Adapter for CSDI — delegates to ``models/csdi/utils.train``
    and ``models/csdi/utils.evaluate`` for native training/eval.

    CSDI's own Adam(lr=1e-3, wd=1e-6), MultiStepLR scheduler, and
    per-20-epoch validation strategy are preserved exactly.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.target_dim: int = config["target_dim"]
        self.csdi_config = _deep_merge(_DEFAULT_CSDI_CONFIG, config.get("csdi", {}))
        self.train_cfg = config.get("training", {})

    # -- model construction ----------------------------------------------

    @property
    def model(self) -> "CSDI_base":
        return super().model  # type: ignore[return-value]

    def build_model(self) -> nn.Module:
        from main_model import CSDI_base

        return CSDI_base(
            target_dim=self.target_dim,
            config=self.csdi_config,
            device=self.device,
        )

    # -- data wrapping ---------------------------------------------------

    def prepare_native_loaders(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
    ) -> Dict[str, Any]:
        bs = self.config.get("batch_size", 16)
        return {
            "train": DataLoader(
                _CSDIBridgeDataset(train_ds), batch_size=bs,
                shuffle=True, collate_fn=_csdi_collate, num_workers=0,
            ),
            "val": DataLoader(
                _CSDIBridgeDataset(val_ds), batch_size=bs,
                shuffle=False, collate_fn=_csdi_collate, num_workers=0,
            ),
            "test": DataLoader(
                _CSDIBridgeDataset(test_ds), batch_size=bs,
                shuffle=False, collate_fn=_csdi_collate, num_workers=0,
            ),
        }

    # -- native training (delegates to CSDI's own code) ------------------

    def native_train(self, loaders: Dict[str, Any], save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        from utils import train as csdi_train

        csdi_cfg = {"epochs": self.train_cfg.get("epochs", 200)}
        csdi_train(
            self.model,
            csdi_cfg,
            loaders["train"],
            valid_loader=loaders["val"],
            valid_epoch_interval=self.train_cfg.get("valid_epoch_interval", 20),
            foldername=save_dir,
        )

    # -- native evaluation -----------------------------------------------

    def native_evaluate(
        self,
        loaders: Dict[str, Any],
        n_samples: int,
        save_dir: str,
    ) -> Dict[str, float]:
        os.makedirs(save_dir, exist_ok=True)
        from utils import evaluate as csdi_evaluate

        csdi_evaluate(
            self.model, loaders["test"],
            nsample=n_samples, scaler=1, mean_scaler=0,
            foldername=save_dir,
        )
        # Read metrics that CSDI saved to disk
        result_path = os.path.join(save_dir, f"result_nsample{n_samples}.pk")
        if os.path.exists(result_path):
            with open(result_path, "rb") as f:
                rmse, mae, crps = pickle.load(f)
            return {"rmse": float(rmse), "mae": float(mae), "crps": float(crps)}
        return {}

    # -- single-batch predict (for diagnostics) --------------------------

    @torch.no_grad()
    def predict(self, batch: UnifiedBatch, n_samples: int = 1) -> UnifiedOutput:
        self.model.eval()
        batch = batch.to(self.device)

        gt_mask = (batch.observed_mask - batch.target_mask).clamp(min=0.0)

        observed_data = batch.observed_data.permute(0, 2, 1)   # (B, K, L)
        observed_mask = batch.observed_mask.permute(0, 2, 1)
        gt_mask_kl = gt_mask.permute(0, 2, 1)
        observed_tp = batch.timepoints

        cond_mask = gt_mask_kl
        target_mask = (observed_mask - cond_mask).clamp(min=0.0)

        side_info = self.model.get_side_info(observed_tp, cond_mask)
        samples_kl = self.model.impute(observed_data, cond_mask, side_info, n_samples)
        samples = samples_kl.permute(0, 1, 3, 2)   # (B, N, L, K)

        return UnifiedOutput(
            samples=samples,
            target_mask=target_mask.permute(0, 2, 1),
            observed_data=batch.observed_data,
            observed_mask=batch.observed_mask,
            timepoints=observed_tp,
        )
