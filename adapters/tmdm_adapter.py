"""
TMDM adapter — thin pipeline wrapper around the Transformer-Modulated
Diffusion Model for probabilistic time-series forecasting.

Data flow:
    UnifiedDataset → prepare_native_loaders → (batch_x, batch_y, x_mark, y_mark)
    native_train   → delegates to Exp_Main.train() (with overridden data)
    native_evaluate → delegates to Exp_Main.test()
    predict         → single-batch inference for diagnostics

TMDM trains the diffusion model + NS-Transformer conditional predictor
jointly with a single Adam optimiser, antithetic diffusion timestep
sampling, and early stopping on validation loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Make TMDM importable
_TMDM_ROOT = str(Path(__file__).resolve().parent.parent / "models" / "tmdm")
if _TMDM_ROOT not in sys.path:
    sys.path.insert(0, _TMDM_ROOT)

if TYPE_CHECKING:
    from model9_NS_transformer.diffusion_models.diffuMTS import Model as DiffuMTSModel
    from model9_NS_transformer.ns_models.ns_Transformer import Model as NSTransformerModel

from adapters.base import BaseAdapter, UnifiedBatch, UnifiedOutput


# ---------------------------------------------------------------------------
# Default TMDM args (mirrors runner9_NS_transformer.py argparse defaults)
# ---------------------------------------------------------------------------
_DEFAULT_TMDM_ARGS: Dict[str, Any] = {
    "enc_in": 7, "dec_in": 7, "c_out": 7,
    "seq_len": 96, "label_len": 48, "pred_len": 192,
    "d_model": 512, "n_heads": 8,
    "e_layers": 2, "d_layers": 1,
    "d_ff": 2048, "factor": 3,
    "dropout": 0.05, "activation": "gelu",
    "embed": "timeF", "freq": "h",
    "output_attention": False,
    "k_z": 1e-2, "k_cond": 1.0, "d_z": 8,
    "p_hidden_dims": [64, 64], "p_hidden_layers": 2,
    "diffusion_config_dir": str(
        Path(__file__).resolve().parent.parent
        / "models" / "tmdm" / "model9_NS_transformer" / "configs" / "toy_8gauss.yml"
    ),
    "timesteps": 1000,
    "CART_input_x_embed_dim": 32,
    "learning_rate": 1e-4,
    "learning_rate_Cond": 1e-4,
    "train_epochs": 200,
    "patience": 15,
    "use_gpu": False, "use_multi_gpu": False,
    "gpu": 0, "devices": "0",
    "use_amp": False,
    "features": "M",
    "checkpoints": "./checkpoints/",
    "moving_avg": 25, "distil": True,
    "MLP_diffusion_net": False,
    "mse_timestep": 0,
    "batch_size": 32, "test_batch_size": 8,
    "num_workers": 0,
    "target": "OT",
    "model_id": "pifmdm_tmdm",
}


def _dict_to_namespace(d: dict) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# Bridge dataset: wraps unified data into TMDM's 4-tuple format.
# ---------------------------------------------------------------------------

class _TMDMBridgeDataset(Dataset):
    """Yields ``(seq_x, seq_y, seq_x_mark, seq_y_mark)`` 4-tuples."""

    def __init__(self, unified_ds: Dataset, seq_len: int, label_len: int,
                 pred_len: int):
        self.ds = unified_ds
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.ds)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        item = self.ds[idx]
        data = item["observed_data"]   # (L_total, K)
        tp = item["timepoints"]        # (L_total,)
        sl, ll, pl = self.seq_len, self.label_len, self.pred_len

        seq_x = data[:sl]                                        # (sl, K)
        seq_y = data[sl - ll: sl + pl]                            # (ll+pl, K)
        # Time marks: use normalised position as a 1-dim feature
        if "time_marks" in item and item["time_marks"] is not None:
            tm = item["time_marks"]
            seq_x_mark = tm[:sl]
            seq_y_mark = tm[sl - ll: sl + pl]
        else:
            seq_x_mark = tp[:sl].unsqueeze(-1)                    # (sl, 1)
            seq_y_mark = tp[sl - ll: sl + pl].unsqueeze(-1)       # (ll+pl, 1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class TMDMAdapter(BaseAdapter):
    """Adapter for TMDM — delegates to ``Exp_Main.train()`` and
    ``Exp_Main.test()`` with an overridden ``_get_data`` hook so our
    unified DataLoaders are used while preserving TMDM's entire joint
    Adam optimiser, early-stopping, antithetic sampling, and diffusion
    reverse-sampling code unmodified."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.target_dim: int = config["target_dim"]

        merged = _DEFAULT_TMDM_ARGS.copy()
        merged.update(config.get("tmdm", {}))
        merged["enc_in"] = self.target_dim
        merged["dec_in"] = self.target_dim
        merged["c_out"] = self.target_dim
        merged["use_gpu"] = device.type == "cuda"
        merged["gpu"] = device.index or 0

        self.args = _dict_to_namespace(merged)
        self._cond_model: "NSTransformerModel | None" = None
        self._cond_model_train: "NSTransformerModel | None" = None

    # -- model construction -----------------------------------------------

    @property
    def model(self) -> "DiffuMTSModel":
        return super().model  # type: ignore[return-value]

    @property
    def cond_model(self) -> "NSTransformerModel":
        if self._cond_model is None:
            self._cond_model = self._build_cond_model()
        return self._cond_model  # type: ignore[return-value]

    def build_model(self) -> nn.Module:
        from model9_NS_transformer.diffusion_models import diffuMTS
        return diffuMTS.Model(self.args, self.device).float()

    def _build_cond_model(self) -> "NSTransformerModel":
        from model9_NS_transformer.ns_models import ns_Transformer
        return ns_Transformer.Model(self.args).float().to(self.device)

    # -- data wrapping ----------------------------------------------------

    def prepare_native_loaders(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
    ) -> Dict[str, Any]:
        sl = self.args.seq_len
        ll = self.args.label_len
        pl = self.args.pred_len
        bs = self.args.batch_size
        tbs = self.args.test_batch_size

        return {
            "train": (
                _TMDMBridgeDataset(train_ds, sl, ll, pl),
                DataLoader(
                    _TMDMBridgeDataset(train_ds, sl, ll, pl),
                    batch_size=bs, shuffle=True, drop_last=True, num_workers=0,
                ),
            ),
            "val": (
                _TMDMBridgeDataset(val_ds, sl, ll, pl),
                DataLoader(
                    _TMDMBridgeDataset(val_ds, sl, ll, pl),
                    batch_size=tbs, shuffle=False, drop_last=True, num_workers=0,
                ),
            ),
            "test": (
                _TMDMBridgeDataset(test_ds, sl, ll, pl),
                DataLoader(
                    _TMDMBridgeDataset(test_ds, sl, ll, pl),
                    batch_size=tbs, shuffle=False, drop_last=True, num_workers=0,
                ),
            ),
        }

    # -- native training (delegates to Exp_Main) -------------------------

    def native_train(self, loaders: Dict[str, Any], save_dir: str) -> None:
        """Create a patched Exp_Main that uses our DataLoaders, then
        call its ``train()`` method — preserving the joint Adam optimiser,
        early stopping, and diffusion training loop exactly."""
        os.makedirs(save_dir, exist_ok=True)

        from model9_NS_transformer.exp.exp_main import Exp_Main

        # We subclass at runtime so _get_data returns our loaders
        captured_loaders = loaders
        parent_device = self.device

        class _PatchedExp(Exp_Main):
            def _acquire_device(self):  # type: ignore[override]
                return parent_device

            def _get_data(self, flag):  # type: ignore[override]
                key = {"train": "train", "val": "val", "test": "test"}.get(flag, flag)
                if key in captured_loaders:
                    return captured_loaders[key]  # (dataset, dataloader)
                return captured_loaders["test"]

        # Override checkpoints dir to use save_dir
        self.args.checkpoints = save_dir
        setting = "pifmdm"
        exp = _PatchedExp(self.args)

        # Replace internal models with ours so weights are shared
        exp.model = self.model
        exp.cond_pred_model = self.cond_model

        exp.train(setting)

        # Sync back: exp.train loads best checkpoint into exp.model
        self._model = exp.model
        self._cond_model = exp.cond_pred_model

    # -- native evaluation ------------------------------------------------

    def native_evaluate(
        self,
        loaders: Dict[str, Any],
        n_samples: int,
        save_dir: str,
    ) -> Dict[str, float]:
        """Run TMDM's own test() logic with reverse diffusion sampling."""
        os.makedirs(save_dir, exist_ok=True)
        from model9_NS_transformer.diffusion_models.diffusion_utils import p_sample_loop

        model = self.model
        cond_model = self.cond_model
        model.eval()
        cond_model.eval()

        _, test_loader = loaders["test"]
        preds, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                _, y_0_hat, _, _ = cond_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                sample_list = []
                for _ in range(n_samples):
                    y_seq = p_sample_loop(
                        model, batch_x, batch_x_mark,
                        y_0_hat, y_0_hat,
                        model.num_timesteps,
                        model.alphas, model.one_minus_alphas_bar_sqrt,
                    )
                    sample = y_seq[-1][:, -self.args.pred_len:, :].cpu()
                    sample_list.append(sample)

                samples_np = torch.stack(sample_list, dim=1).numpy()  # (B, N, pl, K)
                true_np = batch_y[:, -self.args.pred_len:, :].cpu().numpy()
                preds.append(samples_np)
                trues.append(true_np)

        preds = np.concatenate(preds, axis=0)   # (N_total, n_samples, pl, K)
        trues = np.concatenate(trues, axis=0)   # (N_total, pl, K)

        mean_pred = preds.mean(axis=1)
        diff = mean_pred - trues
        mse = float((diff ** 2).mean())
        mae_val = float(np.abs(diff).mean())
        rmse_val = float(np.sqrt(mse))

        metrics = {"mse": mse, "mae": mae_val, "rmse": rmse_val, "n_samples": n_samples}
        np.save(os.path.join(save_dir, "pred.npy"), preds)
        np.save(os.path.join(save_dir, "true.npy"), trues)
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  MSE: {mse:.6f}  MAE: {mae_val:.6f}  RMSE: {rmse_val:.6f}")
        return metrics

    # -- single-batch predict (for diagnostics) ---------------------------

    @torch.no_grad()
    def predict(self, batch: UnifiedBatch, n_samples: int = 1) -> UnifiedOutput:
        from model9_NS_transformer.diffusion_models.diffusion_utils import p_sample_loop

        model = self.model
        cond_model = self.cond_model
        model.eval()
        cond_model.eval()
        batch = batch.to(self.device)

        data = batch.observed_data
        B, L_total, K = data.shape
        sl = self.args.seq_len
        pl = self.args.pred_len
        ll = self.args.label_len

        batch_x = data[:, :sl, :]
        batch_y = data[:, sl - ll: sl + pl, :]

        if batch.time_marks is not None:
            batch_x_mark = batch.time_marks[:, :sl, :]
            batch_y_mark = batch.time_marks[:, sl - ll: sl + pl, :]
        else:
            tp = batch.timepoints
            batch_x_mark = tp[:, :sl].unsqueeze(-1)
            batch_y_mark = tp[:, sl - ll: sl + pl].unsqueeze(-1)

        dec_inp = torch.zeros(B, pl, K, device=self.device)
        dec_inp = torch.cat([batch_y[:, :ll, :], dec_inp], dim=1)

        _, y_0_hat, _, _ = cond_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        all_samples = []
        for _ in range(n_samples):
            y_seq = p_sample_loop(
                model, batch_x, batch_x_mark,
                y_0_hat, y_0_hat,
                model.num_timesteps,
                model.alphas, model.one_minus_alphas_bar_sqrt,
            )
            sample = y_seq[-1][:, -pl:, :]
            all_samples.append(sample)

        samples = torch.stack(all_samples, dim=1)  # (B, N, pl, K)

        # Pad to full length: observed prefix + predicted suffix
        prefix = data[:, :sl, :].unsqueeze(1).expand(-1, n_samples, -1, -1)
        if L_total > sl + pl:
            pad = torch.zeros(B, n_samples, L_total - sl - pl, K, device=self.device)
            full_samples = torch.cat([prefix, samples, pad], dim=2)
        else:
            full_samples = torch.cat([prefix, samples], dim=2)

        return UnifiedOutput(
            samples=full_samples,
            target_mask=batch.target_mask,
            observed_data=batch.observed_data,
            observed_mask=batch.observed_mask,
            timepoints=batch.timepoints,
        )

    # -- checkpoints ------------------------------------------------------

    def save(self, path: str) -> None:
        base = Path(path)
        torch.save(self.model.state_dict(), str(base.with_suffix(".diffusion.pt")))
        torch.save(self.cond_model.state_dict(), str(base.with_suffix(".cond.pt")))

    def load(self, path: str, strict: bool = True) -> None:
        base = Path(path)
        diff_path = str(base.with_suffix(".diffusion.pt"))
        cond_path = str(base.with_suffix(".cond.pt"))
        if os.path.exists(diff_path):
            self.model.load_state_dict(
                torch.load(diff_path, map_location=self.device, weights_only=False), strict=strict)
        if os.path.exists(cond_path):
            self.cond_model.load_state_dict(
                torch.load(cond_path, map_location=self.device, weights_only=False), strict=strict)
