"""
PIDM adapter — thin pipeline wrapper around the Physics-Informed Diffusion Model.

Data flow:
    UnifiedDataset → prepare_native_loaders → (B, L, H, W) image DataLoader
    native_train   → replicates PIDM's main.py training loop faithfully:
                     Adam(lr=1e-4), grad_clip=1.0, cosine schedule, x0-prediction,
                     p2 loss weighting, EMA(0.99, start_iter=1000)
    native_evaluate → p_sample_loop from denoising_utils
    predict         → single-batch sampling for diagnostics

PIDM natively works with 2-D spatial fields ``(B, C, H, W)``.  We exploit
the fact that our spatial features *K* originate from an *H × W* grid and
**unflatten** them back, placing the *L* temporal steps on the channel axis:

    Unified ``(B, L, K)`` → ``(B, L, H, W)``  where ``K = H * W``

This way the UNet's spatial convolutions operate on the **actual** spatial
neighbours and each "channel" is one timestep.  The config must supply
``spatial_shape: [H, W]`` so the adapter knows how to reshape.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Make PIDM importable
_PIDM_ROOT = str(Path(__file__).resolve().parent.parent / "models" / "pidm")
if _PIDM_ROOT not in sys.path:
    sys.path.insert(0, _PIDM_ROOT)

if TYPE_CHECKING:
    from src.unet_model import Unet3D

from adapters.base import BaseAdapter, UnifiedBatch, UnifiedOutput


_DEFAULT_PIDM_CONFIG: Dict[str, Any] = {
    "diff_steps": 100,
    "unet": {
        "dim": 64,
        "dim_mults": [1, 2, 4],
        "attn_heads": 4,
        "attn_dim_head": 32,
        "padding_mode": "zeros",
    },
    # Physics loss weights (default: pure data-driven baseline)
    "c_data": 1.0,
    "c_residual": 0.0,
    "c_ineq": 0.0,
    "lambda_opt": 0.0,
    # Training (matching main.py defaults)
    "train_iterations": 300_000,
    "lr": 1e-4,
    "grad_clip": 1.0,
    "ema_decay": 0.99,
    "ema_start": 1000,
    "log_freq": 500,
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
# Bridge dataset: converts (L, K) time‐series into (L, H, W) spatial images.
# ---------------------------------------------------------------------------

class _PIDMBridgeDataset(Dataset):
    """Wraps unified items into ``(L, H, W)`` image tensors for the UNet.

    *L* timesteps become image channels;  *K* features are unflattened
    back to the original ``H × W`` spatial grid.
    """

    def __init__(self, unified_ds: Dataset, spatial_shape: tuple[int, int]):
        self.ds = unified_ds
        self.H, self.W = spatial_shape

    def __len__(self):
        return len(self.ds)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        item = self.ds[idx]
        data = item["observed_data"]  # (L, K)
        L, K = data.shape
        # → (L, H, W): temporal channels, actual spatial grid
        return data.view(L, self.H, self.W)  # (L, H, W)


# ---------------------------------------------------------------------------
# Simple residual wrapper so we can reuse DenoisingDiffusion.model_estimation_loss
# for Darcy-like flow without actual physics.
# ---------------------------------------------------------------------------

class _DummyResidual:
    """Just runs the UNet and returns zero residual so we can use
    ``DenoisingDiffusion.model_estimation_loss`` unchanged."""

    gov_eqs = "darcy"

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_residual(self, inputs, *, reduce="per-batch",
                         return_model_out=True, return_optimizer=False,
                         return_inequality=False, sample=False,
                         ddim_func=None):
        (noisy_in, t) = inputs[0]
        x0_pred = self.model(noisy_in, t)
        B = x0_pred.shape[0]
        out = {
            "model_out": x0_pred,
            "residual": torch.zeros(B, device=x0_pred.device),
        }
        if return_optimizer:
            out["optimizer"] = torch.zeros(B, device=x0_pred.device)
        if return_inequality:
            out["inequality"] = torch.zeros(B, device=x0_pred.device)
        return out


class PIDMAdapter(BaseAdapter):
    """Adapter for PIDM — faithfully replicates PIDM's ``main.py`` training
    loop: Adam(lr=1e-4), grad_clip=1.0, cosine diffusion schedule,
    x₀-prediction with p2 loss weighting, EMA(0.99)."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.target_dim: int = config["target_dim"]
        self.seq_len: int = config.get("seq_len", 1)
        self.pidm_config = _deep_merge(_DEFAULT_PIDM_CONFIG, config.get("pidm", {}))
        self.train_cfg = config.get("training", {})

        # Spatial shape — required so we can unflatten K → (H, W)
        sp = config.get("spatial_shape")
        if sp is None:
            # Fallback: assume square grid  (K must be a perfect square)
            import math
            side = int(math.isqrt(self.target_dim))
            if side * side != self.target_dim:
                raise ValueError(
                    f"PIDM requires spatial_shape in config (target_dim={self.target_dim} "
                    f"is not a perfect square)."
                )
            sp = [side, side]
        self.spatial_shape: tuple[int, int] = (int(sp[0]), int(sp[1]))
        H, W = self.spatial_shape
        if H * W != self.target_dim:
            raise ValueError(
                f"spatial_shape {self.spatial_shape} does not match "
                f"target_dim={self.target_dim} (H*W={H*W})."
            )

    # -- model construction -----------------------------------------------

    @property
    def model(self) -> "Unet3D":
        return super().model  # type: ignore[return-value]

    def build_model(self) -> nn.Module:
        from src.unet_model import Unet3D

        ucfg = self.pidm_config["unet"]
        # channels = L (temporal steps as image channels)
        # spatial dims = H × W (actual spatial grid)
        return Unet3D(
            dim=ucfg["dim"],
            out_dim=self.seq_len,
            dim_mults=tuple(ucfg["dim_mults"]),
            channels=self.seq_len,
            attn_heads=ucfg.get("attn_heads", 4),
            attn_dim_head=ucfg.get("attn_dim_head", 32),
            padding_mode=ucfg.get("padding_mode", "zeros"),
        )

    # -- data wrapping ----------------------------------------------------

    def prepare_native_loaders(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
    ) -> Dict[str, Any]:
        bs = self.config.get("batch_size", 16)
        sp = self.spatial_shape
        return {
            "train": DataLoader(
                _PIDMBridgeDataset(train_ds, sp), batch_size=bs,
                shuffle=True, num_workers=0,
            ),
            "val": DataLoader(
                _PIDMBridgeDataset(val_ds, sp), batch_size=bs,
                shuffle=False, num_workers=0,
            ),
            "test": DataLoader(
                _PIDMBridgeDataset(test_ds, sp), batch_size=bs,
                shuffle=False, num_workers=0,
            ),
        }

    # -- native training (replicates main.py's loop) ----------------------

    def native_train(self, loaders: Dict[str, Any], save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        from src.denoising_utils import DenoisingDiffusion, EMA
        from src.data_utils import cycle

        pcfg = self.pidm_config
        diff_steps = pcfg["diff_steps"]
        train_iters = self.train_cfg.get(
            "iterations", pcfg.get("train_iterations", 300_000)
        )
        lr = pcfg.get("lr", 1e-4)
        grad_clip = pcfg.get("grad_clip", 1.0)
        ema_decay = pcfg.get("ema_decay", 0.99)
        ema_start = pcfg.get("ema_start", 1000)
        log_freq = pcfg.get("log_freq", 500)

        model = self.model
        diffusion = DenoisingDiffusion(diff_steps, self.device, residual_grad_guidance=False)
        residual_func = _DummyResidual(model)
        ema = EMA(ema_decay)
        ema.register(model)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        dl_train = cycle(loaders["train"])
        dl_val = cycle(loaders["val"]) if loaders.get("val") else None

        pbar = tqdm(range(train_iters + 1), desc="PIDM training")
        for iteration in pbar:
            model.train()
            cur_batch = next(dl_train).to(self.device)

            loss, data_loss, *_ = diffusion.model_estimation_loss(
                cur_batch, residual_func=residual_func,
                c_data=pcfg["c_data"], c_residual=pcfg["c_residual"],
                c_ineq=pcfg["c_ineq"], lambda_opt=pcfg["lambda_opt"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if iteration > ema_start:
                ema.update(model)

            if iteration % log_freq == 0:
                pbar.set_description(f"loss: {loss.item():.3e}")

            # Periodic checkpoint
            if iteration > 0 and iteration % 50_000 == 0:
                ema.ema(model)
                torch.save({"model": model.state_dict()},
                           os.path.join(save_dir, f"checkpoint_{iteration}.pt"))
                ema.restore(model)

        # Final checkpoint with EMA weights
        ema.ema(model)
        torch.save({"model": model.state_dict()},
                   os.path.join(save_dir, "checkpoint_final.pt"))


    def native_evaluate(
        self,
        loaders: Dict[str, Any],
        n_samples: int,
        save_dir: str,
    ) -> Dict[str, float]:
        os.makedirs(save_dir, exist_ok=True)
        from src.denoising_utils import DenoisingDiffusion

        diff_steps = self.pidm_config["diff_steps"]
        diffusion = DenoisingDiffusion(diff_steps, self.device, residual_grad_guidance=False)
        model = self.model
        model.eval()

        H, W = self.spatial_shape

        # Sample unconditionally and compare with test data
        all_samples, all_targets = [], []
        for batch_img in loaders["test"]:
            batch_img = batch_img.to(self.device)  # (B, L, H, W)
            B, L = batch_img.shape[0], batch_img.shape[1]

            samples_batch = []
            for _ in range(n_samples):
                cur_x = torch.randn(B, L, H, W, device=self.device)
                for i in reversed(range(diff_steps)):
                    t = torch.full((B,), i, device=self.device, dtype=torch.long)
                    x0_pred = model(cur_x, t)
                    if x0_pred.dim() == 5:
                        x0_pred = x0_pred.squeeze(2)
                    from src.denoising_utils import extract
                    mean = (
                        extract(diffusion.diff_dict["posterior_mean_coef1"], t, cur_x) * x0_pred
                        + extract(diffusion.diff_dict["posterior_mean_coef2"], t, cur_x) * cur_x
                    )
                    if i > 0:
                        sigma = extract(diffusion.diff_dict["betas"], t, cur_x).sqrt()
                        cur_x = mean + sigma * torch.randn_like(cur_x)
                    else:
                        cur_x = mean
                # cur_x: (B, L, H, W) → (B, L, K)
                samples_batch.append(cur_x.view(B, L, -1).cpu())

            samples = torch.stack(samples_batch, dim=1)  # (B, N, L, K)
            targets = batch_img.view(B, L, -1).cpu()     # (B, L, K)
            all_samples.append(samples)
            all_targets.append(targets)

        all_samples = torch.cat(all_samples, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        median = all_samples.median(dim=1).values
        diff = median - all_targets
        rmse = (diff ** 2).mean().sqrt().item()
        mae_val = diff.abs().mean().item()

        metrics = {"rmse": rmse, "mae": mae_val, "n_samples": n_samples}
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        np.save(os.path.join(save_dir, "samples.npy"), all_samples.numpy())
        print(f"  RMSE: {rmse:.6f}  MAE: {mae_val:.6f}")
        return metrics


    @torch.no_grad()
    def predict(self, batch: UnifiedBatch, n_samples: int = 1) -> UnifiedOutput:
        from src.denoising_utils import DenoisingDiffusion, extract

        self.model.eval()
        batch = batch.to(self.device)

        B, L, K = batch.observed_data.shape
        H, W = self.spatial_shape
        # (B, L, K) → (B, L, H, W)
        data = batch.observed_data.view(B, L, H, W)

        diff_steps = self.pidm_config["diff_steps"]
        diffusion = DenoisingDiffusion(diff_steps, self.device, residual_grad_guidance=False)

        all_samples = []
        for _ in range(n_samples):
            cur_x = torch.randn_like(data)
            for i in reversed(range(diff_steps)):
                t = torch.full((B,), i, device=self.device, dtype=torch.long)
                x0_pred = self.model(cur_x, t)
                if x0_pred.dim() == 5:
                    x0_pred = x0_pred.squeeze(2)
                mean = (
                    extract(diffusion.diff_dict["posterior_mean_coef1"], t, cur_x) * x0_pred
                    + extract(diffusion.diff_dict["posterior_mean_coef2"], t, cur_x) * cur_x
                )
                if i > 0:
                    sigma = extract(diffusion.diff_dict["betas"], t, cur_x).sqrt()
                    cur_x = mean + sigma * torch.randn_like(cur_x)
                else:
                    cur_x = mean
            # (B, L, H, W) → (B, L, K)
            all_samples.append(cur_x.view(B, L, K))

        samples = torch.stack(all_samples, dim=1)  # (B, N, L, K)

        return UnifiedOutput(
            samples=samples,
            target_mask=batch.target_mask,
            observed_data=batch.observed_data,
            observed_mask=batch.observed_mask,
            timepoints=batch.timepoints,
        )


    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"], strict=strict)
        else:
            self.model.load_state_dict(ckpt, strict=strict)
