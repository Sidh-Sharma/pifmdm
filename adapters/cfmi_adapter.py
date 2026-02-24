"""
CFMI adapter — thin pipeline wrapper around the Conditional-Flow-Matching
Imputation codebase.

Data flow:
    UnifiedDataset  → prepare_native_loaders → (X, M, idx) tuples
    native_train    → PyTorch Lightning Trainer.fit()
    native_evaluate → model.sample_imputations() + CFMI metrics
    predict         → single-batch sample_imputations for diagnostics

CFMI expects boolean masks (True = observed) and works with ``(B, L, K)``
for time-series data.
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Make CFMI importable
_CFMI_ROOT = str(Path(__file__).resolve().parent.parent / "models" / "cfmi")
if _CFMI_ROOT not in sys.path:
    sys.path.insert(0, _CFMI_ROOT)

if TYPE_CHECKING:
    from imp_cfm.models.cfm import ConditionalFlowMatching

try:
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
except ImportError:
    from lightning.pytorch.loggers import TensorBoardLogger  # type: ignore[no-redef]

from adapters.base import BaseAdapter, UnifiedBatch, UnifiedOutput


_DEFAULT_CFMI_CONFIG: Dict[str, Any] = {
    "vector_field": {
        "type": "tashiro",
        "channels": 64,
        "emb_time_dim": 128,
        "feature_id_emb_dim": 16,
        "embedding_dim": 128,
        "nheads": 8,
        "layers": 4,
        "is_linear": False,
    },
    "time_embedding": {
        "embedding_dim": 128,
        "frequency_multiplier": 50,
    },
    "flow_matcher": {"sigma": 0.0},
    "mask_generator": {
        "type": "uniform_random",
        "target_probability": None,
    },
    "solver": {"type": "euler", "num_t": 101},
    "opt_lr": 1e-3,
    "opt_weight_decay": 1e-6,
    # Lightning Trainer defaults
    "trainer": {
        "max_epochs": 200,
        "accelerator": "auto",
        "devices": 1,
        "gradient_clip_val": None,
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



# Bridge dataset: yields (X, M, idx) tuples that CFMI's training_step expects
class _CFMIBridgeDataset(Dataset):
    """Wraps unified items into ``(X, M, idx)`` tuples for CFMI."""

    def __init__(self, unified_ds: Dataset):
        self.ds = unified_ds

    def __len__(self):
        return len(self.ds)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        item = self.ds[idx]
        X = item["observed_data"]              # (L, K) float
        M = item["observed_mask"].bool()       # (L, K) bool — True=observed
        return X, M, torch.tensor(idx)


class CFMIAdapter(BaseAdapter):
    """Adapter for CFMI — delegates to PyTorch Lightning for training
    and ``sample_imputations`` for evaluation, preserving the model's
    own AdamW optimiser and LR scheduler."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.target_dim: int = config["target_dim"]
        self.seq_len: int = config.get("seq_len", 0)
        self.cfmi_config = _deep_merge(_DEFAULT_CFMI_CONFIG, config.get("cfmi", {}))
        self.train_cfg = config.get("training", {})

    def _build_mask_generator(self):
        from imp_cfm.utils.mask_generator import (
            UniformRandomMaskGenerator,
            HistoricalMaskGenerator,
        )
        mg_cfg = self.cfmi_config["mask_generator"]
        if mg_cfg["type"] == "uniform_random":
            return UniformRandomMaskGenerator(target_probability=mg_cfg.get("target_probability"))
        elif mg_cfg["type"] == "historical":
            return HistoricalMaskGenerator(
                strategy=mg_cfg.get("strategy", "mix"),
                mix_probability=mg_cfg.get("mix_probability", 0.5),
            )
        raise ValueError(f"Unknown mask_generator type: {mg_cfg['type']}")

    def _build_vector_field_net(self):
        vf_cfg = self.cfmi_config["vector_field"]
        te_cfg = self.cfmi_config["time_embedding"]
        from imp_cfm.models.neural_nets import TimeEmbeddingNet

        time_emb = TimeEmbeddingNet(
            embedding_dim=te_cfg["embedding_dim"],
            frequency_multiplier=te_cfg.get("frequency_multiplier", 50),
        )

        if vf_cfg["type"] == "tashiro":
            from imp_cfm.models.tashiro_timeseries_transformer_net import TashiroTransformerNet
            return TashiroTransformerNet(
                channels=vf_cfg["channels"],
                emb_time_dim=vf_cfg["emb_time_dim"],
                target_dim=self.target_dim,
                feature_id_emb_dim=vf_cfg["feature_id_emb_dim"],
                embedding_dim=vf_cfg["embedding_dim"],
                nheads=vf_cfg["nheads"],
                layers=vf_cfg["layers"],
                is_linear=vf_cfg.get("is_linear", False),
                time_embedding_net=time_emb,
            )
        elif vf_cfg["type"] == "residual_fc":
            from imp_cfm.models.neural_nets import ResidualFCNetwork, VelocityNet
            flat_dim = self.seq_len * self.target_dim if self.seq_len else self.target_dim
            input_dim = flat_dim * 3
            net = ResidualFCNetwork(
                input_dim=input_dim,
                output_dim=flat_dim,
                num_residual_blocks=vf_cfg.get("num_residual_blocks", 6),
                residual_block_dim=vf_cfg.get("residual_block_dim", 512),
            )
            return VelocityNet(net=net, time_embedding_net=time_emb)
        raise ValueError(f"Unknown vector_field type: {vf_cfg['type']}")

    def _get_solver(self):
        from imp_cfm.utils.solver import (
            FlowSolver, EulerWithArgs, MidpointWithArgs,
            DormandPrince45WithArgs, RungeKutta4WithArgs,
        )
        mapping = {
            "euler": EulerWithArgs,
            "midpoint": MidpointWithArgs,
            "dopri45": DormandPrince45WithArgs,
            "rk4": RungeKutta4WithArgs,
        }
        name = self.cfmi_config["solver"]["type"]
        if name not in mapping:
            raise ValueError(f"Unknown solver: {name}")
        return partial(FlowSolver, ode_solver=mapping[name](), atol=1e-5, rtol=1e-5)

    # -- model construction ----------------------------------------------

    @property
    def model(self) -> "ConditionalFlowMatching":
        return super().model  # type: ignore[return-value]

    def build_model(self) -> nn.Module:
        import torchcfm  # type: ignore[import-not-found]
        from imp_cfm.models.cfm import ConditionalFlowMatching

        fm_cfg = self.cfmi_config["flow_matcher"]
        return ConditionalFlowMatching(
            dim=self.target_dim,
            mask_generator=self._build_mask_generator(),
            flow_matcher=torchcfm.ConditionalFlowMatcher(sigma=fm_cfg.get("sigma", 0.0)),
            vector_field_net=self._build_vector_field_net(),
            opt_lr=self.cfmi_config.get("opt_lr", 1e-3),
            opt_weight_decay=self.cfmi_config.get("opt_weight_decay", 1e-6),
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
                _CFMIBridgeDataset(train_ds), batch_size=bs,
                shuffle=True, num_workers=0, pin_memory=True,
            ),
            "val": DataLoader(
                _CFMIBridgeDataset(val_ds), batch_size=bs,
                shuffle=False, num_workers=0, pin_memory=True,
            ),
            "test": DataLoader(
                _CFMIBridgeDataset(test_ds), batch_size=bs,
                shuffle=False, num_workers=0, pin_memory=True,
            ),
        }

    def native_train(self, loaders: Dict[str, Any], save_dir: str) -> None:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint

        os.makedirs(save_dir, exist_ok=True)
        tr_cfg = self.cfmi_config.get("trainer", {})

        checkpoint_cb = ModelCheckpoint(
            dirpath=save_dir,
            save_top_k=1,
            save_last=True,
            monitor="loss/val",
        )

        trainer = pl.Trainer(
            max_epochs=tr_cfg.get("max_epochs", self.train_cfg.get("epochs", 200)),
            accelerator=tr_cfg.get("accelerator", "auto"),
            devices=tr_cfg.get("devices", 1),
            gradient_clip_val=tr_cfg.get("gradient_clip_val"),
            default_root_dir=save_dir,
            callbacks=[checkpoint_cb],
            logger=TensorBoardLogger(save_dir, name="cfmi"),
        )
        trainer.fit(
            self.model,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["val"],
        )

    
    def native_evaluate(
        self,
        loaders: Dict[str, Any],
        n_samples: int,
        save_dir: str,
    ) -> Dict[str, float]:
        os.makedirs(save_dir, exist_ok=True)
        from imp_cfm.utils.imputation_metrics import rmse, mae, crps

        solver = self._get_solver()
        num_t = self.cfmi_config["solver"]["num_t"]
        self.model.eval()
        self.model.to(self.device)

        all_imps, all_X, all_M = [], [], []
        with torch.inference_mode():
            for X, M, idx in loaders["test"]:
                X, M = X.to(self.device), M.to(self.device)
                imps = self.model.sample_imputations(
                    X=X, M=M, Y=None,
                    num_t=num_t, num_imps=n_samples, solver=solver,
                ).detach().cpu()
                all_imps.append(imps)
                all_X.append(X.cpu())
                all_M.append(M.cpu())

        X_all = torch.cat(all_X, dim=0).numpy()
        M_all = torch.cat(all_M, dim=0).numpy()
        imps_all = torch.cat(all_imps, dim=0).numpy()

        metrics: Dict[str, float] = {}
        # target locations = missing in M
        target_M = ~M_all
        if target_M.any():
            metrics["rmse"] = float(rmse(imps_all, X_all, target_M))
            metrics["mae"] = float(mae(imps_all, X_all, target_M))
            flat_X = X_all.reshape(X_all.shape[0], -1)
            flat_imp = imps_all.reshape(imps_all.shape[0], imps_all.shape[1], -1)
            flat_M = target_M.reshape(target_M.shape[0], -1)
            metrics["crps"] = float(crps(flat_imp, flat_X, flat_M).mean())

        np.save(os.path.join(save_dir, "imputations.npy"), imps_all)
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        return metrics

    @torch.no_grad()
    def predict(self, batch: UnifiedBatch, n_samples: int = 1) -> UnifiedOutput:
        self.model.eval()
        batch = batch.to(self.device)

        X = batch.observed_data
        M = batch.observed_mask.bool()

        solver = self._get_solver()
        num_t = self.cfmi_config["solver"]["num_t"]

        X_imp = self.model.sample_imputations(
            X=X, M=M, Y=None,
            num_imps=n_samples, num_t=num_t, solver=solver,
        )

        target_mask = (~M).float()
        tp = torch.arange(X.shape[1], device=X.device).float().unsqueeze(0).expand(X.shape[0], -1)

        return UnifiedOutput(
            samples=X_imp,
            target_mask=target_mask,
            observed_data=X,
            observed_mask=M.float(),
            timepoints=tp,
        )


    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, strict: bool = True) -> None:
        from imp_cfm.models.cfm import ConditionalFlowMatching
        # Try loading as a Lightning checkpoint first
        try:
            loaded = ConditionalFlowMatching.load_from_checkpoint(
                checkpoint_path=path,
                dim=self.target_dim,
                mask_generator=self._build_mask_generator(),
                flow_matcher=self.model.flow_matcher,
                vector_field_net=self.model.vector_field_net,
                opt_lr=self.cfmi_config.get("opt_lr", 1e-3),
                opt_weight_decay=self.cfmi_config.get("opt_weight_decay", 1e-6),
            )
            self._model = loaded.to(self.device)
        except Exception:
            state = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state, strict=strict)
