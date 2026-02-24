"""
Base adapter interface for all diffusion models.

Every model adapter is a **thin pipeline wrapper** that:
1. Converts unified data → whatever format the model's code expects.
2. Delegates training to the model's **own** training loop (preserving the
   original optimiser, schedule, grad-clipping settings, etc.).
3. Delegates evaluation to the model's **own** evaluation code.
4. Wraps the native output back into a common format for our diagnostics.

GPU support comes from each model's existing device-handling code; the
adapter merely forwards the requested device.

Unified batch format (produced by datasets.base.UnifiedDataset):
    observed_data : (B, L, K) float  — observed values (0 where missing)
    observed_mask : (B, L, K) float  — 1 where observed, 0 where missing
    target_mask   : (B, L, K) float  — 1 where the model must predict
    timepoints    : (B, L)   float  — normalised time indices [0, 1]
    features      : (B, K)   float  — optional feature ids (default: arange)
    time_marks    : (B, L, D) float — optional time feature encoding (for TMDM)

Unified output format (returned by adapter.predict()):
    samples       : (B, N, L, K) float — N generated samples
    target_mask   : (B, L, K)    float — mask of predicted positions
    observed_data : (B, L, K)    float — original observed data
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Unified data containers
@dataclass
class UnifiedBatch:
    """Container produced by the project-level dataset and consumed by adapters."""

    observed_data: torch.Tensor   # (B, L, K)
    observed_mask: torch.Tensor   # (B, L, K)
    target_mask: torch.Tensor     # (B, L, K)
    timepoints: torch.Tensor      # (B, L)
    features: Optional[torch.Tensor] = None   # (B, K)  or (K,)
    time_marks: Optional[torch.Tensor] = None # (B, L, D)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.observed_data.shape[0]

    @property
    def seq_len(self) -> int:
        return self.observed_data.shape[1]

    @property
    def num_features(self) -> int:
        return self.observed_data.shape[2]

    def to(self, device: torch.device) -> "UnifiedBatch":
        """Move all tensors to *device* and return self."""
        self.observed_data = self.observed_data.to(device)
        self.observed_mask = self.observed_mask.to(device)
        self.target_mask = self.target_mask.to(device)
        self.timepoints = self.timepoints.to(device)
        if self.features is not None:
            self.features = self.features.to(device)
        if self.time_marks is not None:
            self.time_marks = self.time_marks.to(device)
        return self


@dataclass
class UnifiedOutput:
    """Container returned by every adapter's ``predict`` method."""

    samples: torch.Tensor         # (B, N, L, K) — N generated samples
    target_mask: torch.Tensor     # (B, L, K)
    observed_data: torch.Tensor   # (B, L, K)
    observed_mask: torch.Tensor   # (B, L, K)
    timepoints: torch.Tensor      # (B, L)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Abstract adapter
class BaseAdapter(abc.ABC):
    """Unified wrapper around any diffusion model.

    Sub-classes must implement the abstract methods below.  The adapter is a
    thin pipeline: it wraps data in, delegates to the model's **native**
    training / evaluation code, and wraps output back out.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self._model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Abstract interface — every adapter MUST implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """Instantiate the underlying model from ``self.config`` and return it."""
        ...

    @abc.abstractmethod
    def prepare_native_loaders(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
    ) -> Dict[str, Any]:
        """Convert unified dataset splits into DataLoaders in the format the
        model's **native** training code expects.

        Returns a dict with at least ``"train"``, ``"val"``, ``"test"`` keys.
        The values may be DataLoaders, LightningDataModules, or any object
        the model's native train/eval code accepts.
        """
        ...

    @abc.abstractmethod
    def native_train(self, loaders: Dict[str, Any], save_dir: str) -> None:
        """Run the model's **own** full training loop.

        The adapter must not impose its own optimiser, scheduler, or
        gradient-clipping — those come from the original model code.
        """
        ...

    @abc.abstractmethod
    def native_evaluate(
        self,
        loaders: Dict[str, Any],
        n_samples: int,
        save_dir: str,
    ) -> Dict[str, float]:
        """Run the model's **own** evaluation / sampling code.

        Returns a dict of scalar metrics (e.g. ``{"rmse": ..., "mae": ...}``).
        """
        ...

    @abc.abstractmethod
    def predict(self, batch: UnifiedBatch, n_samples: int = 1) -> UnifiedOutput:
        """Single-batch inference returning :class:`UnifiedOutput`.

        Used by the diagnostics pipeline.  Implementations should handle
        moving the batch to ``self.device`` and converting the native
        output back to ``(B, N, L, K)`` unified format.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers (override where needed)
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self.build_model()
            self._model.to(self.device)
        return self._model

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state, strict=strict)
