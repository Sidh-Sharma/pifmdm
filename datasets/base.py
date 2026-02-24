"""
Unified dataset abstraction for the pifmdm project.

All project-level datasets should produce :class:`UnifiedBatch` objects so
that every adapter receives data in exactly the same format regardless of
the underlying data source.

Included:
    * ``UnifiedDataset``  — a thin PyTorch Dataset wrapper around
      pre-loaded tensors.
    * ``collate_unified``— collate function that assembles individual
      samples into a ``UnifiedBatch``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from adapters.base import UnifiedBatch


class UnifiedDataset(Dataset):
    """A generic dataset that yields dicts convertible to :class:`UnifiedBatch`.

    Parameters
    ----------
    data : torch.Tensor
        Full data tensor of shape ``(N, L, K)`` — N samples, L time steps,
        K features.
    masks : dict, optional
        ``{"observed_mask": (N, L, K), "target_mask": (N, L, K)}``.
        If not given, ``observed_mask`` is all-ones and ``target_mask`` is
        derived from the ``miss_ratio`` and ``mask_strategy`` parameters.
    timepoints : torch.Tensor, optional
        ``(N, L)`` or ``(L,)`` time indices.  Defaults to ``arange(L)/L``.
    time_marks : torch.Tensor, optional
        ``(N, L, D)`` time feature encoding (needed by TMDM).
    miss_ratio : float
        Fraction of observed points to mask as prediction targets.
    mask_strategy : str
        ``"random"`` — uniform random masking per sample.
        ``"future"`` — mask the last ``ceil(miss_ratio * L)`` time steps
        (forecasting setup).
    """

    def __init__(
        self,
        data: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        timepoints: Optional[torch.Tensor] = None,
        time_marks: Optional[torch.Tensor] = None,
        miss_ratio: float = 0.0,
        mask_strategy: str = "random",
    ):
        super().__init__()
        assert data.ndim == 3, f"Expected (N, L, K), got {data.shape}"
        self.data = data.float()
        N, L, K = data.shape

        if masks is not None:
            self.observed_mask = masks["observed_mask"].float()
            self.target_mask = masks.get(
                "target_mask",
                torch.zeros(N, L, K),
            ).float()
        else:
            self.observed_mask = torch.ones(N, L, K)
            self.target_mask = self._make_target_mask(N, L, K, miss_ratio, mask_strategy)

        if timepoints is not None:
            self.timepoints = timepoints.float()
            if self.timepoints.ndim == 1:
                self.timepoints = self.timepoints.unsqueeze(0).expand(N, -1)
        else:
            tp = torch.arange(L, dtype=torch.float32) / max(L - 1, 1)
            self.timepoints = tp.unsqueeze(0).expand(N, -1)

        self.time_marks = time_marks.float() if time_marks is not None else None

    # ------------------------------------------------------------------
    @staticmethod
    def _make_target_mask(
        N: int, L: int, K: int, ratio: float, strategy: str
    ) -> torch.Tensor:
        if ratio <= 0:
            return torch.zeros(N, L, K)
        if strategy == "future":
            mask = torch.zeros(N, L, K)
            cut = int(np.ceil(ratio * L))
            mask[:, -cut:, :] = 1.0
            return mask
        # random per sample
        mask = torch.zeros(N, L, K)
        for i in range(N):
            flat = torch.rand(L * K)
            mask[i] = (flat < ratio).float().view(L, K)
        return mask

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "observed_data": self.data[idx],          # (L, K)
            "observed_mask": self.observed_mask[idx],  # (L, K)
            "target_mask": self.target_mask[idx],      # (L, K)
            "timepoints": self.timepoints[idx],        # (L,)
        }
        if self.time_marks is not None:
            item["time_marks"] = self.time_marks[idx]  # (L, D)
        return item


def collate_unified(batch: list) -> UnifiedBatch:
    """Stack a list of per-sample dicts into a single :class:`UnifiedBatch`."""
    out = {
        "observed_data": torch.stack([s["observed_data"] for s in batch]),
        "observed_mask": torch.stack([s["observed_mask"] for s in batch]),
        "target_mask": torch.stack([s["target_mask"] for s in batch]),
        "timepoints": torch.stack([s["timepoints"] for s in batch]),
    }
    time_marks = None
    if "time_marks" in batch[0]:
        time_marks = torch.stack([s["time_marks"] for s in batch])

    return UnifiedBatch(
        observed_data=out["observed_data"],
        observed_mask=out["observed_mask"],
        target_mask=out["target_mask"],
        timepoints=out["timepoints"],
        time_marks=time_marks,
    )


def make_dataloader(
    dataset: UnifiedDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Convenience wrapper that uses :func:`collate_unified`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_unified,
        **kwargs,
    )
