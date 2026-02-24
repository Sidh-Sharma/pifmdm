"""
Advection dataset loader for the pifmdm unified pipeline.

The raw data produced by ``advection_dataset_generator.py`` has shape
``(T, Nx, Ny)`` — *T* timesteps on an *Nx × Ny* spatial grid.

This module converts it to the unified format ``(N, L, K)`` by:

1. **Temporal subsampling** — keep every ``temporal_stride``-th timestep.
2. **Spatial subsampling** — keep every ``spatial_stride``-th grid point
   along each spatial axis.  Stride 1 = full resolution; stride 2 =
   every other point; etc.  No interpolation — just index slicing.
3. **Flattening** — the 2-D spatial grid is flattened into *K* features,
   where ``K = (Nx // spatial_stride) * (Ny // spatial_stride)``.
4. **Windowing** — a sliding window of length ``window_size`` is moved
   along the time axis with step ``window_stride``, producing *N* samples.

All subsampling / windowing parameters are **selectable** at load time.
No hard-coded downsampling is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from datasets.base import UnifiedDataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_advection(
    path: str | Path = "datasets/synthetic/advection_data_2d.npy",
    *,
    window_size: int = 64,
    window_stride: int = 1,
    spatial_stride: int = 1,
    temporal_stride: int = 1,
    normalise: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Load raw advection data and return ``(N, L, K)`` tensor.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npy`` file with shape ``(T, Nx, Ny)``.
    window_size : int
        Number of (possibly subsampled) timesteps per sample.
    window_stride : int
        Step between consecutive windows (along time axis).
    spatial_stride : int
        Stride for spatial subsampling along **both** x and y axes.
        ``1`` keeps the full grid; ``2`` keeps every other point, etc.
    temporal_stride : int
        Stride for temporal subsampling.  ``1`` keeps every timestep.
    normalise : bool
        If ``True`` shift-scale data to zero-mean unit-variance and return
        the stats in a dict.

    Returns
    -------
    data : torch.Tensor
        Shape ``(N, L, K)`` — *N* sliding-window samples, *L* timesteps
        per window (= ``window_size``), *K* flattened spatial points.
    stats : dict
        ``{"mean": float, "std": float, "spatial_stride": int,
          "temporal_stride": int, "nx_sub": int, "ny_sub": int,
          "nx_orig": int, "ny_orig": int, "T_orig": int}``.
    """
    raw = np.load(str(path))                              # (T, Nx, Ny)
    assert raw.ndim == 3, f"Expected 3-D array (T, Nx, Ny), got shape {raw.shape}"
    T_orig, Nx_orig, Ny_orig = raw.shape

    # --- temporal subsampling ---
    data = raw[::temporal_stride]                          # (T', Nx, Ny)

    # --- spatial subsampling (stride-based, no interpolation) ---
    data = data[:, ::spatial_stride, ::spatial_stride]     # (T', Nx', Ny')
    T, Nx_sub, Ny_sub = data.shape
    K = Nx_sub * Ny_sub

    # --- flatten spatial dims ---
    data = data.reshape(T, K)                              # (T', K)

    # --- normalise ---
    stats: Dict[str, float] = {
        "spatial_stride": spatial_stride,
        "temporal_stride": temporal_stride,
        "nx_sub": Nx_sub,
        "ny_sub": Ny_sub,
        "nx_orig": Nx_orig,
        "ny_orig": Ny_orig,
        "T_orig": T_orig,
    }
    if normalise:
        mean = float(data.mean())
        std = float(data.std())
        data = (data - mean) / (std + 1e-8)
        stats["mean"] = mean
        stats["std"] = std
    else:
        stats["mean"] = 0.0
        stats["std"] = 1.0

    # --- sliding windows ---
    data_t = torch.from_numpy(data).float()                # (T', K)
    n_windows = (T - window_size) // window_stride + 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough timesteps ({T}) for window_size={window_size} "
            f"with temporal_stride={temporal_stride}.  "
            f"Reduce window_size or temporal_stride."
        )

    indices = [
        range(i, i + window_size)
        for i in range(0, n_windows * window_stride, window_stride)
    ]
    windows = torch.stack([data_t[list(idx)] for idx in indices])  # (N, L, K)

    stats["n_windows"] = int(windows.shape[0])
    stats["window_size"] = window_size
    stats["window_stride"] = window_stride
    return windows, stats


def make_advection_dataset(
    path: str | Path = "datasets/synthetic/advection_data_2d.npy",
    *,
    window_size: int = 64,
    window_stride: int = 1,
    spatial_stride: int = 1,
    temporal_stride: int = 1,
    normalise: bool = True,
    miss_ratio: float = 0.1,
    mask_strategy: str = "random",
) -> Tuple[UnifiedDataset, Dict[str, float]]:
    """Convenience wrapper: load data → ``UnifiedDataset``.

    Returns ``(dataset, stats)`` where *stats* contains normalisation
    parameters and grid metadata.
    """
    data, stats = load_advection(
        path,
        window_size=window_size,
        window_stride=window_stride,
        spatial_stride=spatial_stride,
        temporal_stride=temporal_stride,
        normalise=normalise,
    )
    ds = UnifiedDataset(data, miss_ratio=miss_ratio, mask_strategy=mask_strategy)
    return ds, stats
