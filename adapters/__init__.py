"""
Adapter registry — single entry point for creating any model adapter.

Usage::

    from adapters import create_adapter

    adapter = create_adapter("csdi", config={"target_dim": 35, ...}, device=device)
    loss    = adapter.train_step(batch)
    output  = adapter.predict(batch, n_samples=10)
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from adapters.base import BaseAdapter, UnifiedBatch, UnifiedOutput  # re-export


# Lazy imports to avoid pulling in heavy deps until needed
_REGISTRY: Dict[str, str] = {
    "csdi": "adapters.csdi_adapter.CSDIAdapter",
    "cfmi": "adapters.cfmi_adapter.CFMIAdapter",
    "pidm": "adapters.pidm_adapter.PIDMAdapter",
    "tmdm": "adapters.tmdm_adapter.TMDMAdapter",
}


def _import_class(dotted_path: str):
    module_path, cls_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def create_adapter(
    model_name: str,
    config: Dict[str, Any],
    device: torch.device | str = "cpu",
) -> BaseAdapter:
    """Instantiate a model adapter by name.

    Parameters
    ----------
    model_name : str
        One of ``"csdi"``, ``"cfmi"``, ``"pidm"``, ``"tmdm"``.
    config : dict
        Must contain at least ``"target_dim"`` (number of features).
        Model-specific keys live under a nested dict keyed by the model
        name (e.g. ``config["csdi"] = {...}``).
    device : torch.device or str
        Device for tensors and model parameters.

    Returns
    -------
    BaseAdapter
    """
    name = model_name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_REGISTRY)}"
        )
    if isinstance(device, str):
        device = torch.device(device)

    cls = _import_class(_REGISTRY[name])
    return cls(config=config, device=device)


def list_models() -> list[str]:
    """Return the list of registered model names."""
    return list(_REGISTRY)


__all__ = [
    "BaseAdapter",
    "UnifiedBatch",
    "UnifiedOutput",
    "create_adapter",
    "list_models",
]
