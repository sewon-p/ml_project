"""Model save/load utilities supporting PyTorch (.pt/.pth) and pickle (.pkl) formats."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

_TORCH_EXTS = {".pt", ".pth"}
_PICKLE_EXTS = {".pkl", ".pickle"}


def save_checkpoint(obj: Any, path: str | Path) -> Path:
    """Save an object to disk. Format is chosen by file extension.

    - ``.pt`` / ``.pth``: uses ``torch.save``  (state dicts, full models, etc.)
    - ``.pkl`` / ``.pickle``: uses ``pickle.dump``  (sklearn, xgboost, lightgbm, etc.)

    Returns the resolved *Path* for convenience.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in _TORCH_EXTS:
        torch.save(obj, path)
    elif ext in _PICKLE_EXTS:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unsupported extension '{ext}'. Use one of {_TORCH_EXTS | _PICKLE_EXTS}.")
    return path.resolve()


def load_checkpoint(path: str | Path, **torch_load_kwargs: Any) -> Any:
    """Load an object from disk. Format is inferred from extension.

    Extra keyword arguments are forwarded to ``torch.load`` (e.g. ``map_location``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ext = path.suffix.lower()

    if ext in _TORCH_EXTS:
        kwargs: dict[str, Any] = {"weights_only": False}
        kwargs.update(torch_load_kwargs)
        return torch.load(path, **kwargs)
    elif ext in _PICKLE_EXTS:
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301
    else:
        raise ValueError(f"Unsupported extension '{ext}'. Use one of {_TORCH_EXTS | _PICKLE_EXTS}.")
