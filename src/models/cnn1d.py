"""1D CNN model for time-series speed patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseEstimator
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for regression on speed sequences."""

    def __init__(
        self,
        in_channels: int = 6,
        seq_len: int = 60,
        n_filters: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        n_conditions: int = 0,
    ):
        super().__init__()
        if n_filters is None:
            n_filters = [32, 64, 128]

        layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out in n_filters:
            layers.extend(
                [
                    nn.Conv1d(ch_in, ch_out, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                ]
            )
            ch_in = ch_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.n_conditions = n_conditions
        self.fc = nn.Linear(n_filters[-1] + n_conditions, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        if self.n_conditions > 0:
            if cond is None:
                cond = x.new_zeros(x.size(0), self.n_conditions)
            x = torch.cat([x, cond], dim=-1)
        return self.fc(x).squeeze(-1)


class CNN1DEstimator(BaseEstimator):
    """Wrapper around CNN1D for the BaseEstimator interface."""

    def __init__(self, **params: Any):
        self.params = params
        self.device = torch.device(_resolve_device(params.get("device", "cpu")))
        _keys = (
            "in_channels",
            "seq_len",
            "n_filters",
            "kernel_size",
            "dropout",
            "n_conditions",
        )
        model_params = {k: v for k, v in params.items() if k in _keys}
        self.model = CNN1D(**model_params).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"status": "use trainer_dl for full training"}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(1)
            preds = self.model(tensor)
        return preds.cpu().numpy()

    def save(self, path: str | Path) -> Path:
        return save_checkpoint(self.model.state_dict(), Path(path).with_suffix(".pt"))

    @classmethod
    def load(cls, path: str | Path, **params: Any) -> CNN1DEstimator:
        instance = cls(**params)
        state_dict = load_checkpoint(path, map_location=instance.device)
        instance.model.load_state_dict(state_dict)
        return instance

    def get_params(self) -> dict[str, Any]:
        return self.params
