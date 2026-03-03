"""LSTM model for 6-channel time-series vehicle dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseEstimator
from src.utils.checkpoint import load_checkpoint, save_checkpoint


class LSTMModel(nn.Module):
    """LSTM-based regression model.

    Accepts input in (B, C, L) format (same as CNN1D / TimeSeriesDataset)
    and internally permutes to (B, L, C) for the LSTM layer.
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        n_conditions: int = 0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        fc_input = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.n_conditions = n_conditions
        self.fc = nn.Linear(fc_input + n_conditions, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # Input: (B, C, L) → permute to (B, L, C) for LSTM
        x = x.permute(0, 2, 1)
        self.lstm.flatten_parameters()
        output, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last = h_n[-1]
        last = self.dropout(last)
        if self.n_conditions > 0:
            if cond is None:
                cond = last.new_zeros(last.size(0), self.n_conditions)
            last = torch.cat([last, cond], dim=-1)
        return self.fc(last).squeeze(-1)


class LSTMEstimator(BaseEstimator):
    """Wrapper around LSTMModel for the BaseEstimator interface."""

    def __init__(self, **params: Any):
        self.params = params
        self.device = torch.device(params.get("device", "cpu"))
        _keys = (
            "input_size",
            "hidden_size",
            "num_layers",
            "dropout",
            "bidirectional",
            "n_conditions",
        )
        model_params = {k: v for k, v in params.items() if k in _keys}
        self.model = LSTMModel(**model_params).to(self.device)

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
                # (N, L) → (N, 1, L) single-channel fallback
                tensor = tensor.unsqueeze(1)
            preds = self.model(tensor)
        return preds.cpu().numpy()

    def save(self, path: str | Path) -> Path:
        return save_checkpoint(self.model.state_dict(), Path(path).with_suffix(".pt"))

    @classmethod
    def load(cls, path: str | Path, **params: Any) -> LSTMEstimator:
        instance = cls(**params)
        state_dict = load_checkpoint(path, map_location=instance.device)
        instance.model.load_state_dict(state_dict)
        return instance

    def get_params(self) -> dict[str, Any]:
        return self.params
