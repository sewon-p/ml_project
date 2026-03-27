"""DeepSets-style multi-probe density estimation models.

Shared encoder processes each probe independently.
Pooling across probes: mean, attention, or CF-score weighted.
MLP head predicts density.
Works for any N (number of probes) without architecture change.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

PoolingType = Literal["mean", "attention", "cf_score"]


class LSTMEncoder(nn.Module):
    """LSTM encoder that outputs embeddings (no prediction head)."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
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
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last = h_n[-1]
        return self.dropout(last)  # (B, embed_dim)


class CNN1DEncoder(nn.Module):
    """CNN-1D encoder that outputs embeddings (no prediction head)."""

    def __init__(
        self,
        in_channels: int = 6,
        n_filters: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
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
        self.embed_dim = n_filters[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.dropout(x)  # (B, embed_dim)


class AttentionPooling(nn.Module):
    """Learned attention weights over probes."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, N, D)
        scores = self.score_net(embeddings)  # (B, N, 1)
        weights = torch.softmax(scores, dim=1)  # (B, N, 1)
        pooled = (weights * embeddings).sum(dim=1)  # (B, D)
        return pooled


class CFScorePooling(nn.Module):
    """Car-following intensity score as attention weight.

    Computes CF intensity from raw timeseries (ax_std, brake_ratio, speed_cv)
    and uses it to weight probe embeddings. Interpretable: probes in
    car-following state get higher weight.

    Expects raw input x alongside embeddings.
    """

    def __init__(self, embed_dim: int, temperature: float = 1.0):
        super().__init__()
        # Learnable scaling on top of handcrafted CF scores
        self.temperature = temperature
        self.scale = nn.Linear(3, 1)  # 3 CF indicators -> 1 score

    def compute_cf_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract car-following indicators from raw (B*N, C, T) timeseries.

        Channels: [VX, VY, AX, AY, speed, brake]
        """
        ax = x[:, 2, :]  # (B*N, T)  AX channel
        speed = x[:, 4, :]  # (B*N, T)
        brake = x[:, 5, :]  # (B*N, T)

        ax_std = ax.std(dim=1)  # (B*N,)
        brake_ratio = (brake > 0).float().mean(dim=1)  # (B*N,)
        speed_mean = speed.mean(dim=1).clamp(min=1e-6)
        speed_cv = speed.std(dim=1) / speed_mean  # (B*N,)

        return torch.stack([ax_std, brake_ratio, speed_cv], dim=-1)  # (B*N, 3)

    def forward(
        self, embeddings: torch.Tensor, x_raw: torch.Tensor
    ) -> torch.Tensor:
        """
        embeddings: (B, N, D)
        x_raw: (B, N, C, T)
        """
        B, N, C, T = x_raw.shape
        cf_feats = self.compute_cf_features(
            x_raw.view(B * N, C, T)
        )  # (B*N, 3)
        cf_feats = cf_feats.view(B, N, 3)

        scores = self.scale(cf_feats) / self.temperature  # (B, N, 1)
        weights = torch.softmax(scores, dim=1)  # (B, N, 1)
        pooled = (weights * embeddings).sum(dim=1)  # (B, D)
        return pooled


class MultiProbeModel(nn.Module):
    """DeepSets: shared encoder -> pooling -> MLP head.

    Pooling methods:
      - "mean": simple mean across probes
      - "attention": learned attention weights
      - "cf_score": car-following intensity weighted (interpretable)

    Accepts (B, N, C, T) for multi-probe or (B, C, T) for single-probe.
    """

    def __init__(
        self,
        encoder_type: str = "lstm",
        n_conditions: int = 0,
        pooling: PoolingType = "mean",
        **encoder_kwargs,
    ):
        super().__init__()
        if encoder_type == "lstm":
            self.encoder = LSTMEncoder(**encoder_kwargs)
        elif encoder_type == "cnn1d":
            self.encoder = CNN1DEncoder(**encoder_kwargs)
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

        embed_dim = self.encoder.embed_dim
        self.n_conditions = n_conditions
        self.pooling_type = pooling

        if pooling == "attention":
            self.pooling = AttentionPooling(embed_dim)
        elif pooling == "cf_score":
            self.pooling = CFScorePooling(embed_dim)
        # mean pooling has no parameters

        self.head = nn.Sequential(
            nn.Linear(embed_dim + n_conditions, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim == 4:
            # Multi-probe: (B, N, C, T)
            B, N, C, T = x.shape
            embeddings = self.encoder(x.view(B * N, C, T))  # (B*N, embed_dim)
            embeddings = embeddings.view(B, N, -1)  # (B, N, embed_dim)

            if self.pooling_type == "mean":
                pooled = embeddings.mean(dim=1)
            elif self.pooling_type == "attention":
                pooled = self.pooling(embeddings)
            elif self.pooling_type == "cf_score":
                pooled = self.pooling(embeddings, x)
            else:
                pooled = embeddings.mean(dim=1)
        else:
            # Single probe: (B, C, T)
            pooled = self.encoder(x)

        if self.n_conditions > 0:
            if cond is None:
                cond = pooled.new_zeros(pooled.size(0), self.n_conditions)
            pooled = torch.cat([pooled, cond], dim=-1)

        return self.head(pooled).squeeze(-1)
