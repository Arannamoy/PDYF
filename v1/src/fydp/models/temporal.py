"""Lightweight GPU-friendly temporal IMU classifier baselines."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        pad = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=pad,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TemporalCNN(nn.Module):
    """Dilated TCN with pooled multi-resolution temporal summaries."""

    def __init__(self, n_channels: int = 6, width: int = 32,
                 dropout: float = 0.15, n_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=9, stride=2, padding=4,
                      bias=False),
            nn.BatchNorm1d(width), nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualTemporalBlock(width, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input from data pipeline is (batch, time, channels).
        x = x.transpose(1, 2)
        h = self.blocks(self.stem(x))
        pooled = torch.cat([self.pool_avg(h).squeeze(-1),
                            self.pool_max(h).squeeze(-1)], dim=1)
        return self.head(pooled).squeeze(-1)


class FeatureMLP(nn.Module):
    def __init__(self, n_features: int, width: int = 64,
                 dropout: float = 0.25, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_features
        for _ in range(depth):
            layers.extend([nn.Linear(d, width), nn.LayerNorm(width),
                           nn.GELU(), nn.Dropout(dropout)])
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
