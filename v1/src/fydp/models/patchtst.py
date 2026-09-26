"""PatchTST-style IMU encoder + classifier (fixed version of notebooks/002_model.ipynb).

Notebook bugs fixed:
- `feedforward` referenced undefined `self.head` -> added Linear head + dropout.
- Method renamed to `forward` so the module is directly callable.
- TransformerEncoder saw patch tokens with NO positional information, so with
  mean pooling the encoder was permutation-invariant in time. Since FOG
  forecasting is inherently order-dependent (pre-onset dynamics, tremor
  build-up), added learned positional embeddings (`posenc=True`).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(
        self,
        n_channels: int = 6,
        patch: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        n_class: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 4,
        max_tokens: int = 256,
        posenc: bool = True,
    ):
        super().__init__()
        self.patch = patch
        self.posenc = posenc
        self.proj = nn.Linear(n_channels * patch, d_model)
        if posenc:
            self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.register_parameter("pos", None)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_class)

    def forward(self, x):
        # x: (B, T, C) -> patch into (B, n, C*patch)
        _b, t, c = x.shape
        n = t // self.patch
        x = x[:, : n * self.patch, :]
        x = x.reshape(x.shape[0], n, c * self.patch)
        h = self.proj(x)
        if self.posenc:
            if h.size(1) > self.pos.size(1):
                raise ValueError(
                    f"sequence has {h.size(1)} patch tokens > max_tokens={self.pos.size(1)}"
                )
            h = h + self.pos[:, : h.size(1), :]
        h = self.enc(h).mean(dim=1)
        h = self.dropout(h)
        return self.head(h)
