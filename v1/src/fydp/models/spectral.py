"""Spectral branch + late-fusion model for cross-subject FOG forecasting.

A logistic probe on window-level freeze-band/locomotion-band power ratios
reaches test AUPRC 0.567 while the raw-waveform PatchTST baseline stalls at
0.50: spectral ratios are dimensionless and transfer across subjects, whereas
raw amplitudes carry subject-specific gain. This module adds a small MLP on
those features and fuses its logits with the PatchTST temporal branch
(naive late fusion of logits, per the Project Overview pipeline).
"""
from __future__ import annotations

import torch.nn as nn

from fydp.models.patchtst import PatchTST


class SpectralMLP(nn.Module):
    def __init__(self, n_feat: int = 18, hidden: int = 64, n_class: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )

    def forward(self, f):
        return self.net(f)


class SpectralOnlyMLP(SpectralMLP):
    """SpectralMLP with a two-argument forward so the trainer can call every
    branch uniformly as model(xb, fb). Uses fb, ignores xb."""

    def forward(self, x, f=None):  # type: ignore[override]
        return self.net(x if f is None else f)


class FusionModel(nn.Module):
    """PatchTST (temporal) + SpectralMLP (spectral) with averaged logits."""

    def __init__(self, spec_dim: int = 18, spec_hidden: int = 64,
                 spec_dropout: float = 0.2, **patch_kwargs):
        super().__init__()
        self.temporal = PatchTST(**patch_kwargs)
        self.spectral = SpectralMLP(spec_dim, spec_hidden,
                                    patch_kwargs.get("n_class", 2), spec_dropout)

    def forward(self, x, f):
        return 0.5 * (self.temporal(x) + self.spectral(f))
