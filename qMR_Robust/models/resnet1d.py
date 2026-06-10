"""
ResNet-1D — Temporal Convolutional backbone for 1-D MR signals.

Architecture:  Stem → 4 residual stages → global average pool → linear head.
The `features_before_head` property exposes the penultimate feature vector so
that algorithm wrappers (CORAL, DANN, IRM) can attach auxiliary heads.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        n_blocks: List[int] | None = None,
        hidden_dim: int = 256,
        output_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_blocks = n_blocks or [2, 2, 2, 2]

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        ch = base_channels
        self.stages = nn.ModuleList()
        for i, nb in enumerate(n_blocks):
            stride = 1 if i == 0 else 2
            out_ch = ch * (1 if i == 0 else 2)
            blocks = []
            for j in range(nb):
                blocks.append(_BasicBlock1D(ch if j == 0 else out_ch, out_ch, stride if j == 0 else 1))
            self.stages.append(nn.Sequential(*blocks))
            ch = out_ch

        self.norm = nn.BatchNorm1d(ch)
        self._feature_dim = ch
        self.head = nn.Sequential(
            nn.Linear(ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate feature vector (B, D)."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = F.relu(self.norm(x))
        return x.mean(dim=-1)  # global average pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return regression predictions (B, output_dim)."""
        return self.head(self.encode(x))
