"""
ERM — Empirical Risk Minimization (baseline).

Standard regression loss with no domain-robustness penalty.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .base import AlgorithmBase


class ERM(AlgorithmBase):
    def __init__(self, backbone: nn.Module, device: torch.device):
        super().__init__(backbone, device)
        self.loss_fn = nn.L1Loss()  # MAE for qMRI regression

    def compute_loss(
        self, signal: Tensor, target: Tensor, domain: Tensor,
    ) -> Dict[str, Tensor]:
        signal = signal.to(self.device)
        target = target.to(self.device)

        pred = self.backbone(signal)
        pred_loss = self.loss_fn(pred, target)

        return {"loss": pred_loss, "pred_loss": pred_loss, "penalty": torch.tensor(0.0)}
