"""
VREx — Variance of Empirical Risks.

Penalises the variance of per-domain losses, encouraging the model to
perform equally well across all training domains.

Reference: Krueger et al., "Out-of-Distribution Generalization via
Risk Extrapolation (VREx)", ICML 2021.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .base import AlgorithmBase


class VREx(AlgorithmBase):
    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        penalty_weight: float = 1e3,
    ):
        super().__init__(backbone, device)
        self.penalty_weight = penalty_weight
        self.loss_fn = nn.L1Loss()

    def compute_loss(
        self, signal: Tensor, target: Tensor, domain: Tensor,
    ) -> Dict[str, Tensor]:
        signal = signal.to(self.device)
        target = target.to(self.device)
        domain = domain.to(self.device)

        pred = self.backbone(signal)

        unique = domain.unique()
        domain_losses = []
        for d in unique:
            mask = domain == d
            if mask.sum() < 1:
                continue
            dl = self.loss_fn(pred[mask], target[mask])
            domain_losses.append(dl)

        if len(domain_losses) < 2:
            pred_loss = self.loss_fn(pred, target)
            return {"loss": pred_loss, "pred_loss": pred_loss, "penalty": torch.tensor(0.0)}

        stacked = torch.stack(domain_losses)
        pred_loss = stacked.mean()
        penalty = stacked.var(unbiased=False)

        total = pred_loss + self.penalty_weight * penalty
        return {"loss": total, "pred_loss": pred_loss, "penalty": penalty}
