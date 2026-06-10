"""
DeepCORAL — Correlation Alignment.

Penalises the difference between second-order feature statistics
(covariance matrices) across domains.

Reference: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep
Domain Adaptation", ECCV 2016.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .base import AlgorithmBase


def _coral_penalty(feat: Tensor, domains: Tensor) -> Tensor:
    """Compute pairwise CORAL loss between domain feature covariances."""
    unique = domains.unique(sorted=True)
    if len(unique) < 2:
        return torch.tensor(0.0, device=feat.device)

    covs = []
    for d in unique:
        mask = domains == d
        f_d = feat[mask]
        if f_d.size(0) < 2:
            continue
        f_d = f_d - f_d.mean(dim=0, keepdim=True)
        cov_d = (f_d.T @ f_d) / (f_d.size(0) - 1)
        covs.append(cov_d)

    if len(covs) < 2:
        return torch.tensor(0.0, device=feat.device)

    loss = torch.tensor(0.0, device=feat.device)
    n_pairs = 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            diff = covs[i] - covs[j]
            loss = loss + (diff ** 2).sum()
            n_pairs += 1
    return loss / max(n_pairs, 1)


class DeepCORAL(AlgorithmBase):
    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        penalty_weight: float = 1.0,
    ):
        super().__init__(backbone, device)
        self.loss_fn = nn.L1Loss()
        self.penalty_weight = penalty_weight

    def compute_loss(
        self, signal: Tensor, target: Tensor, domain: Tensor,
    ) -> Dict[str, Tensor]:
        signal = signal.to(self.device)
        target = target.to(self.device)
        domain = domain.to(self.device)

        feat = self.backbone.encode(signal)
        pred = self.backbone.head(feat)

        pred_loss = self.loss_fn(pred, target)
        penalty = _coral_penalty(feat, domain)

        total = pred_loss + self.penalty_weight * penalty
        return {"loss": total, "pred_loss": pred_loss, "penalty": penalty}
