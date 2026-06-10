"""
IRM — Invariant Risk Minimization.

Penalises the gradient of the loss w.r.t. a dummy classifier, encouraging
features that support an invariant predictor across environments.

Reference: Arjovsky et al., "Invariant Risk Minimization", arXiv 2019.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .base import AlgorithmBase


def _irm_penalty(pred: Tensor, target: Tensor, device: torch.device) -> Tensor:
    """Approximate IRMv1 penalty: ||∇_w L(w·f, y)||² with w=1."""
    scale = torch.ones(1, device=device, requires_grad=True)
    loss = nn.L1Loss()(pred * scale, target)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


class IRM(AlgorithmBase):
    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        penalty_weight: float = 1e3,
        anneal_steps: int = 500,
    ):
        super().__init__(backbone, device)
        self.penalty_weight = penalty_weight
        self.anneal_steps = anneal_steps
        self._step = 0
        self.loss_fn = nn.L1Loss()

    def compute_loss(
        self, signal: Tensor, target: Tensor, domain: Tensor,
    ) -> Dict[str, Tensor]:
        signal = signal.to(self.device)
        target = target.to(self.device)

        pred = self.backbone(signal)
        pred_loss = self.loss_fn(pred, target)

        penalty = torch.tensor(0.0, device=self.device)
        unique = domain.unique()
        for d in unique:
            mask = domain.to(self.device) == d
            if mask.sum() < 2:
                continue
            penalty = penalty + _irm_penalty(pred[mask], target[mask], self.device)
        penalty = penalty / max(len(unique), 1)

        # Anneal penalty weight
        w = self.penalty_weight
        if self._step < self.anneal_steps:
            w = w * (self._step / self.anneal_steps)
        self._step += 1

        total = pred_loss + w * penalty
        return {"loss": total, "pred_loss": pred_loss, "penalty": penalty}
