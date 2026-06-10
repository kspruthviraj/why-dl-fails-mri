"""
DANN — Domain-Adversarial Neural Network.

Learns domain-invariant features via a Gradient Reversal Layer (GRL) and a
domain classifier head attached to the backbone features.

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks",
JMLR 2016.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from .base import AlgorithmBase


# ── Gradient Reversal Layer ───────────────────────────────────────────────────

class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return _GradientReversalFunction.apply(x, self.alpha)


# ── DANN ──────────────────────────────────────────────────────────────────────

class _DomainClassifier(nn.Module):
    """Two-layer MLP domain classifier."""

    def __init__(self, in_dim: int, n_domains: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DANN(AlgorithmBase):
    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        n_domains: int,
        penalty_weight: float = 1.0,
        anneal_steps: int = 500,
    ):
        super().__init__(backbone, device)
        self.n_domains = n_domains
        self.penalty_weight = penalty_weight
        self.anneal_steps = anneal_steps
        self._step = 0

        self.grl = GradientReversalLayer()
        self.domain_classifier = _DomainClassifier(
            backbone.feature_dim, n_domains,
        ).to(device)

        self.pred_loss_fn = nn.L1Loss()
        self.domain_loss_fn = nn.CrossEntropyLoss()

    def _alpha(self) -> float:
        """Annealing coefficient following the original DANN schedule."""
        p = self._step / max(self.anneal_steps, 1)
        return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0

    def compute_loss(
        self, signal: Tensor, target: Tensor, domain: Tensor,
    ) -> Dict[str, Tensor]:
        signal = signal.to(self.device)
        target = target.to(self.device)
        domain = domain.to(self.device)

        self.grl.alpha = self._alpha()

        feat = self.backbone.encode(signal)
        pred = self.backbone.head(feat)

        pred_loss = self.pred_loss_fn(pred, target)

        reversed_feat = self.grl(feat)
        domain_pred = self.domain_classifier(reversed_feat)
        domain_loss = self.domain_loss_fn(domain_pred, domain)

        total = pred_loss + self.penalty_weight * domain_loss

        self._step += 1
        return {
            "loss": total,
            "pred_loss": pred_loss,
            "penalty": domain_loss,
            "alpha": torch.tensor(self.grl.alpha),
        }

    def parameters(self):
        """Return all trainable parameters including the domain classifier."""
        return list(self.backbone.parameters()) + list(self.domain_classifier.parameters())
