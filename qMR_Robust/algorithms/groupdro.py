"""
GroupDRO — Group Distributionally Robust Optimisation.

Up-weights the worst-performing domain during training using an exponential
re-weighting scheme.

Reference: Sagawa et al., "Distributionally Robust Neural Networks for
Group Shifts", ICLR 2020.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .base import AlgorithmBase


class GroupDRO(AlgorithmBase):
    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        n_domains: int,
        eta: float = 1e-2,
    ):
        super().__init__(backbone, device)
        self.loss_fn = nn.L1Loss(reduction="none")
        self.eta = eta
        self.n_domains = n_domains
        self.register_buffer = {}

        self.group_weights = torch.ones(n_domains, device=device) / n_domains
        self.group_weights.requires_grad_(False)

    def compute_loss(
        self, signal: Tensor, target: Tensor, domain: Tensor,
    ) -> Dict[str, Tensor]:
        signal = signal.to(self.device)
        target = target.to(self.device)
        domain = domain.to(self.device)

        pred = self.backbone(signal)
        per_sample_loss = self.loss_fn(pred, target).mean(dim=-1)

        unique = domain.unique(sorted=True)
        present_losses = []
        present_indices = []
        for d in unique:
            d_int = int(d.item()) if isinstance(d, Tensor) else int(d)
            mask = domain == d
            if mask.sum() > 0:
                present_indices.append(d_int)
                present_losses.append(per_sample_loss[mask].mean())

        if not present_losses:
            pred_loss = per_sample_loss.mean()
            return {
                "loss": pred_loss,
                "pred_loss": pred_loss,
                "penalty": torch.tensor(0.0, device=self.device),
            }

        losses = torch.stack(present_losses)
        indices = torch.tensor(present_indices, device=self.device, dtype=torch.long)

        # Update and renormalize only groups represented in this batch. This
        # prevents absent environments from silently diluting the objective.
        with torch.no_grad():
            self.group_weights[indices] *= torch.exp(
                torch.tensor(self.eta, device=self.device) * losses.detach()
            )
            self.group_weights /= self.group_weights.sum()

        batch_weights = self.group_weights[indices]
        batch_weights = batch_weights / batch_weights.sum().clamp_min(1e-8)
        weighted_loss = (batch_weights * losses).sum()

        pred_loss = per_sample_loss.mean()
        return {
            "loss": weighted_loss,
            "pred_loss": pred_loss,
            "penalty": losses.max() - losses.min(),
        }
