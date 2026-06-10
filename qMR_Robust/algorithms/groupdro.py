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

        unique = domain.unique()
        domain_losses = torch.zeros(self.n_domains, device=self.device)
        for d in unique:
            d_int = int(d.item()) if isinstance(d, Tensor) else int(d)
            mask = domain == d
            if mask.sum() > 0:
                domain_losses[d_int] = per_sample_loss[mask].mean()

        # Exponential reweighting
        with torch.no_grad():
            loss_vals = domain_losses.detach()
            self.group_weights = self.group_weights * torch.exp(
                torch.tensor(self.eta, device=self.device) * loss_vals
            )
            self.group_weights = self.group_weights / self.group_weights.sum()

        weighted_loss = (self.group_weights * domain_losses).sum()

        pred_loss = per_sample_loss.mean()
        return {
            "loss": weighted_loss,
            "pred_loss": pred_loss,
            "penalty": domain_losses.max() - domain_losses.min(),
        }
