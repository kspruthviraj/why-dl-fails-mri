"""
Algorithm base class and registry.

Every robustness algorithm wraps a backbone model and returns a combined loss
from `compute_loss(signal, target, domain_label)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AlgorithmBase(ABC):
    """Abstract base for all robustness algorithms."""

    def __init__(self, backbone: nn.Module, device: torch.device):
        self.backbone = backbone
        self.device = device

    @abstractmethod
    def compute_loss(
        self, signal: Tensor, target: Tensor, domain: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        signal : (B, C, L)
        target : (B, T)  regression targets
        domain : (B,)    integer domain labels

        Returns
        -------
        dict with at least {"loss": Tensor, "pred_loss": Tensor}.
        Auxiliary losses (e.g. "penalty") are included for logging.
        """

    def predict(self, signal: Tensor) -> Tensor:
        """Run forward pass without loss computation."""
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone(signal.to(self.device))


def build_algorithm(
    name: str,
    backbone: nn.Module,
    cfg: dict,
    device: torch.device,
    n_domains: int,
) -> AlgorithmBase:
    """Factory function — returns the algorithm wrapper for the given name."""
    from .erm import ERM
    from .coral import DeepCORAL
    from .dann import DANN
    from .irm import IRM
    from .vrex import VREx
    from .groupdro import GroupDRO

    algo_cfg = cfg.get("algorithm", {})
    mapping = {
        "erm": lambda: ERM(backbone, device),
        "erm_aug": lambda: ERM(backbone, device),
        "coral": lambda: DeepCORAL(backbone, device, float(algo_cfg.get("coral_penalty_weight", 1.0))),
        "dann": lambda: DANN(
            backbone, device,
            n_domains=n_domains,
            penalty_weight=float(algo_cfg.get("dann_penalty_weight", 1.0)),
            anneal_steps=int(algo_cfg.get("dann_anneal_steps", 500)),
        ),
        "irm": lambda: IRM(
            backbone, device,
            penalty_weight=float(algo_cfg.get("irm_penalty_weight", 1e3)),
            anneal_steps=int(algo_cfg.get("irm_anneal_steps", 500)),
        ),
        "vrex": lambda: VREx(backbone, device, float(algo_cfg.get("vrex_penalty_weight", 1e3))),
        "groupdro": lambda: GroupDRO(backbone, device, n_domains, float(algo_cfg.get("groupdro_eta", 1e-2))),
    }
    if name not in mapping:
        raise ValueError(f"Unknown algorithm '{name}'. Choose from {list(mapping)}")
    return mapping[name]()
