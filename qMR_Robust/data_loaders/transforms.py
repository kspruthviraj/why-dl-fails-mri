"""
Shared utilities for all data loaders: transforms, collation, and type aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Sample:
    """Canonical sample yielded by every qMR-Robust Dataset."""

    signal: Tensor          # (C, L) — C=2 for real/imag, L = time/frequency points
    target: Tensor          # regression target: (2,) for MRF [T1,T2]; (K,) for MRS [conc…]
    domain_label: int       # integer encoding of the physical domain
    domain_name: str        # human-readable, e.g. "siemens_site03_3T"
    vendor: str             # "siemens" | "philips" | "ge"
    site: str               # site / scanner identifier
    field_strength: float   # 1.5 | 3.0 | 7.0


# ── Standard transforms ──────────────────────────────────────────────────────

class ToTwoChannel:
    """Convert a complex-valued 1-D array to a (2, L) real tensor (real, imag)."""

    def __call__(self, x: np.ndarray) -> Tensor:
        if np.iscomplexobj(x):
            return torch.stack(
                [torch.from_numpy(x.real.copy()), torch.from_numpy(x.imag.copy())]
            ).float()
        return torch.from_numpy(x).float().unsqueeze(0).repeat(2, 1)


class NormalizeSignal:
    """Peak-magnitude normalization (per-signal)."""

    def __call__(self, x: Tensor) -> Tensor:
        peak = x.abs().max()
        return x / peak.clamp(min=1e-8)


class RandomPhaseShift:
    """Apply a random global phase rotation in the complex plane."""

    def __init__(self, max_deg: float = 10.0):
        self.max_deg = max_deg

    def __call__(self, x: Tensor) -> Tensor:
        theta = (torch.rand(1) * 2 - 1) * self.max_deg * (torch.pi / 180)
        real = x[0] * theta.cos() - x[1] * theta.sin()
        imag = x[0] * theta.sin() + x[1] * theta.cos()
        return torch.stack([real, imag])


class AddGaussianNoise:
    """Add complex Gaussian noise at a random SNR."""

    def __init__(self, snr_range: Tuple[float, float] = (10.0, 100.0)):
        self.snr_lo, self.snr_hi = snr_range

    def __call__(self, x: Tensor) -> Tensor:
        snr = torch.empty(1).uniform_(self.snr_lo, self.snr_hi).item()
        sig_power = (x**2).mean()
        noise_power = sig_power / (snr**2) if snr > 0 else torch.tensor(0.0)
        noise = torch.randn_like(x) * noise_power.sqrt()
        return x + noise


class ComposeTransforms:
    """Sequentially apply a list of transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def build_eval_transform() -> ComposeTransforms:
    return ComposeTransforms([ToTwoChannel(), NormalizeSignal()])


def build_train_transform(cfg: dict) -> ComposeTransforms:
    aug = cfg.get("augmentation", {})
    transforms: list = [ToTwoChannel()]
    if aug.get("enabled", True):
        transforms.append(
            RandomPhaseShift(max_deg=aug.get("phase_shift_range", [-10, 10])[1])
        )
        transforms.append(
            AddGaussianNoise(snr_range=tuple(aug.get("noise_snr_range", [10, 100])))
        )
    transforms.append(NormalizeSignal())
    return ComposeTransforms(transforms)


def build_domain_map(vendors: List[str], sites_per_vendor: int) -> Dict[str, int]:
    """Map human-readable domain names to consecutive integer IDs."""
    mapping: Dict[str, int] = {}
    idx = 0
    for v in vendors:
        for s in range(sites_per_vendor):
            mapping[f"{v}_site{s:02d}"] = idx
            idx += 1
    return mapping
