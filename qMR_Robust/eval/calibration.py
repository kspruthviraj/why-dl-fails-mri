"""
Expected Calibration Error (ECE) for regression and Centered Kernel Alignment (CKA).

ECE for regression measures whether the model's uncertainty estimates are
well-calibrated — i.e. when the model says it is X% confident, it is
correct approximately X% of the time.

CKA compares feature representations across domains to quantify whether
the model has learned domain-invariant features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# ECE for Regression
# ──────────────────────────────────────────────────────────────────────────────

def expected_calibration_error_regression(
    pred: Tensor,
    target: Tensor,
    uncertainty: Tensor,
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Compute ECE for regression using prediction intervals.

    Parameters
    ----------
    pred : (N,) or (N, D) predictions
    target : same shape as pred
    uncertainty : (N,) or (N, D) predicted uncertainty (std dev)
    n_bins : number of calibration bins

    Returns
    -------
    dict with "ece", "bin_accuracies", "bin_confidences"
    """
    if pred.dim() > 1:
        pred = pred.flatten()
        target = target.flatten()
        uncertainty = uncertainty.flatten()

    pred = pred.detach().cpu()
    target = target.detach().cpu()
    uncertainty = uncertainty.detach().cpu().clamp(min=1e-8)

    # Standardised residuals — should follow N(0,1) if calibrated
    z_scores = (target - pred) / uncertainty
    abs_z = z_scores.abs()

    bin_edges = torch.linspace(0, min(abs_z.max().item(), 4.0), n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (abs_z >= lo) & (abs_z < hi)
        count = mask.sum().item()
        if count == 0:
            continue

        # Empirical coverage: fraction of points within this z-band
        empirical_coverage = mask.float().mean().item()

        # Expected coverage from Gaussian
        from scipy.stats import norm
        expected_coverage = float(norm.cdf(hi.item()) - norm.cdf(lo.item()))

        bin_accs.append(empirical_coverage)
        bin_confs.append(expected_coverage)
        bin_counts.append(count)

    if not bin_counts:
        return {"ece": 0.0, "mean_abs_z": 0.0, "bin_details": []}

    total = sum(bin_counts)
    ece = sum(
        abs(a - c) * (n / total)
        for a, c, n in zip(bin_accs, bin_confs, bin_counts)
    )

    return {
        "ece": float(ece),
        "mean_abs_z": float(abs_z.mean().item()),
        "std_abs_z": float(abs_z.std().item()),
        "bin_details": [
            {"accuracy": a, "confidence": c, "count": n}
            for a, c, n in zip(bin_accs, bin_confs, bin_counts)
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Centered Kernel Alignment (CKA)
# ──────────────────────────────────────────────────────────────────────────────

def _center_gram(K: Tensor) -> Tensor:
    """Center a Gram matrix."""
    n = K.size(0)
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H


def cka(X: Tensor, Y: Tensor, kernel: str = "linear") -> float:
    """
    Centered Kernel Alignment between two feature matrices.

    Parameters
    ----------
    X : (N, D) feature matrix from domain A
    Y : (N, D) feature matrix from domain B
    kernel : "linear" or "rbf"

    Returns
    -------
    CKA score in [0, 1] — 1 means identical representations.
    """
    X = X.detach().cpu().float()
    Y = Y.detach().cpu().float()

    if kernel == "linear":
        K_X = X @ X.T
        K_Y = Y @ Y.T
    elif kernel == "rbf":
        K_X = _rbf_kernel(X)
        K_Y = _rbf_kernel(Y)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'")

    K_X = _center_gram(K_X)
    K_Y = _center_gram(K_Y)

    hsic_xy = (K_X * K_Y).sum()
    hsic_xx = (K_X * K_X).sum().sqrt()
    hsic_yy = (K_Y * K_Y).sum().sqrt()

    return float((hsic_xy / (hsic_xx * hsic_yy + 1e-8)).item())


def _rbf_kernel(X: Tensor, sigma: Optional[float] = None) -> Tensor:
    """Gaussian RBF kernel with median heuristic for sigma."""
    dists = torch.cdist(X, X) ** 2
    if sigma is None:
        sigma = dists[dists > 0].median().sqrt().item()
    return torch.exp(-dists / (2 * sigma**2 + 1e-8))


def pairwise_cka(
    features_by_domain: Dict[str, Tensor],
    kernel: str = "linear",
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise CKA between all domain feature sets.

    Parameters
    ----------
    features_by_domain : dict mapping domain name → (N, D) feature tensor

    Returns
    -------
    dict mapping (domain_a, domain_b) → CKA score
    """
    domain_names = sorted(features_by_domain.keys())
    results = {}
    for i, a in enumerate(domain_names):
        for j, b in enumerate(domain_names):
            if j <= i:
                continue
            X = features_by_domain[a]
            Y = features_by_domain[b]
            n = min(X.size(0), Y.size(0))
            results[(a, b)] = cka(X[:n], Y[:n], kernel)
    return results
