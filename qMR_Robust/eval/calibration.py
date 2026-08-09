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
    """Compute calibration error from empirical prediction-interval coverage.

    Each bin corresponds to a nominal central Gaussian interval.  The previous
    implementation grouped residuals by their observed z-score and compared
    unconditional frequencies with Gaussian mass, which was not a calibration
    curve.
    """
    if pred.dim() > 1:
        pred = pred.flatten()
        target = target.flatten()
        uncertainty = uncertainty.flatten()

    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    uncertainty = uncertainty.detach().cpu().float().clamp(min=1e-8)
    abs_error = (target - pred).abs()

    nominal = torch.linspace(0.05, 0.95, n_bins)
    from scipy.stats import norm

    empirical = []
    details = []
    for level in nominal.tolist():
        z = float(norm.ppf((1.0 + level) / 2.0))
        covered = (abs_error <= z * uncertainty).float().mean().item()
        empirical.append(covered)
        details.append({
            "nominal_coverage": float(level),
            "empirical_coverage": float(covered),
        })

    ece = float(np.mean(np.abs(np.asarray(empirical) - nominal.numpy())))
    standardized = abs_error / uncertainty
    return {
        "ece": ece,
        "mean_abs_z": float(standardized.mean().item()),
        "std_abs_z": float(standardized.std().item()),
        "bin_details": details,
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
