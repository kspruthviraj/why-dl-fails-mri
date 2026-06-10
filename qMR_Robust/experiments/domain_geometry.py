"""
Domain Geometry Analysis — OOD Error vs. Feature Distance.

The core claim: OOD error is a linear function of representation distance.
If we can prove OOD_Error = f(CKA_distance), then we have identified the
geometric mechanism of failure and given future researchers a concrete target.

This module:
  1. Extracts features from each domain using the trained model.
  2. Computes pairwise CKA (or MMD) between domain feature distributions.
  3. Plots OOD Error (y) vs. Domain Feature Distance (x).
  4. Fits a linear model and reports R².
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from ..eval.calibration import cka, pairwise_cka
from ..eval.metrics import mae

logger = logging.getLogger(__name__)


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Extract features and predictions from a model for a given domain.

    Returns (features, predictions, targets).
    """
    model.eval()
    features, preds, targets = [], [], []

    with torch.no_grad():
        for i, (signal, target, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            signal = signal.to(device)
            feat = model.encode(signal)
            pred = model.head(feat)
            features.append(feat.cpu())
            preds.append(pred.cpu())
            targets.append(target)

    return torch.cat(features), torch.cat(preds), torch.cat(targets)


def compute_domain_distances(
    features_by_domain: Dict[str, Tensor],
    method: str = "cka",
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise domain distances using CKA or MMD.

    Parameters
    ----------
    features_by_domain : dict mapping domain name → (N, D) feature tensor
    method : "cka" or "mmd"

    Returns
    -------
    dict mapping (domain_a, domain_b) → distance
    """
    if method == "cka":
        raw = pairwise_cka(features_by_domain, kernel="linear")
        return {(a, b): 1.0 - v for (a, b), v in raw.items()}
    elif method == "mmd":
        return _pairwise_mmd(features_by_domain)
    else:
        raise ValueError(f"Unknown method '{method}'")


def _pairwise_mmd(features_by_domain: Dict[str, Tensor]) -> Dict[Tuple[str, str], float]:
    """Compute pairwise MMD (Maximum Mean Discrepancy) between domains."""
    domain_names = sorted(features_by_domain.keys())
    results = {}

    for i, a in enumerate(domain_names):
        for j, b in enumerate(domain_names):
            if j <= i:
                continue
            X = features_by_domain[a].float()
            Y = features_by_domain[b].float()
            n = min(X.size(0), Y.size(0))
            X, Y = X[:n], Y[:n]

            mmd_val = _rbf_mmd(X, Y)
            results[(a, b)] = float(mmd_val)

    return results


def _rbf_mmd(X: Tensor, Y: Tensor, sigma: Optional[float] = None) -> Tensor:
    """RBF kernel MMD between two samples."""
    XX = (X @ X.T)
    YY = (Y @ Y.T)
    XY = (X @ Y.T)

    dist_xx = XX.diag().unsqueeze(0) + XX.diag().unsqueeze(1) - 2 * XX
    dist_yy = YY.diag().unsqueeze(0) + YY.diag().unsqueeze(1) - 2 * YY
    dist_xy = XX.diag().unsqueeze(0) + YY.diag().unsqueeze(1) - 2 * XY

    if sigma is None:
        all_dists = torch.cat([dist_xx.flatten(), dist_yy.flatten(), dist_xy.flatten()])
        sigma = all_dists[all_dists > 0].median().sqrt().item()
        sigma = max(sigma, 1e-6)

    Kxx = torch.exp(-dist_xx / (2 * sigma**2))
    Kyy = torch.exp(-dist_yy / (2 * sigma**2))
    Kxy = torch.exp(-dist_xy / (2 * sigma**2))

    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return torch.sqrt(mmd.clamp(min=0))


def fit_error_distance_model(
    distances: Dict[Tuple[str, str], float],
    errors: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """
    Fit a linear model: OOD_Error = a * Distance + b.

    Returns the fit parameters and R².
    """
    common_pairs = set(distances.keys()) & set(errors.keys())
    if len(common_pairs) < 3:
        return {"a": 0, "b": 0, "r_squared": 0, "n_pairs": len(common_pairs)}

    d = np.array([distances[p] for p in common_pairs])
    e = np.array([errors[p] for p in common_pairs])

    A = np.vstack([d, np.ones(len(d))]).T
    result = np.linalg.lstsq(A, e, rcond=None)
    a, b = result[0]

    e_pred = a * d + b
    ss_res = ((e - e_pred) ** 2).sum()
    ss_tot = ((e - e.mean()) ** 2).sum()
    r_squared = 1 - ss_res / max(ss_tot, 1e-8)

    return {
        "slope": float(a),
        "intercept": float(b),
        "r_squared": float(r_squared),
        "n_pairs": len(common_pairs),
        "pearson_r": float(np.corrcoef(d, e)[0, 1]) if len(d) > 2 else 0.0,
    }


def run_domain_geometry_analysis(
    model: nn.Module,
    domain_loaders: Dict[str, DataLoader],
    device: torch.device,
    distance_method: str = "cka",
) -> Dict[str, any]:
    """
    Full domain geometry analysis.

    Returns pairwise distances, pairwise errors, and the linear fit.
    """
    logger.info("Extracting features from %d domains …", len(domain_loaders))

    features_by_domain = {}
    preds_by_domain = {}
    targets_by_domain = {}

    for domain_name, loader in domain_loaders.items():
        feat, pred, target = extract_features(model, loader, device)
        features_by_domain[domain_name] = feat
        preds_by_domain[domain_name] = pred
        targets_by_domain[domain_name] = target

    distances = compute_domain_distances(features_by_domain, method=distance_method)

    errors = {}
    domain_names = sorted(domain_loaders.keys())
    for i, a in enumerate(domain_names):
        for j, b in enumerate(domain_names):
            if j <= i:
                continue
            err_a = mae(preds_by_domain[a], targets_by_domain[a])
            err_b = mae(preds_by_domain[b], targets_by_domain[b])
            errors[(a, b)] = max(err_a, err_b)

    fit = fit_error_distance_model(distances, errors)

    logger.info(
        "Domain geometry: R²=%.3f, Pearson r=%.3f, slope=%.4f",
        fit["r_squared"], fit["pearson_r"], fit["slope"],
    )

    return {
        "distances": {f"{a}↔{b}": v for (a, b), v in distances.items()},
        "errors": {f"{a}↔{b}": v for (a, b), v in errors.items()},
        "linear_fit": fit,
        "distance_method": distance_method,
    }
