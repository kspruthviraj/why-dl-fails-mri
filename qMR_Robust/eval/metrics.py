"""
Regression Metrics for qMRI evaluation.

Implements:
  - MAE   (Mean Absolute Error)
  - RMSE  (Root Mean Squared Error)
  - R²    (Coefficient of Determination)
  - CCC   (Concordance Correlation Coefficient)
  - DS3   (Domain Shift Severity Score — novel to this benchmark)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor


def mae(pred: Tensor, target: Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def rmse(pred: Tensor, target: Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def r_squared(pred: Tensor, target: Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return float((1 - ss_res / ss_tot.clamp(min=1e-8)).item())


def concordance_cc(pred: Tensor, target: Tensor) -> float:
    """Lin's Concordance Correlation Coefficient."""
    pred_m, target_m = pred.mean(), target.mean()
    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)
    covar = ((pred - pred_m) * (target - target_m)).mean()
    ccc = (2 * covar) / (
        pred_var + target_var + (pred_m - target_m) ** 2 + 1e-8
    )
    return float(ccc.item())


def ds3(
    ood_pred: Tensor,
    id_pred: Tensor,
    target: Tensor,
) -> float:
    """
    Domain Shift Severity Score (novel metric).

    DS3 = mean(|ood_pred - target|) / mean(|id_pred - target|)
    A DS3 of 2.0 means the domain shift doubles the estimation error.
    """
    ood_err = torch.abs(ood_pred - target).mean()
    id_err = torch.abs(id_pred - target).mean().clamp(min=1e-8)
    return float((ood_err / id_err).item())


def compute_all_regression_metrics(
    pred: Tensor,
    target: Tensor,
    id_pred: Optional[Tensor] = None,
) -> Dict[str, float]:
    """Compute the full set of regression metrics."""
    results = {
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "r2": r_squared(pred, target),
        "ccc": concordance_cc(pred, target),
    }
    if id_pred is not None:
        results["ds3"] = ds3(pred, id_pred, target)
    return results


def compute_per_parameter_metrics(
    pred: Tensor,
    target: Tensor,
    param_names: list[str],
    id_pred: Optional[Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each regression parameter separately."""
    results = {}
    for i, name in enumerate(param_names):
        p = pred[:, i]
        t = target[:, i]
        ip = id_pred[:, i] if id_pred is not None else None
        results[name] = compute_all_regression_metrics(p, t, ip)
    return results
