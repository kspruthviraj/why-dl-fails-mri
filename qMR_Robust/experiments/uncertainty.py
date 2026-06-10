"""
Uncertainty Quantification & Clinical Tolerance.

Implements:
  1. MC Dropout for epistemic uncertainty estimation.
  2. Clinical tolerance bounds — how many predictions cross diagnostic thresholds.
  3. Uncertainty calibration under shift — does predicted uncertainty track
     actual error when the domain shifts?

This addresses the reviewer concern: "Clinicians care less about MAE and
more about risk."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ── Clinical thresholds (qMRI parameter-specific) ────────────────────────────

CLINICAL_THRESHOLDS = {
    "mrf": {
        "T1": {"unit": "ms", "threshold": 40.0, "description": "Tissue classification boundary"},
        "T2": {"unit": "ms", "threshold": 15.0, "description": "Edema detection boundary"},
    },
    "mrs": {
        "GABA": {"unit": "IU", "threshold": 0.5, "description": "Neurological disorder boundary"},
        "Glu": {"unit": "IU", "threshold": 1.0, "description": "Glutamate excitotoxicity"},
        "NAA": {"unit": "IU", "threshold": 1.5, "description": "Neuronal integrity"},
        "Cr": {"unit": "IU", "threshold": 1.0, "description": "Energy metabolism"},
    },
}


@dataclass
class UncertaintyResult:
    """Results of uncertainty analysis for one domain."""

    domain_name: str
    mean_predicted_uncertainty: float
    mean_actual_error: float
    uncertainty_error_correlation: float
    ece_score: float
    clinical_violation_rate: float
    per_parameter: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain_name,
            "mean_uncertainty": self.mean_predicted_uncertainty,
            "mean_error": self.mean_actual_error,
            "uncertainty_error_corr": self.uncertainty_error_correlation,
            "ece": self.ece_score,
            "clinical_violation_rate": self.clinical_violation_rate,
            "per_parameter": self.per_parameter,
        }


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation.

    Keeps dropout active at inference time and runs T forward passes
    to estimate predictive uncertainty.
    """

    def __init__(self, model: nn.Module, n_samples: int = 30):
        self.model = model
        self.n_samples = n_samples

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_with_uncertainty(
        self, signal: Tensor, device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run MC Dropout inference.

        Parameters
        ----------
        signal : (B, C, L) input tensor
        device : torch.device

        Returns
        -------
        mean_pred : (B, D) mean prediction across MC samples
        std_pred : (B, D) standard deviation (epistemic uncertainty)
        all_preds : (T, B, D) all MC samples
        """
        self.model.eval()
        self._enable_dropout()

        signal = signal.to(device)
        preds = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(signal)
                preds.append(pred.cpu())

        all_preds = torch.stack(preds)  # (T, B, D)
        mean_pred = all_preds.mean(dim=0)  # (B, D)
        std_pred = all_preds.std(dim=0)  # (B, D)

        return mean_pred, std_pred, all_preds


def compute_clinical_violations(
    pred: Tensor,
    target: Tensor,
    param_names: List[str],
    modality: str = "mrf",
) -> Dict[str, Any]:
    """
    Compute the fraction of predictions that cross clinical diagnostic thresholds.

    Parameters
    ----------
    pred : (B, D) predictions
    target : (B, D) ground truth
    param_names : list of parameter names
    modality : "mrf" or "mrs"

    Returns
    -------
    dict with per-parameter violation rates and overall rate
    """
    thresholds = CLINICAL_THRESHOLDS.get(modality, {})
    errors = (pred - target).abs()

    results = {"per_parameter": {}, "total_violations": 0, "total_samples": len(pred)}

    for i, name in enumerate(param_names):
        if name not in thresholds:
            continue

        thresh = thresholds[name]["threshold"]
        violations = (errors[:, i] > thresh).sum().item()
        rate = violations / len(pred)

        results["per_parameter"][name] = {
            "threshold": thresh,
            "unit": thresholds[name]["unit"],
            "description": thresholds[name]["description"],
            "n_violations": violations,
            "violation_rate": rate,
            "mean_error": float(errors[:, i].mean().item()),
            "max_error": float(errors[:, i].max().item()),
        }
        results["total_violations"] += violations

    results["overall_violation_rate"] = (
        results["total_violations"] / (len(pred) * len(param_names))
        if param_names else 0.0
    )
    return results


def compute_uncertainty_error_correlation(
    uncertainty: Tensor,
    actual_error: Tensor,
) -> float:
    """
    Compute Pearson correlation between predicted uncertainty and actual error.

    A well-calibrated model should have high correlation: when uncertainty
    is high, the error should also be high.
    """
    u = uncertainty.flatten().numpy()
    e = actual_error.flatten().numpy()

    if len(u) < 3 or np.std(u) < 1e-8 or np.std(e) < 1e-8:
        return 0.0

    corr = np.corrcoef(u, e)[0, 1]
    return float(corr)


def run_uncertainty_analysis(
    model: nn.Module,
    dataloader,
    device: torch.device,
    param_names: List[str],
    modality: str = "mrf",
    n_mc_samples: int = 30,
    domain_name: str = "unknown",
) -> UncertaintyResult:
    """
    Full uncertainty analysis for a single domain.

    Returns predicted uncertainty, actual error, their correlation,
    ECE, and clinical violation rate.
    """
    from .calibration import expected_calibration_error_regression

    mc_predictor = MCDropoutPredictor(model, n_samples=n_mc_samples)

    all_mean_pred = []
    all_std_pred = []
    all_target = []

    for signal, target, _ in dataloader:
        mean_pred, std_pred, _ = mc_predictor.predict_with_uncertainty(signal, device)
        all_mean_pred.append(mean_pred)
        all_std_pred.append(std_pred)
        all_target.append(target)

    mean_pred = torch.cat(all_mean_pred)
    std_pred = torch.cat(all_std_pred)
    target = torch.cat(all_target)

    actual_error = (mean_pred - target).abs()

    ue_corr = compute_uncertainty_error_correlation(std_pred, actual_error)

    ece_result = expected_calibration_error_regression(
        mean_pred, target, std_pred, n_bins=15,
    )

    clinical = compute_clinical_violations(
        mean_pred, target, param_names, modality,
    )

    per_param = {}
    for i, name in enumerate(param_names):
        per_param[name] = {
            "mean_uncertainty": float(std_pred[:, i].mean().item()),
            "mean_error": float(actual_error[:, i].mean().item()),
            "uncertainty_error_corr": compute_uncertainty_error_correlation(
                std_pred[:, i], actual_error[:, i]
            ),
        }

    return UncertaintyResult(
        domain_name=domain_name,
        mean_predicted_uncertainty=float(std_pred.mean().item()),
        mean_actual_error=float(actual_error.mean().item()),
        uncertainty_error_correlation=ue_corr,
        ece_score=ece_result["ece"],
        clinical_violation_rate=clinical["overall_violation_rate"],
        per_parameter=per_param,
    )
