"""
Validation loop and JSON leaderboard writer.

Runs a trained model against every evaluation domain (BigGABA vendor splits,
cMRF scanner splits), computes all metrics, and writes a structured JSON
leaderboard that can be compared across runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .metrics import compute_all_regression_metrics, compute_per_parameter_metrics
from .calibration import (
    expected_calibration_error_regression,
    pairwise_cka,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Run inference over a DataLoader and collect predictions + targets."""
    model.eval()
    all_pred, all_target, all_domain = [], [], []

    for signal, target, domain in loader:
        signal = signal.to(device)
        pred = model(signal)
        all_pred.append(pred.cpu())
        all_target.append(target)
        if isinstance(domain, torch.Tensor):
            all_domain.append(domain)

    return {
        "pred": torch.cat(all_pred),
        "target": torch.cat(all_target),
        "domain": torch.cat(all_domain) if all_domain else torch.zeros(0),
    }


@torch.no_grad()
def _collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tensor:
    """Collect penultimate features for CKA."""
    model.eval()
    features = []
    for signal, _, _ in loader:
        signal = signal.to(device)
        feat = model.encode(signal)
        features.append(feat.cpu())
    return torch.cat(features)


def evaluate_model(
    model: nn.Module,
    eval_loaders: Dict[str, DataLoader],
    device: torch.device,
    cfg: dict,
    algorithm: str = "erm",
    modality: str = "mrf",
    id_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    model : trained model
    eval_loaders : dict mapping domain name → DataLoader
    device : torch.device
    cfg : full config dict
    algorithm : algorithm name (for logging)
    modality : "mrf" or "mrs"
    id_loader : optional in-distribution loader for DS3 computation

    Returns
    -------
    results dict ready for JSON serialisation
    """
    eval_cfg = cfg.get("evaluation", {})
    param_names = (
        ["T1", "T2"] if modality == "mrf"
        else cfg.get("simulation", {}).get("mrs", {}).get("metabolites", [])
    )

    # In-distribution predictions for DS3
    id_pred = None
    if id_loader is not None:
        id_results = _collect_predictions(model, id_loader, device)
        id_pred = id_results["pred"]

    # Per-domain evaluation
    domain_results = {}
    features_by_domain = {}

    for domain_name, loader in eval_loaders.items():
        out = _collect_predictions(model, loader, device)
        pred, target = out["pred"], out["target"]

        metrics = compute_all_regression_metrics(pred, target, id_pred)
        per_param = compute_per_parameter_metrics(pred, target, param_names, id_pred)

        # ECE if we can get uncertainty (use residual magnitude as proxy)
        uncertainty = (pred - target).abs().mean(dim=-1, keepdim=True).expand_as(pred)
        ece_result = expected_calibration_error_regression(
            pred, target, uncertainty, n_bins=eval_cfg.get("ece_n_bins", 15)
        )

        domain_results[domain_name] = {
            "n_samples": len(pred),
            "overall": metrics,
            "per_parameter": per_param,
            "ece": ece_result,
        }

        # Collect features for CKA
        features_by_domain[domain_name] = _collect_features(model, loader, device)

        logger.info(
            "  %s: MAE=%.4f  RMSE=%.4f  R²=%.4f  CCC=%.4f",
            domain_name, metrics["mae"], metrics["rmse"], metrics["r2"], metrics["ccc"],
        )

    # Pairwise CKA
    cka_results = {}
    if len(features_by_domain) >= 2:
        raw_cka = pairwise_cka(features_by_domain, kernel="linear")
        cka_results = {f"{a}↔{b}": v for (a, b), v in raw_cka.items()}

    # Aggregate
    all_maes = [d["overall"]["mae"] for d in domain_results.values()]
    worst_domain = max(domain_results, key=lambda d: domain_results[d]["overall"]["mae"])
    best_domain = min(domain_results, key=lambda d: domain_results[d]["overall"]["mae"])

    leaderboard_entry = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": algorithm,
        "architecture": cfg.get("model", {}).get("architecture", "unknown"),
        "modality": modality,
        "overall": {
            "mean_mae": float(sum(all_maes) / len(all_maes)),
            "worst_mae": float(max(all_maes)),
            "best_mae": float(min(all_maes)),
            "worst_domain": worst_domain,
            "best_domain": best_domain,
            "robustness_score": float(min(all_maes) / max(all_maes, 1e-8)),
        },
        "per_domain": domain_results,
        "cka": cka_results,
        "config": {
            "algorithm": algorithm,
            "architecture": cfg.get("model", {}).get("architecture"),
            "training": cfg.get("training", {}),
        },
    }

    return leaderboard_entry


def save_leaderboard(
    entries: List[Dict[str, Any]],
    output_path: str,
):
    """Append entries to the JSON leaderboard file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if path.exists():
        with open(path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    if not isinstance(existing, list):
        existing = [existing]

    existing.extend(entries)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info("Leaderboard updated → %s  (%d entries)", path, len(existing))
