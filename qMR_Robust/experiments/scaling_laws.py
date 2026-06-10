"""
Scaling Laws Experiment — Does OOD robustness scale with synthetic data?

Because we use synthetic data, we have a superpower: infinite scaling.
Train on 10k, 100k, 1M, 5M synthetic signals and plot DS3 vs. dataset size.

If OOD error follows Error = a * N^(-b), we have a predictable scaling law.
If it plateaus, data alone cannot solve physics shifts — a profound finding.

Reference: Fan et al., "Scaling Laws of Synthetic Images", CVPR 2024.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Results at a single dataset size."""

    n_training_samples: int
    ood_mae: float
    id_mae: float
    ds3: float
    training_time_seconds: float


@dataclass
class ScalingLawFit:
    """Fit of the power law: Error = a * N^(-b) + c."""

    a: float
    b: float
    c: float
    r_squared: float
    n_points: int


@dataclass
class ScalingLawResult:
    """Full scaling law analysis."""

    algorithm: str
    architecture: str
    modality: str
    results: List[ScalingResult] = field(default_factory=list)
    fit_ood: Optional[ScalingLawFit] = None
    fit_ds3: Optional[ScalingLawFit] = None
    plateau_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "modality": self.modality,
            "scaling_points": [
                {
                    "n_samples": r.n_training_samples,
                    "ood_mae": r.ood_mae,
                    "id_mae": r.id_mae,
                    "ds3": r.ds3,
                    "time_s": r.training_time_seconds,
                }
                for r in self.results
            ],
            "fit_ood": {
                "a": self.fit_ood.a,
                "b": self.fit_ood.b,
                "c": self.fit_ood.c,
                "r_squared": self.fit_ood.r_squared,
            } if self.fit_ood else None,
            "fit_ds3": {
                "a": self.fit_ds3.a,
                "b": self.fit_ds3.b,
                "c": self.fit_ds3.c,
                "r_squared": self.fit_ds3.r_squared,
            } if self.fit_ds3 else None,
            "plateau_detected": self.plateau_detected,
        }


def _power_law(N: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Power law model: f(N) = a * N^(-b) + c."""
    return a * np.power(N, -b) + c


def fit_power_law(
    n_samples: List[int],
    values: List[float],
) -> ScalingLawFit:
    """
    Fit a power law: value = a * N^(-b) + c.

    Uses log-space linear regression on the residual after subtracting
    the asymptotic value.
    """
    from scipy.optimize import curve_fit

    N = np.array(n_samples, dtype=np.float64)
    V = np.array(values, dtype=np.float64)

    try:
        popt, _ = curve_fit(
            _power_law, N, V,
            p0=[V[0], 0.5, V[-1] * 0.5],
            bounds=([0, 0, 0], [np.inf, 5.0, np.inf]),
            maxfev=5000,
        )
        a, b, c = popt

        V_pred = _power_law(N, a, b, c)
        ss_res = ((V - V_pred) ** 2).sum()
        ss_tot = ((V - V.mean()) ** 2).sum()
        r_squared = 1 - ss_res / max(ss_tot, 1e-8)

        return ScalingLawFit(
            a=float(a), b=float(b), c=float(c),
            r_squared=float(r_squared), n_points=len(N),
        )
    except Exception as e:
        logger.warning("Power law fit failed: %s", e)
        return ScalingLawFit(a=0, b=0, c=0, r_squared=0, n_points=len(N))


def detect_plateau(
    n_samples: List[int],
    values: List[float],
    threshold: float = 0.05,
) -> bool:
    """
    Detect if the scaling curve has plateaued.

    A plateau is detected if the relative improvement from the second-to-last
    to the last point is less than the threshold.
    """
    if len(values) < 3:
        return False
    relative_change = abs(values[-1] - values[-2]) / max(abs(values[-2]), 1e-8)
    return relative_change < threshold


def run_scaling_law_experiment(
    train_fn,
    eval_fn,
    data_generator,
    sample_sizes: Optional[List[int]] = None,
    algorithm: str = "erm",
    architecture: str = "vit1d",
    modality: str = "mrf",
    device: torch.device = torch.device("cpu"),
) -> ScalingLawResult:
    """
    Run the full scaling law experiment.

    Parameters
    ----------
    train_fn : callable(n_samples) → trained_model
        Function that trains a model on n_samples synthetic signals.
    eval_fn : callable(model) → (id_mae, ood_mae, ds3)
        Function that evaluates a model and returns ID MAE, OOD MAE, and DS3.
    data_generator : callable(n_samples) → DataLoader
        Function that generates n_samples synthetic training data.
    sample_sizes : list of int, optional
        Dataset sizes to test. Default: [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000].
    algorithm : str
    architecture : str
    modality : str
    device : torch.device

    Returns
    -------
    ScalingLawResult
    """
    import time

    if sample_sizes is None:
        sample_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]

    result = ScalingLawResult(
        algorithm=algorithm,
        architecture=architecture,
        modality=modality,
    )

    for n in sample_sizes:
        logger.info("Scaling experiment: N = %s", f"{n:,}")

        start_time = time.time()
        model = train_fn(n)
        train_time = time.time() - start_time

        id_mae, ood_mae, ds3_val = eval_fn(model)

        result.results.append(ScalingResult(
            n_training_samples=n,
            ood_mae=ood_mae,
            id_mae=id_mae,
            ds3=ds3_val,
            training_time_seconds=train_time,
        ))

        logger.info(
            "  N=%s  ID_MAE=%.4f  OOD_MAE=%.4f  DS3=%.3f  time=%.1fs",
            f"{n:,}", id_mae, ood_mae, ds3_val, train_time,
        )

    ns = [r.n_training_samples for r in result.results]
    ood_maes = [r.ood_mae for r in result.results]
    ds3s = [r.ds3 for r in result.results]

    result.fit_ood = fit_power_law(ns, ood_maes)
    result.fit_ds3 = fit_power_law(ns, ds3s)
    result.plateau_detected = detect_plateau(ns, ood_maes)

    logger.info(
        "Scaling law (OOD MAE): a=%.4f, b=%.4f, c=%.4f, R²=%.3f",
        result.fit_ood.a, result.fit_ood.b, result.fit_ood.c, result.fit_ood.r_squared,
    )
    logger.info(
        "Scaling law (DS3): a=%.4f, b=%.4f, c=%.4f, R²=%.3f",
        result.fit_ds3.a, result.fit_ds3.b, result.fit_ds3.c, result.fit_ds3.r_squared,
    )
    logger.info("Plateau detected: %s", result.plateau_detected)

    return result
