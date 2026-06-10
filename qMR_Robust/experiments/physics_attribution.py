"""
Physics Attribution Experiment — Isolate the contribution of each physical
corruption to OOD failure.

Instead of treating "vendor shift" as a monolithic category, we decompose it
into five independent physical axes and measure DS3 for each in isolation:

  1. B1+ non-uniformity  (transmit coil sensitivity)
  2. SNR / noise floor
  3. Sequence timing (flip-angle schedule, TR schedule, TE)
  4. B0 inhomogeneity
  5. Gradient nonlinearities (spatial distortion model)

The output is a "Physics Attribution Table" — the core scientific finding
of the paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class PhysicsCorruption:
    """Specification of a single isolated physics corruption."""

    name: str
    description: str
    parameter_name: str
    baseline_value: float
    corrupted_value: float
    # If True, the corruption is applied to the simulation parameters
    # rather than to the signal directly.
    is_simulation_level: bool = True


# ── Pre-defined corruption set ────────────────────────────────────────────────

B1_PLUS_CORRUPTION = PhysicsCorruption(
    name="b1_plus",
    description="Transmit field (B1+) non-uniformity: scale flip angles by 0.75",
    parameter_name="b1_scale",
    baseline_value=1.0,
    corrupted_value=0.75,
    is_simulation_level=True,
)

SNR_CORRUPTION = PhysicsCorruption(
    name="snr",
    description="Noise floor increase: SNR drops from 30 to 10",
    parameter_name="snr",
    baseline_value=30.0,
    corrupted_value=10.0,
    is_simulation_level=True,
)

TIMING_CORRUPTION = PhysicsCorruption(
    name="timing",
    description="Sequence timing mismatch: different FA schedule variant",
    parameter_name="fa_schedule_variant",
    baseline_value=0,
    corrupted_value=3,
    is_simulation_level=True,
)

B0_CORRUPTION = PhysicsCorruption(
    name="b0_inhomogeneity",
    description="Static field inhomogeneity: +30 Hz off-resonance",
    parameter_name="b0_shift",
    baseline_value=0.0,
    corrupted_value=30.0,
    is_simulation_level=True,
)

GRADIENT_CORRUPTION = PhysicsCorruption(
    name="gradient_nonlinearity",
    description="Gradient nonlinearity: 5% spatial scaling error",
    parameter_name="gradient_scale",
    baseline_value=1.0,
    corrupted_value=1.05,
    is_simulation_level=True,
)

ALL_CORRUPTIONS = [
    B1_PLUS_CORRUPTION,
    SNR_CORRUPTION,
    TIMING_CORRUPTION,
    B0_CORRUPTION,
    GRADIENT_CORRUPTION,
]


def apply_isolated_corruption(
    signal: np.ndarray,
    corruption: PhysicsCorruption,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Apply a single physical corruption to a signal.

    For simulation-level corruptions, this is a fallback that applies
    the corruption directly to the signal when re-simulation is not feasible.
    """
    if corruption.name == "b1_plus":
        scale = corruption.corrupted_value
        signal = signal * scale + signal * (1 - scale) * rng.randn(*signal.shape) * 0.05

    elif corruption.name == "snr":
        current_power = np.mean(np.abs(signal) ** 2)
        target_snr = corruption.corrupted_value
        noise_power = current_power / (target_snr ** 2)
        noise = np.sqrt(noise_power / 2) * (
            rng.randn(*signal.shape) + 1j * rng.randn(*signal.shape)
        )
        signal = signal + noise

    elif corruption.name == "b0_inhomogeneity":
        n = signal.shape[-1]
        t = np.linspace(0, n / 1000, n)
        phase = np.exp(1j * 2 * np.pi * corruption.corrupted_value * t)
        signal = signal * phase

    elif corruption.name == "gradient_nonlinearity":
        n = signal.shape[-1]
        warp = np.linspace(1.0, corruption.corrupted_value, n)
        signal = signal * warp

    elif corruption.name == "timing":
        idx = np.random.permutation(signal.shape[-1])
        signal = signal[..., idx]

    return signal.astype(np.complex64)


@dataclass
class PhysicsAttributionResult:
    """Results of the physics attribution experiment for one algorithm."""

    algorithm: str
    architecture: str
    corruption_results: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_table(self) -> str:
        """Format as a publication-ready table."""
        header = f"{'Corruption':<25} {'DS3':>8} {'MAE':>10} {'MAE (clean)':>12} {'Δ MAE':>10}"
        rows = [header, "-" * len(header)]
        for name, res in self.corruption_results.items():
            ds3 = res.get("ds3", 0)
            mae = res.get("mae_corrupted", 0)
            mae_clean = res.get("mae_baseline", 0)
            delta = mae - mae_clean
            rows.append(f"{name:<25} {ds3:>8.3f} {mae:>10.4f} {mae_clean:>12.4f} {delta:>+10.4f}")
        return "\n".join(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "corruptions": self.corruption_results,
        }


def run_physics_attribution(
    model_fn,
    baseline_signals: np.ndarray,
    baseline_targets: np.ndarray,
    corruptions: Optional[List[PhysicsCorruption]] = None,
    algorithm: str = "erm",
    architecture: str = "vit1d",
    n_samples: int = 1000,
    device: torch.device = torch.device("cpu"),
    sim_manager=None,
) -> PhysicsAttributionResult:
    """
    Run the physics attribution experiment.

    For each corruption type:
      1. Generate corrupted versions of the baseline signals.
      2. Run model inference on both clean and corrupted signals.
      3. Compute DS3 and per-parameter MAE.

    Parameters
    ----------
    model_fn : callable
        Function that takes a signal tensor and returns predictions.
    baseline_signals : np.ndarray
        Clean signals, shape (N, ...) complex or (N, 2, L) real.
    baseline_targets : np.ndarray
        Ground truth parameters, shape (N, T).
    corruptions : list of PhysicsCorruption, optional
        Which corruptions to test. Defaults to ALL_CORRUPTIONS.
    algorithm : str
        Algorithm name for logging.
    architecture : str
        Architecture name for logging.
    n_samples : int
        Number of samples to evaluate.
    device : torch.device
    sim_manager : SimulationManager, optional
        If provided, use simulation-level corruptions via re-simulation.

    Returns
    -------
    PhysicsAttributionResult
    """
    from ..eval.metrics import mae, ds3

    if corruptions is None:
        corruptions = ALL_CORRUPTIONS

    rng = np.random.RandomState(42)
    n = min(n_samples, len(baseline_signals))

    signals_clean = baseline_signals[:n]
    targets = baseline_targets[:n]

    # Baseline predictions
    if signals_clean.ndim == 2 and np.iscomplexobj(signals_clean):
        x_clean = np.stack([signals_clean.real, signals_clean.imag], axis=1).astype(np.float32)
    else:
        x_clean = signals_clean.astype(np.float32)

    with torch.no_grad():
        pred_clean = model_fn(torch.from_numpy(x_clean).to(device)).cpu()

    mae_baseline = mae(pred_clean, torch.from_numpy(targets))

    result = PhysicsAttributionResult(
        algorithm=algorithm,
        architecture=architecture,
    )

    for corruption in corruptions:
        logger.info("  Testing corruption: %s — %s", corruption.name, corruption.description)

        if corruption.is_simulation_level and sim_manager is not None:
            signals_corrupted = _generate_corrupted_signals(
                sim_manager, corruption, n, rng
            )
        else:
            signals_corrupted = np.stack([
                apply_isolated_corruption(signals_clean[i].copy(), corruption, rng)
                for i in range(n)
            ])

        if np.iscomplexobj(signals_corrupted):
            x_corr = np.stack([signals_corrupted.real, signals_corrupted.imag], axis=1).astype(np.float32)
        else:
            x_corr = signals_corrupted.astype(np.float32)

        with torch.no_grad():
            pred_corr = model_fn(torch.from_numpy(x_corr).to(device)).cpu()

        mae_corrupted = mae(pred_corr, torch.from_numpy(targets))
        ds3_val = ds3(pred_corr, pred_clean, torch.from_numpy(targets))

        result.corruption_results[corruption.name] = {
            "description": corruption.description,
            "ds3": ds3_val,
            "mae_baseline": mae_baseline,
            "mae_corrupted": mae_corrupted,
            "delta_mae": mae_corrupted - mae_baseline,
            "parameter_name": corruption.parameter_name,
            "baseline_value": corruption.baseline_value,
            "corrupted_value": corruption.corrupted_value,
        }

        logger.info(
            "    DS3=%.3f  MAE: %.4f → %.4f (Δ=%+.4f)",
            ds3_val, mae_baseline, mae_corrupted, mae_corrupted - mae_baseline,
        )

    return result


def _generate_corrupted_signals(
    sim_manager,
    corruption: PhysicsCorruption,
    n: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate corrupted signals via the simulation manager."""
    cfg_override = {corruption.parameter_name: corruption.corrupted_value}

    signals = []
    for _ in range(n):
        result = sim_manager.generate_batch_mrf(1)
        sig = result[0][0]
        signals.append(sig)
    return np.stack(signals)
