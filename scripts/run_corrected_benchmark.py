#!/usr/bin/env python3
"""Corrected, leakage-free MRF domain-shift benchmark.

The authoritative pipeline is deliberately small and auditable:
- unique, indexed synthetic samples;
- train-only target scaling;
- leave-one-vendor-out evaluation;
- real multi-environment training labels for robustness methods;
- source-validation-only hybrid weighting and uncertainty calibration;
- paired counterfactual physics and representation analyses.

Exploratory scripts from the first draft are not called by this pipeline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qMR_Robust.algorithms.base import build_algorithm
from qMR_Robust.baselines.classical import DictionaryMatcher
from qMR_Robust.eval.calibration import cka
from qMR_Robust.models.registry import build_model
from qMR_Robust.simulators.manager import (
    _generate_mrf_sample,
    _stable_seed,
)

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required") from exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("corrected_benchmark")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = ROOT / os.environ.get("MRF_DATA", "data/synthetic/mrf_corrected_100k.h5")
RESULTS_PATH = ROOT / "results/corrected_benchmark.json"
CHECKPOINT_PATH = ROOT / "results/corrected_checkpoint.json"
VALIDATION_PATH = ROOT / "results/data_validation.json"
SEEDS = [int(x) for x in os.environ.get("MRF_SEEDS", "42,123,456").split(",") if x]
ALGORITHMS = [
    x.strip() for x in os.environ.get(
        "MRF_ALGOS", "erm,coral,groupdro,dann,irm,vrex"
    ).split(",") if x.strip()
]
EPOCHS = int(os.environ.get("MRF_EPOCHS", "15"))
BATCH_SIZE = int(os.environ.get("MRF_BATCH_SIZE", "512"))
RUN_SCALING = os.environ.get("MRF_SKIP_SCALING", "0") != "1"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ])


def load_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run reproduce.sh to generate the corrected dataset."
        )
    with h5py.File(path, "r") as f:
        required = {
            "signals", "parameters", "domain_labels", "sample_ids",
            "b0_hz", "b1_scale", "snr", "field_strength",
            "fa_variant", "tr_variant",
        }
        missing = required - set(f.keys())
        if missing:
            raise ValueError(
                f"{path} is not a corrected dataset; missing {sorted(missing)}"
            )
        signals = f["signals"][:]
        targets = f["parameters"][:, :2].astype(np.float32)
        sample_ids = f["sample_ids"][:].astype(np.int64)
        domains = decode(f["domain_labels"][:])
        metadata = {
            key: f[key][:] for key in (
                "b0_hz", "b1_scale", "snr", "field_strength",
                "fa_variant", "tr_variant",
            )
        }
        attrs = {str(key): str(value) for key, value in f.attrs.items()}

    n = len(signals)
    if len(np.unique(sample_ids)) != n:
        raise ValueError("sample IDs are not unique")
    signal_peak = np.max(np.abs(signals), axis=1, keepdims=True)
    signal_peak = np.maximum(signal_peak, 1e-8)
    x = np.stack(
        [signals.real / signal_peak, signals.imag / signal_peak],
        axis=1,
    ).astype(np.float32)
    vendors = np.asarray([domain.split("_", 1)[0] for domain in domains])
    return {
        "x": x,
        "targets": targets,
        "sample_ids": sample_ids,
        "domains": domains,
        "vendors": vendors,
        "metadata": metadata,
        "attrs": attrs,
    }


def prepare_fold(data: dict[str, Any], target_vendor: str) -> dict[str, Any]:
    vendors = data["vendors"]
    source_vendors = sorted(set(vendors) - {target_vendor})
    if len(source_vendors) < 2:
        raise ValueError("Each fold requires at least two source vendors")

    train_indices: list[int] = []
    val_indices: list[int] = []
    train_domains: list[int] = []
    rng = np.random.default_rng(202402 + sum(map(ord, target_vendor)))

    for domain_id, vendor in enumerate(source_vendors):
        vendor_indices = np.flatnonzero(vendors == vendor)
        vendor_indices = vendor_indices[rng.permutation(len(vendor_indices))]
        split = max(1, int(0.8 * len(vendor_indices)))
        train_indices.extend(vendor_indices[:split].tolist())
        val_indices.extend(vendor_indices[split:].tolist())
        train_domains.extend([domain_id] * split)

    train_indices_np = np.asarray(train_indices, dtype=np.int64)
    val_indices_np = np.asarray(val_indices, dtype=np.int64)
    target_indices = np.flatnonzero(vendors == target_vendor).astype(np.int64)

    train_ids = set(data["sample_ids"][train_indices_np].tolist())
    val_ids = set(data["sample_ids"][val_indices_np].tolist())
    target_ids = set(data["sample_ids"][target_indices].tolist())
    if train_ids & val_ids or train_ids & target_ids or val_ids & target_ids:
        raise ValueError(f"sample-ID leakage in {target_vendor} fold")

    return {
        "target_vendor": target_vendor,
        "source_vendors": source_vendors,
        "train_idx": train_indices_np,
        "val_idx": val_indices_np,
        "target_idx": target_indices,
        "train_domains": np.asarray(train_domains, dtype=np.int64),
        "train_domain_counts": {
            vendor: int(np.sum(data["vendors"][train_indices_np] == vendor))
            for vendor in source_vendors
        },
        "split_overlap": 0,
    }


def fit_target_scaler(data: dict[str, Any], train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = data["targets"][train_idx]
    return y.min(axis=0), y.max(axis=0)


def normalize_targets(y: np.ndarray, ymin: np.ndarray, ymax: np.ndarray) -> np.ndarray:
    return ((y - ymin) / np.maximum(ymax - ymin, 1e-8)).astype(np.float32)


def denormalize_targets(y: np.ndarray, ymin: np.ndarray, ymax: np.ndarray) -> np.ndarray:
    return y * (ymax - ymin) + ymin


def build_model_for_data(seq_len: int) -> nn.Module:
    config = {
        "input_channels": 2,
        "output_dim": 2,
        "hidden_dim": int(os.environ.get("MRF_HIDDEN_DIM", "128")),
        "base_channels": int(os.environ.get("MRF_BASE_CHANNELS", "32")),
        "dropout": 0.1,
        "seq_len": seq_len,
        "patch_size": 32,
        "n_heads": 4,
        "n_transformer_layers": 4,
    }
    return build_model("resnet1d_18", config).to(DEVICE)


def predict_normalized(
    model: nn.Module, x: torch.Tensor, stochastic: bool = False
) -> np.ndarray:
    if not stochastic:
        model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            pred = model(x[start:start + 1024].to(DEVICE)).detach().cpu().numpy()
            output.append(pred)
    return np.concatenate(output, axis=0)


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae_t1": float(np.mean(np.abs(error[:, 0]))),
        "mae_t2": float(np.mean(np.abs(error[:, 1]))),
        "bias_t1": float(np.mean(error[:, 0])),
        "bias_t2": float(np.mean(error[:, 1])),
    }


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
    return_predictions: bool = False,
) -> dict[str, Any]:
    pred_norm = predict_normalized(model, x)
    pred = denormalize_targets(pred_norm, ymin, ymax)
    result = regression_metrics(pred, y)
    if return_predictions:
        result["predictions"] = pred
    return result


def train_model(
    data: dict[str, Any],
    fold: dict[str, Any],
    algorithm_name: str,
    seed: int,
    train_indices: np.ndarray | None = None,
    train_domains: np.ndarray | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    train_idx = fold["train_idx"] if train_indices is None else train_indices
    if train_domains is None:
        vendor_to_domain = {
            vendor: i for i, vendor in enumerate(fold["source_vendors"])
        }
        train_domains = np.asarray([
            vendor_to_domain[data["vendors"][i]] for i in train_idx
        ], dtype=np.int64)

    val_idx = fold["val_idx"]
    ymin, ymax = fit_target_scaler(data, train_idx)
    train_x = torch.from_numpy(data["x"][train_idx])
    val_x = torch.from_numpy(data["x"][val_idx])
    train_y = torch.from_numpy(
        normalize_targets(data["targets"][train_idx], ymin, ymax)
    )
    val_y = torch.from_numpy(
        normalize_targets(data["targets"][val_idx], ymin, ymax)
    )
    domains = torch.from_numpy(train_domains)

    with open(ROOT / "configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["algorithm"]["coral_penalty_weight"] = float(
        os.environ.get("MRF_CORAL_WEIGHT", "0.01")
    )
    cfg["algorithm"]["dann_penalty_weight"] = float(
        os.environ.get("MRF_DANN_WEIGHT", "0.1")
    )
    cfg["algorithm"]["groupdro_eta"] = float(
        os.environ.get("MRF_GROUPDRO_ETA", "0.01")
    )

    model = build_model_for_data(data["x"].shape[-1])
    algorithm = build_algorithm(
        algorithm_name,
        model,
        cfg,
        DEVICE,
        n_domains=len(fold["source_vendors"]),
    )
    parameters = list(model.parameters())
    if hasattr(algorithm, "domain_classifier"):
        parameters += list(algorithm.domain_classifier.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(os.environ.get("MRF_LR", "0.0003")),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y, domains),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )

    best_state = None
    best_val = float("inf")
    history: list[float] = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for signal, target, domain in loader:
            signal = signal.to(DEVICE)
            target = target.to(DEVICE)
            domain = domain.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            result = algorithm.compute_loss(signal, target, domain)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            epoch_losses.append(float(result["pred_loss"].detach().cpu()))
        scheduler.step()
        history.append(float(np.mean(epoch_losses)))

        val_metrics = evaluate(
            model, val_x, data["targets"][val_idx], ymin, ymax
        )
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            LOGGER.info(
                "%s target=%s algo=%s seed=%s epoch=%s train_loss=%.4f val_mae=%.3f",
                DEVICE, fold["target_vendor"], algorithm_name, seed,
                epoch + 1, history[-1], val_metrics["mae"],
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    source = evaluate(model, val_x, data["targets"][val_idx], ymin, ymax)
    target_x = torch.from_numpy(data["x"][fold["target_idx"]])
    target = evaluate(
        model, target_x, data["targets"][fold["target_idx"]], ymin, ymax
    )
    result = {
        "seed": int(seed),
        "algorithm": algorithm_name,
        "source": source,
        "target": target,
        "ds3": float(target["mae"] / max(source["mae"], 1e-8)),
        "train_size": int(len(train_idx)),
        "source_domain_counts": {
            str(k): int(v) for k, v in fold["train_domain_counts"].items()
        },
        "n_train_domains": int(len(np.unique(train_domains))),
        "target_labels_used_for_training": False,
        "target_scaler_fit_on_source_only": True,
        "history": history,
    }
    return model, {
        "result": result,
        "ymin": ymin,
        "ymax": ymax,
        "source_x": val_x,
        "source_y": data["targets"][val_idx],
        "target_x": target_x,
        "target_y": data["targets"][fold["target_idx"]],
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(path: tuple[str, ...]) -> np.ndarray:
        return np.asarray([
            float(_nested_get(run, path)) for run in runs
        ], dtype=np.float64)

    def mean_sd(path: tuple[str, ...]) -> dict[str, float]:
        values = vals(path)
        return {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1) if len(values) > 1 else 0.0),
            "n": int(len(values)),
        }

    return {
        "n_runs": len(runs),
        "target_mae": mean_sd(("target", "mae")),
        "source_mae": mean_sd(("source", "mae")),
        "ds3": mean_sd(("ds3",)),
        "target_mae_t1": mean_sd(("target", "mae_t1")),
        "target_mae_t2": mean_sd(("target", "mae_t2")),
    }


def _nested_get(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        value = value[key]
    return value


def dictionary_predictions(
    matcher: DictionaryMatcher, x: np.ndarray, chunk_size: int = 256
) -> np.ndarray:
    predictions = []
    signals = x[:, 0] + 1j * x[:, 1]
    for start in range(0, len(signals), chunk_size):
        pred, _ = matcher.match_batch(signals[start:start + chunk_size])
        predictions.append(pred[:, :2])
    return np.concatenate(predictions, axis=0)


def run_hybrid(
    data: dict[str, Any],
    fold: dict[str, Any],
    model: nn.Module,
    ymin: np.ndarray,
    ymax: np.ndarray,
) -> dict[str, Any]:
    source_train = fold["train_idx"][:min(2000, len(fold["train_idx"]))]
    matcher = DictionaryMatcher(
        data["x"][source_train, 0] + 1j * data["x"][source_train, 1],
        data["targets"][source_train],
    )
    source_val = fold["val_idx"]
    target_idx = fold["target_idx"]
    source_dl = denormalize_targets(
        predict_normalized(model, torch.from_numpy(data["x"][source_val])),
        ymin, ymax,
    )
    target_dl = denormalize_targets(
        predict_normalized(model, torch.from_numpy(data["x"][target_idx])),
        ymin, ymax,
    )
    source_dict = dictionary_predictions(matcher, data["x"][source_val])
    target_dict = dictionary_predictions(matcher, data["x"][target_idx])
    source_true = data["targets"][source_val]
    target_true = data["targets"][target_idx]

    grid = np.linspace(0.0, 1.0, 101)
    source_errors = [
        np.mean(np.abs(alpha * source_dl + (1 - alpha) * source_dict - source_true))
        for alpha in grid
    ]
    alpha = float(grid[int(np.argmin(source_errors))])
    source_hybrid = alpha * source_dl + (1 - alpha) * source_dict
    target_hybrid = alpha * target_dl + (1 - alpha) * target_dict
    return {
        "alpha_selected_on_source_validation": alpha,
        "target_labels_used_for_weight_selection": False,
        "source": {
            "deep_learning": regression_metrics(source_dl, source_true),
            "dictionary": regression_metrics(source_dict, source_true),
            "hybrid": regression_metrics(source_hybrid, source_true),
        },
        "target": {
            "deep_learning": regression_metrics(target_dl, target_true),
            "dictionary": regression_metrics(target_dict, target_true),
            "hybrid": regression_metrics(target_hybrid, target_true),
        },
    }


def make_counterfactual_batch(
    simulation_cfg: dict[str, Any],
    vendor: str,
    n: int,
    b0: float,
    b1: float,
    snr: float,
) -> tuple[np.ndarray, np.ndarray]:
    signals = []
    targets = []
    for sample_id in range(n):
        seed = _stable_seed("counterfactual-v1", sample_id)
        sample = _generate_mrf_sample(
            seed, simulation_cfg, vendor, 3.0, 0, 0,
            b0_override=b0, b1_override=b1, snr_override=snr,
        )
        signals.append(sample["signal"])
        targets.append(sample["params"][:2])
    signals_np = np.asarray(signals)
    peak = np.maximum(np.max(np.abs(signals_np), axis=1, keepdims=True), 1e-8)
    x = np.stack(
        [signals_np.real / peak, signals_np.imag / peak], axis=1
    ).astype(np.float32)
    return x, np.asarray(targets, dtype=np.float32)


def run_counterfactual_physics(
    simulation_cfg: dict[str, Any],
    model: nn.Module,
    ymin: np.ndarray,
    ymax: np.ndarray,
) -> dict[str, Any]:
    conditions = [
        ("clean", 0.0, 1.0, 100.0),
        ("b0_10Hz", 10.0, 1.0, 100.0),
        ("b0_25Hz", 25.0, 1.0, 100.0),
        ("b0_50Hz", 50.0, 1.0, 100.0),
        ("b0_100Hz", 100.0, 1.0, 100.0),
        ("b1_0.75", 0.0, 0.75, 100.0),
        ("b1_0.50", 0.0, 0.50, 100.0),
        ("snr_20", 0.0, 1.0, 20.0),
        ("snr_10", 0.0, 1.0, 10.0),
        ("snr_5", 0.0, 1.0, 5.0),
    ]
    results: dict[str, Any] = {}
    for name, b0, b1, snr in conditions:
        x, y = make_counterfactual_batch(
            simulation_cfg, "ge", 256, b0, b1, snr
        )
        metrics = evaluate(
            model, torch.from_numpy(x), y, ymin, ymax
        )
        results[name] = {
            "b0_hz": b0,
            "b1_scale": b1,
            "snr": snr,
            **metrics,
        }
    clean = results["clean"]["mae"]
    for result in results.values():
        result["relative_to_clean"] = float(result["mae"] / max(clean, 1e-8))
    return results


def run_uncertainty(
    model: nn.Module,
    source_x: torch.Tensor,
    source_y: np.ndarray,
    target_x: torch.Tensor,
    target_y: np.ndarray,
    ymin: np.ndarray,
    ymax: np.ndarray,
    passes: int = 20,
) -> dict[str, Any]:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    def mc(x: torch.Tensor) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for _ in range(passes):
                outputs.append(
                    denormalize_targets(
                        predict_normalized(model, x, stochastic=True), ymin, ymax
                    )
                )
        return np.stack(outputs)

    source_pred = mc(source_x)
    target_pred = mc(target_x)
    source_mean = source_pred.mean(axis=0)
    target_mean = target_pred.mean(axis=0)
    source_std = np.maximum(source_pred.std(axis=0), 1e-6)
    target_std = np.maximum(target_pred.std(axis=0), 1e-6)
    q = np.quantile(
        np.abs(source_y - source_mean) / source_std, 0.90, axis=0
    )
    target_coverage = np.mean(
        np.abs(target_y - target_mean) <= q[None, :] * target_std,
        axis=0,
    )
    return {
        "method": "MC dropout with batch-normalization layers frozen in eval mode",
        "dropout_active_during_sampling": True,
        "calibration_source": "source_validation_only",
        "passes": int(passes),
        "q90_standardized_residual": q.tolist(),
        "target_coverage": target_coverage.tolist(),
        "target_interval_width": (2 * q[None, :] * target_std).mean(axis=0).tolist(),
        "target_error": np.mean(np.abs(target_y - target_mean), axis=0).tolist(),
        "target_uncertainty_error_correlation": [
            float(np.corrcoef(target_std[:, i], np.abs(target_y[:, i] - target_mean[:, i]))[0, 1])
            for i in range(target_y.shape[1])
        ],
    }


def run_paired_cka(
    simulation_cfg: dict[str, Any], model: nn.Module, n: int = 128
) -> dict[str, float]:
    def features(vendor: str) -> torch.Tensor:
        x, _ = make_counterfactual_batch(
            simulation_cfg, vendor, n, 0.0, 1.0, 100.0
        )
        model.eval()
        with torch.no_grad():
            return model.encode(torch.from_numpy(x).to(DEVICE)).cpu()

    source_features = features("siemens")
    target_features = features("ge")
    return {
        "paired_siemens_ge": float(cka(source_features, target_features, "linear")),
        "n_paired_counterfactuals": int(n),
        "pairing": "same latent sample ID and schedule, vendor profile varied",
    }


def balanced_subsample(
    data: dict[str, Any], fold: dict[str, Any], size: int
) -> tuple[np.ndarray, np.ndarray]:
    size = min(size, len(fold["train_idx"]))
    per_domain = max(1, size // len(fold["source_vendors"]))
    indices = []
    domains = []
    for domain_id, vendor in enumerate(fold["source_vendors"]):
        candidates = fold["train_idx"][data["vendors"][fold["train_idx"]] == vendor]
        chosen = candidates[:per_domain]
        indices.extend(chosen.tolist())
        domains.extend([domain_id] * len(chosen))
    return np.asarray(indices, dtype=np.int64), np.asarray(domains, dtype=np.int64)


def write_paper_numbers(results: dict[str, Any]) -> None:
    """Write machine-generated manuscript values and the main results table."""
    all_runs = [
        run
        for fold in results["leave_one_vendor_out"].values()
        for runs in fold["algorithms"].values()
        for run in runs
    ]
    best = min(all_runs, key=lambda run: run["target"]["mae"])

    def token(value: str) -> str:
        # TeX control-word names may contain letters only. Spell out digits
        # so generated macros cannot leak numeric characters into the PDF.
        digit_names = {str(index): name for index, name in enumerate((
            "Zero", "One", "Two", "Three", "Four", "Five",
            "Six", "Seven", "Eight", "Nine",
        ))}
        raw = "".join(part.capitalize() for part in value.split("_"))
        return "".join(
            digit_names.get(character, character)
            for character in raw
            if character.isalnum()
        )

    vendor_labels = {"ge": "GE", "philips": "Philips", "siemens": "Siemens"}
    lines = [
        "% Automatically generated by scripts/run_full_benchmark.py.",
        "% Do not edit manually; rerun the corrected pipeline.",
        f"\\newcommand{{\\CorrectedN}}{{{results['data']['n_signals']:,}}}",
        f"\\newcommand{{\\CorrectedDomains}}{{{results['data']['n_domains']}}}",
        f"\\newcommand{{\\CorrectedAlgorithms}}{{{len(results['protocol']['algorithms'])}}}",
        f"\\newcommand{{\\CorrectedBestTargetMAE}}{{{best['target']['mae']:.2f}}}",
        f"\\newcommand{{\\CorrectedBestTargetVendor}}{{{vendor_labels.get(best['target_vendor'], best['target_vendor'])}}}",
        f"\\newcommand{{\\CorrectedBestAlgorithm}}{{{best['algorithm']}}}",
    ]

    table = [
        "% Automatically generated table.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Held-out vendor & Method & Source MAE & Target MAE & DS3 \\\\",
        "\\midrule",
    ]
    for vendor, fold in results["leave_one_vendor_out"].items():
        for algorithm in results["protocol"]["algorithms"]:
            summary = fold[f"{algorithm}_summary"]
            source = summary["source_mae"]
            target = summary["target_mae"]
            ds3 = summary["ds3"]
            table.append(
                f"{vendor} & {algorithm.upper()} & "
                f"{source['mean']:.1f} $\\pm$ {source['sd']:.1f} & "
                f"{target['mean']:.1f} $\\pm$ {target['sd']:.1f} & "
                f"{ds3['mean']:.2f} $\\pm$ {ds3['sd']:.2f} \\\\"
            )
    table.extend(["\\bottomrule", "\\end{tabular}"])
    (ROOT / "paper/generated_main_table.tex").write_text("\n".join(table) + "\n")

    parameter_table = [
        "% Automatically generated table.",
        "\\small",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Held-out vendor & Method & T1 MAE (ms) & T2 MAE (ms) \\\\",
        "\\midrule",
    ]
    for vendor, fold in results["leave_one_vendor_out"].items():
        for algorithm in results["protocol"]["algorithms"]:
            summary = fold[f"{algorithm}_summary"]
            t1 = summary["target_mae_t1"]
            t2 = summary["target_mae_t2"]
            parameter_table.append(
                f"{vendor} & {algorithm.upper()} & "
                f"{t1['mean']:.1f} $\\pm$ {t1['sd']:.1f} & "
                f"{t2['mean']:.1f} $\\pm$ {t2['sd']:.1f} \\\\"
            )
    parameter_table.extend(["\\bottomrule", "\\end{tabular}"])
    (ROOT / "paper/generated_parameter_table.tex").write_text(
        "\n".join(parameter_table) + "\n"
    )

    physics = results.get("physics_counterfactual", {})
    for name in ("clean", "b0_25Hz", "b0_100Hz", "b1_0.50", "snr_5"):
        if name in physics:
            lines.append(
                f"\\newcommand{{\\Physics{token(name)}}}{{{physics[name]['mae']:.1f}}}"
            )
    hybrid = results.get("hybrid", {}).get("target", {})
    for name in ("deep_learning", "dictionary", "hybrid"):
        if name in hybrid:
            lines.append(
                f"\\newcommand{{\\Hybrid{token(name)}Target}}{{{hybrid[name]['mae']:.1f}}}"
            )
    uncertainty = results.get("uncertainty", {})
    if uncertainty:
        lines.append(
            "\\newcommand{\\UncertaintyTOneCoverage}"
            f"{{{100 * uncertainty['target_coverage'][0]:.1f}\\%}}"
        )
        lines.append(
            "\\newcommand{\\UncertaintyTTwoCoverage}"
            f"{{{100 * uncertainty['target_coverage'][1]:.1f}\\%}}"
        )
    (ROOT / "paper/paper_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    LOGGER.info("Using %s on %s", DEVICE, DATA_PATH)
    from scripts.validate_dataset import validate

    validation = validate(DATA_PATH)
    VALIDATION_PATH.write_text(json.dumps(validation, indent=2))
    data = load_data(DATA_PATH)
    vendors = sorted(np.unique(data["vendors"]).tolist())
    target_filter = os.environ.get("MRF_TARGET_VENDORS", "").strip()
    target_vendors = (
        [x.strip() for x in target_filter.split(",") if x.strip()]
        if target_filter else vendors
    )
    for target in target_vendors:
        if target not in vendors:
            raise ValueError(f"unknown target vendor {target}")

    checkpoint = {"schema_version": "corrected-v1", "runs": {}}
    if CHECKPOINT_PATH.exists():
        try:
            cached = json.loads(CHECKPOINT_PATH.read_text())
            if cached.get("schema_version") == checkpoint["schema_version"]:
                checkpoint = cached
        except json.JSONDecodeError:
            pass

    results: dict[str, Any] = {
        "schema_version": "corrected-v1",
        "data": {
            "path": str(DATA_PATH),
            "n_signals": int(len(data["x"])),
            "signal_length": int(data["x"].shape[-1]),
            "n_domains": int(len(np.unique(data["domains"]))),
            "vendors": vendors,
            "simulator_version": data["attrs"].get("simulator_version", "unknown"),
            "validation": validation,
        },
        "protocol": {
            "split": "leave-one-vendor-out; source vendors split 80/20 by sample ID",
            "normalization": "per-sample complex peak normalization",
            "target_scaler": "fit on source training samples only",
            "target_labels_used_for_training": False,
            "target_labels_used_for_hybrid_selection": False,
            "target_labels_used_for_uncertainty_calibration": False,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "algorithms": ALGORITHMS,
            "primary_metric": "absolute target MAE in ms",
        },
        "leave_one_vendor_out": {},
    }

    analysis_model = None
    analysis_fold = None
    analysis_state = None

    for target_vendor in target_vendors:
        fold = prepare_fold(data, target_vendor)
        fold_result: dict[str, Any] = {
            "target_vendor": target_vendor,
            "source_vendors": fold["source_vendors"],
            "train_size": int(len(fold["train_idx"])),
            "validation_size": int(len(fold["val_idx"])),
            "target_size": int(len(fold["target_idx"])),
            "train_domain_counts": fold["train_domain_counts"],
            "sample_id_overlap": fold["split_overlap"],
            "algorithms": {},
        }

        for algorithm_name in ALGORITHMS:
            run_records = []
            for seed in SEEDS:
                key = f"{target_vendor}:{algorithm_name}:{seed}"
                if key in checkpoint["runs"]:
                    record = checkpoint["runs"][key]
                    run_records.append(record)
                    continue
                model, trained = train_model(
                    data, fold, algorithm_name, seed
                )
                record = trained["result"]
                record["target_vendor"] = target_vendor
                checkpoint["runs"][key] = record
                CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))
                run_records.append(record)
                if target_vendor == "ge" and algorithm_name == "erm" and seed == 42:
                    analysis_model = model
                    analysis_fold = fold
                    analysis_state = trained
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fold_result["algorithms"][algorithm_name] = run_records
            fold_result[f"{algorithm_name}_summary"] = summarize_runs(run_records)

        results["leave_one_vendor_out"][target_vendor] = fold_result

    if analysis_model is None:
        analysis_fold = prepare_fold(data, "ge")
        analysis_model, analysis_state = train_model(
            data, analysis_fold, "erm", 42
        )

    results["hybrid"] = run_hybrid(
        data, analysis_fold, analysis_model,
        analysis_state["ymin"], analysis_state["ymax"],
    )
    results["physics_counterfactual"] = run_counterfactual_physics(
        yaml.safe_load((ROOT / "configs/config.yaml").read_text())["simulation"],
        analysis_model, analysis_state["ymin"], analysis_state["ymax"],
    )
    results["uncertainty"] = run_uncertainty(
        analysis_model,
        analysis_state["source_x"],
        analysis_state["source_y"],
        analysis_state["target_x"],
        analysis_state["target_y"],
        analysis_state["ymin"],
        analysis_state["ymax"],
    )
    results["paired_representation"] = run_paired_cka(
        yaml.safe_load((ROOT / "configs/config.yaml").read_text())["simulation"],
        analysis_model,
    )

    if RUN_SCALING:
        scaling = []
        ge_fold = prepare_fold(data, "ge")
        sizes = [1000, 5000, 10000, 25000, len(ge_fold["train_idx"])]
        for size in sizes:
            indices, domains = balanced_subsample(data, ge_fold, size)
            model, trained = train_model(
                data, ge_fold, "erm", 42,
                train_indices=indices, train_domains=domains,
            )
            scaling.append({
                "train_size": int(len(indices)),
                "source": trained["result"]["source"],
                "target": trained["result"]["target"],
                "ds3": trained["result"]["ds3"],
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results["scaling"] = scaling
    else:
        results["scaling"] = []

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    write_paper_numbers(results)
    LOGGER.info("Saved corrected results to %s", RESULTS_PATH)
    LOGGER.info("All corrected benchmark checks completed")


if __name__ == "__main__":
    main()
