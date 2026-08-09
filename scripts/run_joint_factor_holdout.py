#!/usr/bin/env python3
"""Test generalization to an unseen combination of acquisition factors.

The target combination (3.0 T, flip-angle variant 2, TR variant 1) is removed
as a complete cell of the factorial design. Every individual level remains
present in the source set through other combinations. ERM is trained on the
remaining cells, with a deterministic source split and source-only target
scaling, then evaluated once on the held-out cell.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_corrected_benchmark import (  # noqa: E402
    EPOCHS,
    SEEDS,
    load_data,
    train_model,
)

OUTPUT = ROOT / "results/corrected_joint_factor_holdout.json"
TABLE = ROOT / "paper/generated_joint_factor_holdout_table.tex"

TARGET_COMBINATION = {
    "field_strength": 3.0,
    "fa_variant": 2,
    "tr_variant": 1,
}


def split_source(indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split only the source cell complement using a deterministic permutation."""
    rng = np.random.default_rng(seed)
    shuffled = indices[rng.permutation(len(indices))]
    n_train = max(1, int(0.8 * len(shuffled)))
    return shuffled[:n_train], shuffled[n_train:]


def combination_mask(data: dict[str, Any], combination: dict[str, float | int]) -> np.ndarray:
    mask = np.ones(len(data["sample_ids"]), dtype=bool)
    for factor, value in combination.items():
        mask &= data["metadata"][factor] == value
    return mask


def run_holdout(data: dict[str, Any]) -> dict[str, Any]:
    target_mask = combination_mask(data, TARGET_COMBINATION)
    source_indices = np.flatnonzero(~target_mask)
    target_indices = np.flatnonzero(target_mask)
    train_idx, val_idx = split_source(source_indices, seed=20260809)

    train_ids = set(data["sample_ids"][train_idx].tolist())
    val_ids = set(data["sample_ids"][val_idx].tolist())
    target_ids = set(data["sample_ids"][target_indices].tolist())
    overlap = (
        len(train_ids & val_ids)
        + len(train_ids & target_ids)
        + len(val_ids & target_ids)
    )
    if overlap:
        raise ValueError(f"sample-ID overlap detected: {overlap}")

    # This is the key compositional check: each target factor level is seen in
    # the source data, but their joint cell is not.
    source_level_presence = {
        factor: bool(np.any(data["metadata"][factor][source_indices] == value))
        for factor, value in TARGET_COMBINATION.items()
    }
    if not all(source_level_presence.values()):
        raise ValueError(
            "the proposed target combination is not compositional: "
            f"missing source levels {source_level_presence}"
        )

    fold = {
        "target_vendor": "unseen_acquisition_combination",
        "source_vendors": ["source_acquisition_combinations"],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "target_idx": target_indices,
        "train_domain_counts": {
            "source_acquisition_combinations": int(len(train_idx))
        },
        "split_overlap": 0,
        "sample_id_overlap": 0,
    }

    records: list[dict[str, Any]] = []
    for model_seed in SEEDS:
        train_domains = np.zeros(len(train_idx), dtype=np.int64)
        _, trained = train_model(
            data,
            fold,
            "erm",
            model_seed,
            train_indices=train_idx,
            train_domains=train_domains,
        )
        record = trained["result"]
        records.append(
            {
                "seed": int(model_seed),
                "source": record["source"],
                "target": record["target"],
                "ds3": record["ds3"],
                "train_size": record["train_size"],
                "validation_size": int(len(val_idx)),
                "target_size": int(len(target_indices)),
                "target_labels_used_for_training": False,
                "target_scaler_fit_on_source_only": True,
            }
        )

    def summary(path: tuple[str, ...]) -> dict[str, float | int]:
        values = []
        for row in records:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1) if len(array) > 1 else 0.0),
            "n": int(len(array)),
        }

    target_domains = sorted(
        {
            str(domain)
            for domain in data["domains"][target_indices]
        }
    )
    source_domains = sorted(
        {
            str(domain)
            for domain in data["domains"][source_indices]
        }
    )
    return {
        "target_combination": TARGET_COMBINATION,
        "n_total": int(len(data["sample_ids"])),
        "n_source": int(len(source_indices)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_target": int(len(target_indices)),
        "n_source_domains": int(len(source_domains)),
        "n_target_domains": int(len(target_domains)),
        "source_contains_each_target_level": source_level_presence,
        "target_domains": target_domains,
        "split_overlap": 0,
        "sample_id_overlap": 0,
        "records": records,
        "summary": {
            "source_mae": summary(("source", "mae")),
            "target_mae": summary(("target", "mae")),
            "target_mae_t1": summary(("target", "mae_t1")),
            "target_mae_t2": summary(("target", "mae_t2")),
            "ds3": summary(("ds3",)),
        },
    }


def write_table(result: dict[str, Any]) -> None:
    summary = result["summary"]
    source = (
        f"{summary['source_mae']['mean']:.1f} $\\pm$ "
        f"{summary['source_mae']['sd']:.1f}"
    )
    target = (
        f"{summary['target_mae']['mean']:.1f} $\\pm$ "
        f"{summary['target_mae']['sd']:.1f}"
    )
    t1 = (
        f"{summary['target_mae_t1']['mean']:.1f} $\\pm$ "
        f"{summary['target_mae_t1']['sd']:.1f}"
    )
    t2 = (
        f"{summary['target_mae_t2']['mean']:.1f} $\\pm$ "
        f"{summary['target_mae_t2']['sd']:.1f}"
    )
    ds3 = (
        f"{summary['ds3']['mean']:.2f} $\\pm$ "
        f"{summary['ds3']['sd']:.2f}"
    )
    lines = [
        "% Generated by scripts/run_joint_factor_holdout.py.",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Held-out combination & Source MAE & Target MAE & Target T1 MAE & Target T2 MAE & DS3 \\\\",
        "\\midrule",
        f"3.0 T + FA 2 + TR 1 & {source} & {target} & {t1} & {t2} & {ds3} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "}",
    ]
    TABLE.write_text("\n".join(lines) + "\n")


def main() -> None:
    data = load_data(
        ROOT / os.environ.get(
            "MRF_DATA", "data/synthetic/mrf_corrected_100k.h5"
        )
    )
    result = run_holdout(data)
    output = {
        "schema_version": "corrected-joint-factor-holdout-v1",
        "protocol": {
            "model": "same ResNet1D-18 configuration as corrected benchmark",
            "algorithm": "ERM",
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "source_definition": (
                "all factorial acquisition combinations except the target cell"
            ),
            "source_split": "80/20 by deterministic sample-ID permutation",
            "target_scaler": "source training only",
            "target_labels_used_for_training": False,
            "target_labels_used_for_model_selection": False,
            "target_combination_is_compositional": True,
            "primary_metric": "absolute MAE in ms",
        },
        "result": result,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    write_table(result)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
