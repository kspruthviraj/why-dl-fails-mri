#!/usr/bin/env python3
"""Generate figures from corrected_benchmark.json only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    results = json.loads((ROOT / "results/corrected_benchmark.json").read_text())
    folds = results["leave_one_vendor_out"]
    algorithms = results["protocol"]["algorithms"]

    # Leave-one-vendor-out target MAE.
    means, sds = [], []
    for algorithm in algorithms:
        values = [
            run["target"]["mae"]
            for fold in folds.values()
            for run in fold["algorithms"][algorithm]
        ]
        means.append(np.mean(values))
        sds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(algorithms, means, yerr=sds, capsize=4, color="#4472C4")
    ax.set_ylabel("Held-out vendor MAE (ms)")
    ax.set_xlabel("Algorithm")
    ax.set_title("Leave-one-vendor-out generalization")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig4_algorithm_comparison")

    # Paired counterfactual physics response.
    physics = results["physics_counterfactual"]
    b0_names = ["clean", "b0_10Hz", "b0_25Hz", "b0_50Hz", "b0_100Hz"]
    x = [physics[name]["b0_hz"] for name in b0_names]
    y = [physics[name]["mae"] for name in b0_names]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, y, marker="o", linewidth=2, color="#C0504D")
    ax.set_xlabel("$B_0$ offset (Hz)")
    ax.set_ylabel("MAE (ms)")
    ax.set_title("Paired counterfactual $B_0$ response")
    ax.grid(alpha=0.25)
    save(fig, "fig1_b0_dose_response")

    # Scaling law.
    scaling = results.get("scaling", [])
    if scaling:
        sizes = [row["train_size"] for row in scaling]
        src = [row["source"]["mae"] for row in scaling]
        tgt = [row["target"]["mae"] for row in scaling]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(sizes, src, marker="o", label="Source validation")
        ax.plot(sizes, tgt, marker="o", label="Held-out vendor")
        ax.set_xscale("log")
        ax.set_xlabel("Unique source training samples")
        ax.set_ylabel("MAE (ms)")
        ax.set_title("Absolute error versus source-data size")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        save(fig, "fig5_scaling_law")

    # Calibrated uncertainty coverage and interval width.
    uncertainty = results["uncertainty"]
    coverage = np.asarray(uncertainty["target_coverage"]) * 100
    width = np.asarray(uncertainty["target_interval_width"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    axes[0].bar(["T1", "T2"], coverage, color="#70AD47")
    axes[0].axhline(90, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Target coverage (%)")
    axes[0].set_title("Calibrated 90% intervals")
    axes[1].bar(["T1", "T2"], width, color="#ED7D31")
    axes[1].set_ylabel("Mean interval width (ms)")
    axes[1].set_title("Uncertainty width")
    save(fig, "fig10_uncertainty")

    print(f"generated corrected figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
