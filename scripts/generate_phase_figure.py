#!/usr/bin/env python3
"""
generate_phase_figure.py — Generate phase diagram and joint perturbation figures.

Usage: PYTHONPATH=. python3 scripts/generate_phase_figure.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("paper/figures")
RESULTS_DIR = Path("results")


def fig_phase_diagram():
    """Phase diagram: Data size × B₀ shift → DS3 heatmap."""
    r = json.load(open(RESULTS_DIR / "joint_and_phase.json"))
    phase = r["phase_diagram"]

    data_sizes = [5000, 10000, 26659]
    b0_shifts = [0, 25, 50, 100, 150]

    matrix = np.zeros((len(data_sizes), len(b0_shifts)))
    for i, n in enumerate(data_sizes):
        for j, b0 in enumerate(b0_shifts):
            key = f"N{n}_B0_{b0}"
            matrix[i, j] = phase[key]["ds3"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=1.0, vmax=3.5)

    ax.set_xticks(range(len(b0_shifts)))
    ax.set_xticklabels([f"{b0}" for b0 in b0_shifts])
    ax.set_yticks(range(len(data_sizes)))
    ax.set_yticklabels([f"{n:,}" for n in data_sizes])
    ax.set_xlabel("B₀ Offset (Hz)", fontsize=13)
    ax.set_ylabel("Training Size", fontsize=13)
    ax.set_title("Phase Diagram: Data Size × B₀ Shift", fontsize=15)

    for i in range(len(data_sizes)):
        for j in range(len(b0_shifts)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                   fontsize=12, fontweight="bold",
                   color="white" if matrix[i, j] > 2.5 else "black")

    fig.colorbar(im, ax=ax, label="DS3", shrink=0.8)
    ax.annotate("More data helps here", xy=(1, 2), xytext=(2.5, 2.3),
               arrowprops=dict(arrowstyle="->", color="green", lw=2),
               fontsize=10, color="green", fontweight="bold")
    ax.annotate("Data scaling\nfails here", xy=(3, 0), xytext=(3.5, 0.7),
               arrowprops=dict(arrowstyle="->", color="red", lw=2),
               fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_phase_diagram.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_phase_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_phase_diagram.pdf/png")


def fig_joint_perturbations():
    """Joint perturbation bar chart."""
    r = json.load(open(RESULTS_DIR / "joint_and_phase.json"))
    joint = r["joint_perturbations"]

    names = ["B₀ 25Hz", "SNR=5", "B₁ 0.75", "Timing",
             "B₀+SNR", "B₀+B₁", "B₀+Timing", "All"]
    keys = ["B0_25Hz_only", "SNR_5_only", "B1_0.75_only", "Timing_only",
            "B0_25Hz + SNR_5", "B0_25Hz + B1_0.75", "B0_25Hz + Timing", "All_combined"]
    ds3_vals = [joint[k]["ds3"] for k in keys]

    colors = ["#1f77b4"] * 4 + ["#ff7f0e"] * 3 + ["#d62728"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, ds3_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="DS3=1 (no effect)")
    ax.set_ylabel("DS3 (error increase factor)", fontsize=13)
    ax.set_title("Joint Perturbations: Sub-Additive Effects", fontsize=15)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, ds3_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=10)

    # Add sum annotation
    individual_sum = sum(joint[k]["ds3"] for k in keys[:4]) - 3  # subtract baseline
    ax.annotate(f"Sum of individuals: {individual_sum + 3:.2f}\nActual combined: {joint['All_combined']['ds3']:.2f}\nRatio: {joint['additivity_ratio']:.2f}",
               xy=(7, joint["All_combined"]["ds3"]), xytext=(5.5, 2.2),
               arrowprops=dict(arrowstyle="->", color="black"),
               fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat"))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_joint_perturbations.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_joint_perturbations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_joint_perturbations.pdf/png")


if __name__ == "__main__":
    fig_phase_diagram()
    fig_joint_perturbations()
