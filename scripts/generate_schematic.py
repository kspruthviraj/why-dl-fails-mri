#!/usr/bin/env python3
"""
generate_schematic.py — Generate the paper's schematic overview figure (Fig 1).

Shows: Bloch simulation → vendor perturbations → source training →
isolated corruption testing → DS3 + downstream metrics.

Usage: PYTHONPATH=. python3 scripts/generate_schematic.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIGURES_DIR = Path("paper/figures")


def main():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Colors
    blue = "#1f77b4"
    orange = "#ff7f0e"
    green = "#2ca02c"
    red = "#d62728"
    gray = "#7f7f7f"
    light_blue = "#aec7e8"
    light_orange = "#ffbb78"
    light_green = "#98df8a"

    # ── Box 1: Bloch Simulation ──────────────────────────────────────
    box1 = FancyBboxPatch((0.3, 2.5), 2.4, 2.5, boxstyle="round,pad=0.15",
                          facecolor=light_blue, edgecolor=blue, linewidth=2)
    ax.add_patch(box1)
    ax.text(1.5, 4.5, "Bloch Equation", fontsize=11, ha="center", fontweight="bold")
    ax.text(1.5, 4.0, "Simulation", fontsize=11, ha="center", fontweight="bold")
    ax.text(1.5, 3.3, "100k synthetic", fontsize=9, ha="center")
    ax.text(1.5, 2.9, "MRF signals", fontsize=9, ha="center")

    # ── Box 2: Vendor Perturbations ──────────────────────────────────
    box2 = FancyBboxPatch((3.3, 2.5), 2.4, 2.5, boxstyle="round,pad=0.15",
                          facecolor=light_orange, edgecolor=orange, linewidth=2)
    ax.add_patch(box2)
    ax.text(4.5, 4.5, "Vendor", fontsize=11, ha="center", fontweight="bold")
    ax.text(4.5, 4.0, "Perturbations", fontsize=11, ha="center", fontweight="bold")
    ax.text(4.5, 3.4, "B₀: 0-150 Hz", fontsize=9, ha="center")
    ax.text(4.5, 3.0, "B₁: 0.5-1.0×", fontsize=9, ha="center")
    ax.text(4.5, 2.6, "SNR: 2-50", fontsize=9, ha="center")  # Adjusted y position

    # ── Box 3: Source Training ────────────────────────────────────────
    box3 = FancyBboxPatch((6.3, 2.5), 2.4, 2.5, boxstyle="round,pad=0.15",
                          facecolor=light_green, edgecolor=green, linewidth=2)
    ax.add_patch(box3)
    ax.text(7.5, 4.5, "Source-Only", fontsize=11, ha="center", fontweight="bold")
    ax.text(7.5, 4.0, "Training", fontsize=11, ha="center", fontweight="bold")
    ax.text(7.5, 3.4, "Siemens", fontsize=9, ha="center")
    ax.text(7.5, 3.0, "26,659 signals", fontsize=9, ha="center")
    ax.text(7.5, 2.6, "9 algorithms", fontsize=9, ha="center")

    # ── Box 4: Isolated Testing ──────────────────────────────────────
    box4 = FancyBboxPatch((9.3, 2.5), 2.4, 2.5, boxstyle="round,pad=0.15",
                          facecolor="#ffdddd", edgecolor=red, linewidth=2)
    ax.add_patch(box4)
    ax.text(10.5, 4.5, "Isolated", fontsize=11, ha="center", fontweight="bold")
    ax.text(10.5, 4.0, "Corruption", fontsize=11, ha="center", fontweight="bold")
    ax.text(10.5, 3.4, "9 factors", fontsize=9, ha="center")
    ax.text(10.5, 3.0, "one-at-a-time", fontsize=9, ha="center")
    ax.text(10.5, 2.6, "+ joint effects", fontsize=9, ha="center")

    # ── Box 5: DS3 + Clinical ────────────────────────────────────────
    box5 = FancyBboxPatch((12.3, 2.5), 3.0, 2.5, boxstyle="round,pad=0.15",
                          facecolor="#e8e8e8", edgecolor=gray, linewidth=2)
    ax.add_patch(box5)
    ax.text(13.8, 4.5, "Evaluation", fontsize=11, ha="center", fontweight="bold")
    ax.text(13.8, 3.8, "DS3 metric", fontsize=9, ha="center")
    ax.text(13.8, 3.4, "CKA analysis", fontsize=9, ha="center")
    ax.text(13.8, 3.0, "Calibration", fontsize=9, ha="center")
    ax.text(13.8, 2.6, "Segmentation Dice", fontsize=9, ha="center")

    # ── Arrows ────────────────────────────────────────────────────────
    for x1, x2 in [(2.7, 3.3), (5.7, 6.3), (8.7, 9.3), (11.7, 12.3)]:
        ax.annotate("", xy=(x2, 3.75), xytext=(x1, 3.75),
                   arrowprops=dict(arrowstyle="->", lw=2, color=gray))

    # ── Key Findings ─────────────────────────────────────────────────
    ax.text(1.5, 1.5, "B₀ non-monotonic\n(worst at 25 Hz)", fontsize=8, ha="center",
           color=red, fontstyle="italic")
    ax.text(7.5, 1.5, "All robustness\nmethods fail", fontsize=8, ha="center",
           color=red, fontstyle="italic")
    ax.text(10.5, 1.5, "More data\n≠ better OOD", fontsize=8, ha="center",
           color=red, fontstyle="italic")
    ax.text(13.8, 1.5, "Hybrid: 193 ms\nvs 240 ms ERM", fontsize=8, ha="center",
           color=green, fontweight="bold")

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(8.0, 5.5, "Physics Attribution Framework for Cross-Vendor qMRI Failure",
           fontsize=14, ha="center", fontweight="bold")

    fig.savefig(FIGURES_DIR / "fig0_schematic.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig0_schematic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig0_schematic.pdf/png")


if __name__ == "__main__":
    main()
