#!/usr/bin/env python3
"""
generate_conceptual_figure.py — Conceptual comparison: semantic vs physics-induced shift.

This is the figure people will reuse in talks and review papers.

Usage: PYTHONPATH=. python3 scripts/generate_conceptual_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

FIGURES_DIR = Path("paper/figures")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Conventional ML Shift ──────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Conventional ML Shift", fontsize=14, fontweight="bold", pad=20)

    items = [
        ("More data helps", "✓", "#2ca02c", 9),
        ("Routing/gating works", "✓", "#2ca02c", 7.5),
        ("Calibration degrades slowly", "✓", "#2ca02c", 6),
        ("Representations stable", "✓", "#2ca02c", 4.5),
        ("ERM ≈ robustness methods", "✓", "#2ca02c", 3),
    ]

    for text, symbol, color, y in items:
        box = FancyBboxPatch((0.5, y - 0.5), 9, 1.0, boxstyle="round,pad=0.1",
                            facecolor="#e8f5e9", edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(1.0, y, symbol, fontsize=16, ha="center", va="center", color=color, fontweight="bold")
        ax.text(1.5, y, text, fontsize=11, ha="left", va="center")

    ax.text(5, 1.5, "Examples: hospital A → B,\ndifferent demographics,\nbackground textures",
           fontsize=9, ha="center", va="center", fontstyle="italic", color="gray")

    # ── Right: Physics-Induced Shift ─────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Physics-Induced Shift (This Paper)", fontsize=14, fontweight="bold", pad=20)

    items = [
        ("More data: phase boundary", "!", "#d62728", 9),
        ("Routing collapses (OOD paradox)", "✗", "#d62728", 7.5),
        ("Calibration collapses (95%→12%)", "✗", "#d62728", 6),
        ("Representations degenerate", "✗", "#d62728", 4.5),
        ("All robustness methods fail", "✗", "#d62728", 3),
    ]

    for text, symbol, color, y in items:
        box = FancyBboxPatch((0.5, y - 0.5), 9, 1.0, boxstyle="round,pad=0.1",
                            facecolor="#ffebee", edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(1.0, y, symbol, fontsize=16, ha="center", va="center", color=color, fontweight="bold")
        ax.text(1.5, y, text, fontsize=11, ha="left", va="center")

    ax.text(5, 1.5, "Examples: MRI vendor shift,\nCT scanner differences,\nultrasound probe variations",
           fontsize=9, ha="center", va="center", fontstyle="italic", color="gray")

    fig.suptitle("Physics-Induced Domain Shifts Obey Different Rules Than Semantic Shifts",
                fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_conceptual_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_conceptual_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_conceptual_comparison.pdf/png")


if __name__ == "__main__":
    main()
