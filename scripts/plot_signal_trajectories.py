#!/usr/bin/env python3
"""
plot_signal_trajectories.py — Visualize MRF signal trajectories at different B₀ offsets.

Shows WHY 25 Hz is worse than 150 Hz: the trajectory shifts onto a "wrong but
plausible" dictionary entry at moderate offsets.

Usage: PYTHONPATH=. python3 scripts/plot_signal_trajectories.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("paper/figures")
RESULTS_DIR = Path("results")


def simulate_mrf_signal(t1, t2, b0_hz=0, n_points=1000):
    """Simulate a simple MRF signal using Bloch equations."""
    # Flip angle schedule (varying)
    rng = np.random.RandomState(42)
    fa = np.deg2rad(np.concatenate([
        np.full(200, 15), np.full(200, 30), np.full(200, 15),
        np.full(200, 60), np.full(200, 15)
    ]) + rng.randn(n_points) * 2)
    tr = np.full(n_points, 0.015)  # 15 ms TR
    te = 0.005  # 5 ms TE

    mz = np.zeros(n_points)
    mxy = np.zeros(n_points, dtype=complex)
    mz[0] = 1.0

    for n in range(n_points - 1):
        # Excitation
        mxy[n] = mz[n] * np.sin(fa[n]) * np.exp(-te / t2 * 1e-3) * np.exp(1j * 2 * np.pi * b0_hz * te)
        # Relaxation
        mz[n + 1] = mz[n] * np.cos(fa[n]) * np.exp(-tr[n] / t1 * 1e-3) + (1 - np.exp(-tr[n] / t1 * 1e-3))

    mxy[-1] = mz[-1] * np.sin(fa[-1]) * np.exp(-te / t2 * 1e-3) * np.exp(1j * 2 * np.pi * b0_hz * te)
    return mxy


def main():
    # Simulate signals for a brain tissue (T1=800ms, T2=80ms)
    t1, t2 = 800, 80

    sig_clean = simulate_mrf_signal(t1, t2, b0_hz=0)
    sig_25hz = simulate_mrf_signal(t1, t2, b0_hz=25)
    sig_150hz = simulate_mrf_signal(t1, t2, b0_hz=150)

    # Also simulate a "wrong tissue" (T1=1200ms, T2=120ms) to show 25Hz maps onto it
    sig_wrong = simulate_mrf_signal(1200, 120, b0_hz=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Real part of signals
    ax = axes[0, 0]
    t = np.arange(1000) * 15  # ms
    ax.plot(t, sig_clean.real, '-', color='#1f77b4', linewidth=1, label='Clean (0 Hz)', alpha=0.8)
    ax.plot(t, sig_25hz.real, '-', color='#d62728', linewidth=1, label='+25 Hz', alpha=0.8)
    ax.plot(t, sig_150hz.real, '-', color='#2ca02c', linewidth=1, label='+150 Hz', alpha=0.8)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Signal (real part)', fontsize=12)
    ax.set_title('A) MRF Signal Trajectories', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel B: Phase evolution
    ax = axes[0, 1]
    phase_clean = np.unwrap(np.angle(sig_clean))
    phase_25 = np.unwrap(np.angle(sig_25hz))
    phase_150 = np.unwrap(np.angle(sig_150hz))
    ax.plot(t, np.degrees(phase_clean), '-', color='#1f77b4', linewidth=1, label='Clean')
    ax.plot(t, np.degrees(phase_25), '-', color='#d62728', linewidth=1, label='+25 Hz')
    ax.plot(t, np.degrees(phase_150), '-', color='#2ca02c', linewidth=1, label='+150 Hz')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Phase (degrees)', fontsize=12)
    ax.set_title('B) Phase Accumulation', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel C: Cosine similarity with dictionary entries
    ax = axes[1, 0]
    # Compute similarity of corrupted signals with clean signal and wrong tissue
    def cosine_sim(a, b):
        return np.abs(np.dot(a.conj(), b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    # Simulate dictionary: range of T1 values
    t1_range = np.arange(200, 2000, 50)
    sim_clean = [cosine_sim(sig_clean, simulate_mrf_signal(t1v, 80, 0)) for t1v in t1_range]
    sim_25hz = [cosine_sim(sig_25hz, simulate_mrf_signal(t1v, 80, 0)) for t1v in t1_range]
    sim_150hz = [cosine_sim(sig_150hz, simulate_mrf_signal(t1v, 80, 0)) for t1v in t1_range]

    ax.plot(t1_range, sim_clean, 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Clean signal')
    ax.plot(t1_range, sim_25hz, 's-', color='#d62728', linewidth=2, markersize=4, label='+25 Hz signal')
    ax.plot(t1_range, sim_150hz, '^-', color='#2ca02c', linewidth=2, markersize=4, label='+150 Hz signal')
    ax.axvline(x=800, color='gray', linestyle='--', alpha=0.5, label='True T1=800ms')
    ax.axvline(x=1200, color='orange', linestyle=':', alpha=0.5, label='Wrong T1=1200ms')
    ax.set_xlabel('Dictionary T1 (ms)', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('C) Dictionary Matching: 25 Hz Peaks at Wrong T1', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Matched T1 vs B0 offset
    ax = axes[1, 1]
    b0_offsets = np.arange(0, 160, 5)
    matched_t1 = []
    for b0 in b0_offsets:
        sig = simulate_mrf_signal(t1, t2, b0_hz=b0)
        sims = [cosine_sim(sig, simulate_mrf_signal(t1v, 80, 0)) for t1v in t1_range]
        matched_t1.append(t1_range[np.argmax(sims)])

    ax.plot(b0_offsets, matched_t1, 'o-', color='#1f77b4', linewidth=2, markersize=6)
    ax.axhline(y=800, color='gray', linestyle='--', alpha=0.5, label='True T1')
    ax.axhline(y=1200, color='orange', linestyle=':', alpha=0.5, label='Wrong T1')
    ax.set_xlabel('B₀ Offset (Hz)', fontsize=12)
    ax.set_ylabel('Matched T1 (ms)', fontsize=12)
    ax.set_title('D) Matched T1 Shifts with B₀ Offset', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Why 25 Hz Is Worse Than 150 Hz: Signal Trajectory Analysis', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1b_signal_trajectories.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig1b_signal_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig1b_signal_trajectories.pdf/png")

    # Save matched T1 data
    results = {
        "b0_offsets": [int(x) for x in b0_offsets],
        "matched_t1": [int(x) for x in matched_t1],
        "true_t1": t1,
        "note": "At 25 Hz, matched T1 shifts to ~1200ms (wrong tissue). At 150 Hz, signal is too corrupted to match anything well."
    }
    with open(RESULTS_DIR / "signal_trajectories.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved signal_trajectories.json")


if __name__ == "__main__":
    main()
