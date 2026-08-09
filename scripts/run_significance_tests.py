#!/usr/bin/env python3
"""
run_significance_tests.py — Statistical significance tests for key comparisons.

Tests:
  1. ERM vs Mixup (DS3)
  2. ERM vs Hybrid (OOD MAE)
  3. Hybrid vs Dictionary (OOD MAE)
  4. ERM vs CORAL (DS3)
  5. ERM vs GroupDRO (DS3)

Methods: paired bootstrap, Wilcoxon signed-rank, permutation test.

Usage: PYTHONPATH=. python3 scripts/run_significance_tests.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def load_json(path):
    with open(RESULTS_DIR / path) as f:
        return json.load(f)


def paired_bootstrap_diff(a, b, n_boot=10000, seed=42):
    """Paired bootstrap test for difference in means."""
    rng = np.random.RandomState(seed)
    assert len(a) == len(b), "Need paired samples"
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diffs.append(np.mean([a[i] for i in idx]) - np.mean([b[i] for i in idx]))
    diffs = np.array(diffs)
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_95_low": float(np.percentile(diffs, 2.5)),
        "ci_95_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(p_value),
    }


def permutation_test(a, b, n_perm=10000, seed=42):
    """Two-sample permutation test for difference in means."""
    rng = np.random.RandomState(seed)
    observed_diff = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return {
        "observed_diff": float(observed_diff),
        "p_value": float(count / n_perm),
    }


def main():
    ckpt1 = load_json("checkpoint.json")
    ckpt2 = load_json("checkpoint_v2.json")
    enh = load_json("enhancements.json")

    seeds = [42, 123, 456]

    results = {}

    # ── 1. ERM vs Mixup (DS3_ph) ──────────────────────────────────────
    logger.info("Test 1: ERM vs Mixup (DS3)")
    erm_ds3 = [ckpt1["results"][f"algo_resnet1d_18_erm_seed{s}"]["ds3_philips"] for s in seeds]
    mixup_ds3 = [r["ds3_ph"] for r in enh["mixup"]]

    # Need paired samples — use the 3 seeds
    boot = paired_bootstrap_diff(erm_ds3, mixup_ds3)
    perm = permutation_test(erm_ds3, mixup_ds3)
    wilcox = sp_stats.wilcoxon(erm_ds3, mixup_ds3) if len(erm_ds3) == len(mixup_ds3) else None

    results["erm_vs_mixup_ds3"] = {
        "erm_mean": float(np.mean(erm_ds3)),
        "mixup_mean": float(np.mean(mixup_ds3)),
        "bootstrap": boot,
        "permutation": perm,
        "wilcoxon_p": float(wilcox.pvalue) if wilcox else None,
    }
    logger.info("  ERM DS3=%.1f vs Mixup DS3=%.1f", np.mean(erm_ds3), np.mean(mixup_ds3))
    logger.info("  Bootstrap p=%.4f, Permutation p=%.4f", boot["p_value"], perm["p_value"])

    # ── 2. ERM vs Hybrid (Philips MAE) ────────────────────────────────
    logger.info("Test 2: ERM vs Hybrid (Philips MAE)")
    erm_ph = [ckpt1["results"][f"algo_resnet1d_18_erm_seed{s}"]["philips"]["mae"] for s in seeds]
    hybrid_ph = [ckpt2["results"]["gated_hybrid"]["gated_hybrid_mae"]]  # single value

    # For hybrid, we only have 1 run — use bootstrap on ERM to show hybrid is outside CI
    erm_boot = []
    rng = np.random.RandomState(42)
    for _ in range(10000):
        idx = rng.choice(len(erm_ph), len(erm_ph), replace=True)
        erm_boot.append(np.mean([erm_ph[i] for i in idx]))
    erm_ci = (np.percentile(erm_boot, 2.5), np.percentile(erm_boot, 97.5))

    results["erm_vs_hybrid_ph"] = {
        "erm_mean": float(np.mean(erm_ph)),
        "erm_ci_95": [float(erm_ci[0]), float(erm_ci[1])],
        "hybrid": float(hybrid_ph[0]),
        "hybrid_below_erm_ci": hybrid_ph[0] < erm_ci[0],
    }
    logger.info("  ERM MAE=%.1f [%.1f, %.1f] vs Hybrid=%.1f",
                np.mean(erm_ph), erm_ci[0], erm_ci[1], hybrid_ph[0])
    logger.info("  Hybrid below ERM CI: %s", hybrid_ph[0] < erm_ci[0])

    # ── 3. ERM vs CORAL (DS3) ─────────────────────────────────────────
    logger.info("Test 3: ERM vs CORAL (DS3)")
    erm_ds3 = [ckpt1["results"][f"algo_resnet1d_18_erm_seed{s}"]["ds3_philips"] for s in seeds]
    coral_ds3 = [ckpt1["results"][f"algo_resnet1d_18_coral_seed{s}"]["ds3_philips"] for s in seeds]

    boot = paired_bootstrap_diff(erm_ds3, coral_ds3)
    wilcox = sp_stats.wilcoxon(erm_ds3, coral_ds3)
    results["erm_vs_coral_ds3"] = {
        "erm_mean": float(np.mean(erm_ds3)),
        "coral_mean": float(np.mean(coral_ds3)),
        "bootstrap_p": boot["p_value"],
        "wilcoxon_p": float(wilcox.pvalue),
    }
    logger.info("  ERM=%.1f vs CORAL=%.1f, p=%.4f", np.mean(erm_ds3), np.mean(coral_ds3), boot["p_value"])

    # ── 4. ERM vs GroupDRO (DS3) ──────────────────────────────────────
    logger.info("Test 4: ERM vs GroupDRO (DS3)")
    grp_ds3 = [r["ds3_ph"] for r in ckpt2["results"]["groupdro"]]

    boot = paired_bootstrap_diff(erm_ds3, grp_ds3)
    wilcox = sp_stats.wilcoxon(erm_ds3, grp_ds3)
    results["erm_vs_groupdro_ds3"] = {
        "erm_mean": float(np.mean(erm_ds3)),
        "groupdro_mean": float(np.mean(grp_ds3)),
        "bootstrap_p": boot["p_value"],
        "wilcoxon_p": float(wilcox.pvalue),
    }
    logger.info("  ERM=%.1f vs GroupDRO=%.1f, p=%.4f", np.mean(erm_ds3), np.mean(grp_ds3), boot["p_value"])

    # ── 5. ERM vs Mixup (Source MAE) ──────────────────────────────────
    logger.info("Test 5: ERM vs Mixup (Source MAE) — does Mixup hurt source?")
    erm_src = [ckpt1["results"][f"algo_resnet1d_18_erm_seed{s}"]["source"]["mae"] for s in seeds]
    mixup_src = [r["src"]["mae"] for r in enh["mixup"]]

    boot = paired_bootstrap_diff(erm_src, mixup_src)
    results["erm_vs_mixup_src"] = {
        "erm_mean": float(np.mean(erm_src)),
        "mixup_mean": float(np.mean(mixup_src)),
        "bootstrap_p": boot["p_value"],
        "significant": boot["p_value"] < 0.05,
    }
    logger.info("  ERM src=%.1f vs Mixup src=%.1f, p=%.4f", np.mean(erm_src), np.mean(mixup_src), boot["p_value"])

    # ── 6. T2 > T1 variability (real data) ────────────────────────────
    logger.info("Test 6: T2 > T1 variability (real data)")
    real = load_json("real_data_validation.json")
    t1_pcts = []
    t2_pcts = []
    for pair in ["scanner_1_vs_scanner_2", "scanner_1_vs_scanner_3", "scanner_2_vs_scanner_3"]:
        t1_pcts.append(real["cross_scanner"][pair]["t1_pct"])
        t2_pcts.append(real["cross_scanner"][pair]["t2_pct"])

    # Paired t-test: T2 variability > T1 variability
    t_stat, t_p = sp_stats.ttest_rel(t2_pcts, t1_pcts)
    results["t2_gt_t1_variability"] = {
        "t1_mean_pct": float(np.mean(t1_pcts)),
        "t2_mean_pct": float(np.mean(t2_pcts)),
        "ratio": float(np.mean(t2_pcts) / np.mean(t1_pcts)),
        "t_statistic": float(t_stat),
        "p_value": float(t_p),
    }
    logger.info("  T1 variability=%.1f%% vs T2 variability=%.1f%% (ratio=%.1f×), p=%.4f",
                np.mean(t1_pcts), np.mean(t2_pcts), np.mean(t2_pcts)/np.mean(t1_pcts), t_p)

    # ── Save ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "significance_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else o)
    logger.info("Saved → %s", out_path)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SIGNIFICANCE TEST SUMMARY")
    print("=" * 60)
    for test, r in results.items():
        p = r.get("bootstrap_p") or r.get("permutation", {}).get("p_value") or r.get("p_value")
        sig = "***" if p and p < 0.001 else "**" if p and p < 0.01 else "*" if p and p < 0.05 else "ns"
        print(f"  {test}: p={p:.4f} {sig}" if p else f"  {test}: N/A")


if __name__ == "__main__":
    main()
