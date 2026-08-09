#!/usr/bin/env python3
"""Independent integrity checks for the corrected benchmark outputs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_dataset import validate
RESULTS = ROOT / "results/corrected_benchmark.json"
DATA = ROOT / "data/synthetic/mrf_corrected_100k.h5"
MULTISCANNER_EXTERNAL = ROOT / "results/external_cmrf_multiscanner.json"
JOINT_FACTOR_HOLDOUT = ROOT / "results/corrected_joint_factor_holdout.json"


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def verify_multiscanner_external():
    if not MULTISCANNER_EXTERNAL.exists():
        return None
    external = json.loads(MULTISCANNER_EXTERNAL.read_text())
    assert external["schema_version"] == "external-cmrf-multiscanner-v1"
    task = external["task"]
    assert task["model_training_performed"] is False
    assert task["neural_benchmark_tuning_performed"] is False
    assert task["target_labels_used"] is False
    assert task["reference_scans_used_for_dictionary_or_training"] is False
    assert task["reference_scans_used_only_for_final_evaluation"] is True
    scanners = external["scanners"]
    assert len(scanners) == 4
    for scanner in scanners:
        metadata = scanner["metadata"]
        assert metadata["vendor"].lower() == "siemens"
        assert len(metadata["cMRF_md5"]) == 32
        assert len(metadata["ref_t1_md5"]) == 32
        assert len(metadata["ref_t2_md5"]) == 32
        assert len(metadata["mask_md5"]) == 32
        assert scanner["cmrf_reconstruction"]["n_acquisitions"] == 705
        assert scanner["cmrf_reconstruction"]["n_windows"] == 45
        assert scanner["t1"]["tube_mean_metrics_ms"]["n"] == 9
        assert scanner["t2"]["tube_mean_metrics_ms"]["n"] == 9
        assert finite(scanner)
    aggregate = external["aggregate"]
    assert aggregate["n_scanners"] == 4
    assert aggregate["n_tubes_per_scanner"] == 9
    assert aggregate["n_tube_means"] == 36
    assert finite(aggregate)
    return len(scanners)


def verify_joint_factor_holdout():
    if not JOINT_FACTOR_HOLDOUT.exists():
        return None
    report = json.loads(JOINT_FACTOR_HOLDOUT.read_text())
    assert report["schema_version"] == "corrected-joint-factor-holdout-v1"
    protocol = report["protocol"]
    assert protocol["algorithm"] == "ERM"
    assert protocol["source_definition"] == (
        "all factorial acquisition combinations except the target cell"
    )
    assert protocol["source_split"] == "80/20 by deterministic sample-ID permutation"
    assert protocol["target_scaler"] == "source training only"
    assert protocol["target_labels_used_for_training"] is False
    assert protocol["target_labels_used_for_model_selection"] is False
    assert protocol["target_combination_is_compositional"] is True
    assert len(protocol["seeds"]) == 3

    result = report["result"]
    assert result["target_combination"] == {
        "field_strength": 3.0,
        "fa_variant": 2,
        "tr_variant": 1,
    }
    assert result["n_total"] == 100000
    assert result["n_source"] == 91667
    assert result["n_train"] == 73333
    assert result["n_validation"] == 18334
    assert result["n_target"] == 8333
    assert result["n_source_domains"] == 33
    assert result["n_target_domains"] == 3
    assert all(result["source_contains_each_target_level"].values())
    assert result["split_overlap"] == 0
    assert result["sample_id_overlap"] == 0
    assert len(result["records"]) == len(protocol["seeds"])
    assert finite(result)
    for record in result["records"]:
        assert record["target_labels_used_for_training"] is False
        assert record["target_scaler_fit_on_source_only"] is True
        assert record["target"]["mae"] >= 0
    return 1


def verify_method_effects():
    path = ROOT / "results/corrected_method_effects.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    assert report["schema_version"] == "corrected-method-effects-v1"
    protocol = report["protocol"]
    assert protocol["reference"] == "ERM"
    assert protocol["target_labels_used"] is False
    assert protocol["bootstrap_replicates"] >= 1000
    for method, details in report["methods"].items():
        assert method != "erm"
        assert len(details["paired_units"]) == 9
        for metric in ("mae", "mae_t1", "mae_t2"):
            summary = details["pooled"][metric]
            assert summary["n_pairs"] == 9
            assert len(summary["bootstrap_ci95_ms"]) == 2
            assert finite(summary)
    return len(report["methods"])


def verify_factor_holdout():
    path = ROOT / "results/corrected_factor_holdout.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    assert report["schema_version"] == "corrected-factor-holdout-v1"
    protocol = report["protocol"]
    assert protocol["algorithm"] == "ERM"
    assert protocol["source_split"] == "80/20 by deterministic sample-ID permutation"
    assert protocol["target_scaler"] == "source training only"
    assert protocol["target_labels_used_for_training"] is False
    assert protocol["target_labels_used_for_model_selection"] is False
    assert len(protocol["seeds"]) == 3
    conditions = report["conditions"]
    assert len(conditions) == 3
    names = {condition["name"] for condition in conditions}
    assert names == {
        "Field strength (3.0 T)",
        "Flip-angle schedule (variant 2)",
        "TR schedule (variant 1)",
    }
    for condition in conditions:
        assert condition["split_overlap"] == 0
        assert condition["sample_id_overlap"] == 0
        assert condition["n_source"] > condition["n_train"] > 0
        assert condition["n_validation"] > 0
        assert condition["n_target"] > 0
        assert len(condition["records"]) == len(protocol["seeds"])
        assert finite(condition)
        for record in condition["records"]:
            assert record["target_labels_used_for_training"] is False
            assert record["target_scaler_fit_on_source_only"] is True
            assert record["target"]["mae"] >= 0
    return len(conditions)


def main():
    if not RESULTS.exists():
        raise SystemExit(f"missing {RESULTS}")
    report = validate(DATA)
    if not report["valid"]:
        raise SystemExit("dataset validation failed")
    if report["duplicate_signal_rows"] != 0:
        raise SystemExit("duplicate signal rows detected")

    results = json.loads(RESULTS.read_text())
    assert results["schema_version"] == "corrected-v1"
    assert results["data"]["n_signals"] == report["n_signals"]
    assert results["data"]["validation"]["n_unique_signal_hashes"] == report["n_signals"]

    protocol = results["protocol"]
    assert protocol["target_labels_used_for_training"] is False
    assert protocol["target_labels_used_for_hybrid_selection"] is False
    assert protocol["target_labels_used_for_uncertainty_calibration"] is False

    expected_algorithms = set(protocol["algorithms"])
    if len(expected_algorithms) < 2:
        raise SystemExit("fewer than two algorithms were evaluated")

    n_folds = 0
    n_runs = 0
    for vendor, fold in results["leave_one_vendor_out"].items():
        n_folds += 1
        assert fold["sample_id_overlap"] == 0
        if set(fold["source_vendors"]) | {vendor} != set(results["data"]["vendors"]):
            raise SystemExit(f"invalid vendor fold {vendor}")
        if set(fold["algorithms"]) != expected_algorithms:
            raise SystemExit(f"incomplete algorithm grid in fold {vendor}")
        for algorithm, runs in fold["algorithms"].items():
            if len(runs) != len(protocol["seeds"]):
                raise SystemExit(f"incomplete seeds for {vendor}/{algorithm}")
            for run in runs:
                n_runs += 1
                assert run["target_labels_used_for_training"] is False
                assert run["target_scaler_fit_on_source_only"] is True
                assert run["n_train_domains"] >= 2
                assert run["source"]["mae"] >= 0
                assert run["target"]["mae"] >= 0
                if not finite(run):
                    raise SystemExit(f"non-finite result in {vendor}/{algorithm}")

    assert n_folds == len(results["data"]["vendors"])
    assert finite(results["hybrid"])
    assert results["hybrid"]["target_labels_used_for_weight_selection"] is False
    assert finite(results["physics_counterfactual"])
    assert "clean" in results["physics_counterfactual"]
    assert finite(results["uncertainty"])
    assert results["uncertainty"]["calibration_source"] == "source_validation_only"
    assert results["uncertainty"]["dropout_active_during_sampling"] is True
    assert finite(results["paired_representation"])
    assert finite(results["scaling"])

    external_count = verify_multiscanner_external()
    factor_count = verify_factor_holdout()
    joint_factor_count = verify_joint_factor_holdout()
    effect_count = verify_method_effects()
    suffix = (
        f", {external_count}-scanner analytical external check"
        if external_count is not None
        else ""
    )
    if factor_count is not None:
        suffix += f", {factor_count} acquisition-factor holdouts"
    if joint_factor_count is not None:
        suffix += f", {joint_factor_count} compositional acquisition holdout"
    if effect_count is not None:
        suffix += f", {effect_count} paired method-effect analyses"
    print(
        f"ALL CORRECTED CHECKS PASSED: {n_folds} folds, "
        f"{n_runs} model runs, {report['n_signals']} unique signals{suffix}"
    )


if __name__ == "__main__":
    main()
