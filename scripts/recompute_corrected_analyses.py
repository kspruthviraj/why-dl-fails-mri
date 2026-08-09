#!/usr/bin/env python3
"""Recompute non-training analyses with the corrected stochastic uncertainty code.

The benchmark checkpoint is not changed. One deterministic ERM analysis model
is retrained for the GE fold so hybrid, counterfactual, uncertainty, and CKA
outputs use the current implementation while the cached benchmark and scaling
results are preserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_corrected_benchmark import (
    DATA_PATH,
    ROOT,
    load_data,
    prepare_fold,
    run_counterfactual_physics,
    run_hybrid,
    run_paired_cka,
    run_uncertainty,
    train_model,
    write_paper_numbers,
)

RESULTS_PATH = ROOT / "results/corrected_benchmark.json"


def main() -> None:
    data = load_data(DATA_PATH)
    fold = prepare_fold(data, "ge")
    model, state = train_model(data, fold, "erm", 42)
    simulation_cfg = yaml.safe_load(
        (ROOT / "configs/config.yaml").read_text()
    )["simulation"]

    results = json.loads(RESULTS_PATH.read_text())
    results["hybrid"] = run_hybrid(
        data, fold, model, state["ymin"], state["ymax"]
    )
    results["physics_counterfactual"] = run_counterfactual_physics(
        simulation_cfg, model, state["ymin"], state["ymax"]
    )
    results["uncertainty"] = run_uncertainty(
        model,
        state["source_x"],
        state["source_y"],
        state["target_x"],
        state["target_y"],
        state["ymin"],
        state["ymax"],
    )
    results["paired_representation"] = run_paired_cka(
        simulation_cfg, model
    )
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    write_paper_numbers(results)
    print("Recomputed corrected analysis fields and generated paper values.")


if __name__ == "__main__":
    main()
