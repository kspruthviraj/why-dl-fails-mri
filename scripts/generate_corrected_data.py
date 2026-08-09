#!/usr/bin/env python3
"""Generate the versioned corrected synthetic MRF dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from qMR_Robust.simulators.manager import SimulationManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic/mrf_corrected_100k.h5")
    parser.add_argument("--n-signals", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["simulation"]["mrf"].update({
        "n_signals": args.n_signals,
        "n_workers": args.workers,
        "vendors": ["siemens", "philips", "ge"],
        "field_strengths": [1.5, 3.0],
        "fa_schedule_variants": 3,
        "tr_schedule_variants": 2,
    })
    output = Path(args.output)
    SimulationManager(cfg).generate_mrf(str(output), n_signals=args.n_signals)
    print(f"generated {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
