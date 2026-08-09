#!/usr/bin/env python3
"""Audit the public VENUS summary without mixing it into the MRF benchmark.

VENUS provides derived multi-scanner qMRI summary values, not raw temporal MRF
signals or independent reference maps. This script therefore reports only
descriptive paired shifts and explicitly marks the result as contextual.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_session(value: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"sub-([^_]+)_ses-(.+)", value)
    if not match:
        raise ValueError(f"unexpected Session value: {value}")
    subject, session = match.groups()
    if session.startswith("rth") and session.endswith("rev"):
        arm = "rth"
        scanner = session[3:-3]
    elif session.startswith("vendor") and session.endswith("rev"):
        arm = "vendor"
        scanner = session[6:-3]
    else:
        raise ValueError(f"unexpected session label: {value}")
    if not scanner:
        raise ValueError(f"missing scanner suffix: {value}")
    return subject, arm, scanner


def summarize(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    variance = (
        sum((value - mean) ** 2 for value in ordered) / (n - 1)
        if n > 1 else 0.0
    )
    return {
        "n": n,
        "mean_delta": mean,
        "sd_delta": variance ** 0.5,
        "median_delta": median,
        "min_delta": min(ordered),
        "max_delta": max(ordered),
        "mean_absolute_delta": sum(abs(value) for value in ordered) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/venus/VENUS_summary_merged.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/venus_summary_audit.json"),
    )
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"missing {args.input}")

    variables = ["T1 (avg-all)", "MTsat (avg)", "MTR (avg)"]
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("VENUS summary is empty")
    missing = set(["Session", "Region", *variables]) - set(rows[0])
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")

    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        subject, arm, scanner = parse_session(row["Session"])
        key = (subject, scanner, row["Region"], arm)
        if key in records:
            raise ValueError(f"duplicate row: {key}")
        record = {
            "subject": subject,
            "scanner_suffix": scanner,
            "region": row["Region"],
            "arm": arm,
            "session": row["Session"],
        }
        for variable in variables:
            record[variable] = float(row[variable])
        records[key] = record

    pairs: list[dict[str, Any]] = []
    for subject, scanner, region, arm in sorted(records):
        if arm != "rth":
            continue
        rth = records[(subject, scanner, region, "rth")]
        vendor_key = (subject, scanner, region, "vendor")
        if vendor_key not in records:
            raise ValueError(f"missing paired vendor row: {vendor_key}")
        vendor = records[vendor_key]
        pair = {
            "subject": subject,
            "scanner_suffix": scanner,
            "region": region,
            "rth_session": rth["session"],
            "vendor_session": vendor["session"],
            "delta_definition": "vendor minus rth",
        }
        for variable in variables:
            pair[variable] = {
                "rth": rth[variable],
                "vendor": vendor[variable],
                "delta": vendor[variable] - rth[variable],
            }
        pairs.append(pair)

    by_variable: dict[str, Any] = {}
    for variable in variables:
        deltas = [pair[variable]["delta"] for pair in pairs]
        by_scanner = {}
        for scanner in sorted({pair["scanner_suffix"] for pair in pairs}):
            by_scanner[scanner] = summarize(
                [pair[variable]["delta"] for pair in pairs if pair["scanner_suffix"] == scanner]
            )
        by_variable[variable] = {
            "all_pairs": summarize(deltas),
            "by_scanner_suffix": by_scanner,
        }

    result = {
        "schema_version": "venus-summary-audit-v1",
        "provenance": {
            "dataset": "VENUS summary CSV",
            "official_project": "https://qmrlab.org/VENUS/",
            "osf_record": "https://osf.io/5n3cu/",
            "input_file": str(args.input),
            "input_md5": md5(args.input),
            "source_units": "Values are preserved in the units and scale of the supplied CSV.",
        },
        "scope_boundary": {
            "used_in_synthetic_training": False,
            "used_in_neural_model_selection": False,
            "used_in_primary_mrf_endpoint": False,
            "raw_temporal_mrf_available_in_this_audit": False,
            "independent_reference_maps_available_in_this_audit": False,
            "interpretation": (
                "Contextual descriptive evidence of derived qMRI measurement "
                "shift; not MRF accuracy validation and not clinical inference."
            ),
        },
        "input": {
            "n_rows": len(rows),
            "n_subjects": len({pair["subject"] for pair in pairs}),
            "n_scanner_suffixes": len({pair["scanner_suffix"] for pair in pairs}),
            "n_regions": len({pair["region"] for pair in pairs}),
            "n_paired_rows": len(pairs),
            "variables": variables,
        },
        "paired_rows": pairs,
        "summary": by_variable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
