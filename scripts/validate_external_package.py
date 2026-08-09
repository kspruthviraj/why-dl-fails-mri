#!/usr/bin/env python3
"""Validate a frozen external temporal-MRF package before neural evaluation.

The validator is intentionally conservative. It checks a manifest, a per-scan
CSV index, relative file references, hashes when supplied, and the metadata
needed to detect subject or scan leakage. It does not infer that a final map
archive is a temporal fingerprint dataset. Missing packages produce a report
with status "missing"; --strict turns any status other than ready into a
non-zero exit code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data/external/external_mrf_package"
DEFAULT_OUTPUT = ROOT / "results/external_package_validation.json"
EXPECTED_MODEL_TIMEPOINTS = 1000

REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "package_id",
    "source_citation",
    "license",
    "sample_index",
    "expected_n_timepoints",
    "files",
    "data_contract",
}
REQUIRED_CONTRACT_FLAGS = {
    "raw_complex_fingerprints",
    "exact_acquisition_schedule",
    "independent_t1_t2_reference",
    "scanner_vendor_site_metadata",
    "scan_rescan_or_repeat",
    "raw_metadata_export",
}
REQUIRED_COLUMNS = {
    "subject_id",
    "scan_id",
    "site_id",
    "vendor",
    "scanner_model",
    "field_strength_t",
    "sequence_id",
    "repeat_id",
    "n_timepoints",
    "complex_encoding",
    "fingerprint_path",
    "schedule_path",
    "reference_t1_path",
    "reference_t2_path",
    "reference_independent",
    "split",
}
PATH_COLUMNS = {
    "fingerprint_path",
    "schedule_path",
    "reference_t1_path",
    "reference_t2_path",
    "mask_path",
}
TRUE_VALUES = {"1", "true", "yes", "y", "independent"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a frozen external temporal-MRF package."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Package directory (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Do not compute SHA-256 values for manifest files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless the package is ready for the frozen protocol.",
    )
    return parser.parse_args()


def validate_package(package_root: Path, skip_hash: bool = False) -> dict[str, Any]:
    package_root = package_root.expanduser()
    report: dict[str, Any] = {
        "schema_version": "external-mrf-intake-v1",
        "protocol": "frozen external temporal-MRF neural-evaluation intake",
        "root": str(package_root),
        "status": "missing",
        "checks": [],
        "blocking_items": [],
        "warnings": [],
        "recommendations": [],
        "counts": {},
    }

    def check(name: str, passed: bool, detail: str, blocking: bool = True) -> None:
        report["checks"].append(
            {"name": name, "passed": bool(passed), "detail": detail}
        )
        if not passed:
            (report["blocking_items"] if blocking else report["warnings"]).append(
                f"{name}: {detail}"
            )

    if not package_root.exists():
        report["recommendations"].append(
            "Place the author package at this path or pass --root explicitly."
        )
        return report
    if not package_root.is_dir():
        check("package_directory", False, "root exists but is not a directory")
        report["status"] = "invalid"
        return report

    report["status"] = "invalid"
    manifest_path = package_root / "manifest.json"
    if not manifest_path.exists():
        check("manifest_present", False, "manifest.json is missing")
        return report
    check("manifest_present", True, "manifest.json found")

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        check("manifest_json", False, f"cannot parse manifest.json: {exc}")
        return report
    check("manifest_json", True, "manifest.json is valid JSON")
    if not isinstance(manifest, dict):
        check("manifest_object", False, "manifest root must be an object")
        return report
    missing_keys = sorted(REQUIRED_MANIFEST_KEYS - set(manifest))
    check(
        "manifest_keys",
        not missing_keys,
        "all required keys present"
        if not missing_keys
        else f"missing {missing_keys}",
    )
    if manifest.get("schema_version") != "external-mrf-package-v1":
        check(
            "manifest_schema",
            False,
            "schema_version must be external-mrf-package-v1",
        )
    else:
        check("manifest_schema", True, "external-mrf-package-v1")

    contract = manifest.get("data_contract")
    if not isinstance(contract, dict):
        check("data_contract_object", False, "data_contract must be an object")
        contract = {}
    else:
        check("data_contract_object", True, "data_contract is an object")

    for flag in sorted(REQUIRED_CONTRACT_FLAGS):
        value = contract.get(flag) is True
        check(
            f"contract_{flag}",
            value,
            "declared true"
            if value
            else "must be explicitly true for neural external evaluation",
        )
    check(
        "external_evaluation_only",
        manifest.get("external_evaluation_only") is True,
        "package must be declared evaluation-only",
    )

    expected_timepoints = manifest.get("expected_n_timepoints")
    expected_ok = isinstance(expected_timepoints, int) and expected_timepoints > 0
    check(
        "expected_timepoints_integer",
        expected_ok,
        "expected_n_timepoints must be a positive integer",
    )
    if expected_ok:
        check(
            "model_timepoint_compatibility",
            expected_timepoints == EXPECTED_MODEL_TIMEPOINTS,
            f"current neural model requires {EXPECTED_MODEL_TIMEPOINTS} time points",
        )

    file_entries = manifest.get("files")
    if not isinstance(file_entries, list) or not file_entries:
        check("manifest_files", False, "files must be a non-empty list")
        file_entries = []
    else:
        check("manifest_files", True, f"{len(file_entries)} file entries declared")

    declared_paths: set[str] = set()
    package_resolved = package_root.resolve()
    for index, entry in enumerate(file_entries):
        if not isinstance(entry, dict):
            check(f"file_entry_{index}", False, "entry must be an object")
            continue
        relative = safe_relative(entry.get("path"))
        if relative is None:
            check(
                f"file_entry_{index}_path",
                False,
                "path must be a non-empty relative path without '..'",
            )
            continue
        key = relative.as_posix()
        declared_paths.add(key)
        resolved = (package_root / relative).resolve()
        try:
            resolved.relative_to(package_resolved)
            inside = True
        except ValueError:
            inside = False
        check(
            f"file_entry_{index}_inside_root",
            inside,
            "path resolves inside the package root",
        )
        if not inside or not resolved.exists() or not resolved.is_file():
            check(
                f"file_entry_{index}_exists",
                False,
                f"missing regular file: {key}",
            )
            continue
        check(f"file_entry_{index}_exists", True, key)
        expected_hash = entry.get("sha256")
        if expected_hash:
            if skip_hash:
                report["warnings"].append(
                    f"file_entry_{index}_hash: skipped by --skip-hash"
                )
            else:
                actual_hash = sha256_file(resolved)
                check(
                    f"file_entry_{index}_sha256",
                    actual_hash.lower() == str(expected_hash).lower(),
                    f"expected {expected_hash}, observed {actual_hash}",
                )
        else:
            report["warnings"].append(
                f"file_entry_{index}_sha256: no SHA-256 supplied"
            )

    sample_index_value = manifest.get("sample_index")
    sample_index = safe_relative(sample_index_value)
    sample_index_ok = sample_index is not None
    check(
        "sample_index_relative",
        sample_index_ok,
        "sample_index must be a relative path",
    )
    rows: list[dict[str, str]] = []
    if sample_index_ok:
        sample_path = (package_root / sample_index).resolve()
        try:
            sample_path.relative_to(package_resolved)
            sample_inside = True
        except ValueError:
            sample_inside = False
        check("sample_index_inside_root", sample_inside, "index is inside root")
        check(
            "sample_index_exists",
            sample_inside and sample_path.is_file(),
            f"sample index: {sample_index}",
        )
        if sample_inside and sample_path.is_file():
            if sample_index.as_posix() not in declared_paths:
                report["warnings"].append(
                    "sample_index_declared: samples.csv is not listed in manifest.files"
                )
            try:
                with sample_path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    headers = set(reader.fieldnames or [])
                    missing_columns = sorted(REQUIRED_COLUMNS - headers)
                    check(
                        "sample_index_columns",
                        not missing_columns,
                        "all required columns present"
                        if not missing_columns
                        else f"missing {missing_columns}",
                    )
                    rows = list(reader)
            except OSError as exc:
                check("sample_index_readable", False, str(exc))
            if not rows:
                check(
                    "sample_index_rows",
                    False,
                    "samples.csv must contain at least one scan row",
                )
            else:
                check("sample_index_rows", True, f"{len(rows)} scan rows")

    if rows:
        path_errors: set[str] = set()
        pair_keys: list[tuple[str, str]] = []
        subjects: set[str] = set()
        vendors: set[str] = set()
        sites: set[str] = set()
        scanners: set[str] = set()
        sequences: set[str] = set()
        repeats_by_subject: dict[str, set[str]] = {}
        reference_independence = True
        timepoint_values: set[int] = set()

        for row_number, row in enumerate(rows, start=2):
            def nonempty(column: str) -> str:
                value = str(row.get(column, "")).strip()
                if not value:
                    path_errors.add(f"row {row_number}: {column} is empty")
                return value

            subject = nonempty("subject_id")
            scan = nonempty("scan_id")
            site = nonempty("site_id")
            vendor = nonempty("vendor")
            scanner = nonempty("scanner_model")
            sequence = nonempty("sequence_id")
            repeat = nonempty("repeat_id")
            subjects.add(subject)
            vendors.add(vendor)
            sites.add(site)
            scanners.add(scanner)
            sequences.add(sequence)
            repeats_by_subject.setdefault(subject, set()).add(repeat)
            pair_keys.append((subject, scan))

            try:
                field_strength = float(nonempty("field_strength_t"))
                if field_strength <= 0:
                    raise ValueError
            except ValueError:
                path_errors.add(
                    f"row {row_number}: field_strength_t must be positive"
                )
            try:
                n_timepoints = int(nonempty("n_timepoints"))
                if n_timepoints <= 0:
                    raise ValueError
                timepoint_values.add(n_timepoints)
                if expected_ok and n_timepoints != expected_timepoints:
                    path_errors.add(
                        f"row {row_number}: n_timepoints={n_timepoints}, "
                        f"expected {expected_timepoints}"
                    )
            except ValueError:
                path_errors.add(
                    f"row {row_number}: n_timepoints must be a positive integer"
                )

            if not nonempty("complex_encoding"):
                path_errors.add(f"row {row_number}: complex_encoding is empty")
            if row.get("reference_independent", "").strip().lower() not in TRUE_VALUES:
                reference_independence = False
                path_errors.add(
                    f"row {row_number}: reference_independent must be true"
                )
            if row.get("split", "").strip() != "frozen_external_test":
                path_errors.add(
                    f"row {row_number}: split must be frozen_external_test"
                )

            for column in sorted(PATH_COLUMNS):
                value = str(row.get(column, "")).strip()
                if not value:
                    if column == "mask_path":
                        continue
                    path_errors.add(f"row {row_number}: {column} is empty")
                    continue
                relative = safe_relative(value)
                if relative is None:
                    path_errors.add(
                        f"row {row_number}: {column} is not a safe relative path"
                    )
                    continue
                resolved = (package_root / relative).resolve()
                try:
                    resolved.relative_to(package_resolved)
                    inside = True
                except ValueError:
                    inside = False
                if not inside or not resolved.is_file():
                    path_errors.add(
                        f"row {row_number}: missing {column} file {value}"
                    )

        duplicate_pairs = len(pair_keys) - len(set(pair_keys))
        check(
            "unique_subject_scan_pairs",
            duplicate_pairs == 0,
            f"{duplicate_pairs} duplicate subject/scan pairs",
        )
        check(
            "independent_reference_flag",
            reference_independence,
            "all rows declare independent references",
        )
        check(
            "row_level_integrity",
            not path_errors,
            "all rows have valid metadata and file references"
            if not path_errors
            else f"{len(path_errors)} row-level issues",
        )
        if path_errors:
            report["blocking_items"].extend(sorted(path_errors)[:100])
            if len(path_errors) > 100:
                report["blocking_items"].append(
                    f"... {len(path_errors) - 100} additional row-level issues"
                )
        report["counts"] = {
            "n_scans": len(rows),
            "n_subjects": len(subjects),
            "n_vendors": len(vendors),
            "n_sites": len(sites),
            "n_scanners": len(scanners),
            "n_sequences": len(sequences),
            "n_repeat_ids": len(
                {repeat for values in repeats_by_subject.values() for repeat in values}
            ),
            "n_timepoint_values": sorted(timepoint_values),
        }
        if len(subjects) < 20:
            report["warnings"].append(
                "fewer than 20 independent subjects: treat results as pilot evidence"
            )
        if len(vendors) < 2:
            report["blocking_items"].append(
                "fewer than two vendors: cannot support a cross-vendor neural claim"
            )
        if len(sites) < 2:
            report["warnings"].append(
                "fewer than two sites: site generalization remains untested"
            )
        if not any(len(repeats) >= 2 for repeats in repeats_by_subject.values()):
            report["warnings"].append(
                "no subject has multiple repeat IDs: scan-rescan reliability is unavailable"
            )
    else:
        report["counts"] = {}

    if report["blocking_items"]:
        report["status"] = "invalid"
    else:
        counts = report["counts"]
        if counts.get("n_vendors", 0) >= 2:
            report["status"] = (
                "ready_with_warnings" if report["warnings"] else "ready"
            )
        else:
            report["status"] = "ready_but_insufficient_for_cross_vendor_claim"

    report["recommendations"].extend(
        [
            "Freeze model weights, seeds, preprocessing, and primary endpoints before reading external references.",
            "Hold out complete subjects and, where possible, an entire scanner/site for the final external test.",
            "Report subject- or tube-clustered confidence intervals, bias, MAE, RMSE, Bland-Altman limits, and scan-rescan repeatability.",
            "Keep map-only archives as analytical or reproducibility context; do not feed final maps into the temporal neural model.",
        ]
    )
    return report


def main() -> int:
    args = parse_args()
    report = validate_package(args.root, skip_hash=args.skip_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"External package status: {report['status']} "
        f"(report: {args.output})"
    )
    if report["blocking_items"]:
        print("Blocking items:", file=sys.stderr)
        for item in report["blocking_items"][:20]:
            print(f" - {item}", file=sys.stderr)
    if args.strict and report["status"] not in {"ready", "ready_with_warnings"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
