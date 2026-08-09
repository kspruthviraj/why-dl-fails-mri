#!/usr/bin/env python3
"""Validate the corrected synthetic MRF dataset.

This is intentionally a hard gate: the benchmark must not run on a file with
duplicate sample IDs, duplicate signal rows, missing physical metadata, or
non-finite values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


REQUIRED = {
    "signals",
    "parameters",
    "domain_labels",
    "sample_ids",
    "b0_hz",
    "b1_scale",
    "snr",
    "field_strength",
    "fa_variant",
    "tr_variant",
}


def _decode(values):
    return np.asarray([
        x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in values
    ])


def _row_hashes(values):
    return [
        hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).hexdigest()
        for row in values
    ]


def validate(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        missing = sorted(REQUIRED - set(f.keys()))
        if missing:
            raise ValueError(f"missing required datasets: {missing}")

        signals = f["signals"][:]
        parameters = f["parameters"][:, :2].astype(np.float32)
        sample_ids = f["sample_ids"][:].astype(np.int64)
        domains = _decode(f["domain_labels"][:])
        metadata = {
            name: f[name][:] for name in (
                "b0_hz", "b1_scale", "snr", "field_strength",
                "fa_variant", "tr_variant",
            )
        }
        attrs = {str(k): str(v) for k, v in f.attrs.items()}

    n = len(signals)
    if signals.ndim != 2 or signals.shape[1] < 8:
        raise ValueError(f"unexpected signal shape: {signals.shape}")
    if parameters.shape != (n, 2):
        raise ValueError("parameter rows do not match signals")
    if len(np.unique(sample_ids)) != n:
        raise ValueError("sample_ids are not unique")
    if not np.isfinite(signals.real).all() or not np.isfinite(signals.imag).all():
        raise ValueError("signals contain non-finite values")
    if not np.isfinite(parameters).all():
        raise ValueError("parameters contain non-finite values")
    for name, values in metadata.items():
        if len(values) != n or not np.isfinite(values).all():
            raise ValueError(f"invalid metadata dataset: {name}")

    signal_hashes = _row_hashes(signals)
    parameter_hashes = _row_hashes(parameters)
    unique_signal_hashes = len(set(signal_hashes))
    unique_parameter_hashes = len(set(parameter_hashes))
    duplicate_signal_rows = n - unique_signal_hashes
    duplicate_parameter_rows = n - unique_parameter_hashes
    if duplicate_signal_rows:
        raise ValueError(
            "duplicate signal rows detected: "
            f"signals={duplicate_signal_rows}"
        )

    domain_counts = {
        str(domain): int(np.sum(domains == domain))
        for domain in sorted(np.unique(domains))
    }
    vendor_counts = {}
    for domain, count in domain_counts.items():
        vendor = domain.split("_", 1)[0]
        vendor_counts[vendor] = vendor_counts.get(vendor, 0) + count

    report = {
        "path": str(path),
        "n_signals": int(n),
        "signal_shape": list(signals.shape),
        "n_unique_sample_ids": int(len(np.unique(sample_ids))),
        "n_unique_signal_hashes": int(unique_signal_hashes),
        "n_unique_parameter_rows": int(unique_parameter_hashes),
        "duplicate_signal_rows": int(duplicate_signal_rows),
        "duplicate_parameter_rows": int(duplicate_parameter_rows),
        "n_domains": int(len(domain_counts)),
        "domain_counts": domain_counts,
        "vendor_counts": vendor_counts,
        "parameter_min": parameters.min(axis=0).tolist(),
        "parameter_max": parameters.max(axis=0).tolist(),
        "metadata_ranges": {
            name: [float(np.min(values)), float(np.max(values))]
            for name, values in metadata.items()
        },
        "attrs": attrs,
        "valid": True,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/synthetic/mrf_corrected_100k.h5")
    parser.add_argument("--output", default="results/data_validation.json")
    args = parser.parse_args()

    report = validate(Path(args.path))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
