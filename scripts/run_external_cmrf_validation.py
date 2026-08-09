#!/usr/bin/env python3
"""Reconstruct and validate the public cardiac-MRF phantom reference scan.

This is an optional external validation stage. It intentionally does not train
or tune the synthetic neural benchmark. It reconstructs the raw ISMRMRD/HDF5
file using the public MRpro implementation, performs the published cMRF
sliding-window dictionary matching, and compares tube-level estimates with the
independent reference maps supplied by Zenodo.

Run with the isolated environment created for this project, for example:
    MRPRO_PYTHON=.venv_mrpro/bin/python bash reproduce.sh external
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def paired_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    mean_bias = float(np.mean(diff))
    sd_bias = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    return {
        "n": int(diff.size),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mean_bias": mean_bias,
        "bias_sd": sd_bias,
        "bland_altman_lower": float(mean_bias - 1.96 * sd_bias),
        "bland_altman_upper": float(mean_bias + 1.96 * sd_bias),
        "pearson_r": _pearson(pred, ref),
        "r_squared": float(1.0 - np.sum(diff**2) / max(np.sum((ref - np.mean(ref)) ** 2), 1e-12)),
    }


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_maps(input_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import torch
        import mrpro
        from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
    except ImportError as exc:
        raise SystemExit(
            "External validation requires MRpro and its ISMRMRD dependencies. "
            "Set MRPRO_PYTHON to the isolated project environment."
        ) from exc

    torch.set_num_threads(int(os.environ.get("MRPRO_THREADS", "8")))
    kdata = mrpro.data.KData.from_file(input_path, KTrajectoryIsmrmrd())
    n_acquisitions = int(kdata.data.shape[-2])
    n_acq_per_block = int(os.environ.get("CMRF_ACQ_PER_BLOCK", "47"))
    if n_acquisitions % n_acq_per_block:
        raise ValueError(
            f"{input_path} has {n_acquisitions} acquisitions, which is not "
            f"divisible by {n_acq_per_block}"
        )
    n_blocks = n_acquisitions // n_acq_per_block
    n_acq_per_image = int(os.environ.get("CMRF_ACQ_PER_IMAGE", "20"))
    n_overlap = int(os.environ.get("CMRF_OVERLAP", "10"))

    idx_in_block = torch.arange(n_acq_per_block).unfold(
        0, n_acq_per_image, n_acq_per_image - n_overlap
    )
    split_indices = (
        n_acq_per_block * torch.arange(n_blocks)[:, None, None] + idx_in_block
    ).flatten(end_dim=1)
    kdata_split = kdata[..., split_indices, :]

    average_reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(kdata)
    reconstruction = mrpro.algorithms.reconstruction.DirectReconstruction(
        kdata_split, csm=average_reconstruction.csm_op
    )
    image = reconstruction(kdata_split)

    model = mrpro.operators.AveragingOp(dim=0, idx=split_indices) @ mrpro.operators.models.CardiacFingerprinting(
        kdata.header.acq_info.acquisition_time_stamp.squeeze(),
        echo_time=0.001555,
        repetition_time=0.01,
        t2_prep_echo_times=(0.03, 0.05, 0.1),
    )
    dictionary = mrpro.operators.DictionaryMatchOp(
        model, index_of_scaling_parameter=0
    )
    dictionary.append(
        torch.tensor(1.0),
        torch.arange(0.05, 2.0, 0.01)[:, None],
        torch.arange(0.006, 0.2, 0.002)[None, :],
    )
    _, t1_match, t2_match = dictionary(image.data[:, 0, 0, 0])

    t1 = t1_match.detach().cpu().numpy().astype(np.float32) * 1000.0
    t2 = t2_match.detach().cpu().numpy().astype(np.float32) * 1000.0
    diagnostics = {
        "mrpro_version": str(getattr(mrpro, "__version__", "unknown")),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "input_data_shape": [int(x) for x in kdata.data.shape],
        "reconstructed_window_shape": [int(x) for x in image.data.shape],
        "n_acquisitions": n_acquisitions,
        "n_blocks": n_blocks,
        "n_acq_per_block": n_acq_per_block,
        "n_acq_per_image": n_acq_per_image,
        "window_overlap": n_overlap,
        "finite_reconstruction": bool(torch.isfinite(image.data).all().item()),
        "finite_t1": bool(np.isfinite(t1).all()),
        "finite_t2": bool(np.isfinite(t2).all()),
    }
    return t1, t2, diagnostics


def tube_summary(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    tube_ids = [int(value) for value in np.unique(mask) if value > 0]
    pred_means = []
    ref_means = []
    rows = []
    for tube_id in tube_ids:
        selection = mask == tube_id
        pred_values = pred[selection]
        ref_values = ref[selection]
        pred_mean = float(np.mean(pred_values))
        ref_mean = float(np.mean(ref_values))
        pred_means.append(pred_mean)
        ref_means.append(ref_mean)
        metrics = paired_metrics(pred_values, ref_values)
        rows.append({
            "tube": tube_id,
            "n_voxels": int(np.sum(selection)),
            "predicted_mean_ms": pred_mean,
            "reference_mean_ms": ref_mean,
            "mean_bias_ms": float(np.mean(pred_values - ref_values)),
            "voxel_mae_ms": metrics["mae"],
            "voxel_rmse_ms": metrics["rmse"],
        })

    pred_means_array = np.asarray(pred_means, dtype=np.float64)
    ref_means_array = np.asarray(ref_means, dtype=np.float64)
    roi_metrics = paired_metrics(pred_means_array, ref_means_array)
    voxel_metrics = paired_metrics(pred[mask > 0], ref[mask > 0])
    return {
        "tube_ids": tube_ids,
        "per_tube": rows,
        "roi_mean_metrics_ms": roi_metrics,
        "voxel_metrics_ms": voxel_metrics,
    }, {"predicted_means_ms": pred_means, "reference_means_ms": ref_means}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/cmrf_reference/cMRF.h5"),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/external/cmrf_reference"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/external_cmrf_validation.json"),
    )
    args = parser.parse_args()
    for required in (
        args.input,
        args.reference_dir / "mask.npy",
        args.reference_dir / "ref_t1.npy",
        args.reference_dir / "ref_t2.npy",
    ):
        if not required.exists():
            raise SystemExit(f"missing external validation file: {required}")

    t1_pred, t2_pred, reconstruction = reconstruct_maps(args.input)
    mask = np.load(args.reference_dir / "mask.npy").astype(np.int16)
    t1_ref = np.load(args.reference_dir / "ref_t1.npy").astype(np.float32)
    t2_ref = np.load(args.reference_dir / "ref_t2.npy").astype(np.float32)
    if t1_pred.shape != mask.shape or t2_pred.shape != mask.shape:
        raise ValueError(
            f"map/reference shape mismatch: T1={t1_pred.shape}, "
            f"T2={t2_pred.shape}, mask={mask.shape}"
        )

    t1_summary, t1_vectors = tube_summary(t1_pred, t1_ref, mask)
    t2_summary, t2_vectors = tube_summary(t2_pred, t2_ref, mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    map_dir = args.output.parent / "external_cmrf_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    np.save(map_dir / "t1_pred_ms.npy", t1_pred)
    np.save(map_dir / "t2_pred_ms.npy", t2_pred)

    result = {
        "schema_version": "external-cmrf-v1",
        "provenance": {
            "dataset": "Open-Source Cardiac MR Fingerprinting",
            "zenodo_record": "15726937",
            "doi": "10.5281/zenodo.15726937",
            "input_file": str(args.input),
            "input_md5": md5(args.input),
            "reference_files": {
                name: {
                    "path": str(args.reference_dir / name),
                    "md5": md5(args.reference_dir / name),
                }
                for name in ("mask.npy", "ref_t1.npy", "ref_t2.npy")
            },
            "license_note": "Use and cite the Zenodo record and associated cMRF publication.",
        },
        "task": {
            "description": "Independent raw cMRF reconstruction and dictionary matching against supplied reference maps",
            "model_training_performed": False,
            "neural_benchmark_tuning_performed": False,
            "reference_maps_used_for_dictionary_or_training": False,
            "reference_maps_used_only_for_final_evaluation": True,
            "units": "milliseconds",
        },
        "reconstruction": reconstruction,
        "t1": t1_summary,
        "t2": t2_summary,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    (args.output.parent / "external_cmrf_tube_vectors.json").write_text(
        json.dumps({"t1": t1_vectors, "t2": t2_vectors}, indent=2) + "\n"
    )

    # Generate only letter-based TeX macro names; numeric characters would be
    # parsed as visible text when used in control words.
    t1_roi = t1_summary["roi_mean_metrics_ms"]
    t2_roi = t2_summary["roi_mean_metrics_ms"]
    macro_lines = [
        "% Generated by scripts/run_external_cmrf_validation.py.",
        f"\\newcommand{{\\ExternalTOneRoiMAE}}{{{t1_roi['mae']:.1f}}}",
        f"\\newcommand{{\\ExternalTOneRoiBias}}{{{t1_roi['mean_bias']:.1f}}}",
        f"\\newcommand{{\\ExternalTOneRoiLoALower}}{{{t1_roi['bland_altman_lower']:.1f}}}",
        f"\\newcommand{{\\ExternalTOneRoiLoAUpper}}{{{t1_roi['bland_altman_upper']:.1f}}}",
        f"\\newcommand{{\\ExternalTTwoRoiMAE}}{{{t2_roi['mae']:.1f}}}",
        f"\\newcommand{{\\ExternalTTwoRoiBias}}{{{t2_roi['mean_bias']:.1f}}}",
        f"\\newcommand{{\\ExternalTTwoRoiLoALower}}{{{t2_roi['bland_altman_lower']:.1f}}}",
        f"\\newcommand{{\\ExternalTTwoRoiLoAUpper}}{{{t2_roi['bland_altman_upper']:.1f}}}",
        f"\\newcommand{{\\ExternalTOneRoiRtwo}}{{{t1_roi['r_squared']:.3f}}}",
        f"\\newcommand{{\\ExternalTTwoRoiRtwo}}{{{t2_roi['r_squared']:.3f}}}",
    ]
    (Path("paper") / "external_numbers.tex").write_text("\n".join(macro_lines) + "\n")
    table_lines = [
        "% Generated by scripts/run_external_cmrf_validation.py.",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Parameter & Tube MAE & Mean bias & LoA lower & LoA upper \\\\",
        "\\midrule",
        f"$T_1$ (ms) & {t1_roi['mae']:.1f} & {t1_roi['mean_bias']:.1f} & {t1_roi['bland_altman_lower']:.1f} & {t1_roi['bland_altman_upper']:.1f} \\\\",
        f"$T_2$ (ms) & {t2_roi['mae']:.1f} & {t2_roi['mean_bias']:.1f} & {t2_roi['bland_altman_lower']:.1f} & {t2_roi['bland_altman_upper']:.1f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "}",
    ]
    (Path("paper") / "generated_external_summary.tex").write_text(
        "\n".join(table_lines) + "\n"
    )
    print(json.dumps({"output": str(args.output), "t1": t1_roi, "t2": t2_roi}, indent=2))


if __name__ == "__main__":
    main()
