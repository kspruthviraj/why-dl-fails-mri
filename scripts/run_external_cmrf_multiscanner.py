#!/usr/bin/env python3
"""Reproduce the public four-scanner cMRF analytical validation.

This follows the public PTB-MR/cMRF scanner-comparison notebook:
raw cMRF data are reconstructed with MRpro, cMRF dictionaries are generated
with the published EPG model and flip-angle schedule, and independent raw
spin-echo T1/T2 scans are reconstructed and matched for reference maps.

The experiment is intentionally analytical. It never trains, tunes, or
evaluates the synthetic neural benchmark and it does not use target labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = pred - ref
    mean_bias = float(np.mean(diff))
    sd_bias = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    ref_ss = float(np.sum((ref - np.mean(ref)) ** 2))
    return {
        "n": int(diff.size),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mean_bias": mean_bias,
        "bias_sd": sd_bias,
        "bland_altman_lower": float(mean_bias - 1.96 * sd_bias),
        "bland_altman_upper": float(mean_bias + 1.96 * sd_bias),
        "pearson_r": (
            float(np.corrcoef(pred, ref)[0, 1])
            if diff.size > 1 and np.std(pred) > 0 and np.std(ref) > 0
            else float("nan")
        ),
        "r_squared": float(1.0 - np.sum(diff**2) / max(ref_ss, 1e-12)),
    }


def _import_mrpro() -> dict[str, Any]:
    try:
        import torch
        from einops import rearrange
        from mrpro.algorithms.reconstruction import DirectReconstruction
        from mrpro.data import CsmData, DcfData, IData, KData
        from mrpro.data.traj_calculators import (
            KTrajectoryCartesian,
            KTrajectoryIsmrmrd,
        )
        from mrpro.operators import MagnitudeOp
        from mrpro.operators.models import InversionRecovery, MonoExponentialDecay
        from mrpro.operators.models.EPG import EpgMrfFispWithPreparation
        from mrpro.utils import split_idx
    except ImportError as exc:
        raise SystemExit(
            "Multi-scanner validation requires the public cMRF MRpro branch. "
            "Set MRPRO_PYTHON to .venv_mrpro/bin/python and ensure "
            "data/external/mrpro_cmrf/src is on PYTHONPATH."
        ) from exc

    return {
        "torch": torch,
        "rearrange": rearrange,
        "DirectReconstruction": DirectReconstruction,
        "CsmData": CsmData,
        "DcfData": DcfData,
        "IData": IData,
        "KData": KData,
        "KTrajectoryCartesian": KTrajectoryCartesian,
        "KTrajectoryIsmrmrd": KTrajectoryIsmrmrd,
        "MagnitudeOp": MagnitudeOp,
        "InversionRecovery": InversionRecovery,
        "MonoExponentialDecay": MonoExponentialDecay,
        "EpgMrfFispWithPreparation": EpgMrfFispWithPreparation,
        "split_idx": split_idx,
    }


def load_dictionary_grid(torch: Any) -> tuple[Any, Any]:
    t1 = torch.cat(
        (
            torch.arange(50, 2000 + 10, 10, dtype=torch.float32),
            torch.arange(2020, 3000 + 20, 20, dtype=torch.float32),
            torch.arange(3050, 5000 + 50, 50, dtype=torch.float32),
        )
    )
    t2 = torch.cat(
        (
            torch.arange(6, 100 + 2, 2, dtype=torch.float32),
            torch.arange(105, 200 + 5, 5, dtype=torch.float32),
            torch.arange(220, 500 + 20, 20, dtype=torch.float32),
        )
    )
    t1_grid, t2_grid = torch.broadcast_tensors(t1[None, :], t2[:, None])
    t1_all = t1_grid.flatten()
    t2_all = t2_grid.flatten()
    keep = t1_all >= t2_all
    return t1_all[keep], t2_all[keep]


def load_flip_angles(path: Path, torch: Any) -> Any:
    values = [float(line) for line in path.read_text().splitlines() if line.strip()]
    return torch.as_tensor(values, dtype=torch.float32) / 180.0 * torch.pi


def multi_image_reconstruction(kdata: Any, mr: dict[str, Any]) -> Any:
    DirectReconstruction = mr["DirectReconstruction"]
    IData = mr["IData"]
    CsmData = mr["CsmData"]

    first_reconstruction = DirectReconstruction(kdata=kdata, csm=None)
    first_image = first_reconstruction(kdata)
    first_slice = IData(data=first_image.data[0, None], header=first_image.header)
    csm = CsmData.from_idata_inati(first_slice)
    reconstruction = DirectReconstruction(kdata=kdata, csm=csm)
    return reconstruction(kdata)


def match_reference(
    image: Any,
    model: Any,
    dictionary_values: Any,
    mask: np.ndarray,
    mr: dict[str, Any],
    *,
    absolute_input: bool,
) -> np.ndarray:
    torch = mr["torch"]
    _, _, _, height, width = image.data.shape
    coords = np.flatnonzero(mask.reshape(-1) > 0)
    y = coords // width
    x = coords % width
    image_data = image.data[:, 0, 0]
    if absolute_input:
        image_data = image_data.abs()
    selected = image_data[:, y, x].T.contiguous()

    with torch.no_grad():
        (dictionary,) = model(torch.ones(1), dictionary_values)
        dictionary = dictionary.to(dtype=selected.dtype)
        norms = torch.linalg.vector_norm(dictionary, dim=0).clamp_min(1e-8)
        dictionary = dictionary / norms
        scores = torch.mm(selected, dictionary)
        best = torch.argmax(torch.abs(scores), dim=1)
        values = dictionary_values[best].detach().cpu().numpy().astype(np.float32)

    result = np.full((height, width), np.nan, dtype=np.float32)
    result.reshape(-1)[coords] = values
    return result


def reconstruct_reference_maps(
    scanner_dir: Path,
    mask: np.ndarray,
    t1_values: Any,
    t2_values: Any,
    mr: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch = mr["torch"]
    KData = mr["KData"]
    KTrajectoryCartesian = mr["KTrajectoryCartesian"]
    MonoExponentialDecay = mr["MonoExponentialDecay"]
    InversionRecovery = mr["InversionRecovery"]
    MagnitudeOp = mr["MagnitudeOp"]

    t2_kdata = KData.from_file(
        scanner_dir / "ref_t2.h5", KTrajectoryCartesian()
    )
    t2_kdata.header.recon_matrix.y = 128
    t2_image = multi_image_reconstruction(t2_kdata, mr)
    t2_model = MonoExponentialDecay(decay_time=t2_image.header.te * 1000)
    t2_map = match_reference(
        t2_image,
        t2_model,
        t2_values,
        mask,
        mr,
        absolute_input=False,
    )

    t1_kdata = KData.from_file(
        scanner_dir / "ref_t1.h5", KTrajectoryCartesian()
    )
    t1_kdata.header.recon_matrix.y = 128
    t1_kdata.header.ti = torch.as_tensor(
        [25, 50, 300, 600, 1200, 2400, 4800], dtype=torch.float32
    ) / 1000.0
    t1_image = multi_image_reconstruction(t1_kdata, mr)
    t1_model = MagnitudeOp() @ InversionRecovery(ti=t1_image.header.ti * 1000)
    t1_map = match_reference(
        t1_image,
        t1_model,
        t1_values,
        mask,
        mr,
        absolute_input=True,
    )
    diagnostics = {
        "t1_raw_shape": [int(x) for x in t1_kdata.data.shape],
        "t2_raw_shape": [int(x) for x in t2_kdata.data.shape],
        "t1_reconstruction_shape": [int(x) for x in t1_image.data.shape],
        "t2_reconstruction_shape": [int(x) for x in t2_image.data.shape],
        "t1_echo_or_inversion_times_ms": [25, 50, 300, 600, 1200, 2400, 4800],
        "t2_echo_times_ms": [
            float(x) for x in (t2_image.header.te * 1000).detach().cpu()
        ],
    }
    return t1_map, t2_map, diagnostics


def reconstruct_cmrf_map(
    scanner_dir: Path,
    mask: np.ndarray,
    flip_angles: Any,
    t1_values: Any,
    t2_values: Any,
    mr: dict[str, Any],
    dictionary_chunk: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch = mr["torch"]
    rearrange = mr["rearrange"]
    KData = mr["KData"]
    KTrajectoryIsmrmrd = mr["KTrajectoryIsmrmrd"]
    DirectReconstruction = mr["DirectReconstruction"]
    DcfData = mr["DcfData"]
    EpgMrfFispWithPreparation = mr["EpgMrfFispWithPreparation"]
    split_idx = mr["split_idx"]

    kdata = KData.from_file(scanner_dir / "cMRF.h5", KTrajectoryIsmrmrd())
    n_acq_per_block = 47
    n_acq_per_image = 20
    n_overlap = 10
    if int(kdata.data.shape[-2]) != 705:
        raise ValueError(
            f"{scanner_dir} has {kdata.data.shape[-2]} cMRF acquisitions; "
            "the published scanner comparison expects 705."
        )

    average_reconstruction = DirectReconstruction(kdata)
    dynamic_indices = split_idx(
        torch.arange(0, n_acq_per_block),
        n_acq_per_image,
        n_overlap,
    )
    dynamic_indices = torch.cat(
        [dynamic_indices + block * n_acq_per_block for block in range(15)],
        dim=0,
    )
    dynamic_kdata = kdata.split_k1_into_other(
        dynamic_indices, other_label="repetition"
    )
    dynamic_reconstruction = DirectReconstruction(
        dynamic_kdata, csm=average_reconstruction.csm
    )
    dcf_data = rearrange(
        average_reconstruction.dcf.data,
        "k2 k1 other k0 -> other k2 k1 k0",
    )
    dcf_data = rearrange(
        dcf_data[dynamic_indices.flatten(), ...],
        "(other k1) 1 k2 k0 -> other k2 k1 k0",
        k1=dynamic_indices.shape[-1],
    )
    dynamic_reconstruction.dcf = DcfData(dcf_data)
    image = dynamic_reconstruction(dynamic_kdata).rss()[:, 0, :, :]

    height, width = int(image.shape[-2]), int(image.shape[-1])
    coords = np.flatnonzero(mask.reshape(-1) > 0)
    y = torch.as_tensor(coords // width, dtype=torch.long)
    x = torch.as_tensor(coords % width, dtype=torch.long)
    selected = image[:, y, x].T.contiguous()
    best_score = torch.full(
        (selected.shape[0],), -float("inf"), dtype=torch.float32
    )
    best_t1 = torch.zeros_like(best_score)
    best_t2 = torch.zeros_like(best_score)

    acquisition_time_ms = (
        kdata.header.acq_info.acquisition_time_stamp[0, 0, :, 0] * 2.5
    )
    delay_between_blocks = [
        acquisition_time_ms[block * n_acq_per_block]
        - acquisition_time_ms[block * n_acq_per_block - 1]
        for block in range(1, 3 * 5)
    ]
    delay_between_blocks.append(delay_between_blocks[-1])
    delay_due_to_preparation = [0, 30, 50, 100, 21] * 3
    delay_after_block = [
        trigger - preparation
        for preparation, trigger in zip(
            delay_due_to_preparation, delay_between_blocks
        )
    ]
    model = EpgMrfFispWithPreparation(
        flip_angles,
        0.0,
        1.52,
        kdata.header.tr * 1000,
        [21, None, None, None, None] * 3,
        [None, None, 30, 50, 100] * 3,
        n_acq_per_block,
        delay_after_block,
    )

    with torch.no_grad():
        for start in range(0, len(t1_values), dictionary_chunk):
            stop = min(start + dictionary_chunk, len(t1_values))
            t1_chunk = t1_values[start:stop]
            t2_chunk = t2_values[start:stop]
            (dictionary,) = model.forward(
                torch.ones_like(t1_chunk), t1_chunk, t2_chunk
            )
            dictionary = rearrange(
                dictionary[dynamic_indices.flatten(), :],
                "(other k1) t -> other t k1",
                k1=n_acq_per_image,
            )
            dictionary = dictionary.mean(dim=-1).abs()
            dictionary = dictionary / torch.linalg.vector_norm(
                dictionary, dim=0
            ).clamp_min(1e-8)
            scores = torch.mm(selected, dictionary)
            values, indices = scores.max(dim=1)
            update = values > best_score
            best_score[update] = values[update]
            best_t1[update] = t1_chunk[indices[update]]
            best_t2[update] = t2_chunk[indices[update]]
            print(f"  dictionary {stop}/{len(t1_values)}", flush=True)

    t1_map = np.full((height, width), np.nan, dtype=np.float32)
    t2_map = np.full((height, width), np.nan, dtype=np.float32)
    flat_t1 = t1_map.reshape(-1)
    flat_t2 = t2_map.reshape(-1)
    flat_t1[coords] = best_t1.detach().cpu().numpy()
    flat_t2[coords] = best_t2.detach().cpu().numpy()
    diagnostics = {
        "raw_shape": [int(x) for x in kdata.data.shape],
        "dynamic_index_shape": [int(x) for x in dynamic_indices.shape],
        "reconstruction_shape": [int(x) for x in image.shape],
        "n_acquisitions": int(kdata.data.shape[-2]),
        "n_receiver_channels": int(kdata.data.shape[-4]),
        "n_windows": int(dynamic_indices.shape[0]),
        "n_acq_per_block": n_acq_per_block,
        "n_acq_per_image": n_acq_per_image,
        "window_overlap": n_overlap,
        "dictionary_entries": int(len(t1_values)),
        "finite_reconstruction": bool(torch.isfinite(image).all().item()),
        "finite_t1": bool(np.isfinite(t1_map[mask > 0]).all()),
        "finite_t2": bool(np.isfinite(t2_map[mask > 0]).all()),
    }
    return t1_map, t2_map, diagnostics


def tube_summary(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    tube_ids = [int(x) for x in np.unique(mask) if x > 0]
    pred_means: list[float] = []
    ref_means: list[float] = []
    rows: list[dict[str, Any]] = []
    for tube_id in tube_ids:
        selection = mask == tube_id
        pred_values = pred[selection]
        ref_values = ref[selection]
        pred_mean = float(np.mean(pred_values))
        ref_mean = float(np.mean(ref_values))
        pred_means.append(pred_mean)
        ref_means.append(ref_mean)
        voxel = paired_metrics(pred_values, ref_values)
        rows.append(
            {
                "tube": tube_id,
                "n_voxels": int(selection.sum()),
                "predicted_mean_ms": pred_mean,
                "reference_mean_ms": ref_mean,
                "mean_bias_ms": float(np.mean(pred_values - ref_values)),
                "voxel_mae_ms": voxel["mae"],
                "voxel_rmse_ms": voxel["rmse"],
            }
        )
    pred_means_np = np.asarray(pred_means, dtype=np.float64)
    ref_means_np = np.asarray(ref_means, dtype=np.float64)
    return {
        "tube_ids": tube_ids,
        "per_tube": rows,
        "tube_mean_metrics_ms": paired_metrics(pred_means_np, ref_means_np),
        "voxel_metrics_ms": paired_metrics(pred[mask > 0], ref[mask > 0]),
    }


def scanner_metadata(scanner_dir: Path, mr: dict[str, Any]) -> dict[str, Any]:
    KData = mr["KData"]
    KTrajectoryIsmrmrd = mr["KTrajectoryIsmrmrd"]
    kdata = KData.from_file(scanner_dir / "cMRF.h5", KTrajectoryIsmrmrd())
    h = kdata.header
    return {
        "scanner_directory": scanner_dir.name,
        "vendor": str(h.vendor),
        "model": str(h.model),
        "protocol_name": str(h.protocol_name),
        "field_strength_t_approx": float(h.lamor_frequency_proton) / 42.577e6,
        "reconstruction_matrix": [
            int(h.recon_matrix.z),
            int(h.recon_matrix.y),
            int(h.recon_matrix.x),
        ],
        "cMRF_input_shape": [int(x) for x in kdata.data.shape],
        "cMRF_md5": md5(scanner_dir / "cMRF.h5"),
        "ref_t1_md5": md5(scanner_dir / "ref_t1.h5"),
        "ref_t2_md5": md5(scanner_dir / "ref_t2.h5"),
        "mask_md5": md5(scanner_dir / "mask.npy"),
    }


def tex_escape(value: str) -> str:
    value = value.replace("\\", r"\textbackslash{}")
    value = value.replace("_", r"\_")
    value = value.replace("&", r"\&")
    return value


def write_tex(results: dict[str, Any], paper_dir: Path) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    t1_all = results["aggregate"]["t1"]["pooled_tube_mean_metrics_ms"]
    t2_all = results["aggregate"]["t2"]["pooled_tube_mean_metrics_ms"]
    count = results["aggregate"]["n_scanners"]
    lines = [
        "% Generated by scripts/run_external_cmrf_multiscanner.py.",
        f"\\newcommand{{\\ExternalMultiScannerCount}}{{{count}}}",
        f"\\newcommand{{\\ExternalMultiScannerTOneMAE}}{{{t1_all['mae']:.1f}}}",
        f"\\newcommand{{\\ExternalMultiScannerTTwoMAE}}{{{t2_all['mae']:.1f}}}",
        f"\\newcommand{{\\ExternalMultiScannerTOneBias}}{{{t1_all['mean_bias']:.1f}}}",
        f"\\newcommand{{\\ExternalMultiScannerTTwoBias}}{{{t2_all['mean_bias']:.1f}}}",
        f"\\newcommand{{\\ExternalMultiScannerTOneLoALower}}{{{t1_all['bland_altman_lower']:.1f}}}",
        f"\\newcommand{{\\ExternalMultiScannerTOneLoAUpper}}{{{t1_all['bland_altman_upper']:.1f}}}",
        f"\\newcommand{{\\ExternalMultiScannerTTwoLoALower}}{{{t2_all['bland_altman_lower']:.1f}}}",
        f"\\newcommand{{\\ExternalMultiScannerTTwoLoAUpper}}{{{t2_all['bland_altman_upper']:.1f}}}",
    ]
    (paper_dir / "external_multiscanner_numbers.tex").write_text(
        "\n".join(lines) + "\n"
    )

    table = [
        "% Generated by scripts/run_external_cmrf_multiscanner.py.",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scanner & Platform & $T_1$ MAE & $T_2$ MAE & Tube means \\\\",
        "\\midrule",
    ]
    for scanner in results["scanners"]:
        t1 = scanner["t1"]["tube_mean_metrics_ms"]
        t2 = scanner["t2"]["tube_mean_metrics_ms"]
        scanner_name = scanner["scanner"]
        platform = tex_escape(scanner["platform"])
        table.append(
            f"{scanner_name} & {platform} & "
            f"{t1['mae']:.1f} & {t2['mae']:.1f} & "
            f"{t1['n']} \\\\"
        )
    table.append(
        f"Pooled & -- & {t1_all['mae']:.1f} & {t2_all['mae']:.1f} & "
        f"{t1_all['n']} \\\\"
    )
    table.extend(["\\bottomrule", "\\end{tabular}", "}"])
    (paper_dir / "generated_external_multiscanner_summary.tex").write_text(
        "\n".join(table) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/external/cmrf_comparison/package"))
    parser.add_argument("--output", type=Path, default=Path("results/external_cmrf_multiscanner.json"))
    parser.add_argument("--scanners", default="1,2,3,4")
    parser.add_argument(
        "--dictionary-chunk",
        type=int,
        default=2000,
        help="EPG dictionary entries matched per bounded-memory chunk",
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if args.dictionary_chunk < 1:
        raise SystemExit("--dictionary-chunk must be positive")
    scanner_ids = [int(x) for x in args.scanners.split(",") if x.strip()]
    for scanner_id in scanner_ids:
        scanner_dir = args.data_dir / f"scanner{scanner_id}"
        for name in ("cMRF.h5", "ref_t1.h5", "ref_t2.h5", "mask.npy"):
            if not (scanner_dir / name).exists():
                raise SystemExit(f"missing {scanner_dir / name}")

    mr = _import_mrpro()
    torch = mr["torch"]
    torch.set_num_threads(args.threads)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"finufft\._interfaces",
    )
    flip_angles = load_flip_angles(args.data_dir / "cMRF_fa_705rep.txt", torch)
    t1_values, t2_values = load_dictionary_grid(torch)
    print(
        f"Using {len(t1_values)} EPG dictionary entries, "
        f"{len(scanner_ids)} scanner(s), chunk={args.dictionary_chunk}",
        flush=True,
    )

    package_zip = Path("data/external/cmrf_comparison/open_source_cmrf_scanner_comparison.zip")
    results: dict[str, Any] = {
        "schema_version": "external-cmrf-multiscanner-v1",
        "provenance": {
            "dataset": "Open-Source Cardiac MR Fingerprinting scanner comparison",
            "zenodo_record": "14251660",
            "expanded_record": "15831511",
            "doi": "10.5281/zenodo.14251660",
            "package_file": str(package_zip),
            "package_md5": md5(package_zip),
            "implementation": "https://github.com/PTB-MR/cMRF",
            "mrpro_source": "https://github.com/PTB-MR/mrpro/tree/cMRF",
            "mrpro_version": "0.260420",
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "dictionary_chunk": args.dictionary_chunk,
            "units": "milliseconds",
        },
        "task": {
            "description": "Analytical cMRF reconstruction versus independently reconstructed spin-echo T1/T2 reference scans",
            "model_training_performed": False,
            "neural_benchmark_tuning_performed": False,
            "target_labels_used": False,
            "reference_scans_used_for_dictionary_or_training": False,
            "reference_scans_used_only_for_final_evaluation": True,
            "clinical_inference_performed": False,
        },
        "scanners": [],
    }

    maps_dir = args.output.parent / "external_cmrf_multiscanner_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for scanner_id in scanner_ids:
        scanner_dir = args.data_dir / f"scanner{scanner_id}"
        print(f"Scanner {scanner_id}: reconstructing reference scans", flush=True)
        mask = np.load(scanner_dir / "mask.npy").astype(np.int16)
        t1_ref, t2_ref, ref_diag = reconstruct_reference_maps(
            scanner_dir, mask, t1_values, t2_values, mr
        )
        print(f"Scanner {scanner_id}: reconstructing cMRF", flush=True)
        t1_pred, t2_pred, cmrf_diag = reconstruct_cmrf_map(
            scanner_dir, mask, flip_angles, t1_values, t2_values, mr,
            args.dictionary_chunk,
        )
        if t1_pred.shape != mask.shape or t1_ref.shape != mask.shape:
            raise ValueError(
                f"shape mismatch for scanner{scanner_id}: "
                f"pred={t1_pred.shape}, ref={t1_ref.shape}, mask={mask.shape}"
            )

        metadata = scanner_metadata(scanner_dir, mr)
        scanner_record = {
            "scanner": f"scanner{scanner_id}",
            "platform": metadata["model"],
            "metadata": metadata,
            "reference_reconstruction": ref_diag,
            "cmrf_reconstruction": cmrf_diag,
            "t1": tube_summary(t1_pred, t1_ref, mask),
            "t2": tube_summary(t2_pred, t2_ref, mask),
        }
        results["scanners"].append(scanner_record)
        np.save(maps_dir / f"scanner{scanner_id}_t1_cmrf_ms.npy", t1_pred)
        np.save(maps_dir / f"scanner{scanner_id}_t2_cmrf_ms.npy", t2_pred)
        np.save(maps_dir / f"scanner{scanner_id}_t1_reference_ms.npy", t1_ref)
        np.save(maps_dir / f"scanner{scanner_id}_t2_reference_ms.npy", t2_ref)

    for parameter in ("t1", "t2"):
        pred_means = []
        ref_means = []
        scanner_mae = []
        scanner_bias = []
        for scanner in results["scanners"]:
            metric = scanner[parameter]["tube_mean_metrics_ms"]
            scanner_mae.append(metric["mae"])
            scanner_bias.append(metric["mean_bias"])
            for row in scanner[parameter]["per_tube"]:
                pred_means.append(row["predicted_mean_ms"])
                ref_means.append(row["reference_mean_ms"])
        pooled = paired_metrics(np.asarray(pred_means), np.asarray(ref_means))
        results.setdefault("aggregate", {})[parameter] = {
            "pooled_tube_mean_metrics_ms": pooled,
            "scanner_mae_ms": {
                "mean": float(np.mean(scanner_mae)),
                "sd": float(np.std(scanner_mae, ddof=1)),
                "min": float(np.min(scanner_mae)),
                "max": float(np.max(scanner_mae)),
            },
            "scanner_bias_ms": {
                "mean": float(np.mean(scanner_bias)),
                "sd": float(np.std(scanner_bias, ddof=1)),
            },
        }
    results["aggregate"]["n_scanners"] = len(results["scanners"])
    results["aggregate"]["n_tubes_per_scanner"] = 9
    results["aggregate"]["n_tube_means"] = 9 * len(results["scanners"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    write_tex(results, Path("paper"))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "aggregate": results["aggregate"],
                "scanner_metrics": [
                    {
                        "scanner": scanner["scanner"],
                        "platform": scanner["platform"],
                        "t1_mae": scanner["t1"]["tube_mean_metrics_ms"]["mae"],
                        "t2_mae": scanner["t2"]["tube_mean_metrics_ms"]["mae"],
                    }
                    for scanner in results["scanners"]
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
