#!/usr/bin/env python3
"""
run_joint_and_phase.py — Joint perturbations + Phase diagram (Data × B₀).

1. Joint perturbations: B0×Noise, B0×B1, B0×Timing (factorial design)
2. Phase diagram: Data size × B₀ shift → OOD error heatmap

Usage: PYTHONPATH=. python3 scripts/run_joint_and_phase.py
"""

import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")


def load_data():
    with h5py.File("data/synthetic/mrf_100k.h5", "r") as f:
        sig_np = f["signals"][:]
        tgt_np = f["parameters"][:, :2].astype(np.float32)
        dom_np = f["domain_labels"][:]

    sig_2ch = np.stack([sig_np.real, sig_np.imag], axis=1).astype(np.float32)
    for i in range(len(sig_2ch)):
        pk = np.abs(sig_2ch[i]).max()
        if pk > 0:
            sig_2ch[i] /= pk

    tgt_min = tgt_np.min(axis=0)
    tgt_max = tgt_np.max(axis=0)
    tgt_norm = (tgt_np - tgt_min) / (tgt_max - tgt_min + 1e-8)

    sig_t = torch.from_numpy(sig_2ch)
    tgt_t = torch.from_numpy(tgt_norm)

    def get_mask(prefix):
        return np.array([
            d.decode().startswith(prefix) if isinstance(d, bytes) else d.startswith(prefix)
            for d in dom_np
        ])

    src_m = get_mask("siemens")
    tgt_m = get_mask("philips")

    src_sig, src_tgt = sig_t[src_m], tgt_t[src_m]
    tgt_sig, tgt_tgt = sig_t[tgt_m], tgt_t[tgt_m]

    idx = torch.randperm(len(src_sig))
    n_tr = int(0.8 * len(src_sig))
    tr_sig, tr_tgt = src_sig[idx[:n_tr]], src_tgt[idx[:n_tr]]
    va_sig, va_tgt = src_sig[idx[n_tr:]], src_tgt[idx[n_tr:]]

    return {
        "tr_sig": tr_sig, "tr_tgt": tr_tgt,
        "va_sig": va_sig, "va_tgt": va_tgt,
        "tgt_sig": tgt_sig, "tgt_tgt": tgt_tgt,
        "tgt_min": tgt_min, "tgt_max": tgt_max,
        "sig_shape": sig_2ch.shape,
    }


def denorm(pred, tgt_min, tgt_max):
    return pred * torch.from_numpy(tgt_max - tgt_min) + torch.from_numpy(tgt_min)


@torch.no_grad()
def eval_model(model, sig, tgt, tgt_min, tgt_max, bs=1024):
    model.eval()
    preds = []
    for i in range(0, len(sig), bs):
        preds.append(model(sig[i:i + bs].to(DEVICE)).cpu())
    pred = torch.cat(preds)
    pred_dn = denorm(pred, tgt_min, tgt_max)
    tgt_dn = denorm(tgt, tgt_min, tgt_max)
    return float(torch.mean(torch.abs(pred_dn - tgt_dn)).item())


def train_erm(data, seed=42, n_ep=25, n_train=None):
    import yaml
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_sig, tr_tgt = data["tr_sig"], data["tr_tgt"]
    if n_train and n_train < len(tr_sig):
        idx = torch.randperm(len(tr_sig))[:n_train]
        tr_sig, tr_tgt = tr_sig[idx], tr_tgt[idx]

    mc = {"input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
          "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8, "n_transformer_layers": 6}
    model = build_model("resnet1d_18", mc).to(DEVICE)
    algo = build_algorithm("erm", model, cfg, DEVICE, n_domains=3)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    dom = torch.zeros(len(tr_sig), dtype=torch.long)
    loader = DataLoader(TensorDataset(tr_sig, tr_tgt, dom), batch_size=512, shuffle=True, drop_last=True)

    for ep in range(n_ep):
        model.train()
        for s, t, d in loader:
            s, t, d = s.to(DEVICE), t.to(DEVICE), d.to(DEVICE)
            opt.zero_grad()
            r = algo.compute_loss(s, t, d)
            r["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sch.step()

    return model


def apply_corruption(sig_np, b0_hz=0, snr=None, b1_scale=1.0, shuffle_timing=False, rng=None):
    """Apply corruption to complex signal."""
    if rng is None:
        rng = np.random.RandomState(42)

    sig = sig_np.copy()

    # B0 phase accumulation
    if b0_hz != 0:
        n_points = sig.shape[-1]
        phase = np.exp(1j * 2 * np.pi * b0_hz * np.linspace(0, n_points / 1000, n_points))
        sig = sig * phase[np.newaxis, :]

    # B1+ scaling
    if b1_scale != 1.0:
        sig = sig * b1_scale

    # SNR degradation
    if snr is not None:
        noise = (rng.randn(*sig.shape) + 1j * rng.randn(*sig.shape))
        noise_power = np.sqrt(np.mean(np.abs(sig) ** 2, axis=-1, keepdims=True) / snr)
        sig = sig + noise * noise_power

    # Timing shuffle
    if shuffle_timing:
        for i in range(len(sig)):
            rng.shuffle(sig[i])

    return sig


def main():
    results = {}
    data = load_data()
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]

    # ── 1. Joint Perturbation Experiment ──────────────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Joint Perturbations")
    logger.info("=" * 60)

    # Train one ERM model
    logger.info("Training ERM model (seed=42) ...")
    model = train_erm(data, seed=42)
    clean_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
    logger.info("  Clean MAE: %.1f", clean_mae)

    # Get Philips signals as complex
    tgt_c = data["tgt_sig"][:, 0].numpy() + 1j * data["tgt_sig"][:, 1].numpy()
    rng = np.random.RandomState(42)

    joint_results = {}
    perturbations = {
        "B0_25Hz_only": {"b0_hz": 25},
        "SNR_5_only": {"snr": 5},
        "B1_0.75_only": {"b1_scale": 0.75},
        "Timing_only": {"shuffle_timing": True},
        "B0_25Hz + SNR_5": {"b0_hz": 25, "snr": 5},
        "B0_25Hz + B1_0.75": {"b0_hz": 25, "b1_scale": 0.75},
        "B0_25Hz + Timing": {"b0_hz": 25, "shuffle_timing": True},
        "B0_25Hz + SNR_5 + B1_0.75": {"b0_hz": 25, "snr": 5, "b1_scale": 0.75},
        "All_combined": {"b0_hz": 25, "snr": 5, "b1_scale": 0.75, "shuffle_timing": True},
    }

    for name, params in perturbations.items():
        cor = apply_corruption(tgt_c, rng=rng, **params)
        cor_2ch = torch.from_numpy(np.stack([cor.real, cor.imag], axis=1).astype(np.float32))
        for i in range(len(cor_2ch)):
            pk = cor_2ch[i].abs().max()
            if pk > 0:
                cor_2ch[i] /= pk

        mae = eval_model(model, cor_2ch, data["tgt_tgt"], tgt_min, tgt_max)
        ds3 = mae / max(clean_mae, 0.01)
        joint_results[name] = {"mae": float(mae), "ds3": float(ds3)}
        logger.info("  %s: MAE=%.1f, DS3=%.2f", name, mae, ds3)

    # Check additivity
    individual_sum = sum(joint_results[k]["ds3"] for k in ["B0_25Hz_only", "SNR_5_only", "B1_0.75_only", "Timing_only"])
    joint_actual = joint_results["All_combined"]["ds3"]
    joint_results["additivity_ratio"] = float(joint_actual / max(individual_sum - 3, 0.01))  # -3 for baseline DS3=1 each
    logger.info("  Additivity ratio: %.2f (1.0=perfect additivity)", joint_results["additivity_ratio"])

    results["joint_perturbations"] = joint_results

    # ── 2. Phase Diagram: Data Size × B₀ Shift ───────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Phase Diagram (Data × B₀)")
    logger.info("=" * 60)

    data_sizes = [5000, 10000, 26659]
    b0_shifts = [0, 25, 50, 100, 150]

    phase_results = {}
    for n_train in data_sizes:
        logger.info("  Training N=%d ...", n_train)
        model = train_erm(data, seed=42, n_train=n_train)
        clean_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)

        for b0 in b0_shifts:
            if b0 == 0:
                mae = clean_mae
            else:
                cor = apply_corruption(tgt_c, b0_hz=b0, rng=rng)
                cor_2ch = torch.from_numpy(np.stack([cor.real, cor.imag], axis=1).astype(np.float32))
                for i in range(len(cor_2ch)):
                    pk = cor_2ch[i].abs().max()
                    if pk > 0:
                        cor_2ch[i] /= pk
                mae = eval_model(model, cor_2ch, data["tgt_tgt"], tgt_min, tgt_max)

            ds3 = mae / max(clean_mae, 0.01)
            key = f"N{n_train}_B0_{b0}"
            phase_results[key] = {"n_train": n_train, "b0_hz": b0, "mae": float(mae), "ds3": float(ds3)}
            logger.info("    N=%d, B0=%dHz: MAE=%.1f, DS3=%.2f", n_train, b0, mae, ds3)

    results["phase_diagram"] = phase_results

    # ── Save ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "joint_and_phase.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved → %s", out_path)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n--- Joint Perturbations ---")
    for k, v in joint_results.items():
        if isinstance(v, dict):
            print(f"  {k}: DS3={v['ds3']:.2f}")
    print(f"  Additivity ratio: {joint_results['additivity_ratio']:.2f}")

    print("\n--- Phase Diagram ---")
    for n in data_sizes:
        row = [f"N={n}"]
        for b0 in b0_shifts:
            key = f"N{n}_B0_{b0}"
            row.append(f"B0={b0}: DS3={phase_results[key]['ds3']:.1f}")
        print("  " + " | ".join(row))


if __name__ == "__main__":
    main()
