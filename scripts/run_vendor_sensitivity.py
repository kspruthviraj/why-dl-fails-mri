#!/usr/bin/env python3
"""
run_vendor_sensitivity.py — Monte Carlo sensitivity analysis for vendor parameterization.

Generate 100 random "virtual vendor" profiles within published ranges.
For each profile, compute DS3 for each corruption type.
Show that B0 dominance is robust across the parameter space.

Usage: PYTHONPATH=. python3 scripts/run_vendor_sensitivity.py
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
    device = pred.device
    return pred * torch.from_numpy(tgt_max - tgt_min).to(device) + torch.from_numpy(tgt_min).to(device)


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


def apply_corruption(sig_2ch, b0_hz=0, snr=None, b1_scale=1.0, shuffle_timing=False, rng=None):
    """Apply corruption to 2-channel signal tensor."""
    if rng is None:
        rng = np.random.RandomState(42)

    sig = sig_2ch.numpy().copy()
    # Convert to complex
    sig_c = sig[:, 0] + 1j * sig[:, 1]

    # B0 phase
    if b0_hz != 0:
        n = sig_c.shape[-1]
        phase = np.exp(1j * 2 * np.pi * b0_hz * np.linspace(0, n / 1000, n))
        sig_c = sig_c * phase[np.newaxis, :]

    # B1 scaling
    if b1_scale != 1.0:
        sig_c = sig_c * b1_scale

    # SNR
    if snr is not None:
        noise = (rng.randn(*sig_c.shape) + 1j * rng.randn(*sig_c.shape))
        noise_power = np.sqrt(np.mean(np.abs(sig_c) ** 2, axis=-1, keepdims=True) / snr)
        sig_c = sig_c + noise * noise_power

    # Timing shuffle
    if shuffle_timing:
        for i in range(len(sig_c)):
            rng.shuffle(sig_c[i])

    # Convert back to 2-channel and normalize
    result = np.stack([sig_c.real, sig_c.imag], axis=1).astype(np.float32)
    result_t = torch.from_numpy(result)
    for i in range(len(result_t)):
        pk = result_t[i].abs().max()
        if pk > 0:
            result_t[i] /= pk
    return result_t


def train_model(data, seed=42, n_ep=25):
    import yaml
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    mc = {"input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
          "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8, "n_transformer_layers": 6}
    model = build_model("resnet1d_18", mc).to(DEVICE)
    algo = build_algorithm("erm", model, cfg, DEVICE, n_domains=3)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
    loader = DataLoader(TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
                       batch_size=512, shuffle=True, drop_last=True)

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


def main():
    data = load_data()
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]

    # Train one model (source = reference vendor)
    logger.info("Training source model (seed=42) ...")
    model = train_model(data, seed=42)
    clean_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
    logger.info("  Clean (reference vendor) MAE: %.1f ms", clean_mae)

    # Define corruption experiments
    corruptions = {
        "B0_25Hz": {"b0_hz": 25},
        "B0_50Hz": {"b0_hz": 50},
        "B0_100Hz": {"b0_hz": 100},
        "SNR_5": {"snr": 5},
        "SNR_10": {"snr": 10},
        "B1_0.90": {"b1_scale": 0.90},
        "B1_0.95": {"b1_scale": 0.95},
        "Timing": {"shuffle_timing": True},
    }

    # Monte Carlo: 100 random vendor profiles
    n_profiles = 100
    rng = np.random.RandomState(42)

    logger.info("Running %d random vendor profiles ...", n_profiles)

    all_results = []
    b0_dominant_count = 0
    timing_second_count = 0

    for i in range(n_profiles):
        # Random perturbation magnitudes within published ranges
        b0_mag = rng.uniform(0, 50)  # 0-50 Hz (published range)
        b1_mag = rng.uniform(0.90, 1.10)  # ±10% (published range)
        snr_mag = rng.uniform(2, 20)  # SNR 2-20
        do_timing = rng.random() < 0.5  # 50% chance of timing corruption

        # Apply each corruption at this profile's magnitude
        profile_ds3 = {}

        # B0
        cor = apply_corruption(data["tgt_sig"], b0_hz=b0_mag, rng=rng)
        mae = eval_model(model, cor, data["tgt_tgt"], tgt_min, tgt_max)
        profile_ds3["B0"] = mae / max(clean_mae, 0.01)

        # B1
        cor = apply_corruption(data["tgt_sig"], b1_scale=b1_mag, rng=rng)
        mae = eval_model(model, cor, data["tgt_tgt"], tgt_min, tgt_max)
        profile_ds3["B1"] = mae / max(clean_mae, 0.01)

        # SNR
        cor = apply_corruption(data["tgt_sig"], snr=snr_mag, rng=rng)
        mae = eval_model(model, cor, data["tgt_tgt"], tgt_min, tgt_max)
        profile_ds3["SNR"] = mae / max(clean_mae, 0.01)

        # Timing
        if do_timing:
            cor = apply_corruption(data["tgt_sig"], shuffle_timing=True, rng=rng)
            mae = eval_model(model, cor, data["tgt_tgt"], tgt_min, tgt_max)
            profile_ds3["Timing"] = mae / max(clean_mae, 0.01)

        # Check if B0 is dominant
        max_factor = max(profile_ds3, key=profile_ds3.get)
        if max_factor == "B0":
            b0_dominant_count += 1

        # Check timing ranking
        sorted_factors = sorted(profile_ds3, key=profile_ds3.get, reverse=True)
        if len(sorted_factors) >= 2 and sorted_factors[1] == "Timing":
            timing_second_count += 1

        all_results.append({
            "profile_id": i,
            "b0_mag": float(b0_mag),
            "b1_mag": float(b1_mag),
            "snr_mag": float(snr_mag),
            "has_timing": do_timing,
            "ds3": {k: float(v) for k, v in profile_ds3.items()},
            "dominant_factor": max_factor,
        })

        if (i + 1) % 20 == 0:
            logger.info("  Completed %d/%d profiles (B0 dominant: %d/%d)",
                       i + 1, n_profiles, b0_dominant_count, i + 1)

    # Compute summary statistics
    b0_ds3s = [r["ds3"]["B0"] for r in all_results]
    b1_ds3s = [r["ds3"]["B1"] for r in all_results]
    snr_ds3s = [r["ds3"]["SNR"] for r in all_results]
    timing_ds3s = [r["ds3"]["Timing"] for r in all_results if "Timing" in r["ds3"]]

    summary = {
        "n_profiles": n_profiles,
        "b0_dominant_frequency": b0_dominant_count / n_profiles,
        "b0_dominant_count": b0_dominant_count,
        "timing_second_frequency": timing_second_count / max(b0_dominant_count, 1),
        "mean_ds3": {
            "B0": float(np.mean(b0_ds3s)),
            "B1": float(np.mean(b1_ds3s)),
            "SNR": float(np.mean(snr_ds3s)),
            "Timing": float(np.mean(timing_ds3s)) if timing_ds3s else None,
        },
        "std_ds3": {
            "B0": float(np.std(b0_ds3s)),
            "B1": float(np.std(b1_ds3s)),
            "SNR": float(np.std(snr_ds3s)),
            "Timing": float(np.std(timing_ds3s)) if timing_ds3s else None,
        },
        "ranking_consistency": {
            "B0_is_top_pct": b0_dominant_count / n_profiles * 100,
        },
    }

    output = {"summary": summary, "profiles": all_results}

    out_path = RESULTS_DIR / "vendor_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Saved → %s", out_path)

    # Print summary
    print("\n" + "=" * 60)
    print("VENDOR SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n{n_profiles} random vendor profiles tested")
    print(f"\nB₀ is the dominant corruption in {b0_dominant_count}/{n_profiles} "
          f"profiles ({b0_dominant_count/n_profiles*100:.0f}%)")
    print(f"\nMean DS3 across all profiles:")
    print(f"  B₀:     {np.mean(b0_ds3s):.2f} ± {np.std(b0_ds3s):.2f}")
    print(f"  B₁:     {np.mean(b1_ds3s):.2f} ± {np.std(b1_ds3s):.2f}")
    print(f"  SNR:    {np.mean(snr_ds3s):.2f} ± {np.std(snr_ds3s):.2f}")
    if timing_ds3s:
        print(f"  Timing: {np.mean(timing_ds3s):.2f} ± {np.std(timing_ds3s):.2f}")
    print(f"\nConclusion: B₀ dominance is robust across {b0_dominant_count/n_profiles*100:.0f}% "
          f"of plausible vendor parameterizations.")


if __name__ == "__main__":
    main()
