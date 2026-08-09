#!/usr/bin/env python3
"""
run_irm_ablation_and_long_training.py — IRM hyperparameter ablation + 100-epoch training.

IRM ablation: λ ∈ {100, 500, 1000, 5000, 10000}
100-epoch training: ERM with 100 epochs vs 25 epochs

Usage: PYTHONPATH=. python3 scripts/run_irm_ablation_and_long_training.py
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
    ge_m = get_mask("ge")

    src_sig, src_tgt = sig_t[src_m], tgt_t[src_m]
    tgt_sig, tgt_tgt = sig_t[tgt_m], tgt_t[tgt_m]
    ge_sig, ge_tgt = sig_t[ge_m], tgt_t[ge_m]

    idx = torch.randperm(len(src_sig))
    n_tr = int(0.8 * len(src_sig))
    tr_sig, tr_tgt = src_sig[idx[:n_tr]], src_tgt[idx[:n_tr]]
    va_sig, va_tgt = src_sig[idx[n_tr:]], src_tgt[idx[n_tr:]]

    return {
        "tr_sig": tr_sig, "tr_tgt": tr_tgt,
        "va_sig": va_sig, "va_tgt": va_tgt,
        "tgt_sig": tgt_sig, "tgt_tgt": tgt_tgt,
        "ge_sig": ge_sig, "ge_tgt": ge_tgt,
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


def main():
    data = load_data()
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]
    results = {}

    # ── 1. IRM Hyperparameter Ablation ────────────────────────────────
    logger.info("=" * 60)
    logger.info("IRM HYPERPARAMETER ABLATION")
    logger.info("=" * 60)

    import yaml
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    irm_ablation = []

    # Ablate penalty weight λ
    for penalty_weight in [100, 500, 1000, 5000, 10000]:
        logger.info("  IRM λ=%d ...", penalty_weight)
        torch.manual_seed(42)
        np.random.seed(42)

        mc = {"input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
              "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8, "n_transformer_layers": 6}
        model = build_model("resnet1d_18", mc).to(DEVICE)

        cfg_copy = dict(cfg)
        cfg_copy["algorithm"] = dict(cfg["algorithm"])
        cfg_copy["algorithm"]["irm_penalty_weight"] = penalty_weight
        algo = build_algorithm("irm", model, cfg_copy, DEVICE, n_domains=3)

        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
        dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
        loader = DataLoader(TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
                           batch_size=512, shuffle=True, drop_last=True)

        t0 = time.time()
        penalties = []
        for ep in range(25):
            model.train()
            ep_penalties = []
            for s, t, d in loader:
                s, t, d = s.to(DEVICE), t.to(DEVICE), d.to(DEVICE)
                opt.zero_grad()
                r = algo.compute_loss(s, t, d)
                r["loss"].backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                ep_penalties.append(float(r["penalty"].item()))
            sch.step()
            penalties.append(np.mean(ep_penalties))
        dt = time.time() - t0

        src_mae = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
        ph_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
        ds3 = ph_mae / max(src_mae, 0.01)

        irm_ablation.append({
            "penalty_weight": penalty_weight,
            "src_mae": float(src_mae),
            "ph_mae": float(ph_mae),
            "ds3": float(ds3),
            "final_penalty": float(penalties[-1]),
            "penalty_at_epoch5": float(penalties[4]) if len(penalties) > 4 else None,
            "time": dt,
        })
        logger.info("  λ=%d: src=%.1f, ph=%.1f, ds3=%.1f, penalty_final=%.4f, penalty_ep5=%.4f (%.0fs)",
                     penalty_weight, src_mae, ph_mae, ds3, penalties[-1],
                     penalties[4] if len(penalties) > 4 else 0, dt)

    results["irm_ablation"] = irm_ablation

    # Ablate learning rate
    logger.info("  IRM LR ablation ...")
    irm_lr = []
    for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
        logger.info("  IRM lr=%g ...", lr)
        torch.manual_seed(42)
        np.random.seed(42)

        model = build_model("resnet1d_18", mc).to(DEVICE)
        algo = build_algorithm("irm", model, cfg, DEVICE, n_domains=3)
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
        dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
        loader = DataLoader(TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
                           batch_size=512, shuffle=True, drop_last=True)

        t0 = time.time()
        for ep in range(25):
            model.train()
            for s, t, d in loader:
                s, t, d = s.to(DEVICE), t.to(DEVICE), d.to(DEVICE)
                opt.zero_grad()
                r = algo.compute_loss(s, t, d)
                r["loss"].backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            sch.step()
        dt = time.time() - t0

        src_mae = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
        ph_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
        ds3 = ph_mae / max(src_mae, 0.01)

        irm_lr.append({"lr": lr, "src_mae": float(src_mae), "ph_mae": float(ph_mae), "ds3": float(ds3), "time": dt})
        logger.info("  lr=%g: src=%.1f, ph=%.1f, ds3=%.1f (%.0fs)", lr, src_mae, ph_mae, ds3, dt)

    results["irm_lr_ablation"] = irm_lr

    # Save after IRM ablation
    with open(RESULTS_DIR / "irm_ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── 2. 100-Epoch Training ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("100-EPOCH TRAINING (vs 25 epochs)")
    logger.info("=" * 60)

    for n_ep in [25, 100]:
        logger.info("  ERM %d epochs, seed=42 ...", n_ep)
        torch.manual_seed(42)
        np.random.seed(42)

        model = build_model("resnet1d_18", mc).to(DEVICE)
        algo = build_algorithm("erm", model, cfg, DEVICE, n_domains=3)
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
        dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
        loader = DataLoader(TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
                           batch_size=512, shuffle=True, drop_last=True)

        t0 = time.time()
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
        dt = time.time() - t0

        src_mae = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
        ph_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
        ge_mae = eval_model(model, data["ge_sig"], data["ge_tgt"], tgt_min, tgt_max)
        ds3_ph = ph_mae / max(src_mae, 0.01)
        ds3_ge = ge_mae / max(src_mae, 0.01)

        results[f"erm_{n_ep}ep"] = {
            "n_epochs": n_ep, "src_mae": float(src_mae), "ph_mae": float(ph_mae),
            "ge_mae": float(ge_mae), "ds3_ph": float(ds3_ph), "ds3_ge": float(ds3_ge), "time": dt,
        }
        logger.info("  %d epochs: src=%.1f, ph=%.1f, ge=%.1f, ds3_ph=%.1f (%.0fs)",
                     n_ep, src_mae, ph_mae, ge_mae, ds3_ph, dt)

    # Save final results
    with open(RESULTS_DIR / "irm_ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("ALL ABLATIONS COMPLETE → results/irm_ablation.json")


if __name__ == "__main__":
    main()
