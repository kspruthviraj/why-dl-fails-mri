#!/usr/bin/env python3
"""
run_calibration_analysis.py — Compute proper calibration metrics (ECE, NLL, coverage).

Usage: PYTHONPATH=. python3 scripts/run_calibration_analysis.py
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def mc_dropout_predict(model, sig, n_passes=20, bs=1024):
    """MC Dropout: run forward pass n_passes times with dropout enabled."""
    model.train()  # Enable dropout
    all_preds = []
    for _ in range(n_passes):
        preds = []
        for i in range(0, len(sig), bs):
            preds.append(model(sig[i:i + bs].to(DEVICE)).cpu())
        all_preds.append(torch.cat(preds))
    model.eval()
    preds_tensor = torch.stack(all_preds)  # (n_passes, N, 2)
    mean_pred = preds_tensor.mean(dim=0)
    std_pred = preds_tensor.std(dim=0)
    return mean_pred, std_pred


def compute_ece_regression(pred_mean, pred_std, target, n_bins=15):
    """ECE for regression: measures if predicted uncertainty matches actual error."""
    pred_mean = pred_mean.detach().cpu()
    pred_std = pred_std.detach().cpu().clamp(min=1e-8)
    target = target.detach().cpu()

    errors = torch.abs(pred_mean - target)
    z_scores = errors / pred_std  # Should be ~N(0,1) if calibrated

    bin_edges = torch.linspace(0, min(z_scores.max().item(), 4.0), n_bins + 1)
    total = len(z_scores.flatten())
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (z_scores >= lo) & (z_scores < hi)
        count = mask.sum().item()
        if count == 0:
            continue

        empirical = mask.float().mean().item()
        from scipy.stats import norm
        expected = float(norm.cdf(hi.item()) - norm.cdf(lo.item()))
        ece += abs(empirical - expected) * (count / total)

    return float(ece)


def compute_nll(pred_mean, pred_std, target):
    """Negative log-likelihood assuming Gaussian predictions."""
    pred_mean = pred_mean.detach().cpu()
    pred_std = pred_std.detach().cpu().clamp(min=1e-8)
    target = target.detach().cpu()

    nll = 0.5 * torch.log(2 * np.pi * pred_std ** 2) + ((target - pred_mean) ** 2) / (2 * pred_std ** 2)
    return float(nll.mean().item())


def compute_coverage(pred_mean, pred_std, target, confidence=0.95):
    """Coverage: what fraction of targets fall within the predicted CI."""
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)

    pred_mean = pred_mean.detach().cpu()
    pred_std = pred_std.detach().cpu().clamp(min=1e-8)
    target = target.detach().cpu()

    lower = pred_mean - z * pred_std
    upper = pred_mean + z * pred_std
    covered = ((target >= lower) & (target <= upper)).float().mean().item()
    return float(covered)


def main():
    data = load_data()
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]

    import yaml
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Train ERM model (seed=42)
    logger.info("Training ERM model (seed=42) ...")
    torch.manual_seed(42)
    np.random.seed(42)

    mc = {"input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
          "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8, "n_transformer_layers": 6}
    model = build_model("resnet1d_18", mc).to(DEVICE)
    algo = build_algorithm("erm", model, cfg, DEVICE, n_domains=3)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
    loader = DataLoader(TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
                       batch_size=512, shuffle=True, drop_last=True)

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

    # MC Dropout on source validation
    logger.info("MC Dropout on source validation (20 passes) ...")
    src_mean, src_std = mc_dropout_predict(model, data["va_sig"])
    src_tgt_dn = denorm(data["va_tgt"], tgt_min, tgt_max)
    src_mean_dn = denorm(src_mean, tgt_min, tgt_max)
    src_std_dn = src_std * torch.from_numpy(tgt_max - tgt_min)

    src_ece = compute_ece_regression(src_mean_dn, src_std_dn, src_tgt_dn)
    src_nll = compute_nll(src_mean_dn, src_std_dn, src_tgt_dn)
    src_cov95 = compute_coverage(src_mean_dn, src_std_dn, src_tgt_dn, 0.95)
    src_corr_t1 = float(np.corrcoef(src_std_dn[:, 0].numpy(), torch.abs(src_mean_dn[:, 0] - src_tgt_dn[:, 0]).numpy())[0, 1])
    src_corr_t2 = float(np.corrcoef(src_std_dn[:, 1].numpy(), torch.abs(src_mean_dn[:, 1] - src_tgt_dn[:, 1]).numpy())[0, 1])

    logger.info("  Source: ECE=%.4f, NLL=%.4f, Coverage@95=%.4f, corr_T1=%.3f, corr_T2=%.3f",
                src_ece, src_nll, src_cov95, src_corr_t1, src_corr_t2)

    # MC Dropout on Philips target
    logger.info("MC Dropout on Philips target (20 passes) ...")
    tgt_mean, tgt_std = mc_dropout_predict(model, data["tgt_sig"])
    tgt_tgt_dn = denorm(data["tgt_tgt"], tgt_min, tgt_max)
    tgt_mean_dn = denorm(tgt_mean, tgt_min, tgt_max)
    tgt_std_dn = tgt_std * torch.from_numpy(tgt_max - tgt_min)

    tgt_ece = compute_ece_regression(tgt_mean_dn, tgt_std_dn, tgt_tgt_dn)
    tgt_nll = compute_nll(tgt_mean_dn, tgt_std_dn, tgt_tgt_dn)
    tgt_cov95 = compute_coverage(tgt_mean_dn, tgt_std_dn, tgt_tgt_dn, 0.95)
    tgt_corr_t1 = float(np.corrcoef(tgt_std_dn[:, 0].numpy(), torch.abs(tgt_mean_dn[:, 0] - tgt_tgt_dn[:, 0]).numpy())[0, 1])
    tgt_corr_t2 = float(np.corrcoef(tgt_std_dn[:, 1].numpy(), torch.abs(tgt_mean_dn[:, 1] - tgt_tgt_dn[:, 1]).numpy())[0, 1])

    logger.info("  Philips: ECE=%.4f, NLL=%.4f, Coverage@95=%.4f, corr_T1=%.3f, corr_T2=%.3f",
                tgt_ece, tgt_nll, tgt_cov95, tgt_corr_t1, tgt_corr_t2)

    results = {
        "source": {
            "ece": src_ece, "nll": src_nll, "coverage_95": src_cov95,
            "corr_T1": src_corr_t1, "corr_T2": src_corr_t2,
        },
        "philips_target": {
            "ece": tgt_ece, "nll": tgt_nll, "coverage_95": tgt_cov95,
            "corr_T1": tgt_corr_t1, "corr_T2": tgt_corr_t2,
        },
    }

    with open(Path("results") / "calibration.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved → results/calibration.json")


if __name__ == "__main__":
    main()
