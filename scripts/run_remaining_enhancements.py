#!/usr/bin/env python3
"""Run remaining enhancements: physics-informed, bootstrap, real distribution."""

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
OUT_PATH = Path("results/enhancements.json")


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
    mae = float(torch.mean(torch.abs(pred_dn - tgt_dn)).item())
    mae_t1 = float(torch.mean(torch.abs(pred_dn[:, 0] - tgt_dn[:, 0])).item())
    mae_t2 = float(torch.mean(torch.abs(pred_dn[:, 1] - tgt_dn[:, 1])).item())
    return {"mae": mae, "mae_t1": mae_t1, "mae_t2": mae_t2}


def bootstrap_ds3(src_maes, ood_maes, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    ds3_samples = []
    for _ in range(n_boot):
        idx_s = rng.choice(len(src_maes), len(src_maes), replace=True)
        idx_o = rng.choice(len(ood_maes), len(ood_maes), replace=True)
        s = np.mean([src_maes[i] for i in idx_s])
        o = np.mean([ood_maes[i] for i in idx_o])
        ds3_samples.append(o / max(s, 1e-8))
    return {
        "mean": float(np.mean(ds3_samples)),
        "ci_95_low": float(np.percentile(ds3_samples, 2.5)),
        "ci_95_high": float(np.percentile(ds3_samples, 97.5)),
    }


def main():
    results = json.load(open(OUT_PATH))
    data = load_data()
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]
    seeds = [42, 123, 456]

    # ── Physics-Informed ──────────────────────────────────────────────
    logger.info("EXP: Physics-Informed Baseline (3 seeds)")
    from qMR_Robust.models.resnet1d import ResNet1D

    class PhysicsInformedResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ResNet1D(
                in_channels=2, base_channels=64, n_blocks=[2, 2, 2],
                hidden_dim=256, output_dim=2, dropout=0.1
            )
            self.physics_head = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))
            self.fusion = nn.Sequential(
                nn.Linear(self.backbone._feature_dim + 32, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid()
            )

        def forward(self, x):
            feat = self.backbone.encode(x)
            phase = torch.atan2(x[:, 1, :], x[:, 0, :])
            dphase = phase[:, 1:] - phase[:, :-1]
            b0_est = dphase.mean(dim=1, keepdim=True)
            phys = self.physics_head(b0_est)
            return self.fusion(torch.cat([feat, phys], dim=1))

        def encode(self, x):
            return self.backbone.encode(x)

    pi_results = []
    for seed in seeds:
        logger.info("  physics_informed seed=%d ...", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = PhysicsInformedResNet().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
        dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
        loader = DataLoader(TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
                           batch_size=512, shuffle=True, drop_last=True)
        loss_fn = nn.L1Loss()

        t0 = time.time()
        for ep in range(25):
            model.train()
            for s, t, d in loader:
                s, t = s.to(DEVICE), t.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(s), t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
        dt = time.time() - t0

        src = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
        ph = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
        ge = eval_model(model, data["ge_sig"], data["ge_tgt"], tgt_min, tgt_max)
        pi_results.append({
            "seed": seed, "src": src, "ph": ph, "ge": ge,
            "ds3_ph": ph["mae"] / max(src["mae"], 0.01),
            "ds3_ge": ge["mae"] / max(src["mae"], 0.01),
            "time": dt,
        })
        logger.info("  seed=%d: src=%.1f, ph=%.1f, ge=%.1f, ds3=%.1f (%.0fs)",
                     seed, src["mae"], ph["mae"], ge["mae"], ph["mae"]/max(src["mae"],0.01), dt)

    results["physics_informed"] = pi_results
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Bootstrap CIs ─────────────────────────────────────────────────
    logger.info("EXP: Bootstrap CIs on DS3")
    ckpt1 = json.load(open("results/checkpoint.json"))
    erm_src = [ckpt1["results"][f"algo_resnet1d_18_erm_seed{s}"]["source"]["mae"] for s in seeds]
    erm_ph = [ckpt1["results"][f"algo_resnet1d_18_erm_seed{s}"]["philips"]["mae"] for s in seeds]
    erm_ge = [ckpt1["results"][f"algo_resnet1d_18_erm_seed{s}"]["ge"]["mae"] for s in seeds]

    results["bootstrap"] = {
        "erm_resnet_ds3_ph": bootstrap_ds3(erm_src, erm_ph),
        "erm_resnet_ds3_ge": bootstrap_ds3(erm_src, erm_ge),
    }
    logger.info("  DS3_ph: %.1f [%.1f, %.1f]",
                results["bootstrap"]["erm_resnet_ds3_ph"]["mean"],
                results["bootstrap"]["erm_resnet_ds3_ph"]["ci_95_low"],
                results["bootstrap"]["erm_resnet_ds3_ph"]["ci_95_high"])

    # ── Real distribution ─────────────────────────────────────────────
    logger.info("EXP: Synthetic vs Real T1/T2 distributions")
    import nibabel as nib
    from scipy import stats as sp_stats

    base = Path("data/real/analysis_images_share")
    real_stats = {}
    for sub_dir in sorted(base.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub_"):
            continue
        for scan_dir in sorted(sub_dir.iterdir()):
            if not scan_dir.is_dir() or "scanner_" not in scan_dir.name:
                continue
            parts = scan_dir.name.split("_")
            scanner = f"{parts[1]}_{parts[2]}"
            if scanner not in real_stats:
                real_stats[scanner] = []
            for acq_dir in sorted(scan_dir.iterdir()):
                if not acq_dir.is_dir():
                    continue
                nifti_dir = acq_dir / "Unprocessed_NIFTI"
                t1p = nifti_dir / "MRF_T1.nii.gz"
                t2p = nifti_dir / "MRF_T2.nii.gz"
                if t1p.exists() and t2p.exists():
                    t1 = np.asarray(nib.load(str(t1p)).dataobj).flatten()
                    t2 = np.asarray(nib.load(str(t2p)).dataobj).flatten()
                    mask = (t1 > 50) & (t1 < 3000) & (t2 > 2) & (t2 < 500)
                    if mask.sum() > 100:
                        real_stats[scanner].append({
                            "t1_mean": float(np.mean(t1[mask])),
                            "t2_mean": float(np.mean(t2[mask])),
                            "n_voxels": int(mask.sum()),
                        })

    # Synthetic distribution
    with h5py.File("data/synthetic/mrf_100k.h5", "r") as f:
        params = f["parameters"][:, :2]
        dom_np = f["domain_labels"][:]
    src_mask = np.array([d.decode().startswith("siemens") if isinstance(d, bytes) else d.startswith("siemens") for d in dom_np])
    synth_t1 = params[src_mask, 0]
    synth_t2 = params[src_mask, 1]

    results["real_distribution"] = real_stats
    results["synthetic_distribution"] = {
        "siemens_t1_mean": float(np.mean(synth_t1)),
        "siemens_t1_std": float(np.std(synth_t1)),
        "siemens_t2_mean": float(np.mean(synth_t2)),
        "siemens_t2_std": float(np.std(synth_t2)),
    }

    for scanner in sorted(real_stats.keys()):
        if real_stats[scanner]:
            real_t1s = [r["t1_mean"] for r in real_stats[scanner]]
            real_t2s = [r["t2_mean"] for r in real_stats[scanner]]
            ks_t1 = sp_stats.ks_2samp(synth_t1[:len(real_t1s)*100], np.repeat(real_t1s, 100))
            ks_t2 = sp_stats.ks_2samp(synth_t2[:len(real_t2s)*100], np.repeat(real_t2s, 100))
            results[f"ks_test_{scanner}"] = {
                "t1_ks": float(ks_t1.statistic), "t1_p": float(ks_t1.pvalue),
                "t2_ks": float(ks_t2.statistic), "t2_p": float(ks_t2.pvalue),
            }
            logger.info("  %s: KS T1=%.3f (p=%.3f), KS T2=%.3f (p=%.3f)",
                        scanner, ks_t1.statistic, ks_t1.pvalue, ks_t2.statistic, ks_t2.pvalue)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("ALL ENHANCEMENTS COMPLETE → %s", OUT_PATH)


if __name__ == "__main__":
    main()
