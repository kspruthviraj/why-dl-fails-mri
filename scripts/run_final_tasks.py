#!/usr/bin/env python3
"""
run_final_tasks.py — All remaining tasks: SAM baseline, CKA for Mixup/VREx, downstream impact.

Usage: PYTHONPATH=. python3 scripts/run_final_tasks.py
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
OUT_PATH = RESULTS_DIR / "final_tasks.json"


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


@torch.no_grad()
def extract_features(model, sig, max_n=2000, bs=512):
    model.eval()
    feats = []
    sig = sig[:max_n]
    for i in range(0, len(sig), bs):
        feats.append(model.encode(sig[i:i + bs].to(DEVICE)).cpu())
    return torch.cat(feats)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                eps = p.grad * scale
                p.add_(eps)
                self.state[p]["eps"] = eps
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["eps"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm())
        return torch.norm(torch.stack(norms))

    def zero_grad(self):
        self.base_optimizer.zero_grad()


def train_sam(data, seed=42, n_ep=25, rho=0.05):
    """Train with SAM optimizer."""
    import yaml
    from qMR_Robust.models.registry import build_model

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    mc = {"input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
          "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8, "n_transformer_layers": 6}
    model = build_model("resnet1d_18", mc).to(DEVICE)

    optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=rho, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=n_ep)
    loss_fn = nn.L1Loss()

    dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
    loader = DataLoader(TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
                       batch_size=512, shuffle=True, drop_last=True)

    for ep in range(n_ep):
        model.train()
        for s, t, d in loader:
            s, t = s.to(DEVICE), t.to(DEVICE)
            # First step
            optimizer.zero_grad()
            loss = loss_fn(model(s), t)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # Second step
            loss2 = loss_fn(model(s), t)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
        scheduler.step()

    return model


def train_erm(data, seed=42, n_ep=25, use_mixup=False, alpha_mixup=0.2, algo_name="erm"):
    """Train ERM model (reused for Mixup/VREx CKA)."""
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
    algo = build_algorithm(algo_name, model, cfg, DEVICE, n_domains=3)
    params = list(model.parameters()) + (
        list(algo.domain_classifier.parameters())
        if hasattr(algo, "domain_classifier") else []
    )
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
    loader = DataLoader(TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
                       batch_size=512, shuffle=True, drop_last=True)

    for ep in range(n_ep):
        model.train()
        for s, t, d in loader:
            s, t, d = s.to(DEVICE), t.to(DEVICE), d.to(DEVICE)
            if use_mixup:
                lam = np.random.beta(alpha_mixup, alpha_mixup)
                idx = torch.randperm(len(s))
                s = lam * s + (1 - lam) * s[idx]
                t = lam * t + (1 - lam) * t[idx]
            opt.zero_grad()
            r = algo.compute_loss(s, t, d)
            r["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sch.step()

    return model


def compute_downstream_impact(t1_pred, t2_pred, t1_true, t2_true, voxel_size=(1.0, 1.0, 1.0)):
    """Simulate downstream impact: how T1/T2 error affects tissue segmentation."""
    # Simulate a 3-class brain segmentation: WM (T1~800, T2~80), GM (T1~1300, T2~110), CSF (T1~4000, T2~2000)
    tissue_params = {
        "WM": {"t1": 800, "t2": 80},
        "GM": {"t1": 1300, "t2": 110},
        "CSF": {"t1": 4000, "t2": 2000},
    }

    def assign_tissue(t1, t2):
        """Assign tissue class based on T1/T2 values."""
        dists = {}
        for name, params in tissue_params.items():
            d = np.sqrt(((t1 - params["t1"]) / params["t1"]) ** 2 +
                       ((t2 - params["t2"]) / params["t2"]) ** 2)
            dists[name] = d
        return min(dists, key=dists.get)

    # Generate synthetic brain with known tissue labels
    rng = np.random.RandomState(42)
    n_voxels = 10000
    tissues = ["WM", "GM", "CSF"]
    true_labels = rng.choice(tissues, n_voxels, p=[0.5, 0.4, 0.1])

    # Generate true T1/T2 with noise
    true_t1 = np.array([tissue_params[t]["t1"] + rng.randn() * 50 for t in true_labels])
    true_t2 = np.array([tissue_params[t]["t2"] + rng.randn() * 10 for t in true_labels])

    # Predicted T1/T2 with error
    pred_t1 = true_t1 + rng.randn(n_voxels) * t1_pred  # MAE-scale error
    pred_t2 = true_t2 + rng.randn(n_voxels) * t2_pred

    # Assign tissues based on predicted values
    pred_labels = [assign_tissue(t1, t2) for t1, t2 in zip(pred_t1, pred_t2)]

    # Compute Dice scores
    dice = {}
    for tissue in tissues:
        true_mask = true_labels == tissue
        pred_mask = np.array(pred_labels) == tissue
        intersection = np.sum(true_mask & pred_mask)
        union = np.sum(true_mask) + np.sum(pred_mask)
        dice[tissue] = float(2 * intersection / max(union, 1))

    # Overall accuracy
    accuracy = float(np.mean(np.array(pred_labels) == true_labels))

    return {"dice": dice, "accuracy": accuracy}


def main():
    results = {}
    data = load_data()
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]
    seeds = [42, 123, 456]

    # ── 1. SAM Baseline ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TASK 1: SAM Baseline (3 seeds)")
    logger.info("=" * 60)

    sam_results = []
    for seed in seeds:
        logger.info("  SAM seed=%d ...", seed)
        t0 = time.time()
        model = train_sam(data, seed=seed)
        dt = time.time() - t0

        src = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
        ph = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
        ge = eval_model(model, data["ge_sig"], data["ge_tgt"], tgt_min, tgt_max)
        ds3_ph = ph["mae"] / max(src["mae"], 0.01)
        ds3_ge = ge["mae"] / max(src["mae"], 0.01)

        sam_results.append({
            "seed": seed, "src": src, "ph": ph, "ge": ge,
            "ds3_ph": ds3_ph, "ds3_ge": ds3_ge, "time": dt,
        })
        logger.info("  seed=%d: src=%.1f, ph=%.1f, ge=%.1f, ds3=%.1f (%.0fs)",
                     seed, src["mae"], ph["mae"], ge["mae"], ds3_ph, dt)

    results["sam"] = sam_results
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── 2. CKA for Mixup and VREx ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TASK 2: CKA for Mixup and VREx")
    logger.info("=" * 60)

    from qMR_Robust.eval.calibration import cka as cka_fn

    for algo_name, use_mixup in [("mixup", True), ("vrex", False)]:
        logger.info("  CKA for %s ...", algo_name)
        cka_results = []
        for seed in seeds:
            logger.info("  %s seed=%d ...", algo_name, seed)
            if use_mixup:
                model = train_erm(data, seed=seed, use_mixup=True)
            else:
                model = train_erm(data, seed=seed, algo_name=algo_name)

            feats_si = extract_features(model, data["va_sig"])
            feats_ph = extract_features(model, data["tgt_sig"])
            n = min(len(feats_si), len(feats_ph))
            cka_val = float(cka_fn(feats_si[:n], feats_ph[:n], "linear"))
            cka_results.append({"seed": seed, "cka_si_ph": cka_val})
            logger.info("  seed=%d: cka=%.4f", seed, cka_val)

        results[f"cka_{algo_name}"] = cka_results

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── 3. Downstream Impact Analysis ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("TASK 3: Downstream Impact Analysis")
    logger.info("=" * 60)

    # Load ERM results for T1/T2 MAE
    ckpt1 = json.load(open("results/checkpoint.json"))
    erm = ckpt1["results"]["algo_resnet1d_18_erm_seed42"]

    # Source performance (good)
    src_t1_mae = erm["source"]["mae_t1"]
    src_t2_mae = erm["source"]["mae_t2"]
    src_impact = compute_downstream_impact(src_t1_mae, src_t2_mae, 0, 0)
    logger.info("  Source: T1_mae=%.1f, T2_mae=%.1f → Dice=%s, Acc=%.3f",
                src_t1_mae, src_t2_mae, src_impact["dice"], src_impact["accuracy"])

    # OOD performance (bad)
    ood_t1_mae = erm["philips"]["mae_t1"]
    ood_t2_mae = erm["philips"]["mae_t2"]
    ood_impact = compute_downstream_impact(ood_t1_mae, ood_t2_mae, 0, 0)
    logger.info("  OOD:    T1_mae=%.1f, T2_mae=%.1f → Dice=%s, Acc=%.3f",
                ood_t1_mae, ood_t2_mae, ood_impact["dice"], ood_impact["accuracy"])

    # Classical baseline
    classical_impact = compute_downstream_impact(251.4 * 0.6, 251.4 * 0.4, 0, 0)  # ~251 ms total
    logger.info("  Classical: Dice=%s, Acc=%.3f", classical_impact["dice"], classical_impact["accuracy"])

    # Hybrid
    hybrid_impact = compute_downstream_impact(193.2 * 0.6, 193.2 * 0.4, 0, 0)
    logger.info("  Hybrid: Dice=%s, Acc=%.3f", hybrid_impact["dice"], hybrid_impact["accuracy"])

    results["downstream_impact"] = {
        "source": {"t1_mae": src_t1_mae, "t2_mae": src_t2_mae, **src_impact},
        "ood_erm": {"t1_mae": ood_t1_mae, "t2_mae": ood_t2_mae, **ood_impact},
        "classical": classical_impact,
        "hybrid": hybrid_impact,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ALL FINAL TASKS COMPLETE")
    logger.info("=" * 60)

    print("\n--- SAM ---")
    for r in results["sam"]:
        print(f"  seed={r['seed']}: src={r['src']['mae']:.1f}, ph={r['ph']['mae']:.1f}, ds3={r['ds3_ph']:.1f}")

    print("\n--- CKA Mixup ---")
    for r in results["cka_mixup"]:
        print(f"  seed={r['seed']}: cka={r['cka_si_ph']:.4f}")

    print("\n--- CKA VREx ---")
    for r in results["cka_vrex"]:
        print(f"  seed={r['seed']}: cka={r['cka_si_ph']:.4f}")

    print("\n--- Downstream Impact ---")
    for k, v in results["downstream_impact"].items():
        if "dice" in v:
            print(f"  {k}: accuracy={v['accuracy']:.3f}, WM Dice={v['dice']['WM']:.3f}, GM Dice={v['dice']['GM']:.3f}")


if __name__ == "__main__":
    main()
