#!/usr/bin/env python3
"""
run_missing_experiments.py — Run all experiments that are missing from the paper.

Missing:
  1. Scaling law: add N=10000 and N=20000 points
  2. IRM: add GE evaluation
  3. Additional GroupDRO/IRM seeds (if time)

Usage: PYTHONPATH=. python3 scripts/run_missing_experiments.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/experiment_log.txt", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint_v2.json"


def load_checkpoint():
    with open(CHECKPOINT_PATH) as f:
        return json.load(f)


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2, default=str)


def load_data():
    """Load and prepare data (same as run_full_benchmark.py)."""
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
    return mae


def train_model(data, algo_name="erm", seed=42, n_ep=25, n_train=None):
    """Train a single model."""
    import yaml
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_sig, tr_tgt = data["tr_sig"], data["tr_tgt"]
    if n_train is not None and n_train < len(tr_sig):
        idx = torch.randperm(len(tr_sig))[:n_train]
        tr_sig, tr_tgt = tr_sig[idx], tr_tgt[idx]

    mc = {
        "input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
        "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8,
        "n_transformer_layers": 6,
    }
    model = build_model("resnet1d_18", mc).to(DEVICE)
    algo = build_algorithm(algo_name, model, cfg, DEVICE, n_domains=3)
    params = list(model.parameters()) + (
        list(algo.domain_classifier.parameters())
        if hasattr(algo, "domain_classifier") else []
    )
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    dom = torch.zeros(len(tr_sig), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(tr_sig, tr_tgt, dom),
        batch_size=512, shuffle=True, drop_last=True,
    )

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
    ckpt = load_checkpoint()
    data = load_data()
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]

    # ── 1. Scaling Law: add N=10000 and N=20000 ───────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Scaling Law (additional points)")
    logger.info("=" * 60)

    existing_points = set(ckpt["results"].get("scaling_law", {}).keys())
    new_points = [10000, 20000]
    needs_update = False

    for n_train in new_points:
        key = str(n_train)
        if key in existing_points:
            logger.info("  N=%d already done, skipping", n_train)
            continue

        logger.info("  Training N=%d ...", n_train)
        t0 = time.time()
        model = train_model(data, "erm", seed=42, n_train=n_train)
        dt = time.time() - t0

        src_mae = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
        ph_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
        ds3 = ph_mae / max(src_mae, 0.01)

        ckpt["results"]["scaling_law"][key] = {
            "n_train": n_train,
            "src_mae": float(src_mae),
            "ph_mae": float(ph_mae),
            "ds3": float(ds3),
            "time": float(dt),
        }
        needs_update = True
        logger.info("  N=%d: src=%.1f, ph=%.1f, ds3=%.1f (%.0fs)", n_train, src_mae, ph_mae, ds3, dt)
        save_checkpoint(ckpt)

    # ── 2. IRM: add GE evaluation ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: IRM GE Evaluation")
    logger.info("=" * 60)

    irm_results = ckpt["results"].get("irm", [])
    has_ge = any("ge" in r for r in irm_results) if irm_results else False

    if not has_ge and irm_results:
        logger.info("  Re-running IRM with GE evaluation (3 seeds) ...")
        irm_with_ge = []

        for r in irm_results:
            seed = r["seed"]
            logger.info("  IRM seed=%d: retraining with GE eval ...", seed)
            t0 = time.time()
            model = train_model(data, "irm", seed=seed)
            dt = time.time() - t0

            src_mae = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
            ph_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
            ge_mae = eval_model(model, data["ge_sig"], data["ge_tgt"], tgt_min, tgt_max)
            ds3_ph = ph_mae / max(src_mae, 0.01)
            ds3_ge = ge_mae / max(src_mae, 0.01)

            irm_with_ge.append({
                "seed": seed,
                "src": float(src_mae),
                "ph": float(ph_mae),
                "ge": float(ge_mae),
                "ds3_ph": float(ds3_ph),
                "ds3_ge": float(ds3_ge),
                "time": float(dt),
            })
            logger.info("  IRM seed=%d: src=%.1f, ph=%.1f, ge=%.1f, ds3_ph=%.2f, ds3_ge=%.2f",
                         seed, src_mae, ph_mae, ge_mae, ds3_ph, ds3_ge)

        ckpt["results"]["irm"] = irm_with_ge
        needs_update = True
        save_checkpoint(ckpt)
    else:
        logger.info("  IRM GE already done or no IRM results")

    # ── 3. GroupDRO: add CKA values ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: GroupDRO CKA")
    logger.info("=" * 60)

    from qMR_Robust.eval.calibration import cka as cka_fn

    grp_results = ckpt["results"].get("groupdro", [])
    has_cka = any("cka" in r for r in grp_results) if grp_results else False

    if not has_cka and grp_results:
        logger.info("  Re-running GroupDRO with CKA (3 seeds) ...")
        grp_with_cka = []

        for r in grp_results:
            seed = r["seed"]
            logger.info("  GroupDRO seed=%d: retraining with CKA ...", seed)
            t0 = time.time()
            model = train_model(data, "groupdro", seed=seed)
            dt = time.time() - t0

            @torch.no_grad()
            def extract_features(m, sig, max_n=2000, bs=512):
                m.eval()
                feats = []
                sig = sig[:max_n]
                for i in range(0, len(sig), bs):
                    feats.append(m.encode(sig[i:i + bs].to(DEVICE)).cpu())
                return torch.cat(feats)

            src_mae = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
            ph_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
            ge_mae = eval_model(model, data["ge_sig"], data["ge_tgt"], tgt_min, tgt_max)
            ds3_ph = ph_mae / max(src_mae, 0.01)
            ds3_ge = ge_mae / max(src_mae, 0.01)

            feats = {
                "siemens": extract_features(model, data["va_sig"]),
                "philips": extract_features(model, data["tgt_sig"]),
            }
            n = min(len(feats["siemens"]), len(feats["philips"]))
            cka_val = float(cka_fn(feats["siemens"][:n], feats["philips"][:n], "linear"))

            grp_with_cka.append({
                "seed": seed,
                "src": float(src_mae),
                "ph": float(ph_mae),
                "ge": float(ge_mae),
                "ds3_ph": float(ds3_ph),
                "ds3_ge": float(ds3_ge),
                "cka_si_ph": cka_val,
                "time": float(dt),
            })
            logger.info("  GroupDRO seed=%d: src=%.1f, ph=%.1f, ge=%.1f, ds3=%.1f, cka=%.4f (%.0fs)",
                         seed, src_mae, ph_mae, ge_mae, ds3_ph, cka_val, dt)

        ckpt["results"]["groupdro"] = grp_with_cka
        needs_update = True
        save_checkpoint(ckpt)
    else:
        logger.info("  GroupDRO CKA already done or no GroupDRO results")

    # ── 4. IRM CKA ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: IRM CKA")
    logger.info("=" * 60)

    irm_results = ckpt["results"].get("irm", [])
    has_cka = any("cka_si_ph" in r for r in irm_results) if irm_results else False

    if not has_cka and irm_results:
        logger.info("  Re-running IRM with CKA (3 seeds) ...")
        irm_with_cka = []

        for r in irm_results:
            seed = r["seed"]
            logger.info("  IRM seed=%d: retraining with CKA ...", seed)
            t0 = time.time()
            model = train_model(data, "irm", seed=seed)
            dt = time.time() - t0

            @torch.no_grad()
            def extract_features(m, sig, max_n=2000, bs=512):
                m.eval()
                feats = []
                sig = sig[:max_n]
                for i in range(0, len(sig), bs):
                    feats.append(m.encode(sig[i:i + bs].to(DEVICE)).cpu())
                return torch.cat(feats)

            src_mae = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
            ph_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
            ge_mae = eval_model(model, data["ge_sig"], data["ge_tgt"], tgt_min, tgt_max)
            ds3_ph = ph_mae / max(src_mae, 0.01)
            ds3_ge = ge_mae / max(src_mae, 0.01)

            feats = {
                "siemens": extract_features(model, data["va_sig"]),
                "philips": extract_features(model, data["tgt_sig"]),
            }
            n = min(len(feats["siemens"]), len(feats["philips"]))
            cka_val = float(cka_fn(feats["siemens"][:n], feats["philips"][:n], "linear"))

            irm_with_cka.append({
                "seed": seed,
                "src": float(src_mae),
                "ph": float(ph_mae),
                "ge": float(ge_mae),
                "ds3_ph": float(ds3_ph),
                "ds3_ge": float(ds3_ge),
                "cka_si_ph": cka_val,
                "time": float(dt),
            })
            logger.info("  IRM seed=%d: src=%.1f, ph=%.1f, ge=%.1f, ds3_ph=%.2f, ds3_ge=%.2f, cka=%.4f (%.0fs)",
                         seed, src_mae, ph_mae, ge_mae, ds3_ph, ds3_ge, cka_val, dt)

        ckpt["results"]["irm"] = irm_with_cka
        needs_update = True
        save_checkpoint(ckpt)
    else:
        logger.info("  IRM CKA already done or no IRM results")

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ALL MISSING EXPERIMENTS COMPLETE")
    logger.info("=" * 60)

    if needs_update:
        save_checkpoint(ckpt)
        logger.info("Updated checkpoint_v2.json")

    # Print summary
    print("\n--- Scaling Law (all points) ---")
    for k in sorted(ckpt["results"]["scaling_law"].keys(), key=lambda x: int(x)):
        v = ckpt["results"]["scaling_law"][k]
        print(f"  N={k}: src={v['src_mae']:.1f}, ph={v['ph_mae']:.1f}, ds3={v['ds3']:.1f}")

    print("\n--- IRM (with GE) ---")
    for r in ckpt["results"]["irm"]:
        ge_str = f"ge={r.get('ge', 'N/A'):.1f}" if 'ge' in r else "ge=N/A"
        cka_str = f"cka={r.get('cka_si_ph', 'N/A'):.4f}" if 'cka_si_ph' in r else "cka=N/A"
        print(f"  seed={r['seed']}: src={r['src']:.1f}, ph={r['ph']:.1f}, {ge_str}, {cka_str}")

    print("\n--- GroupDRO (with GE) ---")
    for r in ckpt["results"]["groupdro"]:
        cka_str = f"cka={r.get('cka_si_ph', 'N/A'):.4f}" if 'cka_si_ph' in r else "cka=N/A"
        print(f"  seed={r['seed']}: src={r['src']:.1f}, ph={r['ph']:.1f}, ge={r['ge']:.1f}, {cka_str}")


if __name__ == "__main__":
    main()
