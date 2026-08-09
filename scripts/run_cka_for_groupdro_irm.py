#!/usr/bin/env python3
"""
run_cka_for_groupdro_irm.py — Add CKA values to GroupDRO and IRM results.
"""

import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
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


@torch.no_grad()
def extract_features(model, sig, max_n=2000, bs=512):
    model.eval()
    feats = []
    sig = sig[:max_n]
    for i in range(0, len(sig), bs):
        feats.append(model.encode(sig[i:i + bs].to(DEVICE)).cpu())
    return torch.cat(feats)


def train_model(data, algo_name, seed, n_ep=25):
    import yaml
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

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
    dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
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
    from qMR_Robust.eval.calibration import cka as cka_fn

    ckpt_path = Path("results/checkpoint_v2.json")
    with open(ckpt_path) as f:
        ckpt = json.load(f)

    data = load_data()
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]

    # ── GroupDRO CKA ──────────────────────────────────────────────────
    grp = ckpt["results"]["groupdro"]
    if "cka_si_ph" not in grp[0]:
        logger.info("Adding CKA to GroupDRO (3 seeds) ...")
        for i, r in enumerate(grp):
            seed = r["seed"]
            logger.info("  GroupDRO seed=%d ...", seed)
            t0 = time.time()
            model = train_model(data, "groupdro", seed)

            src_mae = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
            ph_mae = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
            ge_mae = eval_model(model, data["ge_sig"], data["ge_tgt"], tgt_min, tgt_max)

            feats_si = extract_features(model, data["va_sig"])
            feats_ph = extract_features(model, data["tgt_sig"])
            n = min(len(feats_si), len(feats_ph))
            cka_val = float(cka_fn(feats_si[:n], feats_ph[:n], "linear"))

            grp[i]["cka_si_ph"] = cka_val
            grp[i]["time"] = time.time() - t0
            logger.info("  seed=%d: cka=%.4f (%.0fs)", seed, cka_val, time.time() - t0)

            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2, default=str)
    else:
        logger.info("GroupDRO CKA already done")

    # ── IRM CKA ───────────────────────────────────────────────────────
    irm = ckpt["results"]["irm"]
    if "cka_si_ph" not in irm[0]:
        logger.info("Adding CKA to IRM (3 seeds) ...")
        for i, r in enumerate(irm):
            seed = r["seed"]
            logger.info("  IRM seed=%d ...", seed)
            t0 = time.time()
            model = train_model(data, "irm", seed)

            feats_si = extract_features(model, data["va_sig"])
            feats_ph = extract_features(model, data["tgt_sig"])
            n = min(len(feats_si), len(feats_ph))
            cka_val = float(cka_fn(feats_si[:n], feats_ph[:n], "linear"))

            irm[i]["cka_si_ph"] = cka_val
            logger.info("  seed=%d: cka=%.4f (%.0fs)", seed, cka_val, time.time() - t0)

            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2, default=str)
    else:
        logger.info("IRM CKA already done")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n--- GroupDRO (all fields) ---")
    for r in ckpt["results"]["groupdro"]:
        cka_s = f"cka={r.get('cka_si_ph', 'N/A'):.4f}" if 'cka_si_ph' in r else "cka=MISSING"
        print(f"  seed={r['seed']}: src={r['src']:.1f}, ph={r['ph']:.1f}, ge={r['ge']:.1f}, {cka_s}")

    print("\n--- IRM (all fields) ---")
    for r in ckpt["results"]["irm"]:
        ge_s = f"ge={r['ge']:.1f}" if 'ge' in r else "ge=MISSING"
        cka_s = f"cka={r.get('cka_si_ph', 'N/A'):.4f}" if 'cka_si_ph' in r else "cka=MISSING"
        print(f"  seed={r['seed']}: src={r['src']:.1f}, ph={r['ph']:.1f}, {ge_s}, {cka_s}")


if __name__ == "__main__":
    main()
