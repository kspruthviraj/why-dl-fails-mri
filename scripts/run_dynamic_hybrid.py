#!/usr/bin/env python3
"""
run_dynamic_hybrid.py — Dynamic-gating hybrid: attention-based gate that
selects DL vs dictionary per-sample based on input signal features.

Instead of a single α, learn α(x) = sigmoid(gate_network(signal_features)).

Usage: PYTHONPATH=. python3 scripts/run_dynamic_hybrid.py
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
    mae = float(torch.mean(torch.abs(pred_dn - tgt_dn)).item())
    mae_t1 = float(torch.mean(torch.abs(pred_dn[:, 0] - tgt_dn[:, 0])).item())
    mae_t2 = float(torch.mean(torch.abs(pred_dn[:, 1] - tgt_dn[:, 1])).item())
    return {"mae": mae, "mae_t1": mae_t1, "mae_t2": mae_t2}


class DynamicGatedHybrid(nn.Module):
    """Dynamic gate: α(x) = sigmoid(MLP(features)) per sample."""
    def __init__(self, dl_model, feature_dim=512):
        super().__init__()
        self.dl_model = dl_model
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        with torch.no_grad():
            dl_pred = self.dl_model(x)
            features = self.dl_model.encode(x)
        alpha = self.gate(features)  # (B, 1)
        return dl_pred, alpha


def main():
    data = load_data()
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]

    import yaml
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm
    from qMR_Robust.baselines.classical import DictionaryMatcher

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Train ERM model
    logger.info("Training ERM model (seed=42) ...")
    torch.manual_seed(42)
    np.random.seed(42)

    mc = {"input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
          "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8, "n_transformer_layers": 6}
    dl_model = build_model("resnet1d_18", mc).to(DEVICE)
    algo = build_algorithm("erm", dl_model, cfg, DEVICE, n_domains=3)
    params = list(dl_model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
    loader = DataLoader(TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
                       batch_size=512, shuffle=True, drop_last=True)

    for ep in range(25):
        dl_model.train()
        for s, t, d in loader:
            s, t, d = s.to(DEVICE), t.to(DEVICE), d.to(DEVICE)
            opt.zero_grad()
            r = algo.compute_loss(s, t, d)
            r["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sch.step()

    dl_model.eval()

    # Get DL predictions on Philips
    @torch.no_grad()
    def get_dl_pred(sig):
        dl_model.eval()
        preds = []
        for i in range(0, len(sig), 1024):
            preds.append(dl_model(sig[i:i+1024].to(DEVICE)).cpu())
        return torch.cat(preds)

    # Get dictionary predictions — use ONLY Siemens training signals (not test data)
    logger.info("Running dictionary matching (5k Siemens training signals) ...")
    from qMR_Robust.baselines.classical import DictionaryMatcher
    ds = 5000
    dict_sig = (data["tr_sig"][:ds, 0] + 1j * data["tr_sig"][:ds, 1]).numpy()
    dict_params = denorm(data["tr_tgt"][:ds], tgt_min, tgt_max).numpy()
    dict_matcher = DictionaryMatcher(dict_sig, dict_params)

    # Philips predictions
    tgt_c = (data["tgt_sig"][:, 0] + 1j * data["tgt_sig"][:, 1]).numpy()
    dict_pred_np, _ = dict_matcher.match_batch(tgt_c)
    dict_pred = torch.from_numpy(dict_pred_np[:, :2].astype(np.float32))

    # Denormalize for comparison
    dl_pred_dn = denorm(get_dl_pred(data["tgt_sig"]), tgt_min, tgt_max)
    dict_pred_dn = dict_pred  # already in original scale
    tgt_dn = denorm(data["tgt_tgt"], tgt_min, tgt_max)

    # Static hybrid (ridge regression)
    from sklearn.linear_model import Ridge
    X = np.column_stack([dl_pred_dn.numpy(), dict_pred_dn.numpy()])
    y = tgt_dn.numpy()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    alpha_static = ridge.coef_[:, :2].mean() / (ridge.coef_[:, :2].mean() + ridge.coef_[:, 2:].mean())
    static_pred = torch.from_numpy(ridge.predict(X))
    static_mae = float(torch.mean(torch.abs(static_pred - tgt_dn)).item())

    # Dynamic hybrid
    logger.info("Training dynamic gate ...")
    hybrid = DynamicGatedHybrid(dl_model).to(DEVICE)
    gate_optimizer = torch.optim.Adam(hybrid.gate.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    # Train gate on source validation set
    for ep in range(50):
        hybrid.train()
        idx = torch.randperm(len(data["va_sig"]))
        for i in range(0, len(idx), 256):
            batch_idx = idx[i:i+256]
            s = data["va_sig"][batch_idx].to(DEVICE)
            t = data["va_tgt"][batch_idx].to(DEVICE)
            t_dn = denorm(t, tgt_min, tgt_max).to(DEVICE)

            dl_pred_batch = dl_model(s)
            dl_pred_dn_batch = denorm(dl_pred_batch, tgt_min, tgt_max).to(DEVICE)

            # Get dict predictions for this batch
            s_c = s[:, 0].cpu().numpy() + 1j * s[:, 1].cpu().numpy()
            dict_batch_np, _ = dict_matcher.match_batch(s_c)
            dict_batch = torch.from_numpy(dict_batch_np[:, :2].astype(np.float32)).to(DEVICE)

            features = dl_model.encode(s)
            alpha = hybrid.gate(features)  # (B, 1)
            combined = alpha * dl_pred_dn_batch + (1 - alpha) * dict_batch

            gate_optimizer.zero_grad()
            loss = loss_fn(combined, t_dn)
            loss.backward()
            gate_optimizer.step()

    # Evaluate dynamic hybrid on Philips
    hybrid.eval()
    with torch.no_grad():
        features = []
        for i in range(0, len(data["tgt_sig"]), 1024):
            features.append(dl_model.encode(data["tgt_sig"][i:i+1024].to(DEVICE)).cpu())
        features = torch.cat(features)
        alpha_dynamic = hybrid.gate(features.to(DEVICE)).cpu()

    dynamic_pred = alpha_dynamic * dl_pred_dn + (1 - alpha_dynamic) * dict_pred_dn
    dynamic_mae = float(torch.mean(torch.abs(dynamic_pred - tgt_dn)).item())

    # Also evaluate on GE
    ge_c = data["ge_sig"][:, 0].numpy() + 1j * data["ge_sig"][:, 1].numpy()
    dict_ge_np, _ = dict_matcher.match_batch(ge_c)
    dict_ge = torch.from_numpy(dict_ge_np[:, :2].astype(np.float32))
    dl_ge_dn = denorm(get_dl_pred(data["ge_sig"]), tgt_min, tgt_max)
    ge_dn = denorm(data["ge_tgt"], tgt_min, tgt_max)

    with torch.no_grad():
        ge_features = []
        for i in range(0, len(data["ge_sig"]), 1024):
            ge_features.append(dl_model.encode(data["ge_sig"][i:i+1024].to(DEVICE)).cpu())
        ge_features = torch.cat(ge_features)
        alpha_ge = hybrid.gate(ge_features.to(DEVICE)).cpu()

    dynamic_ge = alpha_ge * dl_ge_dn + (1 - alpha_ge) * dict_ge
    dynamic_ge_mae = float(torch.mean(torch.abs(dynamic_ge - ge_dn)).item())

    # Baselines
    dl_ph_mae = float(torch.mean(torch.abs(dl_pred_dn - tgt_dn)).item())
    dict_ph_mae = float(torch.mean(torch.abs(dict_pred_dn - tgt_dn)).item())
    dl_ge_mae = float(torch.mean(torch.abs(dl_ge_dn - ge_dn)).item())
    dict_ge_mae = float(torch.mean(torch.abs(dict_ge - ge_dn)).item())

    results = {
        "philips": {
            "dl_only": dl_ph_mae,
            "dict_only": dict_ph_mae,
            "static_hybrid": static_mae,
            "dynamic_hybrid": dynamic_mae,
            "static_alpha": float(alpha_static),
            "dynamic_alpha_mean": float(alpha_dynamic.mean()),
            "dynamic_alpha_std": float(alpha_dynamic.std()),
        },
        "ge": {
            "dl_only": dl_ge_mae,
            "dict_only": dict_ge_mae,
            "dynamic_hybrid": dynamic_ge_mae,
            "dynamic_alpha_mean": float(alpha_ge.mean()),
        },
    }

    with open(RESULTS_DIR / "dynamic_hybrid.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results:")
    for domain, vals in results.items():
        logger.info("  %s:", domain)
        for k, v in vals.items():
            logger.info("    %s: %.4f", k, v)


if __name__ == "__main__":
    main()
