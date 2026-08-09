#!/usr/bin/env python3
"""
compute_cka_full.py — Compute complete CKA analysis including within-vendor
and random baselines that are reported in Table 5 of the paper.

Produces results/cka_full.json with:
  - Within-Siemens CKA (split A vs B of Siemens validation set)
  - Within-Philips CKA (split A vs B)
  - Between-vendor CKA (si-ph, si-ge, ph-ge)
  - Random features CKA (same architecture, random weights)

Usage:
    PYTHONPATH=. python3 scripts/compute_cka_full.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
SEEDS = [42, 123, 456]


def load_data():
    """Load and prepare the same data split used in run_full_benchmark.py."""
    import h5py

    h5_path = "data/synthetic/mrf_100k.h5"
    if not Path(h5_path).exists():
        logger.error("Synthetic data not found at %s. Run data generation first.", h5_path)
        sys.exit(1)

    with h5py.File(h5_path, "r") as f:
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
    }


def train_model(data, seed=42, n_ep=25):
    """Train ResNet-1D ERM model (same as run_full_benchmark.py)."""
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    mc = {
        "input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
        "seq_len": data["tr_sig"].shape[-1], "patch_size": 32, "n_heads": 8,
        "n_transformer_layers": 6,
    }
    model = build_model("resnet1d_18", mc).to(DEVICE)
    algo = build_algorithm("erm", model, cfg, DEVICE, n_domains=3)
    params = list(model.parameters())
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


@torch.no_grad()
def extract_features(model, sig, max_n=2000, bs=512):
    model.eval()
    feats = []
    sig = sig[:max_n]
    for i in range(0, len(sig), bs):
        feats.append(model.encode(sig[i:i + bs].to(DEVICE)).cpu())
    return torch.cat(feats)


def main():
    from qMR_Robust.eval.calibration import cka

    logger.info("Loading data...")
    data = load_data()

    results = {}

    for seed in SEEDS:
        logger.info("Training model seed=%d ...", seed)
        model = train_model(data, seed=seed)
        model.eval()

        # Extract features
        feats = {
            "siemens_va": extract_features(model, data["va_sig"]),
            "philips": extract_features(model, data["tgt_sig"]),
            "ge": extract_features(model, data["ge_sig"]),
        }

        # Split Siemens validation into A and B
        n = len(feats["siemens_va"])
        half = n // 2
        feats["siemens_A"] = feats["siemens_va"][:half]
        feats["siemens_B"] = feats["siemens_va"][half:]

        # Split Philips into A and B
        n_p = len(feats["philips"])
        half_p = n_p // 2
        feats["philips_A"] = feats["philips"][:half_p]
        feats["philips_B"] = feats["philips"][half_p:]

        seed_results = {}

        # Within-Siemens CKA
        n_min = min(len(feats["siemens_A"]), len(feats["siemens_B"]))
        cka_within_si = cka(feats["siemens_A"][:n_min], feats["siemens_B"][:n_min], "linear")
        seed_results["within_siemens"] = cka_within_si

        # Within-Philips CKA
        n_min = min(len(feats["philips_A"]), len(feats["philips_B"]))
        cka_within_ph = cka(feats["philips_A"][:n_min], feats["philips_B"][:n_min], "linear")
        seed_results["within_philips"] = cka_within_ph

        # Between-vendor CKA
        for d1, d2 in [("siemens_va", "philips"), ("siemens_va", "ge"), ("philips", "ge")]:
            n = min(len(feats[d1]), len(feats[d2]))
            cka_val = cka(feats[d1][:n], feats[d2][:n], "linear")
            key = f"{d1.replace('_va', '')}_{d2}"
            seed_results[key] = cka_val

        # Random features CKA (same architecture, random weights)
        torch.manual_seed(seed + 1000)
        mc = {
            "input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
            "seq_len": data["va_sig"].shape[-1], "patch_size": 32, "n_heads": 8,
            "n_transformer_layers": 6,
        }
        from qMR_Robust.models.registry import build_model
        random_model = build_model("resnet1d_18", mc).to(DEVICE)
        random_model.eval()
        rand_A = extract_features(random_model, data["va_sig"][:1000])
        rand_B = extract_features(random_model, data["tgt_sig"][:1000])
        cka_random = cka(rand_A, rand_B, "linear")
        seed_results["random"] = cka_random

        results[f"seed_{seed}"] = seed_results
        logger.info("  seed=%d: within_si=%.4f within_ph=%.4f si-ph=%.4f si-ge=%.4f ph-ge=%.4f random=%.4f",
                     seed, cka_within_si, cka_within_ph,
                     seed_results.get("siemens_philips", 0),
                     seed_results.get("siemens_ge", 0),
                     seed_results.get("philips_ge", 0),
                     cka_random)

    # Compute averages
    avg = {}
    for key in results[f"seed_{SEEDS[0]}"]:
        vals = [results[f"seed_{s}"][key] for s in SEEDS]
        avg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                     "per_seed": {str(s): float(results[f"seed_{s}"][key]) for s in SEEDS}}

    output = {"per_seed": results, "averages": avg}

    out_path = RESULTS_DIR / "cka_full.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("CKA results saved → %s", out_path)

    # Print summary
    print("\nCKA Summary (mean ± std across 3 seeds):")
    for key, v in avg.items():
        print(f"  {key}: {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == "__main__":
    main()
