"""
run_v2_experiments.py — Phase 2 experiments with checkpointing.

Resumable. Saves to results/checkpoint_v2.json after each experiment.

New experiments:
  1. B0 dose-response curve (0, 25, 50, 75, 100, 125, 150 Hz)
  2. Scaling law (10k, 50k, 100k)
  3. GroupDRO algorithm
  4. Physics-neural hybrid baseline
  5. B1+ ablation without peak normalization
  6. t-SNE feature visualization data
"""

import json, logging, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/experiment_log_v2.txt", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_PATH = RESULTS_DIR / "checkpoint_v2.json"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_ckpt():
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {"completed": {}, "results": {}}


def save_ckpt(ckpt):
    with open(CKPT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2, default=str)


def is_done(ckpt, key):
    return key in ckpt.get("completed", {})


def mark_done(ckpt, key, result):
    ckpt["completed"][key] = True
    ckpt["results"][key] = result
    save_ckpt(ckpt)
    logger.info("  ✓ %s", key)


def load_data():
    import h5py
    with h5py.File("data/synthetic/mrf_100k.h5", "r") as f:
        sig_np = f["signals"][:]
        tgt_np = f["parameters"][:, :2].astype(np.float32)
        dom_np = f["domain_labels"][:]
    return sig_np, tgt_np, dom_np


def prepare(sig_np, tgt_np, dom_np, normalize_peaks=True):
    sig_2ch = np.stack([sig_np.real, sig_np.imag], axis=1).astype(np.float32)
    if normalize_peaks:
        for i in range(len(sig_2ch)):
            pk = np.abs(sig_2ch[i]).max()
            if pk > 0:
                sig_2ch[i] /= pk
    tgt_min = tgt_np.min(axis=0)
    tgt_max = tgt_np.max(axis=0)
    tgt_norm = (tgt_np - tgt_min) / (tgt_max - tgt_min + 1e-8)
    sig_t = torch.from_numpy(sig_2ch)
    tgt_t = torch.from_numpy(tgt_norm)

    def mask(prefix):
        return np.array([
            d.decode().startswith(prefix) if isinstance(d, bytes) else d.startswith(prefix)
            for d in dom_np
        ])

    src_m, tgt_m, ge_m = mask("siemens"), mask("philips"), mask("ge")
    src_sig, src_tgt = sig_t[src_m], tgt_t[src_m]
    tgt_sig, tgt_tgt = sig_t[tgt_m], tgt_t[tgt_m]
    ge_sig, ge_tgt = sig_t[ge_m], tgt_t[ge_m]
    idx = torch.randperm(len(src_sig))
    n_tr = int(0.8 * len(src_sig))
    return {
        "tr_sig": src_sig[idx[:n_tr]], "tr_tgt": src_tgt[idx[:n_tr]],
        "va_sig": src_sig[idx[n_tr:]], "va_tgt": src_tgt[idx[n_tr:]],
        "tgt_sig": tgt_sig, "tgt_tgt": tgt_tgt,
        "ge_sig": ge_sig, "ge_tgt": ge_tgt,
        "tgt_min": tgt_min, "tgt_max": tgt_max,
        "sig_shape": sig_2ch.shape,
    }


def denorm(p, tgt_min, tgt_max):
    return p * torch.from_numpy(tgt_max - tgt_min) + torch.from_numpy(tgt_min)


@torch.no_grad()
def eval_m(model, sig, tgt, tgt_min, tgt_max, bs=1024):
    model.eval()
    preds = []
    for i in range(0, len(sig), bs):
        preds.append(model(sig[i:i+bs].to(DEVICE)).cpu())
    pred = denorm(torch.cat(preds), tgt_min, tgt_max)
    tgt_dn = denorm(tgt, tgt_min, tgt_max)
    from qMR_Robust.eval.metrics import mae
    return float(mae(pred, tgt_dn))


def train_model(arch, algo_name, data, seed, n_ep=25, lr=1e-3):
    import yaml
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(seed); np.random.seed(seed)
    mc = {"input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
          "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8,
          "n_transformer_layers": 6}
    model = build_model(arch, mc).to(DEVICE)
    algo = build_algorithm(algo_name, model, cfg, DEVICE, n_domains=3)
    params = list(model.parameters()) + (
        list(algo.domain_classifier.parameters())
        if hasattr(algo, "domain_classifier") else []
    )
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
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
    for i in range(0, min(len(sig), max_n), bs):
        feats.append(model.encode(sig[i:i+bs].to(DEVICE)).cpu())
    return torch.cat(feats)


def apply_b0(sig_complex, hz, rng):
    n = sig_complex.shape[-1]
    t = np.linspace(0, n / 1000, n)
    return sig_complex * np.exp(1j * 2 * np.pi * hz * t)[np.newaxis, :]


def apply_snr(sig_complex, snr, rng):
    p = np.mean(np.abs(sig_complex)**2, axis=-1, keepdims=True)
    return sig_complex + (rng.randn(*sig_complex.shape) + 1j * rng.randn(*sig_complex.shape)) * np.sqrt(p / snr)


def to_2ch_norm(sig_complex):
    x = np.stack([sig_complex.real, sig_complex.imag], axis=1).astype(np.float32)
    for i in range(len(x)):
        pk = np.abs(x[i]).max()
        if pk > 0:
            x[i] /= pk
    return torch.from_numpy(x)


def main():
    ckpt = load_ckpt()
    logger.info("Loaded checkpoint: %d completed", len(ckpt.get("completed", {})))

    sig_np, tgt_np, dom_np = load_data()

    # ── Experiment 1: B0 Dose-Response Curve ──────────────────────────────
    key = "b0_dose_response"
    if not is_done(ckpt, key):
        logger.info("EXP 1: B0 Dose-Response Curve (0-150 Hz)")
        data = prepare(sig_np, tgt_np, dom_np)
        model = train_model("resnet1d_18", "erm", data, seed=42)
        tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]
        clean = eval_m(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
        rng = np.random.RandomState(42)
        tgt_c = data["tgt_sig"][:, 0].numpy() + 1j * data["tgt_sig"][:, 1].numpy()
        # Denormalize for corruption
        tgt_c_dn = tgt_c * (tgt_max[0] - tgt_min[0]) + tgt_min[0]

        results = {"clean_mae": clean}
        for hz in [0, 25, 50, 75, 100, 125, 150]:
            cor = apply_b0(tgt_c_dn.copy(), hz, rng)
            cor_2ch = to_2ch_norm(cor)
            mae_val = eval_m(model, cor_2ch, data["tgt_tgt"], tgt_min, tgt_max)
            results[f"b0_{hz}hz"] = {"mae": mae_val, "ds3": mae_val / max(clean, 0.01)}
            logger.info("  B0 +%3dHz: MAE=%.1f  DS3=%.3f", hz, mae_val, mae_val / max(clean, 0.01))
        mark_done(ckpt, key, results)
    else:
        logger.info("EXP 1: B0 Dose-Response — cached")

    # ── Experiment 2: Scaling Law ─────────────────────────────────────────
    key = "scaling_law"
    if not is_done(ckpt, key):
        logger.info("EXP 2: Scaling Law (10k, 50k, 100k)")
        tgt_min_full, tgt_max_full = tgt_np.min(0), tgt_np.max(0)
        results = {}
        for n_src_train in [5000, 15000, 26659]:
            # Subset only the SOURCE training data, keep all target data for eval
            sub_data = prepare(sig_np, tgt_np, dom_np)
            sub_data["tr_sig"] = sub_data["tr_sig"][:n_src_train]
            sub_data["tr_tgt"] = sub_data["tr_tgt"][:n_src_train]
            t0 = time.time()
            model = train_model("resnet1d_18", "erm", sub_data, seed=42)
            dt = time.time() - t0
            src_mae = eval_m(model, sub_data["va_sig"], sub_data["va_tgt"], sub_data["tgt_min"], sub_data["tgt_max"])
            ph_mae = eval_m(model, sub_data["tgt_sig"], sub_data["tgt_tgt"], sub_data["tgt_min"], sub_data["tgt_max"])
            ds3 = ph_mae / max(src_mae, 0.01)
            results[str(n_src_train)] = {"n_train": n_src_train, "src_mae": src_mae, "ph_mae": ph_mae, "ds3": ds3, "time": dt}
            logger.info("  N_train=%6d: src=%.1f ph=%.1f DS3=%.1f (%.0fs)", n_src_train, src_mae, ph_mae, ds3, dt)
        mark_done(ckpt, key, results)
    else:
        logger.info("EXP 2: Scaling Law — cached")

    # ── Experiment 3: GroupDRO ────────────────────────────────────────────
    key = "groupdro"
    if not is_done(ckpt, key):
        logger.info("EXP 3: GroupDRO (3 seeds)")
        data = prepare(sig_np, tgt_np, dom_np)
        tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]
        runs = []
        for seed in [42, 123, 456]:
            t0 = time.time()
            model = train_model("resnet1d_18", "groupdro", data, seed=seed)
            dt = time.time() - t0
            src = eval_m(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
            ph = eval_m(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
            ge = eval_m(model, data["ge_sig"], data["ge_tgt"], tgt_min, tgt_max)
            feats = {
                "siemens": extract_features(model, data["va_sig"]),
                "philips": extract_features(model, data["tgt_sig"]),
            }
            from qMR_Robust.eval.calibration import cka
            n = min(len(feats["siemens"]), len(feats["philips"]))
            cka_val = float(cka(feats["siemens"][:n], feats["philips"][:n], "linear"))
            runs.append({"seed": seed, "src": src, "ph": ph, "ge": ge,
                         "ds3_ph": ph / max(src, 0.01), "cka": cka_val, "time": dt})
            logger.info("  seed=%d: src=%.1f ph=%.1f DS3=%.1f CKA=%.4f (%.0fs)",
                        seed, src, ph, ph / max(src, 0.01), cka_val, dt)
        mark_done(ckpt, key, runs)
    else:
        logger.info("EXP 3: GroupDRO — cached")

    # ── Experiment 4: Physics-Neural Hybrid ───────────────────────────────
    key = "hybrid"
    if not is_done(ckpt, key):
        logger.info("EXP 4: Physics-Neural Hybrid (Dict features + Neural features)")
        data = prepare(sig_np, tgt_np, dom_np)
        tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]

        # Train neural model
        model = train_model("resnet1d_18", "erm", data, seed=42)
        neural_ph = eval_m(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)

        # Classical
        from qMR_Robust.baselines.classical import DictionaryMatcher
        ds = 5000
        dict_sig = (data["tr_sig"][:ds, 0] + 1j * data["tr_sig"][:ds, 1]).numpy()
        dict_params = denorm(data["tr_tgt"][:ds], tgt_min, tgt_max).numpy()
        dm = DictionaryMatcher(dict_sig, dict_params)
        ph_sig = (data["tgt_sig"][:, 0] + 1j * data["tgt_sig"][:, 1]).numpy()
        ph_pred, _ = dm.match_batch(ph_sig)
        from qMR_Robust.eval.metrics import mae
        classical_ph = float(mae(torch.from_numpy(ph_pred[:, :2].astype(np.float32)),
                                 denorm(data["tgt_tgt"], tgt_min, tgt_max)))

        # Hybrid: average predictions
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(data["tgt_sig"]), 1024):
                preds.append(model(data["tgt_sig"][i:i+1024].to(DEVICE)).cpu())
            neural_pred = denorm(torch.cat(preds), tgt_min, tgt_max).numpy()

        hybrid_pred = (neural_pred + ph_pred[:, :2].astype(np.float32)) / 2.0
        hybrid_ph = float(mae(torch.from_numpy(hybrid_pred),
                              denorm(data["tgt_tgt"], tgt_min, tgt_max)))

        results = {
            "neural_mae": neural_ph,
            "classical_mae": classical_ph,
            "hybrid_mae": hybrid_ph,
            "improvement_over_neural": (neural_ph - hybrid_ph) / neural_ph * 100,
            "improvement_over_classical": (classical_ph - hybrid_ph) / classical_ph * 100,
        }
        mark_done(ckpt, key, results)
        logger.info("  Neural: %.1f  Classical: %.1f  Hybrid: %.1f", neural_ph, classical_ph, hybrid_ph)
    else:
        logger.info("EXP 4: Hybrid — cached")

    # ── Experiment 5: B1+ Ablation Without Peak Normalization ─────────────
    key = "b1_ablation_no_peak_norm"
    if not is_done(ckpt, key):
        logger.info("EXP 5: B1+ Ablation WITHOUT peak normalization")
        data_no_peak = prepare(sig_np, tgt_np, dom_np, normalize_peaks=False)
        tgt_min, tgt_max = data_no_peak["tgt_min"], data_no_peak["tgt_max"]
        model = train_model("resnet1d_18", "erm", data_no_peak, seed=42)
        clean = eval_m(model, data_no_peak["tgt_sig"], data_no_peak["tgt_tgt"], tgt_min, tgt_max)

        rng = np.random.RandomState(42)
        tgt_c = data_no_peak["tgt_sig"][:, 0].numpy() + 1j * data_no_peak["tgt_sig"][:, 1].numpy()
        tgt_c_dn = tgt_c * (tgt_max[0] - tgt_min[0]) + tgt_min[0]

        results = {"clean_mae": clean, "peak_norm": False}
        for b1 in [0.50, 0.75, 1.0]:
            cor = tgt_c_dn.copy() * b1
            cor_2ch = np.stack([cor.real, cor.imag], axis=1).astype(np.float32)
            # NO peak normalization
            cor_t = torch.from_numpy(cor_2ch)
            mae_val = eval_m(model, cor_t, data_no_peak["tgt_tgt"], tgt_min, tgt_max)
            results[f"b1_{b1}"] = {"mae": mae_val, "ds3": mae_val / max(clean, 0.01)}
            logger.info("  B1=%.2f (no peak norm): MAE=%.1f DS3=%.3f", b1, mae_val, mae_val / max(clean, 0.01))
        mark_done(ckpt, key, results)
    else:
        logger.info("EXP 5: B1+ ablation — cached")

    # ── Experiment 6: SNR Dose-Response ───────────────────────────────────
    key = "snr_dose_response"
    if not is_done(ckpt, key):
        logger.info("EXP 6: SNR Dose-Response (SNR 2, 5, 10, 20, 50)")
        data = prepare(sig_np, tgt_np, dom_np)
        tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]
        model = train_model("resnet1d_18", "erm", data, seed=42)
        clean = eval_m(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
        rng = np.random.RandomState(42)
        tgt_c = data["tgt_sig"][:, 0].numpy() + 1j * data["tgt_sig"][:, 1].numpy()
        tgt_c_dn = tgt_c * (tgt_max[0] - tgt_min[0]) + tgt_min[0]

        results = {"clean_mae": clean}
        for snr in [2, 5, 10, 20, 50]:
            cor = apply_snr(tgt_c_dn.copy(), snr, rng)
            cor_2ch = to_2ch_norm(cor)
            mae_val = eval_m(model, cor_2ch, data["tgt_tgt"], tgt_min, tgt_max)
            results[f"snr_{snr}"] = {"mae": mae_val, "ds3": mae_val / max(clean, 0.01)}
            logger.info("  SNR=%2d: MAE=%.1f DS3=%.3f", snr, mae_val, mae_val / max(clean, 0.01))
        mark_done(ckpt, key, results)
    else:
        logger.info("EXP 6: SNR dose-response — cached")

    # ── Compile final JSON ────────────────────────────────────────────────
    logger.info("Compiling final results")
    final = ckpt["results"]
    with open(RESULTS_DIR / "v2_results.json", "w") as f:
        json.dump(final, f, indent=2, default=str)
    logger.info("ALL V2 EXPERIMENTS COMPLETE → results/v2_results.json")


if __name__ == "__main__":
    main()
