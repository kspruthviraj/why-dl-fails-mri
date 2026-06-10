"""
run_full_benchmark.py — Resumable, checkpointed experiment pipeline.

Saves results after EVERY individual experiment to results/checkpoint.json.
If the process crashes, re-running the same script resumes from where it left off.

Usage:
    PYTHONPATH=. python scripts/run_full_benchmark.py
"""

import json, logging, os, sys, time, hashlib
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
        logging.FileHandler("results/experiment_log.txt", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
FINAL_PATH = RESULTS_DIR / "experiment_results_v3.json"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECKPOINT HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"completed": {}, "results": {}}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2, default=str)


def is_done(ckpt, key):
    return key in ckpt.get("completed", {})


def mark_done(ckpt, key, result):
    ckpt["completed"][key] = True
    ckpt["results"][key] = result
    save_checkpoint(ckpt)
    logger.info("  ✓ Checkpointed: %s", key)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING (checkpointed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ensure_synthetic_data(ckpt):
    key = "step1_generate_data"
    if is_done(ckpt, key):
        logger.info("STEP 1: Already done, loading cached data")
    else:
        logger.info("STEP 1: Generating 100k synthetic MRF signals")
        import yaml
        from qMR_Robust.simulators.manager import SimulationManager

        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg["simulation"]["mrf"]["n_signals"] = 100_000
        cfg["simulation"]["mrf"]["n_workers"] = 8
        cfg["simulation"]["mrf"]["vendors"] = ["siemens", "philips", "ge"]
        cfg["simulation"]["mrf"]["field_strengths"] = [1.5, 3.0]
        cfg["simulation"]["mrf"]["fa_schedule_variants"] = 3
        cfg["simulation"]["mrf"]["tr_schedule_variants"] = 2

        mgr = SimulationManager(cfg)
        mgr.generate_mrf("data/synthetic/mrf_100k.h5", n_signals=100_000)
        mark_done(ckpt, key, {"n_signals": 100_000})

    import h5py
    with h5py.File("data/synthetic/mrf_100k.h5", "r") as f:
        sig_np = f["signals"][:]
        tgt_np = f["parameters"][:, :2].astype(np.float32)
        dom_np = f["domain_labels"][:]

    return sig_np, tgt_np, dom_np


def prepare_data(sig_np, tgt_np, dom_np):
    """Normalize and split data."""
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING / EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def denorm(pred, tgt_min, tgt_max):
    return pred * torch.from_numpy(tgt_max - tgt_min) + torch.from_numpy(tgt_min)


@torch.no_grad()
def eval_model(model, sig, tgt, tgt_min, tgt_max, bs=1024):
    model.eval()
    preds = []
    for i in range(0, len(sig), bs):
        preds.append(model(sig[i:i+bs].to(DEVICE)).cpu())
    pred = torch.cat(preds)
    pred_dn = denorm(pred, tgt_min, tgt_max)
    tgt_dn = denorm(tgt, tgt_min, tgt_max)
    from qMR_Robust.eval.metrics import mae, rmse, r_squared, concordance_cc
    return {
        "mae": float(mae(pred_dn, tgt_dn)),
        "rmse": float(rmse(pred_dn, tgt_dn)),
        "r2": float(r_squared(pred_dn, tgt_dn)),
        "ccc": float(concordance_cc(pred_dn, tgt_dn)),
        "mae_t1": float(torch.mean(torch.abs(pred_dn[:, 0] - tgt_dn[:, 0])).item()),
        "mae_t2": float(torch.mean(torch.abs(pred_dn[:, 1] - tgt_dn[:, 1])).item()),
    }


@torch.no_grad()
def extract_features(model, sig, max_n=2000, bs=512):
    model.eval()
    feats = []
    sig = sig[:max_n]
    for i in range(0, len(sig), bs):
        feats.append(model.encode(sig[i:i+bs].to(DEVICE)).cpu())
    return torch.cat(feats)


def train_one(arch, algo_name, seed, data, n_ep=25):
    """Train a single model and return results dict."""
    import yaml
    from qMR_Robust.models.registry import build_model
    from qMR_Robust.algorithms.base import build_algorithm

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    tr_sig, tr_tgt = data["tr_sig"], data["tr_tgt"]
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    mc = {
        "input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
        "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8,
        "n_transformer_layers": 6,
    }
    model = build_model(arch, mc).to(DEVICE)
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

    src_res = eval_model(model, data["va_sig"], data["va_tgt"], tgt_min, tgt_max)
    ph_res = eval_model(model, data["tgt_sig"], data["tgt_tgt"], tgt_min, tgt_max)
    ge_res = eval_model(model, data["ge_sig"], data["ge_tgt"], tgt_min, tgt_max)

    from qMR_Robust.eval.calibration import cka
    feats = {
        "siemens": extract_features(model, data["va_sig"]),
        "philips": extract_features(model, data["tgt_sig"]),
        "ge": extract_features(model, data["ge_sig"]),
    }
    ckas = {}
    for d1, d2 in [("siemens", "philips"), ("siemens", "ge"), ("philips", "ge")]:
        n = min(len(feats[d1]), len(feats[d2]))
        ckas[f"cka_{d1}_{d2}"] = float(cka(feats[d1][:n], feats[d2][:n], "linear"))

    ds3_ph = ph_res["mae"] / max(src_res["mae"], 0.01)
    ds3_ge = ge_res["mae"] / max(src_res["mae"], 0.01)

    return {
        "source": src_res, "philips": ph_res, "ge": ge_res,
        "ds3_philips": float(ds3_ph), "ds3_ge": float(ds3_ge),
        "cka": ckas,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHYSICS ATTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_physics_attribution(model, data, rng):
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]
    tgt_sig, tgt_tgt = data["tgt_sig"], data["tgt_tgt"]
    clean_mae = eval_model(model, tgt_sig, tgt_tgt, tgt_min, tgt_max)["mae"]

    tgt_c = tgt_sig[:, 0].numpy() + 1j * tgt_sig[:, 1].numpy()
    results = {"clean_mae": clean_mae}

    corruptions = {
        "B1+_0.50": lambda s: s * 0.50,
        "B1+_0.75": lambda s: s * 0.75,
        "SNR_5": lambda s: s + (rng.randn(*s.shape) + 1j * rng.randn(*s.shape))
            * np.sqrt(np.mean(np.abs(s)**2, axis=-1, keepdims=True) / 5),
        "SNR_10": lambda s: s + (rng.randn(*s.shape) + 1j * rng.randn(*s.shape))
            * np.sqrt(np.mean(np.abs(s)**2, axis=-1, keepdims=True) / 10),
        "Timing_shuffle": lambda s: np.stack(
            [s[i][rng.permutation(s.shape[-1])] for i in range(len(s))]
        ),
        "B0_+50Hz": lambda s: s * np.exp(
            1j * 2 * np.pi * 50 * np.linspace(0, s.shape[-1] / 1000, s.shape[-1])
        )[np.newaxis, :],
        "B0_+100Hz": lambda s: s * np.exp(
            1j * 2 * np.pi * 100 * np.linspace(0, s.shape[-1] / 1000, s.shape[-1])
        )[np.newaxis, :],
        "Gradient_10pct": lambda s: s * np.linspace(1.0, 1.10, s.shape[-1])[np.newaxis, :],
        "Gradient_20pct": lambda s: s * np.linspace(1.0, 1.20, s.shape[-1])[np.newaxis, :],
    }

    for name, fn in corruptions.items():
        cor = fn(tgt_c.copy())
        cor_2ch = torch.from_numpy(
            np.stack([cor.real, cor.imag], axis=1).astype(np.float32)
        )
        for i in range(len(cor_2ch)):
            pk = cor_2ch[i].abs().max()
            if pk > 0:
                cor_2ch[i] /= pk
        cmae = eval_model(model, cor_2ch, tgt_tgt, tgt_min, tgt_max)["mae"]
        results[name] = {"mae": cmae, "ds3": cmae / max(clean_mae, 0.01)}

    # Combined
    cor = tgt_c.copy()
    cor = cor * 0.75  # B1
    cor = cor * np.exp(
        1j * 2 * np.pi * 50 * np.linspace(0, cor.shape[-1] / 1000, cor.shape[-1])
    )[np.newaxis, :]  # B0
    cor = cor + (rng.randn(*cor.shape) + 1j * rng.randn(*cor.shape)) * np.sqrt(
        np.mean(np.abs(cor)**2, axis=-1, keepdims=True) / 10
    )  # SNR
    cor_2ch = torch.from_numpy(
        np.stack([cor.real, cor.imag], axis=1).astype(np.float32)
    )
    for i in range(len(cor_2ch)):
        pk = cor_2ch[i].abs().max()
        if pk > 0:
            cor_2ch[i] /= pk
    cmae = eval_model(model, cor_2ch, tgt_tgt, tgt_min, tgt_max)["mae"]
    results["ALL_combined"] = {"mae": cmae, "ds3": cmae / max(clean_mae, 0.01)}

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLASSICAL BASELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_classical(data):
    from qMR_Robust.baselines.classical import DictionaryMatcher
    from qMR_Robust.eval.metrics import mae

    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]
    ds = 5000

    # Dictionary: normalized signals → denormalized T1/T2 parameters
    dict_sig = (data["tr_sig"][:ds, 0] + 1j * data["tr_sig"][:ds, 1]).numpy()
    dict_params = denorm(data["tr_tgt"][:ds], tgt_min, tgt_max).numpy()
    dm = DictionaryMatcher(dict_sig, dict_params)

    # Evaluate on Philips
    ph_sig = (data["tgt_sig"][:, 0] + 1j * data["tgt_sig"][:, 1]).numpy()
    ph_pred, _ = dm.match_batch(ph_sig)
    ph_dn = denorm(data["tgt_tgt"], tgt_min, tgt_max)
    ph_mae = float(mae(torch.from_numpy(ph_pred[:, :2].astype(np.float32)), ph_dn))

    # Evaluate on GE
    ge_sig = (data["ge_sig"][:, 0] + 1j * data["ge_sig"][:, 1]).numpy()
    ge_pred, _ = dm.match_batch(ge_sig)
    ge_dn = denorm(data["ge_tgt"], tgt_min, tgt_max)
    ge_mae = float(mae(torch.from_numpy(ge_pred[:, :2].astype(np.float32)), ge_dn))

    # Evaluate on source (Siemens val)
    va_sig = (data["va_sig"][:, 0] + 1j * data["va_sig"][:, 1]).numpy()
    va_pred, _ = dm.match_batch(va_sig)
    va_dn = denorm(data["va_tgt"], tgt_min, tgt_max)
    va_mae = float(mae(torch.from_numpy(va_pred[:, :2].astype(np.float32)), va_dn))

    return {
        "source_mae": va_mae, "philips_mae": ph_mae, "ge_mae": ge_mae,
        "ds3_philips": ph_mae / max(va_mae, 0.01),
        "ds3_ge": ge_mae / max(va_mae, 0.01),
        "dict_size": ds,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MC DROPOUT UNCERTAINTY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_uncertainty(model, data):
    tgt_min, tgt_max = data["tgt_min"], data["tgt_max"]
    tgt_sig, tgt_tgt = data["tgt_sig"], data["tgt_tgt"]

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    mc_preds = []
    with torch.no_grad():
        for _ in range(20):
            p = []
            for i in range(0, len(tgt_sig), 1024):
                p.append(model(tgt_sig[i:i+1024].to(DEVICE)).cpu())
            mc_preds.append(torch.cat(p))

    mc_s = torch.stack(mc_preds)
    mc_m_dn = denorm(mc_s.mean(0), tgt_min, tgt_max)
    mc_std_dn = mc_s.std(0) * torch.from_numpy(tgt_max - tgt_min)
    tgt_dn = denorm(tgt_tgt, tgt_min, tgt_max)
    err = (mc_m_dn - tgt_dn).abs()

    return {
        "T1": {
            "sigma_ms": float(mc_std_dn[:, 0].mean()),
            "error_ms": float(err[:, 0].mean()),
            "correlation": float(np.corrcoef(mc_std_dn[:, 0].numpy(), err[:, 0].numpy())[0, 1]),
        },
        "T2": {
            "sigma_ms": float(mc_std_dn[:, 1].mean()),
            "error_ms": float(err[:, 1].mean()),
            "correlation": float(np.corrcoef(mc_std_dn[:, 1].numpy(), err[:, 1].numpy())[0, 1]),
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ckpt = load_checkpoint()
    logger.info("Loaded checkpoint: %d experiments completed", len(ckpt.get("completed", {})))

    # Step 1: Data
    sig_np, tgt_np, dom_np = ensure_synthetic_data(ckpt)
    data = prepare_data(sig_np, tgt_np, dom_np)
    logger.info(
        "Data: Siemens train=%d val=%d | Philips=%d | GE=%d",
        len(data["tr_sig"]), len(data["va_sig"]),
        len(data["tgt_sig"]), len(data["ge_sig"]),
    )

    # Step 2: Algorithm × Architecture grid
    configs = [
        ("resnet1d_18", "erm"),
        ("resnet1d_18", "coral"),
        ("vit1d", "erm"),
        ("vit1d", "coral"),
    ]
    seeds = [42, 123, 456]

    for arch, algo in configs:
        runs = []
        for seed in seeds:
            key = f"algo_{arch}_{algo}_seed{seed}"
            if is_done(ckpt, key):
                runs.append(ckpt["results"][key])
                logger.info("  %s: cached", key)
                continue

            logger.info("  Training %s %s seed=%d ...", arch, algo, seed)
            t0 = time.time()
            result = train_one(arch, algo, seed, data)
            result["time_s"] = time.time() - t0
            result["seed"] = seed
            mark_done(ckpt, key, result)
            runs.append(result)
            logger.info(
                "  %s: src=%.1f ph=%.1f ge=%.1f DS3_ph=%.1f (%.0fs)",
                key, result["source"]["mae"], result["philips"]["mae"],
                result["ge"]["mae"], result["ds3_philips"], result["time_s"],
            )

    # Step 3: Physics Attribution (reuse best ResNet ERM model)
    key = "physics_attribution"
    if not is_done(ckpt, key):
        logger.info("STEP 3: Physics Attribution")
        import yaml
        from qMR_Robust.models.registry import build_model
        from qMR_Robust.algorithms.base import build_algorithm
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        torch.manual_seed(42); np.random.seed(42)
        mc = {
            "input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
            "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8,
            "n_transformer_layers": 6,
        }
        model = build_model("resnet1d_18", mc).to(DEVICE)
        algo = build_algorithm("erm", model, cfg, DEVICE, n_domains=3)
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
        dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
        loader = DataLoader(
            TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
            batch_size=512, shuffle=True, drop_last=True,
        )
        for ep in range(25):
            model.train()
            for s, t, d in loader:
                s, t, d = s.to(DEVICE), t.to(DEVICE), d.to(DEVICE)
                opt.zero_grad(); r = algo.compute_loss(s, t, d); r["loss"].backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            sch.step()
        phys = run_physics_attribution(model, data, np.random.RandomState(42))
        mark_done(ckpt, key, phys)
        for name, v in phys.items():
            if isinstance(v, dict) and "ds3" in v:
                logger.info("  %s: DS3=%.3f", name, v["ds3"])
    else:
        logger.info("STEP 3: Physics Attribution — cached")

    # Step 4: Classical baseline
    key = "classical_baseline"
    if not is_done(ckpt, key):
        logger.info("STEP 4: Classical Baseline (Dictionary Matching)")
        classical = run_classical(data)
        mark_done(ckpt, key, classical)
        logger.info(
            "  Dict: src=%.1f ph=%.1f ge=%.1f DS3_ph=%.2f",
            classical["source_mae"], classical["philips_mae"],
            classical["ge_mae"], classical["ds3_philips"],
        )
    else:
        logger.info("STEP 4: Classical Baseline — cached")

    # Step 5: MC Dropout uncertainty
    key = "uncertainty"
    if not is_done(ckpt, key):
        logger.info("STEP 5: MC Dropout Uncertainty")
        import yaml
        from qMR_Robust.models.registry import build_model
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        torch.manual_seed(42)
        mc = {
            "input_channels": 2, "output_dim": 2, "hidden_dim": 256, "dropout": 0.1,
            "seq_len": data["sig_shape"][-1], "patch_size": 32, "n_heads": 8,
            "n_transformer_layers": 6,
        }
        model = build_model("resnet1d_18", mc).to(DEVICE)
        algo = build_algorithm("erm", model, cfg, DEVICE, n_domains=3)
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
        dom = torch.zeros(len(data["tr_sig"]), dtype=torch.long)
        loader = DataLoader(
            TensorDataset(data["tr_sig"], data["tr_tgt"], dom),
            batch_size=512, shuffle=True, drop_last=True,
        )
        for ep in range(25):
            model.train()
            for s, t, d in loader:
                s, t, d = s.to(DEVICE), t.to(DEVICE), d.to(DEVICE)
                opt.zero_grad(); r = algo.compute_loss(s, t, d); r["loss"].backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            sch.step()
        unc = run_uncertainty(model, data)
        mark_done(ckpt, key, unc)
        for p, v in unc.items():
            logger.info("  %s: σ=%.1f err=%.1f corr=%.3f", p, v["sigma_ms"], v["error_ms"], v["correlation"])
    else:
        logger.info("STEP 5: MC Dropout — cached")

    # Step 6: Compile final JSON
    logger.info("STEP 6: Compiling final results")
    final = {
        "algorithm_comparison": {},
        "physics_attribution": ckpt["results"].get("physics_attribution", {}),
        "classical_baseline": ckpt["results"].get("classical_baseline", {}),
        "uncertainty": ckpt["results"].get("uncertainty", {}),
        "metadata": {
            "n_total": 100_000,
            "vendors": ["siemens", "philips", "ge"],
            "field_strengths": [1.5, 3.0],
            "seeds": seeds,
            "normalization": "per-sample peak + min-max target",
        },
    }

    for arch, algo in configs:
        key_name = f"{arch}_{algo}"
        runs = []
        for seed in seeds:
            ck = f"algo_{arch}_{algo}_seed{seed}"
            if ck in ckpt["results"]:
                runs.append(ckpt["results"][ck])
        if runs:
            final["algorithm_comparison"][key_name] = {
                "source_mae": f"{np.mean([r['source']['mae'] for r in runs]):.1f} ± {np.std([r['source']['mae'] for r in runs]):.1f}",
                "philips_mae": f"{np.mean([r['philips']['mae'] for r in runs]):.1f} ± {np.std([r['philips']['mae'] for r in runs]):.1f}",
                "ge_mae": f"{np.mean([r['ge']['mae'] for r in runs]):.1f} ± {np.std([r['ge']['mae'] for r in runs]):.1f}",
                "ds3_philips": f"{np.mean([r['ds3_philips'] for r in runs]):.1f} ± {np.std([r['ds3_philips'] for r in runs]):.1f}",
                "ds3_ge": f"{np.mean([r['ds3_ge'] for r in runs]):.1f} ± {np.std([r['ds3_ge'] for r in runs]):.1f}",
                "cka_siemens_philips": f"{np.mean([r['cka']['cka_siemens_philips'] for r in runs]):.4f} ± {np.std([r['cka']['cka_siemens_philips'] for r in runs]):.4f}",
                "raw_runs": [
                    {"seed": r["seed"], "source": r["source"], "philips": r["philips"],
                     "ge": r["ge"], "ds3_ph": r["ds3_philips"], "ds3_ge": r["ds3_ge"],
                     "cka": r["cka"], "time_s": r.get("time_s", 0)}
                    for r in runs
                ],
            }

    with open(FINAL_PATH, "w") as f:
        json.dump(final, f, indent=2, default=str)
    logger.info("Final results saved → %s", FINAL_PATH)
    logger.info("ALL EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
