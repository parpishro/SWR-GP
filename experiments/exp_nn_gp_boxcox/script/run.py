#!/usr/bin/env python3
"""NN-GP (Box-Cox λ=0.2) application study experiment — self-contained.

Fits GPNN (two-layer MLP mean + NNGP covariance, GLS iterations) on
Box-Cox-transformed Big Sur streamflow.  Architecture is fixed
(hidden_dim=24, dropout=0.0, weight_decay=1e-3); model selection uses a
time-ordered held-out CRPS on the tail of the training period.
Back-transformation uses the delta-method approximation to the conditional
mean, identical to the SWR-GP Box-Cox procedure.

Outputs (written to output/ sub-dir):
    results.json        training summary + test metrics
    best.pkl            serialised best GPNN model
    pit_sim_test.png    PIT histogram — simulation mode, test period
    pit_krig_test.png   PIT histogram — kriging mode, test period
    qq_sim_test.png     Innovation QQ — simulation mode, test period
    qq_krig_test.png    Innovation QQ — kriging mode, test period

Log written to run.log at experiment root.

Usage (from repository root):
    .venv/bin/python experiments/exp_nn_gp_boxcox/script/run.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import boxcox as scipy_boxcox
from scipy.stats import norm

HERE      = Path(__file__).resolve().parent
EXP_DIR   = HERE.parent
REPO_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from swrgp.bigsur_data import load_bigsur_train_test
from swrgp.diagnostics import compute_gaussian_calibration, compute_standardized_innovations
from swrgp.metrics import compute_crps, compute_metrics
from swrgp.nn_model import GPNN, set_global_seed
from swrgp.nngp import build_nngp_matrices
from swrgp.paths import DATA_DIR

# ── Experiment config ──────────────────────────────────────────────────────
M              = 20
MAX_LAG        = 100
NU             = 1.5
SEED           = 42
NN_HIDDEN_DIM  = 24
NN_DROPOUT     = 0.0
NN_WEIGHT_DECAY = 1e-3
NN_EPOCHS      = 400
NN_GLS_ITERS   = 6
BOXCOX_LAMBDA  = 0.2
METHOD         = "nn_gp"
TRANSFORM      = "boxcox"

# ── Generic helpers ────────────────────────────────────────────────────────

def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("exp_nn_gp_boxcox")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _plot_qq(z: np.ndarray, out_file: Path, title: str) -> None:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 10:
        return
    osm = np.sort(norm.ppf((np.arange(1, z.size + 1) - 0.5) / z.size))
    osr = np.sort(z)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    ax.scatter(osm, osr, s=8, alpha=0.6)
    lx = np.array([osm.min(), osm.max()])
    ax.plot(lx, lx, "r--", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("Theoretical N(0,1)")
    ax.set_ylabel("Innovation quantiles")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def _plot_pit(pit: np.ndarray, out_file: Path, title: str, bins: int = 20) -> None:
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.hist(pit, bins=bins, density=True, alpha=0.7, edgecolor="k")
    ax.axhline(1.0, color="r", ls="--", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def _make_gpnn(**kw) -> GPNN:
    return GPNN(
        lag_window=MAX_LAG,
        m=M,
        nu=NU,
        hidden_dim=NN_HIDDEN_DIM,
        dropout=NN_DROPOUT,
        epochs=NN_EPOCHS,
        lr=1e-3,
        n_gls_iterations=NN_GLS_ITERS,
        boxcox_lambda=BOXCOX_LAMBDA,
        log_transform=False,
        verbose=False,
        patience=20,
        min_delta=1e-5,
        min_epochs=20,
        max_grad_norm=5.0,
        weight_decay=NN_WEIGHT_DECAY,
        seed=SEED,
        **kw,
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    out_dir = EXP_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(EXP_DIR / "run.log")

    logger.info("Experiment: %s__%s  lambda=%.2f", METHOD, TRANSFORM, BOXCOX_LAMBDA)
    logger.info(
        "Config: m=%d  max_lag=%d  nu=%.1f  seed=%d  "
        "hidden_dim=%d  dropout=%.2f  weight_decay=%.4f  "
        "epochs=%d  gls_iters=%d",
        M, MAX_LAG, NU, SEED,
        NN_HIDDEN_DIM, NN_DROPOUT, NN_WEIGHT_DECAY,
        NN_EPOCHS, NN_GLS_ITERS,
    )

    set_global_seed(SEED)

    train_df, test_df = load_bigsur_train_test(DATA_DIR)
    rain_train = train_df["rain"].to_numpy(dtype=float)
    y_train    = train_df["gauge"].to_numpy(dtype=float)
    rain_test  = test_df["rain"].to_numpy(dtype=float)
    y_test     = test_df["gauge"].to_numpy(dtype=float)
    full_rain  = np.concatenate([rain_train, rain_test])
    full_y     = np.concatenate([y_train, y_test])
    n_train    = len(y_train)

    logger.info("Data: train=%d  test=%d", n_train, len(y_test))

    # ── Held-out split for selection score (time-ordered tail) ──────────
    min_val    = max(int(MAX_LAG), 60)
    target_val = max(min_val, int(round(0.2 * n_train)))
    target_val = min(target_val, max(10, n_train // 3))
    sub_end    = n_train - target_val
    min_sub    = max(MAX_LAG + 5, M + 5, 30)
    if sub_end < min_sub:
        sub_end = min_sub
    if sub_end >= n_train - 5:
        logger.error("Not enough training samples for held-out split")
        return 1

    rain_sub    = rain_train[:sub_end]
    rain_val    = rain_train[sub_end:]
    y_sub       = y_train[:sub_end]
    y_val       = y_train[sub_end:]

    logger.info("Held-out split: sub_n=%d  val_n=%d", sub_end, len(y_val))

    # ── Selection model (fit on sub-train, validate on val) ─────────────
    logger.info("Fitting selector model on sub-train ...")
    t_sel = time.perf_counter()
    selector = _make_gpnn()
    selector.fit(rain_sub, y_sub)

    sel_theta       = np.asarray(selector.theta_, dtype=float)
    sigma_sel_t     = float(np.sqrt(max(sel_theta[0] + sel_theta[3], 1e-12)))
    lam             = selector.boxcox_lambda
    shift           = selector.boxcox_shift_

    # Kriging on the sub+val range; keep only the val portion
    krig_subval   = selector.forecast(rain_train, y_train)  # original scale
    krig_val      = np.maximum(krig_subval[sub_end:], 0.0)

    # Box-Cox CRPS for selection (need transformed-scale quantities)
    krig_val_t    = selector.forecast_raw_[sub_end:]
    y_val_t       = scipy_boxcox(y_val + shift, lam)
    selection_score = float(compute_crps(
        y_val, krig_val,
        y_transformed=y_val_t, mu_transformed=krig_val_t,
        family="gaussian", sigma=sigma_sel_t,
        response_transform="boxcox", response_shift=shift, response_lambda=lam,
    ))
    logger.info("Selector done in %.1fs | val krig CRPS=%.4f",
                time.perf_counter() - t_sel, selection_score)

    # ── Final model (fit on full training data) ──────────────────────────
    logger.info("Fitting final model on full training data ...")
    t_fit = time.perf_counter()
    model = _make_gpnn()
    model.fit(rain_train, y_train)
    fit_sec = time.perf_counter() - t_fit
    logger.info("Final model fit in %.1fs", fit_sec)

    n_params_eff = int(sum(p.numel() for p in model.net.parameters()) + 3)
    theta  = np.asarray(model.theta_, dtype=float)
    lam    = model.boxcox_lambda
    shift  = model.boxcox_shift_

    # ── Predictions ──────────────────────────────────────────────────────
    t_wall_start = time.perf_counter()

    # Simulation mode (NN mean, original scale)
    sim_full      = model.predict(full_rain)
    sim_train_arr = np.maximum(sim_full[:n_train], 0.0)
    sim_test_arr  = np.maximum(sim_full[n_train:], 0.0)
    sim_train_t   = model.predict_raw_[:n_train]    # transformed scale
    sim_test_t    = model.predict_raw_[n_train:]

    # Kriging mode (NN mean + GP correction, original scale)
    krig_full  = model.forecast(full_rain, full_y)
    krig_train = np.maximum(krig_full[:n_train], 0.0)
    krig_test  = np.maximum(krig_full[n_train:], 0.0)
    krig_train_t = model.forecast_raw_[:n_train]    # transformed scale
    krig_test_t  = model.forecast_raw_[n_train:]

    # Per-observation kriging sigma (transformed scale)
    sigma_krig_train = np.sqrt(np.maximum(model.F_krig_[:n_train], 1e-12))
    sigma_krig_test  = np.sqrt(np.maximum(model.F_krig_[n_train:], 1e-12))
    sigma_marginal   = model.sigma_marginal_

    # Transformed-scale observed values (for CRPS)
    y_train_t = scipy_boxcox(y_train + shift, lam)
    y_test_t  = scipy_boxcox(y_test  + shift, lam)

    # ── Point metrics (original scale) ───────────────────────────────────
    metrics: dict[str, Any] = {
        "sim_train":  compute_metrics(y_train, sim_train_arr),
        "sim_test":   compute_metrics(y_test,  sim_test_arr),
        "krig_train": compute_metrics(y_train, krig_train),
        "krig_test":  compute_metrics(y_test,  krig_test),
    }

    # CRPS — Box-Cox Monte Carlo back-transform
    _crps_kw = dict(
        family="gaussian",
        response_transform="boxcox", response_shift=shift, response_lambda=lam,
    )
    metrics["sim_train"]["CRPS"] = float(compute_crps(
        y_train, sim_train_arr, y_transformed=y_train_t, mu_transformed=sim_train_t,
        sigma=sigma_marginal, **_crps_kw))
    metrics["sim_test"]["CRPS"] = float(compute_crps(
        y_test, sim_test_arr, y_transformed=y_test_t, mu_transformed=sim_test_t,
        sigma=sigma_marginal, **_crps_kw))
    metrics["krig_train"]["CRPS"] = float(compute_crps(
        y_train, krig_train, y_transformed=y_train_t, mu_transformed=krig_train_t,
        sigma=sigma_krig_train, **_crps_kw))
    metrics["krig_test"]["CRPS"] = float(compute_crps(
        y_test, krig_test, y_transformed=y_test_t, mu_transformed=krig_test_t,
        sigma=sigma_krig_test, **_crps_kw))

    logger.info(
        "Final | SimTest NSE=%.4f CRPS=%.4f | KrigTest NSE=%.4f CRPS=%.4f",
        metrics["sim_test"]["NSE"],  metrics["sim_test"]["CRPS"],
        metrics["krig_test"]["NSE"], metrics["krig_test"]["CRPS"],
    )

    # ── Calibration diagnostics ──────────────────────────────────────────
    calibration = {
        "sim_train": compute_gaussian_calibration(
            y_train, sim_train_arr, np.full(n_train, sigma_marginal),
            transform="boxcox", lam=lam, shift=shift, mu_transformed=sim_train_t),
        "sim_test": compute_gaussian_calibration(
            y_test, sim_test_arr, np.full(len(y_test), sigma_marginal),
            transform="boxcox", lam=lam, shift=shift, mu_transformed=sim_test_t),
        "krig_train": compute_gaussian_calibration(
            y_train, krig_train, sigma_krig_train,
            transform="boxcox", lam=lam, shift=shift, mu_transformed=krig_train_t),
        "krig_test": compute_gaussian_calibration(
            y_test, krig_test, sigma_krig_test,
            transform="boxcox", lam=lam, shift=shift, mu_transformed=krig_test_t),
    }
    innovations = {
        "sim_test": compute_standardized_innovations(
            y_test, sim_test_arr, sigma_marginal),
        "krig_test": compute_standardized_innovations(
            y_test, krig_test, sigma_krig_test),
    }

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_qq(innovations["sim_test"]["z"],
             out_dir / "qq_sim_test.png",  "Innovation QQ: NN-GP-BC sim (test)")
    _plot_qq(innovations["krig_test"]["z"],
             out_dir / "qq_krig_test.png", "Innovation QQ: NN-GP-BC krig (test)")
    _plot_pit(calibration["sim_test"]["pit"],
              out_dir / "pit_sim_test.png",  "PIT: NN-GP-BC sim (test)")
    _plot_pit(calibration["krig_test"]["pit"],
              out_dir / "pit_krig_test.png", "PIT: NN-GP-BC krig (test)")

    # ── Save model ────────────────────────────────────────────────────────
    with open(out_dir / "best.pkl", "wb") as f:
        pickle.dump(model, f)

    wall_sec = time.perf_counter() - t_wall_start + fit_sec

    _diag_summary = lambda mode, cal, inn=None: {
        k: _to_plain(v)
        for k, v in {
            **(
                {k2: v2 for k2, v2 in cal[mode].items()
                 if k2 in {"pit_mean", "pit_var", "rqr_mean", "rqr_std", "n_valid"}}
                if mode in cal else {}
            ),
            **(
                {k2: v2 for k2, v2 in inn[mode].items()
                 if k2 in {"n_valid", "z_mean", "z_std", "z_skew", "z_excess_kurtosis"}}
                if inn and mode in inn else {}
            ),
        }.items()
    }

    payload: dict[str, Any] = {
        "experiment_id": f"{METHOD}__{TRANSFORM}",
        "method": METHOD,
        "transform": TRANSFORM,
        "config": {
            "m": M, "max_lag": MAX_LAG, "nu": NU, "seed": SEED,
            "hidden_dim": NN_HIDDEN_DIM, "dropout": NN_DROPOUT,
            "weight_decay": NN_WEIGHT_DECAY,
            "epochs": NN_EPOCHS, "gls_iterations": NN_GLS_ITERS,
            "boxcox_lambda": BOXCOX_LAMBDA,
            "n_params_eff": n_params_eff,
            "train_split": "hydr_year < 30",
            "test_split":  "hydr_year >= 30",
            "selection": "tail-of-training heldout CRPS (kriging, Box-Cox)",
        },
        "selection": {
            "criterion": "val_CRPS_krig(train_heldout, boxcox)",
            "selection_score": float(selection_score),
            "subtrain_n": int(sub_end),
            "holdout_n":  int(len(y_val)),
            "boxcox_shift": float(shift),
        },
        "best": {
            "theta": _to_plain(theta),
            "n_params_eff": n_params_eff,
            "boxcox_shift": float(shift),
            "metrics": _to_plain(metrics),
            "diagnostics": {
                "calibration": {
                    mode: _diag_summary(mode, calibration)
                    for mode in calibration
                },
                "innovation": {
                    mode: _diag_summary(mode, {}, innovations)
                    for mode in innovations
                },
            },
        },
        "runtime_wall_sec": float(wall_sec),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Experiment complete | val_CRPS=%.4f | wall=%.1fs",
                selection_score, wall_sec)
    logger.info("Artifacts in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
