#!/usr/bin/env python3
"""SWR-GP (no transform) application study experiment — self-contained.

Fits GPSWR (Gaussian kernel, NNGP covariance) on Big Sur daily streamflow.
K ∈ {1,2,3,4}; model selected by BIC(GLS).  Simulation mode = GP mean;
kriging mode = GP mean + residual NNGP correction.

Outputs (written to output/ sub-dir):
    results.json        per-K metrics + selected model summary
    best.pkl            serialised best GPSWR model
    pit_sim_test.png    PIT histogram — simulation mode, test period
    pit_krig_test.png   PIT histogram — kriging mode, test period
    qq_sim_test.png     Innovation QQ — simulation mode, test period
    qq_krig_test.png    Innovation QQ — kriging mode, test period

Log written to run.log at experiment root.

Usage (from repository root):
    .venv/bin/python experiments/exp_swr_gp_none/script/run.py
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
from scipy.stats import norm

HERE      = Path(__file__).resolve().parent
EXP_DIR   = HERE.parent
REPO_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from swrgp.bigsur_data import load_bigsur_train_test
from swrgp.diagnostics import compute_gaussian_calibration, compute_standardized_innovations
from swrgp.metrics import compute_aic, compute_bic, compute_crps, compute_metrics
from swrgp.model import GPSWR
from swrgp.nngp import build_nngp_matrices, gls_log_likelihood
from swrgp.paths import DATA_DIR

# ── Experiment config ──────────────────────────────────────────────────────
K_VALUES    = [1, 2, 3, 4]
M           = 20
MAX_LAG     = 100
NU          = 1.5
SEED        = 42
GP_MAXITER  = 200
KERNEL_TYPE = "gaussian"
METHOD      = "swr_gp"
TRANSFORM   = "none"
LOGGER_NAME = "exp_swr_gp_none"
LOG_FILENAME = "run.log"
OUTPUT_SUBDIR = "output"
EXPERIMENT_ID_SUFFIX = ""
ENFORCE_MAX_KERNEL_OVERLAP_IN_ESTIMATION = True

# ── Helpers ────────────────────────────────────────────────────────────────

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
    logger = logging.getLogger(LOGGER_NAME)
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


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    out_dir = EXP_DIR / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(EXP_DIR / LOG_FILENAME)

    logger.info("Experiment: %s__%s", METHOD, TRANSFORM)
    logger.info("Config: K=%s  m=%d  max_lag=%d  nu=%.1f  seed=%d  maxiter=%d",
                K_VALUES, M, MAX_LAG, NU, SEED, GP_MAXITER)

    train_df, test_df = load_bigsur_train_test(DATA_DIR)
    rain_train = train_df["rain"].to_numpy(dtype=float)
    y_train    = train_df["gauge"].to_numpy(dtype=float)
    rain_test  = test_df["rain"].to_numpy(dtype=float)
    y_test     = test_df["gauge"].to_numpy(dtype=float)
    full_rain  = np.concatenate([rain_train, rain_test])
    full_y     = np.concatenate([y_train, y_test])
    n_train    = len(y_train)

    logger.info("Data: train=%d  test=%d", n_train, len(y_test))

    best: dict[str, Any] | None = None
    best_bic = float("inf")
    per_k: dict[str, Any] = {}
    t_wall_start = time.perf_counter()

    for K in K_VALUES:
        logger.info("Fitting K=%d ...", K)
        t_k = time.perf_counter()

        model = GPSWR(
            K=K,
            m=M,
            nu=NU,
            kernel_type=KERNEL_TYPE,
            max_lag=MAX_LAG,
            maxiter=GP_MAXITER,
            log_transform=False,
            seed=SEED,
            verbose=False,
            enforce_max_kernel_overlap_in_estimation=ENFORCE_MAX_KERNEL_OVERLAP_IN_ESTIMATION,
        )
        model.fit(rain_train, y_train)
        n_params_eff = 3 * K + 3

        # Predictions on transformed scale (identity here) then clip to ≥0
        sim_train = np.maximum(model.predict(rain_train), 0.0)
        sim_test  = np.maximum(model.predict_with_history(rain_train, rain_test), 0.0)
        krig_full = model.forecast(full_rain, full_y)
        krig_train = np.maximum(krig_full[:n_train], 0.0)
        krig_test  = np.maximum(krig_full[n_train:], 0.0)
        theta = np.asarray(model.theta_, dtype=float)

        # Per-observation kriging sigma from NNGP conditional variance
        _, F_full = build_nngp_matrices(len(full_y), M, theta)
        sigma_krig_train = np.sqrt(np.maximum(F_full[:n_train], 1e-12))
        sigma_krig_test  = np.sqrt(np.maximum(F_full[n_train:], 1e-12))
        sigma_marginal   = float(np.sqrt(max(theta[0] + theta[3], 1e-12)))

        # Point metrics
        metrics: dict[str, Any] = {
            "sim_train":  compute_metrics(y_train, sim_train),
            "sim_test":   compute_metrics(y_test,  sim_test),
            "krig_train": compute_metrics(y_train, krig_train),
            "krig_test":  compute_metrics(y_test,  krig_test),
        }
        # CRPS — analytic Gaussian on original scale (no transform)
        metrics["sim_train"]["CRPS"]  = float(compute_crps(y_train, sim_train,  family="gaussian", sigma=sigma_marginal))
        metrics["sim_test"]["CRPS"]   = float(compute_crps(y_test,  sim_test,   family="gaussian", sigma=sigma_marginal))
        metrics["krig_train"]["CRPS"] = float(compute_crps(y_train, krig_train, family="gaussian", sigma=sigma_krig_train))
        metrics["krig_test"]["CRPS"]  = float(compute_crps(y_test,  krig_test,  family="gaussian", sigma=sigma_krig_test))

        # BIC(GLS) for within-model selection
        B, F = build_nngp_matrices(n_train, M, theta)
        ll      = gls_log_likelihood(y_train, sim_train, B, F, M)
        bic_val = compute_bic(ll, n_params_eff, n_train)
        aic_val = compute_aic(ll, n_params_eff)

        fit_sec = time.perf_counter() - t_k
        per_k[f"K{K}"] = {
            "K": K, "n_params_eff": n_params_eff,
            "bic": float(bic_val), "aic": float(aic_val), "loglik": float(ll),
            "ic_type": "BIC(GLS)", "fit_time_sec": float(fit_sec),
            "theta": theta.tolist(),
            "metrics": _to_plain(metrics),
        }
        logger.info(
            "K=%d | BIC(GLS)=%.2f | SimTest NSE=%.4f CRPS=%.4f | "
            "KrigTest NSE=%.4f CRPS=%.4f | %.1fs",
            K, bic_val,
            metrics["sim_test"]["NSE"],  metrics["sim_test"]["CRPS"],
            metrics["krig_test"]["NSE"], metrics["krig_test"]["CRPS"],
            fit_sec,
        )

        if bic_val < best_bic:
            best_bic = bic_val
            best = dict(
                K=K, model=model, theta=theta,
                sim_train=sim_train, sim_test=sim_test,
                krig_train=krig_train, krig_test=krig_test,
                sigma_krig_train=sigma_krig_train, sigma_krig_test=sigma_krig_test,
                sigma_marginal=sigma_marginal,
                metrics=metrics,
            )

    if best is None:
        logger.error("No valid model fitted.")
        return 1

    # Calibration diagnostics for best model
    calibration = {
        "sim_train": compute_gaussian_calibration(
            y_train, best["sim_train"],
            np.full(n_train, best["sigma_marginal"]),
            transform="none", lam=0.0, shift=0.0),
        "sim_test": compute_gaussian_calibration(
            y_test, best["sim_test"],
            np.full(len(y_test), best["sigma_marginal"]),
            transform="none", lam=0.0, shift=0.0),
        "krig_train": compute_gaussian_calibration(
            y_train, best["krig_train"], best["sigma_krig_train"],
            transform="none", lam=0.0, shift=0.0),
        "krig_test": compute_gaussian_calibration(
            y_test, best["krig_test"], best["sigma_krig_test"],
            transform="none", lam=0.0, shift=0.0),
    }
    innovations = {
        "sim_test": compute_standardized_innovations(
            y_test, best["sim_test"], best["sigma_marginal"]),
        "krig_test": compute_standardized_innovations(
            y_test, best["krig_test"], best["sigma_krig_test"]),
    }

    # Plots
    _plot_qq(innovations["sim_test"]["z"],
             out_dir / "qq_sim_test.png",  "Innovation QQ: SWR-GP sim (test)")
    _plot_qq(innovations["krig_test"]["z"],
             out_dir / "qq_krig_test.png", "Innovation QQ: SWR-GP krig (test)")
    _plot_pit(calibration["sim_test"]["pit"],
              out_dir / "pit_sim_test.png",  "SWR-GP, simulation")
    _plot_pit(calibration["krig_test"]["pit"],
              out_dir / "pit_krig_test.png", "SWR-GP, corrected")

    # Save model
    with open(out_dir / "best.pkl", "wb") as f:
        pickle.dump(best["model"], f)

    wall_sec = time.perf_counter() - t_wall_start

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
        "experiment_id": f"{METHOD}__{TRANSFORM}{EXPERIMENT_ID_SUFFIX}",
        "method": METHOD,
        "transform": TRANSFORM,
        "config": {
            "k_values": K_VALUES, "m": M, "max_lag": MAX_LAG, "nu": NU,
            "seed": SEED, "gp_maxiter": GP_MAXITER,
            "kernel_type": KERNEL_TYPE,
            "enforce_max_kernel_overlap_in_estimation":
                ENFORCE_MAX_KERNEL_OVERLAP_IN_ESTIMATION,
            "train_split": "hydr_year < 30",
            "test_split":  "hydr_year >= 30",
        },
        "selection": {
            "criterion": "BIC(GLS)",
            "best_K": int(best["K"]),
            "best_bic": float(best_bic),
        },
        "per_k": _to_plain(per_k),
        "best": {
            "K": int(best["K"]),
            "theta": _to_plain(best["theta"]),
            "metrics": _to_plain(best["metrics"]),
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

    logger.info("Experiment complete | best_K=%d | BIC=%.2f | wall=%.1fs",
                best["K"], best_bic, wall_sec)
    logger.info("Artifacts in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
