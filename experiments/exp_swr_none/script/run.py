#!/usr/bin/env python3
"""SWR (no transform) application study experiment — self-contained.

Fits an OLS kernel-convolution model (SWR) on raw Big Sur streamflow.
K ∈ {1,2,3,4}; model selected by BIC_eff(OLS).
Simulation mode = OLS predictions; kriging mode = Cochrane-Orcutt AR correction.

Outputs (written to output/ sub-dir):
    results.json        per-K metrics + selected model summary
    best.pkl            serialised best SWRKernelOLS model
    pit_sim_test.png    PIT histogram — simulation mode, test period
    pit_krig_test.png   PIT histogram — kriging mode, test period
    qq_sim_test.png     Innovation QQ — simulation mode, test period
    qq_krig_test.png    Innovation QQ — kriging mode, test period

Log written to run.log at experiment root.

Usage (from repository root):
    .venv/bin/python experiments/exp_swr_none/script/run.py
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
from scipy.optimize import minimize, nnls
from scipy.stats import norm

HERE      = Path(__file__).resolve().parent
EXP_DIR   = HERE.parent
REPO_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from swrgp.bigsur_data import load_bigsur_train_test
from swrgp.diagnostics import compute_gaussian_calibration, compute_standardized_innovations
from swrgp.kernels import build_design_matrix
from swrgp.metrics import (
    compute_aic_eff, compute_bic_eff, compute_crps, compute_metrics,
)
from swrgp.nngp import estimate_theta_mom
from swrgp.paths import DATA_DIR

# ── Experiment config ──────────────────────────────────────────────────────
K_VALUES    = [1, 2, 3, 4]
M           = 20
MAX_LAG     = 100
NU          = 1.5
SEED        = 42
KERNEL_TYPE = "gaussian"
METHOD      = "swr"
TRANSFORM   = "none"

# ── SWR kernel-OLS model ──────────────────────────────────────────────────

class SWRKernelOLS:
    """Kernel-convolution OLS: optimise kernel shapes, estimate β by NNLS.

    GP theta (from MOM on residuals) is used for diagnostics only — SWR
    does not use GP for forecasting; the AR correction is applied instead.
    """

    def __init__(self, K: int, kernel_type: str, max_lag: int,
                 m: int, nu: float, seed: int) -> None:
        self.K            = int(K)
        self.kernel_type  = str(kernel_type)
        self.max_lag      = int(max_lag)
        self.m            = int(m)
        self.nu           = float(nu)
        self.seed         = int(seed)
        self.kernel_params_: np.ndarray | None = None
        self.beta_:          np.ndarray | None = None
        self.theta_:         np.ndarray | None = None

    def _bounds(self) -> list[tuple[float, float]]:
        return [(-2.0, 3.0), (-1.0, 4.0)] * self.K

    def fit(self, rain: np.ndarray, y: np.ndarray) -> "SWRKernelOLS":
        rng = np.random.default_rng(self.seed)

        def objective(params: np.ndarray) -> float:
            kp = params.reshape(self.K, 2)
            try:
                X = build_design_matrix(rain, kp, self.kernel_type,
                                        max_length=self.max_lag)
                beta, _ = nnls(X, y)
                return float(np.mean((y - X @ beta) ** 2))
            except Exception:
                return 1e12

        best_val = float("inf")
        best_x: np.ndarray | None = None
        bnds = self._bounds()
        for _ in range(4):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bnds])
            res = minimize(objective, x0=x0, method="L-BFGS-B",
                           bounds=bnds, options={"maxiter": 120})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_x = np.asarray(res.x, dtype=float)

        if best_x is None:
            raise RuntimeError("SWRKernelOLS optimisation failed")

        self.kernel_params_ = best_x.reshape(self.K, 2)
        X = build_design_matrix(rain, self.kernel_params_, self.kernel_type,
                                max_length=self.max_lag)
        self.beta_, _ = nnls(X, y)
        resid = y - X @ self.beta_
        self.theta_ = estimate_theta_mom(resid, self.nu)
        return self

    def predict(self, rain: np.ndarray) -> np.ndarray:
        X = build_design_matrix(rain, self.kernel_params_, self.kernel_type,
                                max_length=self.max_lag)
        return X @ self.beta_


# ── AR-correction helpers (Cochrane-Orcutt progressive AR) ─────────────────

def _durbin_watson(resid: np.ndarray) -> float:
    resid = np.asarray(resid, dtype=float)
    if resid.size < 2:
        return 2.0
    num = np.sum((resid[1:] - resid[:-1]) ** 2)
    den = np.sum(resid ** 2)
    return float(num / den) if den > 1e-12 else 2.0


def _fit_progressive_ar(train_resid: np.ndarray, p_max: int = 8) -> dict:
    train_resid = np.asarray(train_resid, dtype=float)
    best = {"p": 0, "phi": np.array([], dtype=float),
            "dw": _durbin_watson(train_resid)}
    for p in range(1, p_max + 1):
        if train_resid.size <= p + 5:
            break
        y  = train_resid[p:]
        X  = np.column_stack([train_resid[p - j - 1 : -j - 1] for j in range(p)])
        phi, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        dw = _durbin_watson(y - X @ phi)
        if abs(dw - 2.0) < abs(best["dw"] - 2.0) - 0.03:
            best = {"p": p, "phi": np.asarray(phi, dtype=float), "dw": float(dw)}
        else:
            break
    return best


def _apply_ar_correction_from(
    base_pred: np.ndarray,
    y_obs: np.ndarray,
    start_index: int,
    phi: np.ndarray,
) -> np.ndarray:
    out   = np.asarray(base_pred, dtype=float).copy()
    phi   = np.asarray(phi, dtype=float)
    p     = phi.size
    if p == 0:
        return out
    resid = np.asarray(y_obs, dtype=float) - np.asarray(base_pred, dtype=float)
    n     = out.size
    start = max(int(start_index), int(p))
    for t in range(start, n):
        hist  = resid[t - p : t][::-1]
        out[t] += float(np.dot(phi, hist))
    return out


def _swr_ar_innovation_sigma(
    sim_train: np.ndarray,
    y_train: np.ndarray,
    phi: np.ndarray,
) -> float:
    resid = np.asarray(y_train, dtype=float) - np.asarray(sim_train, dtype=float)
    phi   = np.asarray(phi, dtype=float)
    p     = int(phi.size)
    if p == 0 or resid.size <= p:
        return float(max(np.std(resid), 1e-8))
    y_ar  = resid[p:]
    X_ar  = np.column_stack([resid[p - j - 1 : -j - 1] for j in range(p)])
    innov = y_ar - X_ar @ phi
    return float(max(np.std(innov), 1e-8))


# ── OLS log-likelihood ─────────────────────────────────────────────────────

def _ols_loglik(y: np.ndarray, pred: np.ndarray) -> float:
    resid = np.asarray(y) - np.asarray(pred)
    n     = len(resid)
    sigma2 = float(np.mean(resid ** 2))
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return float("-inf")
    return -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma2) - n / 2


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
    logger = logging.getLogger("exp_swr_none")
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
    out_dir = EXP_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(EXP_DIR / "run.log")

    logger.info("Experiment: %s__%s", METHOD, TRANSFORM)
    logger.info("Config: K=%s  m=%d  max_lag=%d  nu=%.1f  seed=%d",
                K_VALUES, M, MAX_LAG, NU, SEED)

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
    best_bic_eff = float("inf")
    per_k: dict[str, Any] = {}
    t_wall_start = time.perf_counter()

    for K in K_VALUES:
        logger.info("Fitting K=%d ...", K)
        t_k = time.perf_counter()

        model = SWRKernelOLS(
            K=K, kernel_type=KERNEL_TYPE, max_lag=MAX_LAG,
            m=M, nu=NU, seed=SEED,
        )
        model.fit(rain_train, y_train)
        # n_params: 2K kernel shapes + K betas + 1 error variance
        n_params_eff = 3 * K + 1

        # Simulation predictions (OLS kernel)
        sim_full  = model.predict(full_rain)
        sim_train = np.maximum(sim_full[:n_train], 0.0)
        sim_test  = np.maximum(sim_full[n_train:], 0.0)
        theta     = np.asarray(model.theta_, dtype=float)

        # Kriging mode = Cochrane-Orcutt AR correction on original scale
        ar_fit = _fit_progressive_ar(y_train - sim_train, p_max=8)
        krig_full_adj = _apply_ar_correction_from(
            np.concatenate([sim_train, sim_test]), full_y,
            start_index=0, phi=ar_fit["phi"])
        krig_train = np.maximum(krig_full_adj[:n_train], 0.0)
        krig_test  = np.maximum(krig_full_adj[n_train:], 0.0)

        sigma_krig = _swr_ar_innovation_sigma(sim_train, y_train, ar_fit["phi"])
        sigma_krig_arr_train = np.full(n_train,      sigma_krig)
        sigma_krig_arr_test  = np.full(len(y_test),  sigma_krig)

        # Marginal sigma from MOM theta for simulation-mode CRPS
        sigma_marginal = float(np.sqrt(max(theta[0] + theta[3], 1e-12)))

        # Point metrics
        metrics: dict[str, Any] = {
            "sim_train":  compute_metrics(y_train, sim_train),
            "sim_test":   compute_metrics(y_test,  sim_test),
            "krig_train": compute_metrics(y_train, krig_train),
            "krig_test":  compute_metrics(y_test,  krig_test),
        }
        # CRPS — analytic Gaussian, simulation with marginal sigma, krig with AR sigma
        metrics["sim_train"]["CRPS"]  = float(compute_crps(y_train, sim_train,  family="gaussian", sigma=sigma_marginal))
        metrics["sim_test"]["CRPS"]   = float(compute_crps(y_test,  sim_test,   family="gaussian", sigma=sigma_marginal))
        metrics["krig_train"]["CRPS"] = float(compute_crps(y_train, krig_train, family="gaussian", sigma=sigma_krig_arr_train))
        metrics["krig_test"]["CRPS"]  = float(compute_crps(y_test,  krig_test,  family="gaussian", sigma=sigma_krig_arr_test))

        # BIC_eff(OLS) for within-model selection
        ll      = _ols_loglik(y_train, sim_train)
        resid   = y_train - sim_train
        bic_eff = compute_bic_eff(ll, n_params_eff, resid)
        aic_eff = compute_aic_eff(ll, n_params_eff, resid)

        fit_sec = time.perf_counter() - t_k
        per_k[f"K{K}"] = {
            "K": K, "n_params_eff": n_params_eff,
            "bic": float(bic_eff), "aic": float(aic_eff), "loglik": float(ll),
            "ic_type": "BIC_eff(OLS)", "fit_time_sec": float(fit_sec),
            "ar_p": int(ar_fit["p"]), "ar_dw": float(ar_fit["dw"]),
            "theta": theta.tolist(),
            "metrics": _to_plain(metrics),
        }
        logger.info(
            "K=%d | BIC_eff(OLS)=%.2f AR(p=%d) | SimTest NSE=%.4f CRPS=%.4f | "
            "KrigTest NSE=%.4f CRPS=%.4f | %.1fs",
            K, bic_eff, ar_fit["p"],
            metrics["sim_test"]["NSE"],  metrics["sim_test"]["CRPS"],
            metrics["krig_test"]["NSE"], metrics["krig_test"]["CRPS"],
            fit_sec,
        )

        if bic_eff < best_bic_eff:
            best_bic_eff = bic_eff
            best = dict(
                K=K, model=model, theta=theta,
                sim_train=sim_train, sim_test=sim_test,
                krig_train=krig_train, krig_test=krig_test,
                sigma_marginal=sigma_marginal,
                sigma_krig_arr_train=sigma_krig_arr_train,
                sigma_krig_arr_test=sigma_krig_arr_test,
                metrics=metrics, ar_fit=ar_fit,
            )

    if best is None:
        logger.error("No valid model fitted.")
        return 1

    # Calibration diagnostics
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
            y_train, best["krig_train"], best["sigma_krig_arr_train"],
            transform="none", lam=0.0, shift=0.0),
        "krig_test": compute_gaussian_calibration(
            y_test, best["krig_test"], best["sigma_krig_arr_test"],
            transform="none", lam=0.0, shift=0.0),
    }
    innovations = {
        "sim_test": compute_standardized_innovations(
            y_test, best["sim_test"], best["sigma_marginal"]),
        "krig_test": compute_standardized_innovations(
            y_test, best["krig_test"], best["sigma_krig_arr_test"]),
    }

    # Plots
    _plot_qq(innovations["sim_test"]["z"],
             out_dir / "qq_sim_test.png",  "Innovation QQ: SWR sim (test)")
    _plot_qq(innovations["krig_test"]["z"],
             out_dir / "qq_krig_test.png", "Innovation QQ: SWR krig (test)")
    _plot_pit(calibration["sim_test"]["pit"],
              out_dir / "pit_sim_test.png",  "SWR, simulation")
    _plot_pit(calibration["krig_test"]["pit"],
              out_dir / "pit_krig_test.png", "SWR, corrected")

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
        "experiment_id": f"{METHOD}__{TRANSFORM}",
        "method": METHOD,
        "transform": TRANSFORM,
        "config": {
            "k_values": K_VALUES, "m": M, "max_lag": MAX_LAG, "nu": NU,
            "seed": SEED, "kernel_type": KERNEL_TYPE,
            "train_split": "hydr_year < 30",
            "test_split":  "hydr_year >= 30",
            "sim_krig_definition": "sim=OLS kernel mean; krig=Cochrane-Orcutt progressive AR",
        },
        "selection": {
            "criterion": "BIC_eff(OLS)",
            "best_K": int(best["K"]),
            "best_bic_eff": float(best_bic_eff),
            "ar_p": int(best["ar_fit"]["p"]),
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

    logger.info("Experiment complete | best_K=%d | BIC_eff=%.2f | wall=%.1fs",
                best["K"], best_bic_eff, wall_sec)
    logger.info("Artifacts in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
