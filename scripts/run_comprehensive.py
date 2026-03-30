"""Comprehensive final experiment runner (6 variants).

Runs 3 model families × 2 response transforms with fixed split/config:
- K in {1,2,3,4}
- m = 20
- split: hydr_year < 30 (train), >= 30 (test)
- predictor/response: rain -> gauge
- kernel type: gaussian (fixed)
- metrics (train/test; simulation/kriging): NSE, KGE, CRPS, RMSE, RelErr
- diagnostics (train/test; simulation/kriging): innovation QQ plot (transformed scale), PIT histogram (original scale)

Outputs under experiments/exp_comprehensive:
- models/: best model per variant
- metrics/: per-variant JSON + combined summary JSON
- logs/: one log per variant + master log
- plots/: diagnostics PNGs
- report.qmd + report.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import boxcox as scipy_boxcox
from scipy.special import inv_boxcox
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swrgp.bigsur_data import load_bigsur_train_test
from swrgp.diagnostics import (
    compute_gaussian_calibration,
    compute_standardized_innovations,
)
from swrgp.kernels import build_design_matrix
from swrgp.metrics import compute_aic, compute_aic_eff, compute_bic, compute_bic_eff, compute_crps, compute_metrics
from swrgp.model import GPSWR
from swrgp.nn_model import GPNN, set_global_seed
from swrgp.nngp import (
    build_nngp_matrices,
    decorrelate,
    decorrelate_matrix,
    estimate_theta_mom,
    gls_log_likelihood,
)
from swrgp.paths import DATA_DIR, ensure_output_dirs_for_root, timestamp


METHODS = ["swr", "swr_gp", "nn_gp"]
KERNEL_TYPE = "gaussian"
TRANSFORMS = ["none", "boxcox"]


@dataclass
class RunConfig:
    k_values: list[int]
    m: int
    max_lag: int
    nu: float
    boxcox_lambda: float
    train_cutoff_year: int
    target_runtime_sec: float
    seed: int
    n_jobs: int
    gp_maxiter: int
    nn_epochs: int
    nn_gls_iterations: int
    nn_hidden_dim: int
    nn_dropout: float
    nn_weight_decay: float
    render_report: bool


@dataclass(frozen=True)
class TimerSnapshot:
    wall: float
    cpu: float


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"exp_comprehensive::{log_file.stem}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def _safe_float(x: Any) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def _timer_now() -> TimerSnapshot:
    # perf_counter is monotonic wall time; process_time tracks active CPU time.
    return TimerSnapshot(wall=time.perf_counter(), cpu=time.process_time())


def _elapsed_seconds(start: TimerSnapshot, end: TimerSnapshot | None = None) -> dict[str, float]:
    stop = end or _timer_now()
    wall = float(max(stop.wall - start.wall, 0.0))
    active = float(max(stop.cpu - start.cpu, 0.0))
    stall = float(max(wall - active, 0.0))
    return {
        "wall_sec": wall,
        "active_sec": active,
        "stall_sec": stall,
        "stall_fraction": float(stall / wall) if wall > 1e-12 else 0.0,
    }


def _runtime_quality_flags(runtime: dict[str, float]) -> list[str]:
    flags: list[str] = []
    wall = float(runtime.get("wall_sec", 0.0))
    active = float(runtime.get("active_sec", 0.0))
    stall = float(runtime.get("stall_sec", 0.0))
    stall_fraction = float(runtime.get("stall_fraction", 0.0))

    # Conservative thresholds to avoid flagging normal I/O overhead.
    if stall >= 60.0 and stall_fraction >= 0.50:
        flags.append("wall_stall_suspected")
    if active > 0.0 and wall / active >= 8.0 and wall >= 120.0:
        flags.append("wall_to_cpu_ratio_extreme")
    return flags


def _detect_per_k_runtime_outliers(per_k: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | bool]]:
    entries: list[tuple[str, float]] = []
    for key, payload in per_k.items():
        runtime = payload.get("fit_time_sec")
        if runtime is None:
            continue
        runtime_f = float(runtime)
        if np.isfinite(runtime_f):
            entries.append((key, runtime_f))

    if len(entries) < 3:
        return {key: {"is_outlier": False, "score": 0.0} for key, _ in entries}

    values = np.asarray([v for _, v in entries], dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))

    out: dict[str, dict[str, float | bool]] = {}
    if mad <= 1e-12:
        for key, runtime in entries:
            # Fallback when dispersion is nearly zero.
            score = float((runtime / med) if med > 1e-12 else 0.0)
            is_outlier = bool(runtime >= med * 5.0 and runtime - med >= 30.0)
            out[key] = {"is_outlier": is_outlier, "score": score}
        return out

    robust_sigma = 1.4826 * mad
    for key, runtime in entries:
        robust_z = float((runtime - med) / robust_sigma)
        is_outlier = bool(robust_z >= 4.5 and runtime - med >= 10.0)
        out[key] = {"is_outlier": is_outlier, "score": robust_z}
    return out


def _format_runtime_flags_for_report(payload: dict[str, Any]) -> str:
    runtime = payload.get("runtime", {})
    flags = list(runtime.get("flags", []))
    per_k_outliers = list(runtime.get("per_k_outliers", []))
    if per_k_outliers:
        flags.append("per_k_outlier:" + ",".join(per_k_outliers))
    return ", ".join(flags) if flags else "none"


def _boxcox_shift(y: np.ndarray) -> float:
    y_min = float(np.nanmin(y))
    return 0.0 if y_min > 0 else abs(y_min) + 1e-3


def _inverse_boxcox_mean_delta(mu_t: np.ndarray, sigma_t: float | np.ndarray, lam: float, shift: float) -> np.ndarray:
    """Approximate E[g^{-1}(Z)] for Z~N(mu_t, sigma_t^2) using delta method.

    sigma_t may be a scalar or per-observation array.
    """
    mu_t = np.asarray(mu_t, dtype=float)
    sigma_t = np.maximum(np.asarray(sigma_t, dtype=float), 1e-8)
    sigma2 = sigma_t ** 2

    base = inv_boxcox(mu_t, lam) - shift
    inner = np.maximum(lam * mu_t + 1.0, 1e-8)
    second = (1.0 / lam - 1.0) * lam * np.power(inner, 1.0 / lam - 2.0)
    out = base + 0.5 * second * sigma2
    return np.maximum(out, 0.0)


def _save_plot_innovation_qq(z: np.ndarray, out_file: Path, title: str) -> None:
    """QQ plot for transformed-scale standardized innovations against N(0,1)."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 10:
        return

    osm = np.sort(norm.ppf((np.arange(1, z.size + 1) - 0.5) / z.size))
    osr = np.sort(z)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    ax.scatter(osm, osr, s=8, alpha=0.6)
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, line_x, "r--", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("Theoretical Quantiles N(0,1)")
    ax.set_ylabel("Standardized Innovation Quantiles")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def _save_plot_pit_hist(pit: np.ndarray, out_file: Path, title: str, bins: int = 20) -> None:
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


class SWRKernelOLS:
    """Simple SWR-like model: kernel+beta by OLS, then GP theta from residuals."""

    def __init__(self, K: int, kernel_type: str, max_lag: int, m: int, nu: float, seed: int):
        self.K = int(K)
        self.kernel_type = str(kernel_type)
        self.max_lag = int(max_lag)
        self.m = int(m)
        self.nu = float(nu)
        self.seed = int(seed)

        self.kernel_params_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None

    def _bounds(self) -> list[tuple[float, float]]:
        bnds: list[tuple[float, float]] = []
        for _ in range(self.K):
            bnds.append((-2.0, 3.0))
            bnds.append((-1.0, 4.0))
        return bnds

    def fit(self, rain: np.ndarray, y: np.ndarray) -> "SWRKernelOLS":
        rng = np.random.default_rng(self.seed)

        def objective(params: np.ndarray) -> float:
            kp = params.reshape(self.K, 2)
            try:
                X = build_design_matrix(rain, kp, self.kernel_type, max_length=self.max_lag)
                beta, _ = nnls(X, y)
                resid = y - X @ beta
                return float(np.mean(resid**2))
            except Exception:
                return 1e12

        best = None
        best_val = float("inf")
        for _ in range(4):
            x0 = []
            for low, high in self._bounds():
                x0.append(rng.uniform(low, high))
            x0 = np.asarray(x0, dtype=float)
            res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=self._bounds(), options={"maxiter": 120})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best = np.asarray(res.x, dtype=float)

        if best is None:
            raise RuntimeError("SWRKernelOLS optimization failed")

        self.kernel_params_ = best.reshape(self.K, 2)
        X = build_design_matrix(rain, self.kernel_params_, self.kernel_type, max_length=self.max_lag)
        self.beta_, _ = nnls(X, y)
        resid = y - X @ self.beta_

        # For SWR we do not use GP forecasting; theta is retained only for scale/diagnostics.
        self.theta_ = estimate_theta_mom(resid, self.nu)
        return self

    def predict(self, rain: np.ndarray) -> np.ndarray:
        X = build_design_matrix(rain, self.kernel_params_, self.kernel_type, max_length=self.max_lag)
        return X @ self.beta_

def _durbin_watson(resid: np.ndarray) -> float:
    resid = np.asarray(resid, dtype=float)
    if resid.size < 2:
        return 2.0
    num = np.sum((resid[1:] - resid[:-1]) ** 2)
    den = np.sum(resid**2)
    if den <= 1e-12:
        return 2.0
    return float(num / den)


def _fit_progressive_ar(train_resid: np.ndarray, p_max: int = 8) -> dict:
    train_resid = np.asarray(train_resid, dtype=float)
    best = {"p": 0, "phi": np.array([], dtype=float), "dw": _durbin_watson(train_resid)}

    for p in range(1, p_max + 1):
        if train_resid.size <= p + 5:
            break
        y = train_resid[p:]
        X = np.column_stack([train_resid[p - j - 1 : -j - 1] for j in range(p)])
        phi, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        fit = X @ phi
        e = y - fit
        dw = _durbin_watson(e)
        # Keep model if DW is closer to 2
        if abs(dw - 2.0) < abs(best["dw"] - 2.0) - 0.03:
            best = {"p": p, "phi": np.asarray(phi, dtype=float), "dw": float(dw)}
        else:
            break

    return best


def _apply_ar_correction(base_pred: np.ndarray, y_obs: np.ndarray, train_n: int, phi: np.ndarray) -> np.ndarray:
    out = np.asarray(base_pred, dtype=float).copy()
    phi = np.asarray(phi, dtype=float)
    p = phi.size
    if p == 0:
        return out

    resid = np.asarray(y_obs, dtype=float) - np.asarray(base_pred, dtype=float)
    n = out.size

    # One-step style correction: use available observed history
    for t in range(train_n, n):
        if t < p:
            continue
        hist = resid[t - p : t][::-1]
        out[t] += float(np.dot(phi, hist))
    return out


def _apply_ar_correction_from(base_pred: np.ndarray, y_obs: np.ndarray, start_index: int, phi: np.ndarray) -> np.ndarray:
    out = np.asarray(base_pred, dtype=float).copy()
    phi = np.asarray(phi, dtype=float)
    p = phi.size
    if p == 0:
        return out

    resid = np.asarray(y_obs, dtype=float) - np.asarray(base_pred, dtype=float)
    n = out.size
    start = max(int(start_index), int(p))

    for t in range(start, n):
        hist = resid[t - p : t][::-1]
        out[t] += float(np.dot(phi, hist))
    return out


def _ols_loglik(y: np.ndarray, pred: np.ndarray) -> float:
    """Gaussian OLS log-likelihood (independence-assuming)."""
    resid = y - pred
    n = len(resid)
    sigma2 = np.mean(resid ** 2)
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return float("-inf")
    return -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma2) - n / 2


def _estimate_ic_train(
    y_train_t: np.ndarray,
    pred_train_t: np.ndarray,
    m: int,
    theta: np.ndarray,
    n_params: int,
    method: str,
) -> tuple[float, float, float]:
    """Compute (aic, bic, log-likelihood) appropriate for the method.

    SWR (OLS model): OLS log-likelihood + BIC_eff (autocorrelation-corrected).
    SWR-GP / NN-GP (GLS models): GLS log-likelihood + standard BIC.
    """
    n = len(y_train_t)
    resid = y_train_t - pred_train_t

    if method == "swr":
        # SWR is an OLS model — residual autocorrelation is unmodeled,
        # so BIC_eff corrects the penalty via n_eff.
        ll = _ols_loglik(y_train_t, pred_train_t)
        aic = compute_aic_eff(ll, n_params, resid)
        bic = compute_bic_eff(ll, n_params, resid)
    else:
        # SWR-GP and NN-GP use GLS likelihood that already accounts for
        # temporal correlation → standard BIC with actual n.
        B, F = build_nngp_matrices(n, m, theta)
        ll = gls_log_likelihood(y_train_t, pred_train_t, B, F, m)
        aic = compute_aic(ll, n_params)
        bic = compute_bic(ll, n_params, n)
    return float(aic), float(bic), float(ll)


def _compute_all_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    sim_train: np.ndarray,
    sim_test: np.ndarray,
    krig_train: np.ndarray,
    krig_test: np.ndarray,
    theta: np.ndarray,
    transform: str,
    shift: float,
    lam: float,
    y_train_t: np.ndarray | None,
    y_test_t: np.ndarray | None,
    sim_train_t: np.ndarray | None,
    sim_test_t: np.ndarray | None,
    krig_train_t: np.ndarray | None,
    krig_test_t: np.ndarray | None,
    sigma_krig_train: np.ndarray | None = None,
    sigma_krig_test: np.ndarray | None = None,
) -> dict:
    sigma_marginal = float(np.sqrt(max(theta[0] + theta[3], 1e-12)))
    # Kriging CRPS uses conditional sigma (per-observation) when available,
    # falling back to marginal sigma for simulation mode.
    sigma_krig_train_used = sigma_krig_train if sigma_krig_train is not None else sigma_marginal
    sigma_krig_test_used = sigma_krig_test if sigma_krig_test is not None else sigma_marginal

    out = {
        "sim_train": compute_metrics(y_train, sim_train),
        "sim_test": compute_metrics(y_test, sim_test),
        "krig_train": compute_metrics(y_train, krig_train),
        "krig_test": compute_metrics(y_test, krig_test),
    }

    if transform == "boxcox":
        sim_crps_kwargs = {
            "family": "gaussian",
            "sigma": sigma_marginal,
            "response_transform": "boxcox",
            "response_shift": shift,
            "response_lambda": lam,
        }
        krig_train_crps_kwargs = {
            "family": "gaussian",
            "sigma": sigma_krig_train_used,
            "response_transform": "boxcox",
            "response_shift": shift,
            "response_lambda": lam,
        }
        krig_test_crps_kwargs = {
            "family": "gaussian",
            "sigma": sigma_krig_test_used,
            "response_transform": "boxcox",
            "response_shift": shift,
            "response_lambda": lam,
        }
        out["sim_train"]["CRPS"] = float(compute_crps(y_train, sim_train, y_transformed=y_train_t, mu_transformed=sim_train_t, **sim_crps_kwargs))
        out["sim_test"]["CRPS"] = float(compute_crps(y_test, sim_test, y_transformed=y_test_t, mu_transformed=sim_test_t, **sim_crps_kwargs))
        out["krig_train"]["CRPS"] = float(compute_crps(y_train, krig_train, y_transformed=y_train_t, mu_transformed=krig_train_t, **krig_train_crps_kwargs))
        out["krig_test"]["CRPS"] = float(compute_crps(y_test, krig_test, y_transformed=y_test_t, mu_transformed=krig_test_t, **krig_test_crps_kwargs))
    else:
        out["sim_train"]["CRPS"] = float(compute_crps(y_train, sim_train, family="gaussian", sigma=sigma_marginal))
        out["sim_test"]["CRPS"] = float(compute_crps(y_test, sim_test, family="gaussian", sigma=sigma_marginal))
        out["krig_train"]["CRPS"] = float(compute_crps(y_train, krig_train, family="gaussian", sigma=sigma_krig_train_used))
        out["krig_test"]["CRPS"] = float(compute_crps(y_test, krig_test, family="gaussian", sigma=sigma_krig_test_used))

    return out


def _calibration_for_mode(
    y: np.ndarray,
    pred: np.ndarray,
    sigma: float | np.ndarray | None,
    transform: str,
    lam: float,
    shift: float,
    pred_transformed: np.ndarray | None = None,
) -> dict:
    sigma_used = sigma if sigma is not None else float(np.std(y - pred))
    return compute_gaussian_calibration(
        y=y,
        mu=pred,
        sigma=sigma_used,
        transform=transform,
        lam=lam,
        shift=shift,
        mu_transformed=pred_transformed,
    )


def _innovation_for_mode(
    y_transformed: np.ndarray,
    pred_transformed: np.ndarray,
    sigma_transformed: float | np.ndarray,
) -> dict:
    return compute_standardized_innovations(
        y_transformed=y_transformed,
        mu_transformed=pred_transformed,
        sigma_transformed=sigma_transformed,
    )


def _compute_swr_ar_innovation_sigma(sim_train: np.ndarray, y_train: np.ndarray, phi: np.ndarray) -> float:
    resid = np.asarray(y_train, dtype=float) - np.asarray(sim_train, dtype=float)
    phi = np.asarray(phi, dtype=float)
    p = int(phi.size)

    if p == 0 or resid.size <= p:
        return float(max(np.std(resid), 1e-8))

    y_ar = resid[p:]
    X_ar = np.column_stack([resid[p - j - 1 : -j - 1] for j in range(p)])
    innov = y_ar - X_ar @ phi
    return float(max(np.std(innov), 1e-8))


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _run_variant(
    method: str,
    transform: str,
    cfg_dict: dict,
    results_root: str,
) -> dict:
    cfg = RunConfig(**cfg_dict)
    _, logs_dir, metrics_dir, models_dir, plots_dir = ensure_output_dirs_for_root(
        results_root,
        experiment_key="comprehensive",
        script_path=__file__,
    )

    exp_id = f"{method}__{transform}"
    logger = _setup_logger(logs_dir / f"{exp_id}.log")
    logger.info("Starting %s", exp_id)
    set_global_seed(cfg.seed)

    train_df, test_df = load_bigsur_train_test(DATA_DIR)
    rain_train = train_df["rain"].to_numpy(dtype=float)
    y_train = train_df["gauge"].to_numpy(dtype=float)
    rain_test = test_df["rain"].to_numpy(dtype=float)
    y_test = test_df["gauge"].to_numpy(dtype=float)

    full_rain = np.concatenate([rain_train, rain_test])
    full_y = np.concatenate([y_train, y_test])

    shift = _boxcox_shift(y_train) if transform == "boxcox" else 0.0
    lam = cfg.boxcox_lambda

    if transform == "boxcox":
        y_train_t = scipy_boxcox(y_train + shift, lam)
        y_test_t = scipy_boxcox(y_test + shift, lam)
        full_y_t = scipy_boxcox(full_y + shift, lam)
    else:
        y_train_t = y_train.copy()
        y_test_t = y_test.copy()
        full_y_t = full_y.copy()

    best: dict[str, Any] | None = None
    best_bic_val = float("inf")
    best_selection_score = float("inf")
    if method == "nn_gp":
        criterion_within = "val_CRPS_krig(train_heldout)"
    elif method == "swr":
        criterion_within = "BIC_eff(OLS)"
    else:
        criterion_within = "BIC(GLS)"

    per_k: dict[str, Any] = {}
    t_total_start = _timer_now()

    # Use a trailing, time-ordered holdout from training only for NN capacity selection
    # to avoid test leakage while preserving causal ordering.
    if method == "nn_gp":
        n_train = len(y_train)
        min_val = max(int(cfg.max_lag), 60)
        target_val = max(min_val, int(round(0.2 * n_train)))
        target_val = min(target_val, max(10, n_train // 3))
        sub_end = n_train - target_val
        min_sub = max(cfg.max_lag + 5, cfg.m + 5, 30)
        if sub_end < min_sub:
            sub_end = min_sub
        if sub_end >= n_train - 5:
            raise RuntimeError("Not enough training samples for NN held-out selection split")

        rain_sub = rain_train[:sub_end]
        rain_val = rain_train[sub_end:]
        rain_subval = rain_train.copy()

        y_sub = y_train[:sub_end]
        y_val = y_train[sub_end:]
        y_sub_t = y_train_t[:sub_end]
        y_val_t = y_train_t[sub_end:]
        y_subval_t = y_train_t.copy()
    else:
        sub_end = None

    for K in cfg.k_values:
        k_start = _timer_now()
        logger.info("%s | K=%d start", exp_id, K)
        complexity_axis_value: int = int(K)
        complexity_axis_label = "K"
        n_params_eff: int
        sigma_krig_train_t: np.ndarray | None = None
        sigma_krig_test_t: np.ndarray | None = None

        if method == "nn_gp":
            selector_model = GPNN(
                lag_window=cfg.max_lag,
                m=cfg.m,
                nu=cfg.nu,
                hidden_dim=cfg.nn_hidden_dim,
                dropout=cfg.nn_dropout,
                epochs=cfg.nn_epochs,
                lr=1e-3,
                n_gls_iterations=cfg.nn_gls_iterations,
                log_transform=False,
                verbose=False,
                patience=20,
                min_delta=1e-5,
                min_epochs=20,
                max_grad_norm=5.0,
                weight_decay=cfg.nn_weight_decay,
                seed=cfg.seed,
            )
            selector_model.fit(rain_sub, y_sub_t)

            sel_theta = np.asarray(selector_model.theta_, dtype=float)
            sigma_sel_t = float(np.sqrt(max(sel_theta[0] + sel_theta[3], 1e-12)))
            krig_subval_t = selector_model.forecast(rain_subval, y_subval_t)
            krig_val_t = krig_subval_t[sub_end:]

            if transform == "boxcox":
                krig_val = _inverse_boxcox_mean_delta(krig_val_t, sigma_sel_t, lam, shift)
                selection_score = float(
                    compute_crps(
                        y_val,
                        krig_val,
                        y_transformed=y_val_t,
                        mu_transformed=krig_val_t,
                        family="gaussian",
                        sigma=sigma_sel_t,
                        response_transform="boxcox",
                        response_lambda=lam,
                        seed=cfg.seed,
                    )
                )
            else:
                krig_val = np.maximum(krig_val_t, 0.0)
                selection_score = float(compute_crps(y_val, krig_val, family="gaussian", sigma=sigma_sel_t))

            # Refit on full training data for final variant metrics/artifacts.
            model = GPNN(
                lag_window=cfg.max_lag,
                m=cfg.m,
                nu=cfg.nu,
                hidden_dim=cfg.nn_hidden_dim,
                dropout=cfg.nn_dropout,
                epochs=cfg.nn_epochs,
                lr=1e-3,
                n_gls_iterations=cfg.nn_gls_iterations,
                log_transform=False,
                verbose=False,
                patience=20,
                min_delta=1e-5,
                min_epochs=20,
                max_grad_norm=5.0,
                weight_decay=cfg.nn_weight_decay,
                seed=cfg.seed,
            )
            model.fit(rain_train, y_train_t)
            n_params_eff = int(sum(p.numel() for p in model.net.parameters()) + 3)
            complexity_axis_label = "hidden_dim"

            sim_full_t = model.predict(full_rain)
            krig_full_t = model.forecast(full_rain, full_y_t)
            sim_train_t = sim_full_t[: len(y_train)]
            sim_test_t = sim_full_t[len(y_train) :]
            krig_train_t = krig_full_t[: len(y_train)]
            krig_test_t = krig_full_t[len(y_train) :]
            theta = np.asarray(model.theta_, dtype=float)

            _, F_full = build_nngp_matrices(len(full_y_t), cfg.m, theta)
            sigma_krig_train_t = np.sqrt(np.maximum(F_full[: len(y_train)], 1e-12))
            sigma_krig_test_t = np.sqrt(np.maximum(F_full[len(y_train) :], 1e-12))

        elif method == "swr_gp":
            model = GPSWR(
                K=K,
                m=cfg.m,
                nu=cfg.nu,
                kernel_type=KERNEL_TYPE,
                max_lag=cfg.max_lag,
                maxiter=cfg.gp_maxiter,
                log_transform=False,
                seed=cfg.seed,
                verbose=False,
            )
            model.fit(rain_train, y_train_t)
            n_params_eff = int(3 * K + 3)

            sim_train_t = model.predict(rain_train)
            sim_test_t = model.predict_with_history(rain_train, rain_test)
            krig_full_t = model.forecast(full_rain, full_y_t)
            krig_train_t = krig_full_t[: len(y_train)]
            krig_test_t = krig_full_t[len(y_train) :]
            theta = np.asarray(model.theta_, dtype=float)

            _, F_full = build_nngp_matrices(len(full_y_t), cfg.m, theta)
            sigma_krig_train_t = np.sqrt(np.maximum(F_full[: len(y_train)], 1e-12))
            sigma_krig_test_t = np.sqrt(np.maximum(F_full[len(y_train) :], 1e-12))

        else:
            model = SWRKernelOLS(
                K=K,
                kernel_type=KERNEL_TYPE,
                max_lag=cfg.max_lag,
                m=cfg.m,
                nu=cfg.nu,
                seed=cfg.seed,
            ).fit(rain_train, y_train_t)
            # SWR OLS params: 2K kernel shapes + K betas + 1 error variance
            n_params_eff = int(3 * K + 1)

            sim_full_t = model.predict(full_rain)
            sim_train_t = sim_full_t[: len(y_train)]
            sim_test_t = sim_full_t[len(y_train) :]
            krig_train_t = sim_train_t.copy()
            krig_test_t = sim_test_t.copy()
            theta = np.asarray(model.theta_, dtype=float)

        sigma_t = float(np.sqrt(max(theta[0] + theta[3], 1e-12)))
        # For kriging back-transform, use conditional sigma when available.
        sigma_krig_train_bt = sigma_krig_train_t if sigma_krig_train_t is not None else sigma_t
        sigma_krig_test_bt = sigma_krig_test_t if sigma_krig_test_t is not None else sigma_t

        if transform == "boxcox":
            sim_train = _inverse_boxcox_mean_delta(sim_train_t, sigma_t, lam, shift)
            sim_test = _inverse_boxcox_mean_delta(sim_test_t, sigma_t, lam, shift)
            krig_train = _inverse_boxcox_mean_delta(krig_train_t, sigma_krig_train_bt, lam, shift)
            krig_test = _inverse_boxcox_mean_delta(krig_test_t, sigma_krig_test_bt, lam, shift)
        else:
            sim_train = np.maximum(sim_train_t, 0.0)
            sim_test = np.maximum(sim_test_t, 0.0)
            krig_train = np.maximum(krig_train_t, 0.0)
            krig_test = np.maximum(krig_test_t, 0.0)

        # For SWR, interpret sim/krig as no-AR vs AR-corrected (Cochrane-Orcutt style)
        if method == "swr":
            if transform == "boxcox":
                # Keep AR correction in transformed space for Box-Cox coherence.
                sim_full_base_t = np.concatenate([sim_train_t, sim_test_t])
                ar_fit_t = _fit_progressive_ar(y_train_t - sim_train_t, p_max=8)
                krig_full_adj_t = _apply_ar_correction_from(sim_full_base_t, full_y_t, start_index=0, phi=ar_fit_t["phi"])

                krig_train_t = krig_full_adj_t[: len(y_train)]
                krig_test_t = krig_full_adj_t[len(y_train) :]
                swr_sigma_krig = _compute_swr_ar_innovation_sigma(sim_train_t, y_train_t, ar_fit_t["phi"])
                krig_train = _inverse_boxcox_mean_delta(krig_train_t, swr_sigma_krig, lam, shift)
                krig_test = _inverse_boxcox_mean_delta(krig_test_t, swr_sigma_krig, lam, shift)
            else:
                sim_full_base = np.concatenate([sim_train, sim_test])
                ar_fit = _fit_progressive_ar(y_train - sim_train, p_max=8)
                krig_full_adj = _apply_ar_correction_from(sim_full_base, full_y, start_index=0, phi=ar_fit["phi"])

                krig_train = krig_full_adj[: len(y_train)]
                krig_test = krig_full_adj[len(y_train) :]
                krig_train_t = krig_train.copy()
                krig_test_t = krig_test.copy()
                swr_sigma_krig = _compute_swr_ar_innovation_sigma(sim_train, y_train, ar_fit["phi"])

            sigma_krig_train_t = np.full_like(y_train_t, swr_sigma_krig, dtype=float)
            sigma_krig_test_t = np.full_like(y_test_t, swr_sigma_krig, dtype=float)

        metrics = _compute_all_metrics(
            y_train=y_train,
            y_test=y_test,
            sim_train=sim_train,
            sim_test=sim_test,
            krig_train=krig_train,
            krig_test=krig_test,
            theta=theta,
            transform=transform,
            shift=shift,
            lam=lam,
            y_train_t=y_train_t,
            y_test_t=y_test_t,
            sim_train_t=sim_train_t,
            sim_test_t=sim_test_t,
            krig_train_t=krig_train_t,
            krig_test_t=krig_test_t,
            sigma_krig_train=sigma_krig_train_t,
            sigma_krig_test=sigma_krig_test_t,
        )

        aic_val, bic_val, ll_val = _estimate_ic_train(
            y_train_t, sim_train_t, cfg.m, theta, n_params=n_params_eff, method=method,
        )

        k_elapsed = _elapsed_seconds(k_start)
        k_payload = {
            "K": K,
            "complexity_axis_label": complexity_axis_label,
            "complexity_axis_value": complexity_axis_value,
            "n_params_eff": int(n_params_eff),
            "bic": bic_val,
            "aic": aic_val,
            "loglik": ll_val,
            "ic_type": "BIC_eff(OLS)" if method == "swr" else "BIC(GLS)",
            "within_selection_criterion": criterion_within,
            "fit_time_sec": float(k_elapsed["active_sec"]),
            "fit_time_wall_sec": float(k_elapsed["wall_sec"]),
            "fit_time_stall_sec": float(k_elapsed["stall_sec"]),
            "fit_time_stall_fraction": float(k_elapsed["stall_fraction"]),
            "theta": theta.tolist(),
            "metrics": _to_plain(metrics),
        }
        if method != "nn_gp":
            selection_score = float(bic_val)

        k_payload["selection_score"] = float(selection_score)
        if method == "nn_gp":
            k_payload["selection_split"] = {
                "subtrain_n": int(sub_end),
                "holdout_n": int(len(y_train) - int(sub_end)),
                "holdout_location": "tail_of_training_period",
            }
        per_k[f"K{K}"] = k_payload

        logger.info(
            "%s | K=%d done | BIC=%.2f (%s) | selector(%s)=%.4f | SimTest NSE=%.4f CRPS=%.4f | KrigTest NSE=%.4f CRPS=%.4f | active=%.1fs wall=%.1fs",
            exp_id,
            K,
            bic_val,
            "BIC_eff" if method == "swr" else "BIC",
            criterion_within,
            selection_score,
            metrics["sim_test"]["NSE"],
            metrics["sim_test"]["CRPS"],
            metrics["krig_test"]["NSE"],
            metrics["krig_test"]["CRPS"],
            k_elapsed["active_sec"],
            k_elapsed["wall_sec"],
        )

        if bic_val < best_bic_val:
            best_bic_val = bic_val

        if selection_score < best_selection_score:
            best_selection_score = selection_score
            best = {
                "K": K,
                "model": model,
                "theta": theta,
                "transform": transform,
                "shift": shift,
                "lambda": lam,
                "sim_train": sim_train,
                "sim_test": sim_test,
                "krig_train": krig_train,
                "krig_test": krig_test,
                "sim_train_t": sim_train_t,
                "sim_test_t": sim_test_t,
                "krig_train_t": krig_train_t,
                "krig_test_t": krig_test_t,
                "sigma_krig_train_t": sigma_krig_train_t,
                "sigma_krig_test_t": sigma_krig_test_t,
                "metrics": metrics,
            }

    if best is None:
        raise RuntimeError(f"No valid model selected for {exp_id}")

    outlier_stats = _detect_per_k_runtime_outliers(per_k)
    per_k_outliers: list[str] = []
    for key, stat in outlier_stats.items():
        is_outlier = bool(stat["is_outlier"])
        per_k[key]["runtime_outlier"] = is_outlier
        per_k[key]["runtime_outlier_score"] = float(stat["score"])
        if is_outlier:
            per_k_outliers.append(key)

    # Diagnostics should reflect predictive distributional assumption.
    # In this experiment family, predictive uncertainty is Gaussian for all variants.
    sigma_best = float(np.sqrt(max(best["theta"][0] + best["theta"][3], 1e-12)))
    sigma_sim_train_best = np.full_like(y_train, sigma_best, dtype=float)
    sigma_sim_test_best = np.full_like(y_test, sigma_best, dtype=float)
    sigma_krig_train_best = np.asarray(best.get("sigma_krig_train_t", sigma_sim_train_best), dtype=float)
    sigma_krig_test_best = np.asarray(best.get("sigma_krig_test_t", sigma_sim_test_best), dtype=float)

    calibration = {
        "sim_train": _calibration_for_mode(
            y_train,
            best["sim_train"],
            sigma_sim_train_best,
            transform=transform,
            lam=lam,
            shift=shift,
            pred_transformed=best["sim_train_t"] if transform == "boxcox" else None,
        ),
        "sim_test": _calibration_for_mode(
            y_test,
            best["sim_test"],
            sigma_sim_test_best,
            transform=transform,
            lam=lam,
            shift=shift,
            pred_transformed=best["sim_test_t"] if transform == "boxcox" else None,
        ),
        "krig_train": _calibration_for_mode(
            y_train,
            best["krig_train"],
            sigma_krig_train_best,
            transform=transform,
            lam=lam,
            shift=shift,
            pred_transformed=best["krig_train_t"] if transform == "boxcox" else None,
        ),
        "krig_test": _calibration_for_mode(
            y_test,
            best["krig_test"],
            sigma_krig_test_best,
            transform=transform,
            lam=lam,
            shift=shift,
            pred_transformed=best["krig_test_t"] if transform == "boxcox" else None,
        ),
    }

    innovations = {
        "sim_train": _innovation_for_mode(y_train_t, best["sim_train_t"], sigma_sim_train_best),
        "sim_test": _innovation_for_mode(y_test_t, best["sim_test_t"], sigma_sim_test_best),
        "krig_train": _innovation_for_mode(y_train_t, best["krig_train_t"], sigma_krig_train_best),
        "krig_test": _innovation_for_mode(y_test_t, best["krig_test_t"], sigma_krig_test_best),
    }

    # Save diagnostics plots for test modes
    qq_sim = plots_dir / f"qq_rqr__{exp_id}__sim_test.png"
    qq_krig = plots_dir / f"qq_rqr__{exp_id}__krig_test.png"
    pit_sim = plots_dir / f"pit_hist__{exp_id}__sim_test.png"
    pit_krig = plots_dir / f"pit_hist__{exp_id}__krig_test.png"

    _save_plot_innovation_qq(np.asarray(innovations["sim_test"]["z"]), qq_sim, f"Innovation QQ: {exp_id} (sim test, transformed scale)")
    _save_plot_innovation_qq(np.asarray(innovations["krig_test"]["z"]), qq_krig, f"Innovation QQ: {exp_id} (krig test, transformed scale)")
    _save_plot_pit_hist(np.asarray(calibration["sim_test"]["pit"]), pit_sim, f"PIT: {exp_id} (sim test, original scale)")
    _save_plot_pit_hist(np.asarray(calibration["krig_test"]["pit"]), pit_krig, f"PIT: {exp_id} (krig test, original scale)")

    model_path = models_dir / f"{exp_id}__best.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best["model"], f)

    elapsed_total = _elapsed_seconds(t_total_start)
    runtime_flags = _runtime_quality_flags(elapsed_total)
    if per_k_outliers:
        runtime_flags.append("per_k_runtime_outlier")

    payload = {
        "experiment_id": exp_id,
        "method": method,
        "kernel_type": KERNEL_TYPE,
        "transform": transform,
        "config": {
            "k_values": cfg.k_values,
            "m": cfg.m,
            "max_lag": cfg.max_lag,
            "nu": cfg.nu,
            "boxcox_lambda": cfg.boxcox_lambda,
            "train_split": "hydr_year < 30",
            "test_split": "hydr_year >= 30",
            "predictor": "rain",
            "response": "gauge",
            "target_runtime_sec": cfg.target_runtime_sec,
            "gp_maxiter": cfg.gp_maxiter,
            "nn_epochs": cfg.nn_epochs,
            "nn_gls_iterations": cfg.nn_gls_iterations,
            "nn_hidden_dim": cfg.nn_hidden_dim,
            "nn_dropout": cfg.nn_dropout,
            "nn_weight_decay": cfg.nn_weight_decay,
            "nn_complexity_interpretation": "For nn_gp, architecture is fixed (hidden_dim, dropout, weight_decay); K sweep axis is not used.",
            "diagnostics_family": "gaussian calibration on original scale (PIT/RQR) + transformed-scale innovation QQ",
            "sim_krig_definition": "SWR: sim=no-AR, krig=AR-corrected (Cochrane-Orcutt auto-p by Durbin-Watson); others: sim=mean, krig=residual-corrected",
        },
        "selection": {
            "criterion_within": criterion_within,
            "criterion_across": "test_CRPS",
            "best_K": int(best["K"]),
            "selector_axis": "hidden_dim" if method == "nn_gp" else "K",
            "best_selection_score": float(best_selection_score),
            "best_bic": float(best_bic_val),
        },
        "per_k": _to_plain(per_k),
        "best": {
            "K": int(best["K"]),
            "theta": _to_plain(best["theta"]),
            "metrics": _to_plain(best["metrics"]),
            "diagnostics": {
                "calibration": {
                    mode: {
                        k: _to_plain(v)
                        for k, v in calibration[mode].items()
                        if k in {"pit_mean", "pit_var", "rqr_mean", "rqr_std", "n_valid"}
                    }
                    for mode in calibration
                },
                "innovation": {
                    mode: {
                        k: _to_plain(v)
                        for k, v in innovations[mode].items()
                        if k in {"n_valid", "z_mean", "z_std", "z_skew", "z_excess_kurtosis"}
                    }
                    for mode in innovations
                },
            },
            "diagnostic_plots": {
                "qq_rqr_sim_test": str(qq_sim),
                "qq_rqr_krig_test": str(qq_krig),
                "pit_sim_test": str(pit_sim),
                "pit_krig_test": str(pit_krig),
            },
        },
        "runtime": {
            "method": "active_cpu_sec_primary",
            "active_sec": float(elapsed_total["active_sec"]),
            "wall_sec": float(elapsed_total["wall_sec"]),
            "stall_sec": float(elapsed_total["stall_sec"]),
            "stall_fraction": float(elapsed_total["stall_fraction"]),
            "flags": runtime_flags,
            "per_k_outliers": sorted(per_k_outliers),
        },
        "runtime_sec": float(elapsed_total["active_sec"]),
        "runtime_wall_sec": float(elapsed_total["wall_sec"]),
    }

    metrics_path = metrics_dir / f"{exp_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        "%s complete | active=%.1fs wall=%.1fs flags=%s",
        exp_id,
        payload["runtime_sec"],
        payload["runtime_wall_sec"],
        ",".join(runtime_flags) if runtime_flags else "none",
    )
    return {
        "exp_id": exp_id,
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "best_K": int(best["K"]),
        "best_test_crps_sim": float(best["metrics"]["sim_test"]["CRPS"]),
        "best_test_crps_krig": float(best["metrics"]["krig_test"]["CRPS"]),
        "runtime_sec": float(payload["runtime_sec"]),
        "runtime_wall_sec": float(payload["runtime_wall_sec"]),
        "runtime_flags": runtime_flags,
    }


def _estimate_budgets(cfg: RunConfig) -> tuple[int, int]:
    set_global_seed(cfg.seed)

    train_df, _ = load_bigsur_train_test(DATA_DIR)
    train_df = train_df.iloc[:1200].copy()
    rain = train_df["rain"].to_numpy(dtype=float)
    y = train_df["gauge"].to_numpy(dtype=float)

    gp_iter_sec = 0.8
    nn_epoch_sec = 0.04

    try:
        t0 = _timer_now()
        m = GPSWR(K=1, m=cfg.m, nu=cfg.nu, kernel_type=KERNEL_TYPE, max_lag=cfg.max_lag, maxiter=10, log_transform=False, seed=cfg.seed, verbose=False)
        m.fit(rain, y)
        gp_iter_sec = max(_elapsed_seconds(t0)["active_sec"] / 10.0, 0.05)
    except Exception:
        pass

    try:
        t0 = _timer_now()
        nn = GPNN(
            lag_window=cfg.max_lag,
            m=cfg.m,
            nu=cfg.nu,
            hidden_dim=cfg.nn_hidden_dim,
            epochs=40,
            n_gls_iterations=2,
            lr=1e-3,
            log_transform=False,
            verbose=False,
            seed=cfg.seed,
        )
        nn.fit(rain, y)
        nn_epoch_sec = max(_elapsed_seconds(t0)["active_sec"] / 40.0, 0.005)
    except Exception:
        pass

    gp_maxiter = int(np.clip(round(cfg.target_runtime_sec / gp_iter_sec), 40, 260))
    nn_epochs = int(np.clip(round(cfg.target_runtime_sec / nn_epoch_sec), 200, 2600))
    return gp_maxiter, nn_epochs


def _write_report_qmd(results: list[dict], exp_dir: Path, cfg: RunConfig) -> Path:
    metrics_dir = exp_dir / "metrics"
    plots_dir = exp_dir / "plots"

    # Load all variant payloads
    payloads = []
    for r in sorted(results, key=lambda x: x["exp_id"]):
        with open(r["metrics_path"], "r") as f:
            payloads.append(json.load(f))

    def _short_name(exp_id: str) -> str:
        mapping = {
            "swr__none": "SWR-N",
            "swr__boxcox": "SWR-BC",
            "swr_gp__none": "SWR-GP-N",
            "swr_gp__boxcox": "SWR-GP-BC",
            "nn_gp__none": "NNGP-N",
            "nn_gp__boxcox": "NNGP-BC",
        }
        return mapping.get(exp_id, exp_id.replace("__", "-"))

    def _fmt(x: Any, d: int = 4) -> str:
        if x is None:
            return "na"
        try:
            xf = float(x)
            if np.isfinite(xf):
                return f"{xf:.{d}f}"
        except Exception:
            return "na"
        return "na"

    def _diag_view(p: dict, mode: str) -> dict:
        # Backward-compatible view for old payloads and new nested diagnostics payloads.
        d = p["best"]["diagnostics"]
        if "calibration" in d:
            cal = d["calibration"].get(mode, {})
            inv = d.get("innovation", {}).get(mode, {})
            return {
                "n_valid": cal.get("n_valid"),
                "pit_mean": cal.get("pit_mean"),
                "pit_var": cal.get("pit_var"),
                "rqr_mean": cal.get("rqr_mean"),
                "rqr_std": cal.get("rqr_std"),
                "z_mean": inv.get("z_mean"),
                "z_std": inv.get("z_std"),
                "z_skew": inv.get("z_skew"),
                "z_excess_kurtosis": inv.get("z_excess_kurtosis"),
            }
        old = d.get(mode, {})
        return {
            "n_valid": old.get("n_valid"),
            "pit_mean": old.get("pit_mean"),
            "pit_var": old.get("pit_var"),
            "rqr_mean": old.get("rqr_mean"),
            "rqr_std": old.get("rqr_std"),
            "z_mean": None,
            "z_std": None,
            "z_skew": None,
            "z_excess_kurtosis": None,
        }

    lines: list[str] = []
    lines.extend(
        [
            "---",
            'title: "Comprehensive Final Experiment Report"',
            'subtitle: "6-model matrix (3 methods × 2 response transforms; Gaussian kernel fixed)"',
            'author: "Par Pishrobat"',
            "date: today",
            "format:",
            "  pdf: default",
            "execute:",
            "  echo: false",
            "  warning: false",
            "  message: false",
            "---",
            "",
            "# Setup",
            "",
            f"- K sweep: `{cfg.k_values}`",
            f"- m: `{cfg.m}`",
            f"- max lag: `{cfg.max_lag}`",
            f"- split: `hydr_year < {cfg.train_cutoff_year}` (train), `hydr_year >= {cfg.train_cutoff_year}` (test)",
            "- data: Big Sur (`rain` predictor, `gauge` response)",
            "- kernel type: `gaussian` (fixed)",
            f"- complexity sweep axis: GP families use kernel count `K`; NN-GP uses fixed architecture (hidden_dim={cfg.nn_hidden_dim}, dropout={cfg.nn_dropout}, weight_decay={cfg.nn_weight_decay})",
            "- metrics: NSE, KGE, CRPS, RMSE, RelErr",
            "- diagnostics: innovation QQ on transformed scale + PIT histogram on original scale (simulation + kriging)",
            "- SWR interpretation: sim=no-AR SWR, krig=AR-corrected SWR (Cochrane-Orcutt auto-p)",
            "- selection: within-model by BIC_eff/OLS (SWR) or BIC/GLS (SWR-GP); by train-heldout krig validation CRPS (NN); across-model by test CRPS",
            "",
            "# Runtime budget calibration",
            "",
            f"- target runtime per model variant: `{cfg.target_runtime_sec:.1f}` sec",
            f"- calibrated GP maxiter: `{cfg.gp_maxiter}`",
            f"- calibrated NN epochs: `{cfg.nn_epochs}` (GLS iterations `{cfg.nn_gls_iterations}`)",
            "",
            "# Training Metrics (all variants)",
            "",
            "| Model | Sim NSE | Sim KGE | Sim CRPS | Sim RMSE | Sim RelErr | Krig NSE | Krig KGE | Krig CRPS | Krig RMSE | Krig RelErr |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for p in payloads:
        m = p["best"]["metrics"]
        lines.append(
            f"| `{_short_name(p['experiment_id'])}` | "
            f"{m['sim_train']['NSE']:.4f} | {m['sim_train']['KGE']:.4f} | {m['sim_train']['CRPS']:.4f} | {m['sim_train']['RMSE']:.4f} | {m['sim_train']['RelErr']:.4f} | "
            f"{m['krig_train']['NSE']:.4f} | {m['krig_train']['KGE']:.4f} | {m['krig_train']['CRPS']:.4f} | {m['krig_train']['RMSE']:.4f} | {m['krig_train']['RelErr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "# Test Metrics (all variants)",
            "",
            "| Model | Sim NSE | Sim KGE | Sim CRPS | Sim RMSE | Sim RelErr | Krig NSE | Krig KGE | Krig CRPS | Krig RMSE | Krig RelErr |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for p in payloads:
        m = p["best"]["metrics"]
        lines.append(
            f"| `{_short_name(p['experiment_id'])}` | "
            f"{m['sim_test']['NSE']:.4f} | {m['sim_test']['KGE']:.4f} | {m['sim_test']['CRPS']:.4f} | {m['sim_test']['RMSE']:.4f} | {m['sim_test']['RelErr']:.4f} | "
            f"{m['krig_test']['NSE']:.4f} | {m['krig_test']['KGE']:.4f} | {m['krig_test']['CRPS']:.4f} | {m['krig_test']['RMSE']:.4f} | {m['krig_test']['RelErr']:.4f} |"
        )

    lines.extend(["", "# Detailed Results and Diagnostics", ""])

    for p in payloads:
        m = p["best"]["metrics"]
        d = p["best"]["diagnostic_plots"]
        cal_sim_train = _diag_view(p, "sim_train")
        cal_sim_test = _diag_view(p, "sim_test")
        cal_krig_train = _diag_view(p, "krig_train")
        cal_krig_test = _diag_view(p, "krig_test")

        if p["method"] == "nn_gp":
            complexity_note = (
                f"- Fixed architecture: hidden_dim={cfg.nn_hidden_dim}, "
                f"dropout={cfg.nn_dropout}, weight_decay={cfg.nn_weight_decay}"
            )
        else:
            complexity_note = f"- Best K: `{p['selection']['best_K']}`"

        lines.extend(
            [
                f"## {_short_name(p['experiment_id'])}",
                "",
                f"- Variant ID: `{p['experiment_id']}`",
                f"- Method: `{p['method']}`",
                "- Kernel type: `gaussian` (fixed)",
                f"- Response transform: `{p['transform']}`",
                complexity_note,
                f"- Runtime active/wall: `{p['runtime_sec']:.1f}s / {p.get('runtime_wall_sec', p['runtime_sec']):.1f}s`",
                f"- Runtime flags: `{_format_runtime_flags_for_report(p)}`",
                "",
                "### Metrics",
                "",
                "| Split | Mode | NSE | KGE | CRPS | RMSE | RelErr |",
                "|---|---|---:|---:|---:|---:|---:|",
                f"| Train | Sim | {m['sim_train']['NSE']:.4f} | {m['sim_train']['KGE']:.4f} | {m['sim_train']['CRPS']:.4f} | {m['sim_train']['RMSE']:.4f} | {m['sim_train']['RelErr']:.4f} |",
                f"| Train | Krig | {m['krig_train']['NSE']:.4f} | {m['krig_train']['KGE']:.4f} | {m['krig_train']['CRPS']:.4f} | {m['krig_train']['RMSE']:.4f} | {m['krig_train']['RelErr']:.4f} |",
                f"| Test | Sim | {m['sim_test']['NSE']:.4f} | {m['sim_test']['KGE']:.4f} | {m['sim_test']['CRPS']:.4f} | {m['sim_test']['RMSE']:.4f} | {m['sim_test']['RelErr']:.4f} |",
                f"| Test | Krig | {m['krig_test']['NSE']:.4f} | {m['krig_test']['KGE']:.4f} | {m['krig_test']['CRPS']:.4f} | {m['krig_test']['RMSE']:.4f} | {m['krig_test']['RelErr']:.4f} |",
                "",
                "### Diagnostics Summary",
                "",
                "| Split | Mode | n_valid | PIT mean | PIT var | RQR mean | RQR std | z mean | z std | z skew | z ex.kurt |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| Train | Sim | {_fmt(cal_sim_train['n_valid'], 0)} | {_fmt(cal_sim_train['pit_mean'])} | {_fmt(cal_sim_train['pit_var'])} | {_fmt(cal_sim_train['rqr_mean'])} | {_fmt(cal_sim_train['rqr_std'])} | {_fmt(cal_sim_train['z_mean'])} | {_fmt(cal_sim_train['z_std'])} | {_fmt(cal_sim_train['z_skew'])} | {_fmt(cal_sim_train['z_excess_kurtosis'])} |",
                f"| Train | Krig | {_fmt(cal_krig_train['n_valid'], 0)} | {_fmt(cal_krig_train['pit_mean'])} | {_fmt(cal_krig_train['pit_var'])} | {_fmt(cal_krig_train['rqr_mean'])} | {_fmt(cal_krig_train['rqr_std'])} | {_fmt(cal_krig_train['z_mean'])} | {_fmt(cal_krig_train['z_std'])} | {_fmt(cal_krig_train['z_skew'])} | {_fmt(cal_krig_train['z_excess_kurtosis'])} |",
                f"| Test | Sim | {_fmt(cal_sim_test['n_valid'], 0)} | {_fmt(cal_sim_test['pit_mean'])} | {_fmt(cal_sim_test['pit_var'])} | {_fmt(cal_sim_test['rqr_mean'])} | {_fmt(cal_sim_test['rqr_std'])} | {_fmt(cal_sim_test['z_mean'])} | {_fmt(cal_sim_test['z_std'])} | {_fmt(cal_sim_test['z_skew'])} | {_fmt(cal_sim_test['z_excess_kurtosis'])} |",
                f"| Test | Krig | {_fmt(cal_krig_test['n_valid'], 0)} | {_fmt(cal_krig_test['pit_mean'])} | {_fmt(cal_krig_test['pit_var'])} | {_fmt(cal_krig_test['rqr_mean'])} | {_fmt(cal_krig_test['rqr_std'])} | {_fmt(cal_krig_test['z_mean'])} | {_fmt(cal_krig_test['z_std'])} | {_fmt(cal_krig_test['z_skew'])} | {_fmt(cal_krig_test['z_excess_kurtosis'])} |",
                "",
                f"![Innovation QQ (sim test, transformed scale)]({Path(d['qq_rqr_sim_test']).relative_to(exp_dir)})",
                "",
                f"![PIT histogram (sim test, original scale)]({Path(d['pit_sim_test']).relative_to(exp_dir)})",
                "",
                f"![Innovation QQ (krig test, transformed scale)]({Path(d['qq_rqr_krig_test']).relative_to(exp_dir)})",
                "",
                f"![PIT histogram (krig test, original scale)]({Path(d['pit_krig_test']).relative_to(exp_dir)})",
                "",
            ]
        )

    lines.extend(
        [
            "# Discussion",
            "",
            "- This report uses matched runtime budgets, but exact parity is not guaranteed because optimization landscapes differ across SWR/GPKR/NN families.",
            "- Model ranking should prioritize held-out CRPS (primary) with NSE/KGE as complementary skill indicators.",
            "- Where runtime imbalance remains, interpret gains jointly with fit-time columns.",
            "",
        ]
    )

    qmd_path = exp_dir / "report.qmd"
    qmd_path.write_text("\n".join(lines), encoding="utf-8")

    summary_path = metrics_dir / "summary_all.json"
    summary = {
        "config": {
            "k_values": cfg.k_values,
            "m": cfg.m,
            "max_lag": cfg.max_lag,
            "nu": cfg.nu,
            "boxcox_lambda": cfg.boxcox_lambda,
            "train_split": "hydr_year < 30",
            "test_split": "hydr_year >= 30",
            "gp_maxiter": cfg.gp_maxiter,
            "nn_epochs": cfg.nn_epochs,
            "nn_gls_iterations": cfg.nn_gls_iterations,
            "nn_hidden_dim": cfg.nn_hidden_dim,
            "nn_dropout": cfg.nn_dropout,
            "nn_weight_decay": cfg.nn_weight_decay,
        },
        "variants": payloads,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return qmd_path


def run_comprehensive(cfg: RunConfig, results_root: str | None = None) -> dict:
    exp_dir, logs_dir, metrics_dir, models_dir, plots_dir = ensure_output_dirs_for_root(
        results_root,
        experiment_key="comprehensive",
        script_path=__file__,
    )
    run_ts = timestamp()
    master_log = logs_dir / f"run_master_{run_ts}.log"
    logger = _setup_logger(master_log)

    logger.info("Starting comprehensive run")
    logger.info("Exp dir: %s", exp_dir)

    if cfg.gp_maxiter <= 0 or cfg.nn_epochs <= 0:
        gp_mi, nn_ep = _estimate_budgets(cfg)
        cfg.gp_maxiter = gp_mi
        cfg.nn_epochs = nn_ep

    methods = getattr(cfg, "methods", METHODS)
    transforms = getattr(cfg, "transforms", TRANSFORMS)
    tasks = [(m, t) for m in methods for t in transforms]
    logger.info("Total variants: %d", len(tasks))

    results: list[dict] = []
    cfg_dict = {
        "k_values": cfg.k_values,
        "m": cfg.m,
        "max_lag": cfg.max_lag,
        "nu": cfg.nu,
        "boxcox_lambda": cfg.boxcox_lambda,
        "train_cutoff_year": cfg.train_cutoff_year,
        "target_runtime_sec": cfg.target_runtime_sec,
        "seed": cfg.seed,
        "n_jobs": cfg.n_jobs,
        "gp_maxiter": cfg.gp_maxiter,
        "nn_epochs": cfg.nn_epochs,
        "nn_gls_iterations": cfg.nn_gls_iterations,
        "nn_hidden_dim": cfg.nn_hidden_dim,
        "nn_dropout": cfg.nn_dropout,
        "nn_weight_decay": cfg.nn_weight_decay,
        "render_report": cfg.render_report,
    }

    t0 = _timer_now()
    if cfg.n_jobs == 1:
        for method, transform in tasks:
            exp_id = f"{method}__{transform}"
            try:
                res = _run_variant(method, transform, cfg_dict, str(results_root) if results_root else None)
                results.append(res)
                logger.info(
                    "DONE %s | bestK=%d | krigCRPS=%.4f | active=%.1fs wall=%.1fs | flags=%s",
                    exp_id,
                    res["best_K"],
                    res["best_test_crps_krig"],
                    res["runtime_sec"],
                    res.get("runtime_wall_sec", res["runtime_sec"]),
                    ",".join(res.get("runtime_flags", [])) if res.get("runtime_flags") else "none",
                )
            except Exception as e:
                logger.exception("FAILED %s: %s", exp_id, e)
                raise
    else:
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as ex:
            futs = {
                ex.submit(_run_variant, method, transform, cfg_dict, str(results_root) if results_root else None): (method, transform)
                for method, transform in tasks
            }
            for fut in as_completed(futs):
                method, transform = futs[fut]
                exp_id = f"{method}__{transform}"
                try:
                    res = fut.result()
                    results.append(res)
                    logger.info(
                        "DONE %s | bestK=%d | krigCRPS=%.4f | active=%.1fs wall=%.1fs | flags=%s",
                        exp_id,
                        res["best_K"],
                        res["best_test_crps_krig"],
                        res["runtime_sec"],
                        res.get("runtime_wall_sec", res["runtime_sec"]),
                        ",".join(res.get("runtime_flags", [])) if res.get("runtime_flags") else "none",
                    )
                except Exception as e:
                    logger.exception("FAILED %s: %s", exp_id, e)
                    raise

    qmd_path = _write_report_qmd(results, exp_dir, cfg)

    if cfg.render_report:
        import subprocess

        cmd = ["quarto", "render", str(qmd_path)]
        logger.info("Rendering report: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    run_elapsed = _elapsed_seconds(t0)
    run_flags = _runtime_quality_flags(run_elapsed)

    payload = {
        "experiment": "exp_comprehensive",
        "timestamp": run_ts,
        "runtime": {
            "method": "active_cpu_sec_primary",
            "active_sec": float(run_elapsed["active_sec"]),
            "wall_sec": float(run_elapsed["wall_sec"]),
            "stall_sec": float(run_elapsed["stall_sec"]),
            "stall_fraction": float(run_elapsed["stall_fraction"]),
            "flags": run_flags,
        },
        "runtime_sec": float(run_elapsed["active_sec"]),
        "runtime_wall_sec": float(run_elapsed["wall_sec"]),
        "n_variants": len(results),
        "results": sorted(results, key=lambda x: x["exp_id"]),
        "report_qmd": str(qmd_path),
        "report_pdf": str(qmd_path.with_suffix(".pdf")),
    }

    with open(metrics_dir / "run_manifest.json", "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        "Comprehensive run complete | active=%.1fs wall=%.1fs flags=%s",
        payload["runtime_sec"],
        payload["runtime_wall_sec"],
        ",".join(run_flags) if run_flags else "none",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run comprehensive 6-variant final experiment (gaussian kernel fixed)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=100)
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--boxcox-lambda", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=min(6, os.cpu_count() or 2))
    parser.add_argument("--target-runtime-sec", type=float, default=45.0)
    parser.add_argument("--gp-maxiter", type=int, default=0, help="If <=0, auto-calibrate")
    parser.add_argument("--nn-epochs", type=int, default=0, help="If <=0, auto-calibrate")
    parser.add_argument("--nn-gls-iterations", type=int, default=6)
    parser.add_argument("--nn-hidden-dim", type=int, default=24)
    parser.add_argument("--nn-dropout", type=float, default=0.0)
    parser.add_argument("--nn-weight-decay", type=float, default=1e-3)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--transforms", type=str, default=",".join(TRANSFORMS))
    parser.add_argument("--no-render-report", action="store_true")
    args = parser.parse_args()

    cfg = RunConfig(
        k_values=list(args.k_values),
        m=int(args.m),
        max_lag=int(args.max_lag),
        nu=float(args.nu),
        boxcox_lambda=float(args.boxcox_lambda),
        train_cutoff_year=30,
        target_runtime_sec=float(args.target_runtime_sec),
        seed=int(args.seed),
        n_jobs=max(1, int(args.n_jobs)),
        gp_maxiter=int(args.gp_maxiter),
        nn_epochs=int(args.nn_epochs),
        nn_gls_iterations=int(args.nn_gls_iterations),
        nn_hidden_dim=int(args.nn_hidden_dim),
        nn_dropout=float(args.nn_dropout),
        nn_weight_decay=float(args.nn_weight_decay),
        render_report=not args.no_render_report,
    )
    cfg.methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    cfg.transforms = [s.strip() for s in args.transforms.split(",") if s.strip()]

    out = run_comprehensive(cfg, results_root=args.results_root)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
