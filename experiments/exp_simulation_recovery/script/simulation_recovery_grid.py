"""Simulation-recovery grid: oracle-K simulation study.

Factorial design:
  - K_true ∈ {1, 2, 3, 4}  (true kernel count)
  - noise  ∈ {low, medium, high}  (SNR = 100, 20, 5)

Each replication:
  1. Draw true Gaussian kernel params with low overlap and distinct centers.
  2. Set total residual variance = signal_var / target_SNR.
  3. Split into σ² (90%) and τ² (10%), sample φ, simulate Matérn GP residuals.
  4. Fit GP-SWR once at the true K.
  5. Report oracle recovery metrics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np
from scipy.optimize import linear_sum_assignment

from swrgp.paths import DATA_DIR
from swrgp.kernels import (
    build_design_matrix,
    kernel_mean_lag,
    kernel_spread,
    sample_constrained_gaussian_kernel_params,
)
from swrgp.model import GPSWR
from swrgp.bigsur_data import load_bigsur_train_test


# ── SNR-based noise tiers ─────────────────────────────────────────────
# target_SNR = Var(μ) / (σ² + τ²)
SNR_TIERS: Dict[str, float] = {
    "low":    100.0,
    "medium":  20.0,
    "high":     5.0,
}

TIER_DISPLAY: Dict[str, str] = {
    "low":    "Low\n(SNR = 100)",
    "medium": "Medium\n(SNR = 20)",
    "high":   "High\n(SNR = 5)",
}

# Fixed nugget fraction of total variance (τ² / (σ² + τ²))
NUGGET_FRACTION = 0.10
SUPPORT_SIGMA_MULTIPLE = 3.0
SUPPORT_LAG_RANGE = (0.0, 90.0)
SIM_SPREAD_RANGE = (2.0, 4.0)
SIM_GAUSSIAN_KERNEL_LOG_BOUNDS = (
    (float(np.log(1.0)), float(np.log(1.0))),
    (float(np.log(100.0)), float(np.log(20.0))),
)
EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = EXP_DIR / "output"


# ── Data class ────────────────────────────────────────────────────────

@dataclass
class TrueParams:
    kernel_params: np.ndarray   # (K, 2) log-delta, log-sigma
    beta: np.ndarray            # (K,)
    theta: np.ndarray           # [sigma_sq, phi, nu, tau_sq]


# ── Data loading ──────────────────────────────────────────────────────

def load_rainfall_series(n_points: Optional[int] = None) -> np.ndarray:
    train, _ = load_bigsur_train_test(DATA_DIR)
    rainfall = train["rain"].to_numpy(dtype=float)
    if n_points is not None and n_points < len(rainfall):
        return rainfall[:n_points]
    return rainfall


# ── Simulation helpers ────────────────────────────────────────────────

def matern_cov_vector(
    max_lag: int, sigma_sq: float, phi: float, nu: float, tau_sq: float,
) -> np.ndarray:
    h = np.arange(max_lag + 1, dtype=float)
    if np.isclose(nu, 0.5):
        cov = sigma_sq * np.exp(-h / phi)
    elif np.isclose(nu, 1.5):
        sqrt3 = np.sqrt(3.0)
        scaled = sqrt3 * h / phi
        cov = sigma_sq * (1.0 + scaled) * np.exp(-scaled)
    elif np.isclose(nu, 2.5):
        sqrt5 = np.sqrt(5.0)
        scaled = sqrt5 * h / phi
        cov = sigma_sq * (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)
    else:
        raise ValueError("nu must be one of 0.5, 1.5, 2.5")
    cov[0] += tau_sq
    return cov


def _simulate_stationary_gp(
    n: int, cov_vec: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Wood & Chan (1994) circulant embedding — exact stationary GP, O(n log n)."""
    N = 1
    while N < 2 * n:
        N *= 2
    c_embed = np.zeros(N)
    c_embed[:n] = cov_vec
    if n > 1:
        c_embed[N - n + 1:] = cov_vec[1:][::-1]
    lam = np.real(np.fft.fft(c_embed))
    lam = np.maximum(lam, 0.0)
    w = rng.standard_normal(N)
    sample = np.real(np.fft.ifft(np.sqrt(lam) * np.fft.fft(w)))
    return sample[:n]


def _sample_separated_kernel_params(
    rng: np.random.Generator,
    K: int,
    min_mean_lag_separation: float,
    max_lag: float = 90.0,
    spread_range: Tuple[float, float] = (2.0, 4.0),
    max_overlap: float = 0.05,
    max_attempts: int = 2000,
) -> np.ndarray:
    """Sample Gaussian kernel params with the shared constraint geometry."""
    return sample_constrained_gaussian_kernel_params(
        rng=rng,
        K=K,
        spread_range=spread_range,
        min_mean_lag_separation=min_mean_lag_separation,
        max_overlap=max_overlap,
        support_sigma_multiple=SUPPORT_SIGMA_MULTIPLE,
        support_lag_range=(0.0, max_lag),
        max_attempts=max_attempts,
    )


def sample_true_params(
    rng: np.random.Generator,
    K: int,
    nu: float,
    target_snr: float,
    rainfall: np.ndarray,
    min_mean_lag_separation: float = 10.0,
    max_overlap: float = 0.05,
) -> TrueParams:
    """Sample parameters with SNR-controlled noise.

    1. Draw kernel params and beta, build true mean, compute Var(μ).
    2. total_var = Var(μ) / target_snr.
    3. τ² = NUGGET_FRACTION × total_var, σ² = (1-NUGGET_FRACTION) × total_var.
    4. φ ~ U(7, 50).
    """
    kernel_params = _sample_separated_kernel_params(
        rng, K=K,
        min_mean_lag_separation=min_mean_lag_separation,
        max_overlap=max_overlap,
    )
    beta = rng.uniform(0.10, 0.35, size=K)

    # Build mean to compute signal variance
    X = build_design_matrix(rainfall, kernel_params, kernel_type="gaussian")
    mu = X @ beta
    signal_var = float(np.var(mu))

    # Derive noise from target SNR
    total_var = signal_var / target_snr
    tau_sq = NUGGET_FRACTION * total_var
    sigma_sq = (1.0 - NUGGET_FRACTION) * total_var

    phi = float(rng.uniform(7.0, 50.0))
    theta = np.array([sigma_sq, phi, nu, tau_sq], dtype=float)
    return TrueParams(kernel_params=kernel_params, beta=beta, theta=theta)


def simulate_response(
    rainfall: np.ndarray, params: TrueParams, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = build_design_matrix(rainfall, params.kernel_params, kernel_type="gaussian")
    mu = X @ params.beta
    n = len(rainfall)
    cov_vec = matern_cov_vector(
        max_lag=n - 1,
        sigma_sq=float(params.theta[0]),
        phi=float(params.theta[1]),
        nu=float(params.theta[2]),
        tau_sq=float(params.theta[3]),
    )
    rng = np.random.default_rng(seed)
    eps = _simulate_stationary_gp(n=n, cov_vec=cov_vec, rng=rng)
    y = mu + eps
    return y, mu, X


# ── Recovery metric helpers ───────────────────────────────────────────

def _extract_kernel_summary(
    kernel_params: np.ndarray, beta: np.ndarray,
) -> List[Dict[str, float]]:
    summary = []
    for k in range(kernel_params.shape[0]):
        ld, ls = float(kernel_params[k, 0]), float(kernel_params[k, 1])
        summary.append({
            "kernel": k + 1,
            "beta": float(beta[k]),
            "delta": float(np.exp(ld)),
            "sigma": float(np.exp(ls)),
            "mean_lag": float(kernel_mean_lag(ld, ls, "gaussian")),
            "spread": float(kernel_spread(ld, ls, "gaussian")),
        })
    return summary


def _match_kernel_order(
    true_summary: List[Dict[str, float]], est_summary: List[Dict[str, float]],
) -> Tuple[List[int], List[int]]:
    K_true = len(true_summary)
    K_est = len(est_summary)
    K = min(K_true, K_est)
    cost = np.zeros((K_true, K_est), dtype=float)
    for i in range(K_true):
        for j in range(K_est):
            cost[i, j] = abs(true_summary[i]["mean_lag"] - est_summary[j]["mean_lag"])
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind.tolist(), col_ind.tolist()


def _safe_rel_err(est: float, truth: float) -> float:
    return abs(est - truth) / max(abs(truth), 1e-8)


def _mse(estimates: List[float], truths: List[float]) -> float:
    e = np.array(estimates, dtype=float)
    t = np.array(truths, dtype=float)
    return float(np.mean((e - t) ** 2))


def _mae(estimates: List[float], truths: List[float]) -> float:
    e = np.array(estimates, dtype=float)
    t = np.array(truths, dtype=float)
    return float(np.mean(np.abs(e - t)))


def compute_nse(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency = 1 - SS_res / SS_tot."""
    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def theoretical_max_nse(target_snr: float) -> float:
    """Best achievable NSE given the noise level.

    If the oracle predicts μ exactly, MSE = Var(ε) = signal_var / SNR.
    Var(y) = signal_var + signal_var / SNR = signal_var (1 + 1/SNR).
    NSE_max = 1 - Var(ε)/Var(y) = 1 - (1/SNR) / (1 + 1/SNR) = SNR / (1 + SNR).
    """
    return target_snr / (1.0 + target_snr)


# ── Single-cell runner ────────────────────────────────────────────────

def run_cell(
    K_true: int,
    noise_tier: str,
    target_snr: float,
    rainfall: np.ndarray,
    reps: int,
    m: int,
    nu: float,
    maxiter: int,
    n_restarts: int,
    min_mean_lag_separation: float,
    max_overlap: float,
    master_rng: np.random.Generator,
    verbose: bool,
) -> Dict[str, Any]:
    """Run one (K_true, noise_tier) cell using oracle K only."""
    rep_results: List[Dict[str, Any]] = []
    nse_max = theoretical_max_nse(target_snr)

    for rep in range(reps):
        rep_seed = int(master_rng.integers(1, 10**9))
        param_rng = np.random.default_rng(rep_seed)

        true_params = sample_true_params(
            param_rng, K=K_true, nu=nu,
            target_snr=target_snr,
            rainfall=rainfall,
            min_mean_lag_separation=min_mean_lag_separation,
            max_overlap=max_overlap,
        )
        y, mu_true, _ = simulate_response(
            rainfall=rainfall, params=true_params, seed=rep_seed + 13,
        )

        # Per-replication diagnostics
        true_mu_std = float(np.std(mu_true))
        signal_var = float(np.var(mu_true))
        noise_var = float(true_params.theta[0] + true_params.theta[3])
        snr_actual = signal_var / noise_var if noise_var > 0 else float("inf")
        snr_db = 10.0 * np.log10(snr_actual) if noise_var > 0 else float("inf")

        model = GPSWR(
            K=K_true, m=m, nu=nu, kernel_type="gaussian",
            maxiter=maxiter, n_restarts=n_restarts, log_transform=False,
            seed=rep_seed, verbose=verbose,
            min_mean_lag_separation=min_mean_lag_separation,
            max_kernel_overlap=max_overlap,
            support_sigma_multiple=SUPPORT_SIGMA_MULTIPLE,
            support_lag_range=SUPPORT_LAG_RANGE,
            restart_spread_range=SIM_SPREAD_RANGE,
            gaussian_kernel_log_bounds=SIM_GAUSSIAN_KERNEL_LOG_BOUNDS,
        )
        model.fit(rainfall, y)

        mu_est = (
            build_design_matrix(rainfall, model.kernel_params_, "gaussian")
            @ model.beta_
        )
        rmse = float(np.sqrt(np.mean((mu_est - mu_true) ** 2)))
        nrmse = rmse / max(true_mu_std, 1e-8)
        nse = compute_nse(y, mu_est)

        # Kernel lag recovery (oracle model at true K)
        true_kernel = _extract_kernel_summary(true_params.kernel_params, true_params.beta)
        est_kernel = model.get_kernel_summary()
        rows, cols = _match_kernel_order(true_kernel, est_kernel)

        delta_rel_errs, sigma_k_rel_errs, lag_rel_errs, beta_rel_errs = [], [], [], []
        delta_est, delta_true_vals = [], []
        sigma_est, sigma_true_vals = [], []
        beta_est, beta_true_vals = [], []
        for i, j in zip(rows, cols):
            lag_rel_errs.append(
                _safe_rel_err(float(est_kernel[j]["mean_lag"]),
                              float(true_kernel[i]["mean_lag"])))
            delta_est.append(float(est_kernel[j]["delta"]))
            delta_true_vals.append(float(true_kernel[i]["delta"]))
            delta_rel_errs.append(
                _safe_rel_err(float(est_kernel[j]["delta"]),
                              float(true_kernel[i]["delta"])))
            sigma_est.append(float(est_kernel[j]["sigma"]))
            sigma_true_vals.append(float(true_kernel[i]["sigma"]))
            sigma_k_rel_errs.append(
                _safe_rel_err(float(est_kernel[j]["sigma"]),
                              float(true_kernel[i]["sigma"])))
            beta_est.append(float(est_kernel[j]["beta"]))
            beta_true_vals.append(float(true_kernel[i]["beta"]))
            beta_rel_errs.append(
                _safe_rel_err(float(est_kernel[j]["beta"]),
                              float(true_kernel[i]["beta"])))

        param_mse = {
            "delta": _mse(delta_est, delta_true_vals),
            "sigma_k": _mse(sigma_est, sigma_true_vals),
            "beta": _mse(beta_est, beta_true_vals),
        }
        param_mae = {
            "delta": _mae(delta_est, delta_true_vals),
            "sigma_k": _mae(sigma_est, sigma_true_vals),
            "beta": _mae(beta_est, beta_true_vals),
        }

        # Covariance recovery
        theta_true = true_params.theta
        theta_est = model.theta_
        cov_rel_errs = {
            "sigma_sq": _safe_rel_err(float(theta_est[0]), float(theta_true[0])),
            "phi": _safe_rel_err(float(theta_est[1]), float(theta_true[1])),
            "tau_sq": _safe_rel_err(float(theta_est[3]), float(theta_true[3])),
        }
        cov_mse = {
            "sigma_sq": float((float(theta_est[0]) - float(theta_true[0])) ** 2),
            "phi": float((float(theta_est[1]) - float(theta_true[1])) ** 2),
            "tau_sq": float((float(theta_est[3]) - float(theta_true[3])) ** 2),
        }
        cov_mae = {
            "sigma_sq": float(abs(float(theta_est[0]) - float(theta_true[0]))),
            "phi": float(abs(float(theta_est[1]) - float(theta_true[1]))),
            "tau_sq": float(abs(float(theta_est[3]) - float(theta_true[3]))),
        }

        rep_results.append({
            "replication": rep + 1,
            "seed": rep_seed,
            "snr_db": snr_db,
            "snr_actual": snr_actual,
            "true_sigma_sq": float(true_params.theta[0]),
            "true_tau_sq": float(true_params.theta[3]),
            "true_mu_std": true_mu_std,
            "nrmse": nrmse,
            "nse": nse,
            # Theoretical ceiling
            "nse_max": nse_max,
            "fit_bic": float(model.bic_),
            "fit_loglik": float(model.log_lik_),
            "mean_lag_rel_error": float(np.mean(lag_rel_errs)),
            "delta_rel_error": float(np.mean(delta_rel_errs)),
            "sigma_k_rel_error": float(np.mean(sigma_k_rel_errs)),
            "beta_rel_error": float(np.mean(beta_rel_errs)),
            "cov_rel_errors": cov_rel_errs,
            "param_mse": param_mse,
            "param_mae": param_mae,
            "cov_mse": cov_mse,
            "cov_mae": cov_mae,
            "true_kernel_summary": true_kernel,
            "estimated_kernel_summary": est_kernel,
            "matched_kernel_pairs": [
                {
                    "true": true_kernel[i],
                    "estimated": est_kernel[j],
                }
                for i, j in zip(rows, cols)
            ],
            "true_theta": {
                "sigma_sq": float(theta_true[0]),
                "phi": float(theta_true[1]),
                "nu": float(theta_true[2]),
                "tau_sq": float(theta_true[3]),
            },
            "estimated_theta": {
                "sigma_sq": float(theta_est[0]),
                "phi": float(theta_est[1]),
                "nu": float(theta_est[2]),
                "tau_sq": float(theta_est[3]),
            },
        })

        for rs in model.restart_summaries_:
            print(
                f"    restart {rs['restart']}/{model.n_restarts}: "
                f"logLik={rs['best_loglik']:.3f}  "
                f"iters={rs['iters']}  seed={rs['seed']}"
            )
        print(
            f"    rep {rep + 1}/{reps}: "
            f"NRMSE={nrmse:.3f}  NSE={nse:.3f} (max={nse_max:.3f})  "
            f"BIC={float(model.bic_):.3f}  logLik={float(model.log_lik_):.3f}  "
            f"lag_err={float(np.mean(lag_rel_errs)):.3f}"
        )
        print(
            "      kernel params: "
            f"delta MSE={param_mse['delta']:.4f} MAE={param_mae['delta']:.4f} | "
            f"sigma_k MSE={param_mse['sigma_k']:.4f} MAE={param_mae['sigma_k']:.4f} | "
            f"beta MSE={param_mse['beta']:.4f} MAE={param_mae['beta']:.4f}"
        )
        print(
            "      covariance: "
            f"sigma_sq MSE={cov_mse['sigma_sq']:.4f} MAE={cov_mae['sigma_sq']:.4f} | "
            f"phi MSE={cov_mse['phi']:.4f} MAE={cov_mae['phi']:.4f} | "
            f"tau_sq MSE={cov_mse['tau_sq']:.4f} MAE={cov_mae['tau_sq']:.4f}"
        )

    # ── Aggregate cell summary ────────────────────────────────────────
    def _agg(vals):
        a = np.array(vals, dtype=float)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "std": float(np.std(a)),
        }

    cell_summary = {
        "K_true": K_true,
        "noise_tier": noise_tier,
        "target_snr": target_snr,
        "nse_max": nse_max,
        "n_reps": reps,
        # SNR
        **{f"snr_db_{k}": v for k, v in _agg([r["snr_db"] for r in rep_results]).items()},
        **{f"nrmse_{k}": v for k, v in _agg([r["nrmse"] for r in rep_results]).items()},
        **{f"nse_{k}": v for k, v in _agg([r["nse"] for r in rep_results]).items()},
        **{f"fit_bic_{k}": v for k, v in _agg([r["fit_bic"] for r in rep_results]).items()},
        **{f"fit_loglik_{k}": v for k, v in _agg([r["fit_loglik"] for r in rep_results]).items()},
        **{f"lag_rel_error_{k}": v for k, v in _agg([r["mean_lag_rel_error"] for r in rep_results]).items()},
        **{f"delta_rel_error_{k}": v for k, v in _agg([r["delta_rel_error"] for r in rep_results]).items()},
        **{f"sigma_k_rel_error_{k}": v for k, v in _agg([r["sigma_k_rel_error"] for r in rep_results]).items()},
        **{f"beta_rel_error_{k}": v for k, v in _agg([r["beta_rel_error"] for r in rep_results]).items()},
        **{f"cov_sigma_sq_rel_error_{k}": v for k, v in _agg([r["cov_rel_errors"]["sigma_sq"] for r in rep_results]).items()},
        **{f"cov_phi_rel_error_{k}": v for k, v in _agg([r["cov_rel_errors"]["phi"] for r in rep_results]).items()},
        **{f"cov_tau_sq_rel_error_{k}": v for k, v in _agg([r["cov_rel_errors"]["tau_sq"] for r in rep_results]).items()},
        **{f"delta_mse_{k}": v for k, v in _agg([r["param_mse"]["delta"] for r in rep_results]).items()},
        **{f"sigma_k_mse_{k}": v for k, v in _agg([r["param_mse"]["sigma_k"] for r in rep_results]).items()},
        **{f"beta_mse_{k}": v for k, v in _agg([r["param_mse"]["beta"] for r in rep_results]).items()},
        **{f"delta_mae_{k}": v for k, v in _agg([r["param_mae"]["delta"] for r in rep_results]).items()},
        **{f"sigma_k_mae_{k}": v for k, v in _agg([r["param_mae"]["sigma_k"] for r in rep_results]).items()},
        **{f"beta_mae_{k}": v for k, v in _agg([r["param_mae"]["beta"] for r in rep_results]).items()},
        **{f"cov_sigma_sq_mse_{k}": v for k, v in _agg([r["cov_mse"]["sigma_sq"] for r in rep_results]).items()},
        **{f"cov_phi_mse_{k}": v for k, v in _agg([r["cov_mse"]["phi"] for r in rep_results]).items()},
        **{f"cov_tau_sq_mse_{k}": v for k, v in _agg([r["cov_mse"]["tau_sq"] for r in rep_results]).items()},
        **{f"cov_sigma_sq_mae_{k}": v for k, v in _agg([r["cov_mae"]["sigma_sq"] for r in rep_results]).items()},
        **{f"cov_phi_mae_{k}": v for k, v in _agg([r["cov_mae"]["phi"] for r in rep_results]).items()},
        **{f"cov_tau_sq_mae_{k}": v for k, v in _agg([r["cov_mae"]["tau_sq"] for r in rep_results]).items()},
    }

    return {"cell_summary": cell_summary, "replications": rep_results}


# ── Plotting ──────────────────────────────────────────────────────────

def plot_nse_small_multiples(grid_results: Dict[str, Any], plots_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = grid_results["grid_cells"]
    K_vals = sorted(set(c["cell_summary"]["K_true"] for c in cells))
    tier_order = ["low", "medium", "high"]

    def _cell(K, tier):
        return next(c["cell_summary"] for c in cells
                    if c["cell_summary"]["K_true"] == K
                    and c["cell_summary"]["noise_tier"] == tier)

    tier_labels = ["SNR = 100", "SNR = 20", "SNR = 5"]
    kernel_colors = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}
    x = np.arange(len(tier_order), dtype=float)

    fig, axes = plt.subplots(1, len(K_vals), figsize=(12.0, 3.5), sharey=True)

    for ax, K in zip(axes, K_vals):
        achieved = np.array([_cell(K, tier)["nse_median"] for tier in tier_order], dtype=float)
        ceiling = np.array([_cell(K, tier)["nse_max"] for tier in tier_order], dtype=float)
        color = kernel_colors[K]
        ax.fill_between(x, achieved, 0, color=color, alpha=0.18)
        ax.plot(x, achieved, color=color, lw=2.0, marker="o")
        ax.plot(x, ceiling, color=color, lw=1.7, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels, fontsize=8)
        ax.set_xlabel("Noise tier", fontsize=8.5)
        ax.set_title(f"$K={K}$", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.72, 1.01)

    axes[0].set_ylabel("NSE", fontsize=9)
    fig.suptitle("Observed-Space NSE by Kernel Count", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = plots_dir / "recovery_nse_by_k.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_nrmse_heatmap(grid_results: Dict[str, Any], plots_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = grid_results["grid_cells"]
    K_vals = sorted(set(c["cell_summary"]["K_true"] for c in cells))
    tier_order = ["low", "medium", "high"]

    def _cell(K, tier):
        return next(c["cell_summary"] for c in cells
                    if c["cell_summary"]["K_true"] == K
                    and c["cell_summary"]["noise_tier"] == tier)

    nrmse = np.array([[_cell(K, t)["nrmse_median"] for t in tier_order] for K in K_vals])
    nrmse_vmax = max(0.20, float(np.max(nrmse)) * 1.05)

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.0))
    im = ax.imshow(nrmse, cmap="YlOrRd", vmin=0, vmax=nrmse_vmax, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Low", "Med", "High"], fontsize=8)
    ax.set_yticks(range(len(K_vals)))
    ax.set_yticklabels([f"K={k}" for k in K_vals], fontsize=8.5)
    ax.set_xlabel("Noise tier", fontsize=8.5)
    ax.set_ylabel(r"True $K$", fontsize=9)
    ax.set_title("NRMSE", fontsize=10)
    ax.tick_params(axis="both", length=0)
    for i in range(len(K_vals)):
        for j in range(3):
            v = nrmse[i, j]
            c = "white" if v > 0.55 * nrmse_vmax else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5, color=c)

    plt.tight_layout(pad=0.7)

    out_path = plots_dir / "recovery_nrmse_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_parameter_recovery_heatmaps(grid_results: Dict[str, Any], plots_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = grid_results["grid_cells"]
    K_vals = sorted(set(c["cell_summary"]["K_true"] for c in cells))
    tier_order = ["low", "medium", "high"]

    def _cell(K, tier):
        return next(c["cell_summary"] for c in cells
                    if c["cell_summary"]["K_true"] == K
                    and c["cell_summary"]["noise_tier"] == tier)

    panels = [
        ("delta_mse_median", r"$\delta$ MSE"),
        ("sigma_k_mse_median", r"$\sigma_k$ MSE"),
        ("beta_mse_median", r"$\beta$ MSE"),
        ("cov_sigma_sq_mse_median", r"$\sigma^2$ MSE"),
        ("cov_phi_mse_median", r"$\phi$ MSE"),
        ("cov_tau_sq_mse_median", r"$\tau^2$ MSE"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), sharey=True)

    for ax, (key, title) in zip(axes.flat, panels):
        data = np.array([[_cell(K, t)[key] for t in tier_order] for K in K_vals])
        vmax = max(1e-6, float(np.max(data)) * 1.05)
        im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Low", "Med", "High"], fontsize=8)
        ax.set_yticks(range(len(K_vals)))
        ax.set_yticklabels([f"K={k}" for k in K_vals], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Noise tier", fontsize=8.5)
        ax.tick_params(axis="both", length=0)
        for i in range(len(K_vals)):
            for j in range(3):
                v = data[i, j]
                c = "white" if v > 0.55 * vmax else "black"
                txt = f"{v:.3f}" if vmax < 10 else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=c)

    axes[0, 0].set_ylabel(r"True $K$", fontsize=9)
    axes[1, 0].set_ylabel(r"True $K$", fontsize=9)
    fig.suptitle("GP-SWR Oracle-K Parameter Recovery", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = plots_dir / "recovery_parameter_heatmaps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main grid runner ──────────────────────────────────────────────────

def run_grid(
    K_values: List[int] = [1, 2, 3, 4],
    snr_tiers: Optional[Dict[str, float]] = None,
    n_points: Optional[int] = None,
    reps: int = 10,
    m: int = 10,
    nu: float = 1.5,
    maxiter: int = 202,
    n_restarts: int = 5,
    seed: int = 42,
    min_mean_lag_separation: float = 10.0,
    max_overlap: float = 0.05,
    results_root: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if snr_tiers is None:
        snr_tiers = SNR_TIERS

    output_dir = DEFAULT_OUTPUT_DIR if results_root is None else Path(results_root)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rainfall = load_rainfall_series(n_points=n_points)
    n_actual = len(rainfall)
    print(f"Using rainfall series of length n={n_actual:,} (Big Sur training split)")
    print("Fitting only the oracle model at each true K")

    master_rng = np.random.default_rng(seed)

    grid_cells: List[Dict[str, Any]] = []
    total_cells = len(K_values) * len(snr_tiers)
    cell_idx = 0

    for K in K_values:
        for tier_name, target_snr in snr_tiers.items():
            cell_idx += 1
            print(f"\n{'='*70}")
            print(f"  Grid cell {cell_idx}/{total_cells}: K_true={K}, "
                  f"noise_tier={tier_name} (SNR={target_snr})")
            print(f"  {reps} oracle fits")
            print(f"{'='*70}")

            t0 = time.time()
            cell_result = run_cell(
                K_true=K,
                noise_tier=tier_name,
                target_snr=target_snr,
                rainfall=rainfall,
                reps=reps,
                m=m, nu=nu, maxiter=maxiter, n_restarts=n_restarts,
                min_mean_lag_separation=min_mean_lag_separation,
                max_overlap=max_overlap,
                master_rng=master_rng,
                verbose=verbose,
            )
            elapsed = time.time() - t0
            print(f"  ✓ Cell complete in {elapsed:.1f}s")

            grid_cells.append(cell_result)

    # ── Build summary table ───────────────────────────────────────────
    summary_table = []
    for cell in grid_cells:
        s = cell["cell_summary"]
        row = {
            "K_true": s["K_true"],
            "noise_tier": s["noise_tier"],
            "target_snr": s["target_snr"],
            "nse_max": round(s["nse_max"], 4),
        }
        for prefix in [
            "nrmse", "nse", "fit_bic", "fit_loglik",
            "lag_rel_error", "delta_rel_error", "sigma_k_rel_error",
            "beta_rel_error",
            "cov_sigma_sq_rel_error", "cov_phi_rel_error", "cov_tau_sq_rel_error",
            "delta_mse", "sigma_k_mse", "beta_mse",
            "cov_sigma_sq_mse", "cov_phi_mse", "cov_tau_sq_mse",
            "delta_mae", "sigma_k_mae", "beta_mae",
            "cov_sigma_sq_mae", "cov_phi_mae", "cov_tau_sq_mae",
        ]:
            for stat in ["mean", "median", "p25", "p75"]:
                key = f"{prefix}_{stat}"
                row[key] = round(s[key], 4)
        summary_table.append(row)

    results = {
        "experiment": "simulation_recovery_grid",
        "config": {
            "K_values": K_values,
            "snr_tiers": snr_tiers,
            "nugget_fraction": NUGGET_FRACTION,
            "n_points": n_actual,
            "reps": reps,
            "m": m,
            "nu": nu,
            "maxiter": maxiter,
            "n_restarts": n_restarts,
            "seed": seed,
            "min_mean_lag_separation": min_mean_lag_separation,
            "max_overlap": max_overlap,
            "kernel_sampling": "gaussian_center_spread_with_overlap_cap",
            "fit_mode": "oracle_true_K_only",
            "fit_constraints": {
                "min_mean_lag_separation": min_mean_lag_separation,
                "max_kernel_overlap": max_overlap,
                "support_sigma_multiple": SUPPORT_SIGMA_MULTIPLE,
                "support_lag_range": SUPPORT_LAG_RANGE,
            },
        },
        "summary_table": summary_table,
        "grid_cells": grid_cells,
    }

    results_file = output_dir / "simulation_recovery_grid.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    summary_csv = output_dir / "simulation_recovery_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_table[0].keys()))
        writer.writeheader()
        writer.writerows(summary_table)

    # Plots
    try:
        nse_path = plot_nse_small_multiples(results, output_dir)
        nrmse_path = plot_nrmse_heatmap(results, output_dir)
        param_path = plot_parameter_recovery_heatmaps(results, output_dir)
        print(f"\nNSE plot saved to: {nse_path}")
        print(f"NRMSE heatmap saved to: {nrmse_path}")
        print(f"Parameter recovery plot saved to: {param_path}")
    except Exception as exc:
        print(f"\nPlot generation failed: {exc}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="GP-SWR oracle-K simulation recovery grid",
    )
    parser.add_argument("--K-values", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--maxiter", type=int, default=202)
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-mean-lag-separation", type=float, default=10.0)
    parser.add_argument("--max-overlap", type=float, default=0.05)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = run_grid(
        K_values=args.K_values,
        n_points=args.n_points,
        reps=args.reps,
        m=args.m,
        nu=args.nu,
        maxiter=args.maxiter,
        n_restarts=args.n_restarts,
        seed=args.seed,
        min_mean_lag_separation=args.min_mean_lag_separation,
        max_overlap=args.max_overlap,
        results_root=args.results_root,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 100)
    print("SIMULATION RECOVERY GRID COMPLETE")
    print("=" * 100)
    cfg = results["config"]
    print(f"\nn={cfg['n_points']:,} obs | "
          f"{len(results['summary_table'])} cells | "
          f"{cfg['reps']} reps each | "
          f"{cfg['n_restarts']} restarts per fit\n")
    print(f"{'K':>3}  {'Tier':>6}  {'SNR':>5}  "
          f"{'NRMSE':>11}  {'NSE':>9}  {'NSE_max':>8}  "
          f"{'LagErr':>8}")
    print("-" * 72)
    for row in results["summary_table"]:
        print(f"{row['K_true']:>3}  {row['noise_tier']:>6}  "
              f"{row['target_snr']:>5.1f}  "
              f"{row['nrmse_median']:>11.4f}  "
              f"{row['nse_median']:>9.4f}  {row['nse_max']:>8.4f}  "
              f"{row['lag_rel_error_median']:>8.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
