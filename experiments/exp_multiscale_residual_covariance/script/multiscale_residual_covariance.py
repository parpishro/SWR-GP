"""Experiment: compare single-scale vs multi-scale residual covariance on Big Sur."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from swrgp.bigsur_data import load_bigsur_train_test
from swrgp.kernels import build_design_matrix, kernel_mean_lag, kernel_spread
from swrgp.metrics import (
    compute_aic,
    compute_aic_eff,
    compute_bic,
    compute_bic_eff,
    compute_crps,
    compute_metrics,
    estimate_effective_sample_size,
)
from swrgp.model import GPSWR
from swrgp.nngp import build_nngp_matrices, decorrelate, decorrelate_matrix
from swrgp.paths import DATA_DIR, ensure_output_dirs_for_root, timestamp


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("multiscale_cov")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _serialize_dict(values: Dict) -> Dict:
    output = {}
    for key, value in values.items():
        if isinstance(value, (np.floating, float, np.integer, int)):
            output[key] = float(value)
        else:
            output[key] = value
    return output


def _sequential_correction(residuals: np.ndarray, B: np.ndarray, m: int) -> np.ndarray:
    n = residuals.shape[0]
    correction = np.zeros(n, dtype=float)
    for t in range(n):
        nn = min(t, m)
        for j in range(nn):
            neighbor_idx = t - nn + j
            correction[t] += B[t, m - nn + j] * residuals[neighbor_idx]
    return correction


@dataclass
class MultiScaleResidualGPSWR:
    K: int = 3
    m: int = 10
    nu: float = 1.5
    kernel_type: str = "gamma"
    maxiter: int = 120
    n_restarts: int = 5
    log_transform: bool = True
    seed: int = 42
    verbose: bool = True

    def __post_init__(self):
        self.epsilon_target = 0.01
        self.kernel_params_ = None
        self.beta_ = None
        self.theta_short_ = None
        self.theta_long_ = None
        self.B_short_ = None
        self.F_short_ = None
        self.B_long_ = None
        self.F_long_ = None
        self.warmup_ = 0

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        n_params = 2 * self.K + 6
        lower = np.zeros(n_params)
        upper = np.zeros(n_params)

        for k in range(self.K):
            lower[2 * k] = -2.0
            upper[2 * k] = 3.0
            lower[2 * k + 1] = -1.0
            upper[2 * k + 1] = 4.0

        idx = 2 * self.K
        lower[idx] = -4.0
        upper[idx] = 4.0
        lower[idx + 1] = 0.0
        upper[idx + 1] = 4.0
        lower[idx + 2] = -6.0
        upper[idx + 2] = 2.0
        lower[idx + 3] = -4.0
        upper[idx + 3] = 4.0
        lower[idx + 4] = -2.0
        upper[idx + 4] = 4.0
        lower[idx + 5] = -6.0
        upper[idx + 5] = 2.0
        return lower, upper

    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_kernel = 2 * self.K
        kernel_params = params[:n_kernel].reshape(self.K, 2)

        idx = n_kernel
        sigma_sq_short = np.exp(params[idx])
        phi_short = np.exp(params[idx + 1])
        tau_sq_short = np.exp(params[idx + 2])

        sigma_sq_long = np.exp(params[idx + 3])
        phi_gap = np.exp(params[idx + 4])
        tau_sq_long = np.exp(params[idx + 5])
        phi_long = phi_short + phi_gap

        theta_short = np.array([sigma_sq_short, phi_short, self.nu, tau_sq_short])
        theta_long = np.array([sigma_sq_long, phi_long, self.nu, tau_sq_long])
        return kernel_params, theta_short, theta_long

    def _objective(self, params: np.ndarray, rainfall: np.ndarray, streamflow: np.ndarray) -> float:
        try:
            y_target = streamflow
            if self.log_transform:
                y_target = np.log(streamflow + self.epsilon_target)

            kernel_params, theta_short, theta_long = self._unpack_params(params)
            X = build_design_matrix(rainfall, kernel_params, self.kernel_type)

            B_short, F_short = build_nngp_matrices(len(y_target), self.m, theta_short)
            Y_short = decorrelate(y_target, B_short, F_short, self.m)
            X_short = decorrelate_matrix(X, B_short, F_short, self.m)

            B_long, F_long = build_nngp_matrices(len(y_target), self.m, theta_long)
            Y_multi = decorrelate(Y_short, B_long, F_long, self.m)
            X_multi = decorrelate_matrix(X_short, B_long, F_long, self.m)

            beta, _, _, _ = np.linalg.lstsq(X_multi, Y_multi, rcond=None)

            pred = X @ beta
            residuals = y_target - pred
            resid_short = decorrelate(residuals, B_short, F_short, self.m)
            resid_multi = decorrelate(resid_short, B_long, F_long, self.m)

            log_det = np.sum(np.log(F_short)) + np.sum(np.log(F_long))
            rss = np.sum(resid_multi**2)
            n = len(y_target)
            log_lik = -0.5 * (n * np.log(2.0 * np.pi) + log_det + rss)
            return -float(log_lik)
        except Exception:
            return 1e10

    def fit(self, rainfall: np.ndarray, streamflow: np.ndarray, baseline_model: GPSWR | None = None):
        self.rainfall_ = rainfall.copy()
        self.streamflow_ = streamflow.copy()
        self.n_evals_ = 0

        y_target = streamflow
        if self.log_transform:
            y_target = np.log(streamflow + self.epsilon_target)

        lower, upper = self._get_bounds()
        n_params = lower.shape[0]
        rng = np.random.default_rng(self.seed)
        best_val = np.inf
        best_params = None

        if baseline_model is not None:
            x0 = np.zeros(n_params)
            x0[: 2 * self.K] = baseline_model.kernel_params_.reshape(-1)
            idx = 2 * self.K
            x0[idx] = np.log(max(baseline_model.theta_[0] * 0.7, 1e-8))
            x0[idx + 1] = np.log(max(baseline_model.theta_[1] * 0.5, 1e-6))
            x0[idx + 2] = np.log(max(baseline_model.theta_[3] * 0.5, 1e-8))
            x0[idx + 3] = np.log(max(baseline_model.theta_[0] * 0.5, 1e-8))
            x0[idx + 4] = np.log(max(baseline_model.theta_[1] * 0.7, 1e-6))
            x0[idx + 5] = np.log(max(baseline_model.theta_[3] * 0.5, 1e-8))
            x0 = np.clip(x0, lower, upper)

            result = minimize(
                lambda p: self._objective(p, rainfall, streamflow),
                x0,
                method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={"maxiter": max(10, self.maxiter // 2)},
            )
            self.n_evals_ += int(result.nfev)
            if result.fun < best_val:
                best_val = float(result.fun)
                best_params = result.x.copy()

        for restart in range(self.n_restarts):
            x0 = lower + rng.random(n_params) * (upper - lower)
            result = minimize(
                lambda p: self._objective(p, rainfall, streamflow),
                x0,
                method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={"maxiter": self.maxiter},
            )
            self.n_evals_ += int(result.nfev)
            if result.fun < best_val:
                best_val = float(result.fun)
                best_params = result.x.copy()
            if self.verbose:
                print(f"  Multi-scale restart {restart + 1}/{self.n_restarts}: f(x)={result.fun:.4f}")

        if best_params is None:
            raise RuntimeError("Multi-scale optimization failed to produce parameters")

        self.kernel_params_, self.theta_short_, self.theta_long_ = self._unpack_params(best_params)
        X = build_design_matrix(rainfall, self.kernel_params_, self.kernel_type)
        self.B_short_, self.F_short_ = build_nngp_matrices(len(y_target), self.m, self.theta_short_)
        self.B_long_, self.F_long_ = build_nngp_matrices(len(y_target), self.m, self.theta_long_)

        Y_short = decorrelate(y_target, self.B_short_, self.F_short_, self.m)
        X_short = decorrelate_matrix(X, self.B_short_, self.F_short_, self.m)
        Y_multi = decorrelate(Y_short, self.B_long_, self.F_long_, self.m)
        X_multi = decorrelate_matrix(X_short, self.B_long_, self.F_long_, self.m)

        self.beta_, _, _, _ = np.linalg.lstsq(X_multi, Y_multi, rcond=None)
        pred = X @ self.beta_
        residuals = y_target - pred
        resid_short = decorrelate(residuals, self.B_short_, self.F_short_, self.m)
        resid_multi = decorrelate(resid_short, self.B_long_, self.F_long_, self.m)

        log_det = np.sum(np.log(self.F_short_)) + np.sum(np.log(self.F_long_))
        rss = np.sum(resid_multi**2)
        n = len(y_target)
        self.log_lik_ = -0.5 * (n * np.log(2.0 * np.pi) + log_det + rss)

        max_lag = 0.0
        for k in range(self.K):
            lag = kernel_mean_lag(self.kernel_params_[k, 0], self.kernel_params_[k, 1], self.kernel_type)
            spread = kernel_spread(self.kernel_params_[k, 0], self.kernel_params_[k, 1], self.kernel_type)
            max_lag = max(max_lag, lag + 3 * spread)
        self.warmup_ = int(np.ceil(max_lag))

        n_effective = 3 * self.K + 6
        self.aic_ = compute_aic(self.log_lik_, n_effective)
        self.bic_ = compute_bic(self.log_lik_, n_effective, n)
        self.n_eff_ = estimate_effective_sample_size(residuals)
        self.aic_eff_ = compute_aic_eff(self.log_lik_, n_effective, residuals)
        self.bic_eff_ = compute_bic_eff(self.log_lik_, n_effective, residuals)
        return self

    def predict(self, rainfall: np.ndarray) -> np.ndarray:
        X = build_design_matrix(rainfall, self.kernel_params_, self.kernel_type)
        pred = X @ self.beta_
        if self.log_transform:
            pred = np.exp(pred) - self.epsilon_target
        return pred

    def predict_with_history(self, train_rain: np.ndarray, test_rain: np.ndarray) -> np.ndarray:
        full_rain = np.concatenate([train_rain, test_rain])
        full_pred = self.predict(full_rain)
        return full_pred[len(train_rain):]

    def forecast(self, rainfall: np.ndarray, observed_streamflow: np.ndarray) -> np.ndarray:
        y_target = observed_streamflow
        if self.log_transform:
            y_target = np.log(observed_streamflow + self.epsilon_target)

        X = build_design_matrix(rainfall, self.kernel_params_, self.kernel_type)
        mu = X @ self.beta_
        epsilon = y_target - mu

        B_short, _ = build_nngp_matrices(len(rainfall), self.m, self.theta_short_)
        B_long, _ = build_nngp_matrices(len(rainfall), self.m, self.theta_long_)
        corr_short = _sequential_correction(epsilon, B_short, self.m)
        residual_short = epsilon - corr_short
        corr_long = _sequential_correction(residual_short, B_long, self.m)
        forecast_val = mu + corr_short + corr_long

        if self.log_transform:
            forecast_val = np.exp(forecast_val) - self.epsilon_target
        return forecast_val


def _compute_eval_metrics(
    model,
    rain_train: np.ndarray,
    y_train: np.ndarray,
    rain_test: np.ndarray,
    y_test: np.ndarray,
    sigma_total: float,
) -> Dict:
    train_sim = model.predict(rain_train)
    test_sim = model.predict_with_history(rain_train, rain_test)

    full_rain = np.concatenate([rain_train, rain_test])
    full_y = np.concatenate([y_train, y_test])
    full_krig = model.forecast(full_rain, full_y)
    train_krig = full_krig[: len(rain_train)]
    test_krig = full_krig[len(rain_train):]

    eps = getattr(model, "epsilon_target", 0.01)
    y_train_t = np.log(y_train + eps)
    y_test_t = np.log(y_test + eps)

    train_sim_t = np.log(np.maximum(train_sim + eps, 1e-12))
    test_sim_t = np.log(np.maximum(test_sim + eps, 1e-12))
    train_krig_t = np.log(np.maximum(train_krig + eps, 1e-12))
    test_krig_t = np.log(np.maximum(test_krig + eps, 1e-12))

    train_sim_metrics = _serialize_dict(compute_metrics(y_train, train_sim))
    test_sim_metrics = _serialize_dict(compute_metrics(y_test, test_sim))
    train_krig_metrics = _serialize_dict(compute_metrics(y_train, train_krig))
    test_krig_metrics = _serialize_dict(compute_metrics(y_test, test_krig))

    crps_kwargs = dict(
        family="gaussian",
        sigma=float(max(sigma_total, 1e-8)),
        response_transform="log",
        response_shift=eps,
    )

    train_sim_metrics["CRPS"] = float(
        compute_crps(y_train, train_sim, y_transformed=y_train_t, mu_transformed=train_sim_t, **crps_kwargs)
    )
    test_sim_metrics["CRPS"] = float(
        compute_crps(y_test, test_sim, y_transformed=y_test_t, mu_transformed=test_sim_t, **crps_kwargs)
    )
    train_krig_metrics["CRPS"] = float(
        compute_crps(y_train, train_krig, y_transformed=y_train_t, mu_transformed=train_krig_t, **crps_kwargs)
    )
    test_krig_metrics["CRPS"] = float(
        compute_crps(y_test, test_krig, y_transformed=y_test_t, mu_transformed=test_krig_t, **crps_kwargs)
    )

    return {
        "train_sim": train_sim_metrics,
        "test_sim": test_sim_metrics,
        "train_krig": train_krig_metrics,
        "test_krig": test_krig_metrics,
    }


def run_experiment(
    K: int = 3,
    m: int = 10,
    nu: float = 1.5,
    maxiter: int = 120,
    seed: int = 42,
    test_mode: bool = False,
    results_root: str | None = None,
    quiet: bool = False,
) -> Dict:
    _, logs_dir, metrics_dir, models_dir, plots_dir = ensure_output_dirs_for_root(
        results_root,
        experiment_key="multiscale_covariance",
        script_path=__file__,
    )
    run_ts = timestamp()
    log_file = logs_dir / f"multiscale_covariance_{run_ts}.log"
    logger = setup_logging(log_file)

    logger.info("=" * 72)
    logger.info("MULTI-SCALE RESIDUAL COVARIANCE EXPERIMENT")
    logger.info("=" * 72)
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"K={K}, m={m}, nu={nu}, maxiter={maxiter}, seed={seed}, test_mode={test_mode}")

    train, test = load_bigsur_train_test(DATA_DIR)
    if test_mode:
        train = train.iloc[:1000]
        test = test.iloc[:300]

    rain_train = train["rain"].to_numpy(dtype=float)
    y_train = train["gauge"].to_numpy(dtype=float)
    rain_test = test["rain"].to_numpy(dtype=float)
    y_test = test["gauge"].to_numpy(dtype=float)

    logger.info(f"Train size={len(train)}, Test size={len(test)}")

    baseline = GPSWR(
        K=K,
        m=m,
        nu=nu,
        maxiter=maxiter,
        log_transform=True,
        seed=seed,
        verbose=not quiet,
    )
    t0 = time.time()
    baseline.fit(rain_train, y_train)
    baseline_fit_time = time.time() - t0

    multiscale = MultiScaleResidualGPSWR(
        K=K,
        m=m,
        nu=nu,
        maxiter=maxiter,
        n_restarts=5 if not test_mode else 2,
        log_transform=True,
        seed=seed,
        verbose=not quiet,
    )
    t1 = time.time()
    multiscale.fit(rain_train, y_train, baseline_model=baseline)
    multiscale_fit_time = time.time() - t1

    sigma_baseline = float(np.sqrt(max(baseline.theta_[0] + baseline.theta_[3], 1e-12)))
    sigma_multiscale = float(
        np.sqrt(
            max(
                multiscale.theta_short_[0]
                + multiscale.theta_short_[3]
                + multiscale.theta_long_[0]
                + multiscale.theta_long_[3],
                1e-12,
            )
        )
    )

    baseline_metrics = _compute_eval_metrics(
        baseline,
        rain_train,
        y_train,
        rain_test,
        y_test,
        sigma_total=sigma_baseline,
    )
    multiscale_metrics = _compute_eval_metrics(
        multiscale,
        rain_train,
        y_train,
        rain_test,
        y_test,
        sigma_total=sigma_multiscale,
    )

    improvement = {
        "test_sim_nse_delta": float(multiscale_metrics["test_sim"]["NSE"] - baseline_metrics["test_sim"]["NSE"]),
        "test_krig_nse_delta": float(multiscale_metrics["test_krig"]["NSE"] - baseline_metrics["test_krig"]["NSE"]),
        "test_sim_crps_delta": float(multiscale_metrics["test_sim"]["CRPS"] - baseline_metrics["test_sim"]["CRPS"]),
        "test_krig_crps_delta": float(multiscale_metrics["test_krig"]["CRPS"] - baseline_metrics["test_krig"]["CRPS"]),
    }

    results = {
        "experiment": "multiscale_residual_covariance",
        "config": {
            "K": int(K),
            "m": int(m),
            "nu": float(nu),
            "maxiter": int(maxiter),
            "seed": int(seed),
            "test_mode": bool(test_mode),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
        "baseline_single_scale": {
            "log_lik": float(baseline.log_lik_),
            "aic": float(baseline.aic_),
            "aic_eff": float(baseline.aic_eff_),
            "bic": float(baseline.bic_),
            "bic_eff": float(baseline.bic_eff_),
            "n_eff": float(baseline.n_eff_),
            "fit_time_seconds": float(baseline_fit_time),
            "theta": [float(v) for v in baseline.theta_],
            "metrics": baseline_metrics,
        },
        "multiscale_residual": {
            "log_lik": float(multiscale.log_lik_),
            "aic": float(multiscale.aic_),
            "aic_eff": float(multiscale.aic_eff_),
            "bic": float(multiscale.bic_),
            "bic_eff": float(multiscale.bic_eff_),
            "n_eff": float(multiscale.n_eff_),
            "fit_time_seconds": float(multiscale_fit_time),
            "theta_short": [float(v) for v in multiscale.theta_short_],
            "theta_long": [float(v) for v in multiscale.theta_long_],
            "metrics": multiscale_metrics,
        },
        "improvement": improvement,
    }

    metrics_file = metrics_dir / "multiscale_covariance_results.json"
    metrics_archive = metrics_dir / f"multiscale_covariance_results_{run_ts}.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(metrics_archive, "w") as f:
        json.dump(results, f, indent=2)

    with open(models_dir / f"baseline_single_scale_K{K}.pkl", "wb") as f:
        pickle.dump(baseline, f)
    with open(models_dir / f"multiscale_residual_K{K}.pkl", "wb") as f:
        pickle.dump(multiscale, f)

    labels = ["Sim NSE", "Krig NSE", "Sim CRPS", "Krig CRPS"]
    base_vals = [
        baseline_metrics["test_sim"]["NSE"],
        baseline_metrics["test_krig"]["NSE"],
        baseline_metrics["test_sim"]["CRPS"],
        baseline_metrics["test_krig"]["CRPS"],
    ]
    multi_vals = [
        multiscale_metrics["test_sim"]["NSE"],
        multiscale_metrics["test_krig"]["NSE"],
        multiscale_metrics["test_sim"]["CRPS"],
        multiscale_metrics["test_krig"]["CRPS"],
    ]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.bar(x - width / 2, base_vals, width, label="Single-scale")
    ax.bar(x + width / 2, multi_vals, width, label="Multi-scale")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Test Metrics: Single-scale vs Multi-scale Residual Covariance")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_file = plots_dir / f"multiscale_covariance_comparison_K{K}_{run_ts}.png"
    fig.savefig(plot_file)
    plt.close(fig)

    logger.info("\nSingle-scale (test): NSE(sim)=%.4f, NSE(krig)=%.4f, CRPS(sim)=%.4f, CRPS(krig)=%.4f",
                baseline_metrics["test_sim"]["NSE"],
                baseline_metrics["test_krig"]["NSE"],
                baseline_metrics["test_sim"]["CRPS"],
                baseline_metrics["test_krig"]["CRPS"])
    logger.info("Multi-scale  (test): NSE(sim)=%.4f, NSE(krig)=%.4f, CRPS(sim)=%.4f, CRPS(krig)=%.4f",
                multiscale_metrics["test_sim"]["NSE"],
                multiscale_metrics["test_krig"]["NSE"],
                multiscale_metrics["test_sim"]["CRPS"],
                multiscale_metrics["test_krig"]["CRPS"])
    logger.info("Deltas (multi - single): NSE(sim)=%.4f, NSE(krig)=%.4f, CRPS(sim)=%.4f, CRPS(krig)=%.4f",
                improvement["test_sim_nse_delta"],
                improvement["test_krig_nse_delta"],
                improvement["test_sim_crps_delta"],
                improvement["test_krig_crps_delta"])

    logger.info(f"Results saved: {metrics_file}")
    logger.info(f"Archived results: {metrics_archive}")
    logger.info(f"Comparison plot: {plot_file}")
    logger.info(f"Log: {log_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare single-scale vs multi-scale residual covariance")
    parser.add_argument("--K", type=int, default=3, help="Number of kernels (default: 3)")
    parser.add_argument("--m", type=int, default=10, help="NNGP neighbors (default: 10)")
    parser.add_argument("--nu", type=float, default=1.5, help="Matérn smoothness (default: 1.5)")
    parser.add_argument("--maxiter", type=int, default=120, help="Optimizer max iterations (default: 120)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--results-root", type=str, default=None, help="Optional output root")
    parser.add_argument("--test", action="store_true", help="Use reduced train/test subsets")
    parser.add_argument("--quiet", action="store_true", help="Reduce optimizer verbosity")
    args = parser.parse_args()

    run_experiment(
        K=args.K,
        m=args.m,
        nu=args.nu,
        maxiter=args.maxiter,
        seed=args.seed,
        test_mode=args.test,
        results_root=args.results_root,
        quiet=args.quiet,
    )
