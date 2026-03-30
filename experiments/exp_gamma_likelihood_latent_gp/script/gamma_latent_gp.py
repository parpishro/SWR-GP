"""Prototype Gamma-likelihood latent GP experiment for Big Sur.

Keeps GP-SWR latent structure (mean + NNGP covariance) and swaps
observation model from Gaussian to Gamma with log-link mean.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import gamma as gamma_dist

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from swrgp.bigsur_data import load_bigsur_train_test
from swrgp.kernels import build_design_matrix
from swrgp.metrics import compute_metrics
from swrgp.model import GPSWR
from swrgp.nngp import build_nngp_matrices
from swrgp.paths import DATA_DIR, ensure_output_dirs_for_root, timestamp


def _latent_forecast_path(model: GPSWR, rainfall: np.ndarray, observed_streamflow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return latent kriging mean path eta_t and NNGP conditional variance F_t."""
    if not model.log_transform:
        raise ValueError("Model must be fitted with log_transform=True for latent gamma prototype")

    y_target = np.log(np.maximum(observed_streamflow + model.epsilon_target, 1e-12))
    X = build_design_matrix(rainfall, model.kernel_params_, model.kernel_type)
    mu = X @ model.beta_

    B, F = build_nngp_matrices(len(rainfall), model.m, model.theta_)
    epsilon = y_target - mu
    correction = np.zeros(len(rainfall), dtype=float)

    for t in range(len(rainfall)):
        nn = min(t, model.m)
        for j in range(nn):
            neighbor_idx = t - nn + j
            correction[t] += B[t, model.m - nn + j] * epsilon[neighbor_idx]

    eta = mu + correction
    return eta, F


def _fit_gamma_shape(y: np.ndarray, eta: np.ndarray) -> float:
    """Fit Gamma shape alpha with mean exp(eta), scale=mean/alpha by MLE."""
    y_pos = np.maximum(np.asarray(y, dtype=float), 1e-10)
    mean_obs = np.maximum(np.exp(np.asarray(eta, dtype=float)), 1e-10)

    def objective(log_alpha: float) -> float:
        alpha = float(np.exp(log_alpha))
        scale = mean_obs / alpha
        ll = gamma_dist.logpdf(y_pos, a=alpha, scale=scale)
        if not np.all(np.isfinite(ll)):
            return 1e12
        return float(-np.sum(ll))

    result = minimize_scalar(objective, bounds=(-6.0, 8.0), method="bounded")
    return float(np.exp(result.x))


def _sample_gamma_predictive(
    eta: np.ndarray,
    f_var: np.ndarray,
    alpha: float,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample predictive Y from latent-normal + gamma-observation hierarchy."""
    rng = np.random.default_rng(seed)
    n = eta.size
    samples = np.empty((n, n_samples), dtype=float)

    std = np.sqrt(np.maximum(f_var, 1e-12))
    for i in range(n):
        eta_s = rng.normal(loc=eta[i], scale=std[i], size=n_samples)
        mean_s = np.exp(eta_s)
        scale_s = np.maximum(mean_s / alpha, 1e-12)
        samples[i, :] = rng.gamma(shape=alpha, scale=scale_s)

    return samples


def _empirical_crps(y_obs: np.ndarray, pred_samples: np.ndarray) -> float:
    """Empirical CRPS from forecast samples: E|X-y| - 0.5 E|X-X'|."""
    y_obs = np.asarray(y_obs, dtype=float)
    if pred_samples.shape[0] != y_obs.size:
        raise ValueError("pred_samples first dimension must match y_obs length")

    n = y_obs.size
    vals = np.empty(n, dtype=float)

    for i in range(n):
        s = pred_samples[i]
        e1 = np.mean(np.abs(s - y_obs[i]))
        e2 = np.mean(np.abs(s[:, None] - s[None, :]))
        vals[i] = e1 - 0.5 * e2

    return float(np.mean(vals))


def run_gamma_latent_gp(
    K: int = 3,
    m: int = 10,
    nu: float = 1.5,
    maxiter: int = 120,
    n_samples: int = 1000,
    seed: int = 42,
    test_mode: bool = False,
    results_root: str | None = None,
    verbose: bool = True,
) -> Dict:
    _, logs_dir, metrics_dir, models_dir, plots_dir = ensure_output_dirs_for_root(
        results_root,
        experiment_key="gamma_latent_gp",
        script_path=__file__,
    )
    run_ts = timestamp()

    train, test = load_bigsur_train_test(DATA_DIR)
    if test_mode:
        train = train.iloc[:1000]
        test = test.iloc[:300]

    rain_train = train["rain"].to_numpy(dtype=float)
    y_train = train["gauge"].to_numpy(dtype=float)
    rain_test = test["rain"].to_numpy(dtype=float)
    y_test = test["gauge"].to_numpy(dtype=float)

    rain_full = np.concatenate([rain_train, rain_test])
    y_full = np.concatenate([y_train, y_test])

    model = GPSWR(
        K=K,
        m=m,
        nu=nu,
        maxiter=maxiter,
        log_transform=True,
        seed=seed,
        verbose=verbose,
    )
    model.fit(rain_train, y_train)

    # Baseline Gaussian-observation kriging (existing model behavior)
    baseline_pred = model.forecast(rain_full, y_full)[len(rain_train):]

    eta_train, f_train = _latent_forecast_path(model, rain_train, y_train)
    eta_full, f_full = _latent_forecast_path(model, rain_full, y_full)
    eta_test = eta_full[len(rain_train):]
    f_test = f_full[len(rain_train):]

    alpha_hat = _fit_gamma_shape(y_train, eta_train)

    point_gamma = np.exp(eta_test)
    samples_gamma = _sample_gamma_predictive(
        eta=eta_test,
        f_var=f_test,
        alpha=alpha_hat,
        n_samples=n_samples,
        seed=seed,
    )

    lo = np.quantile(samples_gamma, 0.025, axis=1)
    hi = np.quantile(samples_gamma, 0.975, axis=1)
    coverage95 = float(np.mean((y_test >= lo) & (y_test <= hi)))
    width95 = float(np.mean(hi - lo))

    metrics_gamma = compute_metrics(y_test, point_gamma)
    metrics_gamma["CRPS"] = _empirical_crps(y_test, samples_gamma)

    metrics_baseline = compute_metrics(y_test, baseline_pred)

    output = {
        "experiment": "gamma_likelihood_latent_gp_prototype",
        "config": {
            "K": K,
            "m": m,
            "nu": nu,
            "maxiter": maxiter,
            "n_samples": n_samples,
            "seed": seed,
            "test_mode": test_mode,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
        "fitted": {
            "alpha_shape": float(alpha_hat),
            "theta": [float(x) for x in model.theta_],
            "epsilon_target": float(model.epsilon_target),
        },
        "results": {
            "baseline_log_gaussian_kriging": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics_baseline.items()},
            "gamma_latent_gp": {
                **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics_gamma.items()},
                "coverage_95": coverage95,
                "pi95_mean_width": width95,
            },
        },
    }

    model_file = models_dir / "model_gamma_latent_gp_base.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    metrics_file = metrics_dir / "gamma_latent_gp_results.json"
    archive_file = metrics_dir / f"gamma_latent_gp_results_{run_ts}.json"
    with open(metrics_file, "w") as f:
        json.dump(output, f, indent=2)
    with open(archive_file, "w") as f:
        json.dump(output, f, indent=2)

    # Save compact NPZ for optional downstream diagnostics
    samples_file = plots_dir / f"gamma_latent_gp_predictive_samples_{run_ts}.npz"
    np.savez_compressed(
        samples_file,
        y_test=y_test,
        baseline_pred=baseline_pred,
        gamma_point_pred=point_gamma,
        pi95_low=lo,
        pi95_high=hi,
    )

    log_file = logs_dir / f"gamma_latent_gp_{run_ts}.log"
    with open(log_file, "w") as f:
        f.write(json.dumps(output["config"], indent=2) + "\n\n")
        f.write(json.dumps(output["fitted"], indent=2) + "\n\n")
        f.write(json.dumps(output["results"], indent=2) + "\n")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype Gamma-likelihood latent GP experiment")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output = run_gamma_latent_gp(
        K=args.K,
        m=args.m,
        nu=args.nu,
        maxiter=args.maxiter,
        n_samples=args.n_samples,
        seed=args.seed,
        test_mode=args.test,
        results_root=args.results_root,
        verbose=not args.quiet,
    )

    print(json.dumps(output["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
