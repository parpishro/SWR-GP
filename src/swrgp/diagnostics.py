"""Diagnostic utilities (RQR/PIT) shared across experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


def _clip_pit(pit: np.ndarray) -> np.ndarray:
    return np.clip(pit, 1e-10, 1 - 1e-10)


def _broadcast_sigma(sigma: float | np.ndarray, n: int) -> np.ndarray:
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        sigma_vec = np.full(n, max(float(sigma_arr), 1e-8), dtype=float)
    elif sigma_arr.shape == (n,):
        sigma_vec = np.maximum(sigma_arr, 1e-8)
    else:
        raise ValueError("sigma must be scalar or shape-(n,) array")
    return sigma_vec


def summarize_pit_rqr(pit: np.ndarray, rqr: np.ndarray) -> dict:
    pit = np.asarray(pit, dtype=float)
    rqr = np.asarray(rqr, dtype=float)
    return {
        "pit_mean": float(np.nanmean(pit)) if pit.size else float("nan"),
        "pit_var": float(np.nanvar(pit)) if pit.size else float("nan"),
        "rqr_mean": float(np.nanmean(rqr)) if rqr.size else float("nan"),
        "rqr_std": float(np.nanstd(rqr)) if rqr.size else float("nan"),
    }


def compute_pit_rqr_gaussian(y: np.ndarray, mu: np.ndarray, sigma: float | np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if y.shape != mu.shape:
        raise ValueError("y and mu must have matching shape")

    sigma_vec = _broadcast_sigma(sigma, y.size)

    valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma_vec)
    yv = y[valid]
    muv = mu[valid]
    sigv = sigma_vec[valid]

    pit = norm.cdf(yv, loc=muv, scale=sigv)
    pit = _clip_pit(pit)
    rqr = norm.ppf(pit)

    return {
        "pit": pit,
        "rqr": rqr,
        "n_valid": int(yv.size),
    }


def compute_gaussian_calibration(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: float | np.ndarray,
    *,
    transform: str,
    lam: float,
    shift: float,
    mu_transformed: np.ndarray | None = None,
) -> dict:
    """Compute PIT/RQR calibration diagnostics under Gaussian predictive assumption.

    For `transform="none"`, diagnostics are computed directly on original response scale.
    For `transform="boxcox"`, diagnostics are reported on original scale via the
    transformed Gaussian CDF mapping F_Y(y) = Phi((g(y+shift)-mu_t)/sigma_t).
    """
    if transform == "boxcox":
        if mu_transformed is None:
            raise ValueError("mu_transformed is required when transform='boxcox'")
        out = compute_pit_rqr_boxcox_gaussian(
            y=y,
            mu_t=mu_transformed,
            sigma_t=sigma,
            lam=lam,
            shift=shift,
        )
    else:
        out = compute_pit_rqr_gaussian(y=y, mu=mu, sigma=sigma)

    out.update(summarize_pit_rqr(out["pit"], out["rqr"]))
    return out


def compute_standardized_innovations(
    y_transformed: np.ndarray,
    mu_transformed: np.ndarray,
    sigma_transformed: float | np.ndarray,
) -> dict:
    """Compute standardized innovations on transformed (Gaussian) scale."""
    y_t = np.asarray(y_transformed, dtype=float)
    mu_t = np.asarray(mu_transformed, dtype=float)
    if y_t.shape != mu_t.shape:
        raise ValueError("y_transformed and mu_transformed must have matching shape")

    sigma_vec = _broadcast_sigma(sigma_transformed, y_t.size)
    valid = np.isfinite(y_t) & np.isfinite(mu_t) & np.isfinite(sigma_vec)
    z = (y_t[valid] - mu_t[valid]) / sigma_vec[valid]
    z = z[np.isfinite(z)]

    if z.size:
        mean = float(np.nanmean(z))
        std = float(np.nanstd(z))
        skew = float(np.nanmean(((z - mean) / max(std, 1e-12)) ** 3))
        kurt = float(np.nanmean(((z - mean) / max(std, 1e-12)) ** 4) - 3.0)
    else:
        mean = float("nan")
        std = float("nan")
        skew = float("nan")
        kurt = float("nan")

    return {
        "z": z,
        "n_valid": int(z.size),
        "z_mean": mean,
        "z_std": std,
        "z_skew": skew,
        "z_excess_kurtosis": kurt,
    }


def compute_pit_rqr_boxcox_gaussian(
    y: np.ndarray,
    mu_t: np.ndarray,
    sigma_t: float | np.ndarray,
    lam: float,
    shift: float,
) -> dict:
    """PIT/RQR for Box-Cox Gaussian predictive model on original response scale.

    If Z ~ N(mu_t, sigma_t^2) on transformed scale and Y = g^{-1}(Z) - shift,
    then F_Y(y) = Phi((g(y + shift) - mu_t) / sigma_t), where g is Box-Cox.
    """
    y = np.asarray(y, dtype=float)
    mu_t = np.asarray(mu_t, dtype=float)
    if y.shape != mu_t.shape:
        raise ValueError("y and mu_t must have matching shape")

    sigma_vec = _broadcast_sigma(sigma_t, y.size)

    y_shifted = y + float(shift)
    if abs(float(lam)) < 1e-8:
        y_t = np.log(np.maximum(y_shifted, 1e-12))
    else:
        y_t = (np.maximum(y_shifted, 1e-12) ** float(lam) - 1.0) / float(lam)

    return compute_pit_rqr_gaussian(y_t, mu_t, sigma_vec)


def fit_gamma_shape_mle(y: np.ndarray, mu: np.ndarray) -> float:
    """Estimate gamma shape alpha under mean parameterization E[Y]=mu.

    Uses a simple profile-likelihood grid on log(alpha) for numerical robustness.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    valid = np.isfinite(y) & np.isfinite(mu) & (y > 0) & (mu > 0)
    yv = y[valid]
    muv = mu[valid]

    if yv.size < 20:
        return 5.0

    log_alpha_grid = np.linspace(np.log(0.2), np.log(200.0), 220)
    best_alpha = 5.0
    best_ll = -np.inf

    for log_alpha in log_alpha_grid:
        alpha = float(np.exp(log_alpha))
        scale = np.maximum(muv / alpha, 1e-12)
        ll = float(np.sum(gamma_dist.logpdf(yv, a=alpha, scale=scale)))
        if np.isfinite(ll) and ll > best_ll:
            best_ll = ll
            best_alpha = alpha

    return float(best_alpha)


def compute_pit_rqr_gamma(y: np.ndarray, mu: np.ndarray, alpha: float) -> dict:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    alpha = max(float(alpha), 1e-8)

    valid = np.isfinite(y) & np.isfinite(mu) & (y > 0) & (mu > 0)
    yv = y[valid]
    muv = mu[valid]

    scale = np.maximum(muv / alpha, 1e-12)
    pit = gamma_dist.cdf(yv, a=alpha, scale=scale)
    pit = _clip_pit(pit)
    rqr = norm.ppf(pit)

    return {
        "pit": pit,
        "rqr": rqr,
        "n_valid": int(yv.size),
    }
