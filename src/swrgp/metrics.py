"""
Metrics module - matches SWR (Schrunner et al., 2023) exactly.
"""

import numpy as np


def compute_metrics(y_obs: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute hydrological metrics matching SWR implementation.
    
    Parameters
    ----------
    y_obs : array (n,)
        Observed values
    y_pred : array (n,)
        Predicted values
        
    Returns
    -------
    dict with: RMSE, NRMSE, MAE, NSE, Bias, PBIAS, KGE, RelErr
    """
    # Remove NaN values
    mask = ~(np.isnan(y_obs) | np.isnan(y_pred))
    y_obs = y_obs[mask]
    y_pred = y_pred[mask]
    
    n = len(y_obs)
    
    # Residuals
    residuals = y_obs - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # NRMSE (normalized by std)
    nrmse = rmse / np.std(y_obs)
    
    # MAE
    mae = np.mean(np.abs(residuals))

    # Relative error (multiplicative-scale friendly):
    # mean absolute relative error with small stabilization.
    rel_err = np.mean(np.abs(residuals) / np.maximum(np.abs(y_obs), 1e-8))
    
    # NSE (Nash-Sutcliffe Efficiency)
    nse = 1 - ss_res / ss_tot
    
    # Bias
    bias = np.mean(residuals)
    
    # Percent Bias
    pbias = 100 * np.sum(residuals) / np.sum(y_obs)
    
    # KGE (Kling-Gupta Efficiency)
    r_corr = np.corrcoef(y_obs, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_obs)
    beta_kge = np.mean(y_pred) / np.mean(y_obs)
    kge = 1 - np.sqrt((r_corr - 1)**2 + (alpha - 1)**2 + (beta_kge - 1)**2)
    
    return {
        'RMSE': rmse,
        'NRMSE': nrmse,
        'MAE': mae,
        'RelErr': rel_err,
        'NSE': nse,
        'Bias': bias,
        'PBIAS': pbias,
        'KGE': kge,
        'N': n
    }


def compute_aic(log_lik: float, n_params: int) -> float:
    """AIC = -2 * log_lik + 2 * k"""
    return -2 * log_lik + 2 * n_params


def compute_bic(log_lik: float, n_params: int, n_obs: int) -> float:
    """BIC = -2 * log_lik + k * log(n)"""
    return -2 * log_lik + n_params * np.log(n_obs)


def estimate_effective_sample_size(residuals: np.ndarray, max_lag: int | None = None) -> float:
    """Estimate effective sample size from residual autocorrelation.

    Uses positive-sequence truncation:
        n_eff = n / (1 + 2 * sum_{h=1}^H rho_h),
    with H at the first non-positive lag autocorrelation.
    """
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    n = residuals.size
    if n < 5:
        return float(n)

    if max_lag is None:
        max_lag = min(100, n // 4)
    max_lag = max(1, min(max_lag, n - 1))

    centered = residuals - np.mean(residuals)
    var = np.dot(centered, centered) / n
    if not np.isfinite(var) or var <= 0:
        return float(n)

    rho = []
    for lag in range(1, max_lag + 1):
        cov = np.dot(centered[:-lag], centered[lag:]) / n
        r = cov / var
        if not np.isfinite(r):
            continue
        if r <= 0:
            break
        rho.append(r)

    if not rho:
        return float(n)

    denom = 1.0 + 2.0 * float(np.sum(rho))
    if not np.isfinite(denom) or denom <= 0:
        return float(n)

    n_eff = n / denom
    n_eff = max(2.0, min(float(n), float(n_eff)))
    return float(n_eff)


def compute_aic_eff(log_lik: float, n_params: int, residuals: np.ndarray) -> float:
    """Autocorrelation-adjusted AIC (SWR-style):

    AIC_eff = -2 * log_lik + 2 * n_params * (n / n_eff)
    where n_eff is estimated from residual autocorrelation.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = int(np.sum(np.isfinite(residuals)))
    if n <= 0:
        return float("nan")
    n_eff = estimate_effective_sample_size(residuals)
    return -2 * log_lik + 2 * n_params * (n / max(n_eff, 1.0))


def compute_bic_eff(log_lik: float, n_params: int, residuals: np.ndarray) -> float:
    """Autocorrelation-adjusted BIC (SWR-style):

    BIC_eff = -2 * log_lik + n_params * log(max(n_eff, 2))
    """
    n_eff = estimate_effective_sample_size(residuals)
    return -2 * log_lik + n_params * np.log(max(n_eff, 2.0))


def compute_crps(
    y: np.ndarray,
    mu: np.ndarray,
    family: str = "gaussian",
    sigma: float | np.ndarray | None = None,
    response_transform: str = "none",
    response_shift: float = 0.0,
    response_lambda: float | None = None,
    y_transformed: np.ndarray | None = None,
    mu_transformed: np.ndarray | None = None,
    n_mc: int = 200,
    seed: int | None = None,
) -> float:
    """SWR-aligned CRPS for Gaussian family on original scale.

    - `response_transform='none'`: analytic Normal CRPS.
    - `response_transform in {'log','sqrt','boxcox'}`: Monte Carlo on transformed
      scale with inverse transform to original scale.

    sigma may be a scalar or a per-observation array matching y.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if y.shape != mu.shape:
        raise ValueError("y and mu must have matching shape")

    # Broadcast sigma to array early so it can be filtered with valid mask.
    if sigma is not None:
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape).copy()

    valid = np.isfinite(y) & np.isfinite(mu)
    if sigma is not None:
        valid = valid & np.isfinite(sigma)
    if y_transformed is not None:
        y_transformed = np.asarray(y_transformed, dtype=float)
        valid = valid & np.isfinite(y_transformed)
    if mu_transformed is not None:
        mu_transformed = np.asarray(mu_transformed, dtype=float)
        valid = valid & np.isfinite(mu_transformed)

    y = y[valid]
    mu = mu[valid]
    if sigma is not None:
        sigma = sigma[valid]
    if y_transformed is not None:
        y_transformed = y_transformed[valid]
    if mu_transformed is not None:
        mu_transformed = mu_transformed[valid]

    if y.size == 0:
        return float("nan")

    if family.lower() != "gaussian":
        raise ValueError("compute_crps currently supports family='gaussian' only")

    if sigma is None:
        sigma = np.full(y.shape, np.sqrt(np.mean((y - mu) ** 2)), dtype=float)
    sigma = np.maximum(sigma, 1e-6)

    transform = response_transform.lower()
    if transform == "none":
        z = (y - mu) / sigma
        phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
        from scipy.special import erf as _erf
        Phi = 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
        crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
        return float(np.mean(crps))

    rng = np.random.default_rng(seed)

    if mu_transformed is None:
        if transform == "log":
            mu_transformed = np.log(np.maximum(mu + response_shift, 1e-12))
        elif transform == "sqrt":
            mu_transformed = np.sqrt(np.maximum(mu + response_shift, 0.0))
        elif transform == "boxcox":
            if response_lambda is None:
                raise ValueError("response_lambda is required for Box-Cox CRPS")
            mu_shifted = mu + response_shift
            if abs(response_lambda) < 1e-4:
                mu_transformed = np.log(np.maximum(mu_shifted, 1e-12))
            else:
                mu_transformed = (np.maximum(mu_shifted, 1e-12) ** response_lambda - 1.0) / response_lambda
        else:
            raise ValueError(f"Unsupported response_transform: {response_transform}")

    def _inv_transform(samples_trans: np.ndarray) -> np.ndarray:
        if transform == "log":
            return np.exp(samples_trans) - response_shift
        if transform == "sqrt":
            return samples_trans**2 - response_shift
        if transform == "boxcox":
            if response_lambda is None:
                raise ValueError("response_lambda is required for Box-Cox CRPS")
            if abs(response_lambda) < 1e-4:
                return np.exp(samples_trans) - response_shift
            inner = np.maximum(response_lambda * samples_trans + 1.0, 1e-12)
            return inner ** (1.0 / response_lambda) - response_shift
        raise ValueError(f"Unsupported response_transform: {response_transform}")

    def _crps_mc(samples: np.ndarray, y_obs: float) -> float:
        e1 = np.mean(np.abs(samples - y_obs))
        e2 = np.mean(np.abs(samples - rng.choice(samples, size=samples.size, replace=True)))
        return float(e1 - 0.5 * e2)

    crps_vals = np.empty(y.size, dtype=float)
    for i in range(y.size):
        samples_t = rng.normal(loc=mu_transformed[i], scale=sigma[i], size=n_mc)
        samples = _inv_transform(samples_t)
        samples = samples[np.isfinite(samples)]
        if samples.size < 10:
            crps_vals[i] = np.nan
        else:
            crps_vals[i] = _crps_mc(samples, y[i])

    return float(np.nanmean(crps_vals))
