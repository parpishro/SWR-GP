"""
Kernels module - Gamma kernel convolution matching SWR (Schrunner et al., 2023).
"""

import numpy as np
from scipy.stats import gamma
from scipy.signal import fftconvolve


def build_kernel(log_delta: float, log_sigma: float, 
                 kernel_type: str = "gamma", max_length: int = 365) -> np.ndarray:
    """
    Build a normalized kernel via CDF differences.
    
    Parameters
    ----------
    log_delta : float
        Log of shape parameter
    log_sigma : float  
        Log of scale parameter
    kernel_type : str
        "gamma", "gaussian", or "triangle"
    max_length : int
        Maximum kernel length
        
    Returns
    -------
    kernel : array
        Normalized kernel weights (sum = 1)
    """
    delta = np.exp(log_delta)
    sigma = np.exp(log_sigma)
    
    if kernel_type == "gamma":
        # CDF differences for discretization
        lags = np.arange(max_length + 1)
        cdf_vals = gamma.cdf(lags, a=delta, scale=sigma)
        kernel = np.diff(cdf_vals)
        
        # Truncate where kernel effectively zero
        cumsum = np.cumsum(kernel)
        cutoff = np.searchsorted(cumsum, 0.999) + 1
        kernel = kernel[:cutoff]
        
    elif kernel_type == "gaussian":
        # Gaussian kernel parameterized directly by center and spread.
        mean = delta
        std = sigma
        lags = np.arange(int(mean + 4*std) + 1)
        kernel = np.exp(-0.5 * ((lags - mean) / std) ** 2)
        
    elif kernel_type == "triangle":
        # Triangle kernel
        peak = delta * sigma
        width = sigma
        lags = np.arange(int(peak + width) + 1)
        kernel = np.maximum(0, 1 - np.abs(lags - peak) / width)
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    return kernel


def convolve_kernel(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Causal convolution of input with kernel.
    
    Parameters
    ----------
    x : array (n,)
        Input time series
    kernel : array (L,)
        Kernel weights
        
    Returns
    -------
    y : array (n,)
        Convolved output
    """
    n = len(x)
    
    # FFT convolution (much faster for long kernels)
    full_conv = fftconvolve(x, kernel, mode='full')
    
    # Take first n elements (causal)
    return full_conv[:n]


def build_design_matrix(x: np.ndarray, 
                        kernel_params: np.ndarray,
                        kernel_type: str = "gamma",
                        max_length: int = 365) -> np.ndarray:
    """
    Build design matrix from kernel convolutions.
    
    Parameters
    ----------
    x : array (n,)
        Rainfall time series
    kernel_params : array (K, 2)
        Each row is [log_delta, log_sigma]
    kernel_type : str
        Kernel type
    max_length : int
        Maximum kernel length used when building each kernel
        
    Returns
    -------
    X : array (n, K)
        Design matrix with convolved signals
    """
    n = len(x)
    K = len(kernel_params)
    X = np.zeros((n, K))
    
    for k in range(K):
        kernel = build_kernel(
            kernel_params[k, 0],
            kernel_params[k, 1],
            kernel_type,
            max_length=max_length,
        )
        X[:, k] = convolve_kernel(x, kernel)
    
    return X


def sort_kernel_params(kernel_params: np.ndarray,
                       kernel_type: str = "gamma") -> np.ndarray:
    """Sort kernel parameters by mean lag."""
    if len(kernel_params) <= 1:
        return np.array(kernel_params, copy=True)
    order = np.argsort([
        kernel_mean_lag(row[0], row[1], kernel_type)
        for row in kernel_params
    ])
    return np.array(kernel_params, copy=True)[order]


def max_pairwise_kernel_overlap(kernel_params: np.ndarray,
                                kernel_type: str = "gaussian") -> float:
    """Maximum pairwise overlap between normalized kernels."""
    kernel_params = sort_kernel_params(kernel_params, kernel_type)
    K = kernel_params.shape[0]
    if K < 2:
        return 0.0

    kernels = [
        build_kernel(kernel_params[k, 0], kernel_params[k, 1], kernel_type)
        for k in range(K)
    ]
    max_overlap = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            L = max(len(kernels[i]), len(kernels[j]))
            ki = np.zeros(L)
            kj = np.zeros(L)
            ki[:len(kernels[i])] = kernels[i]
            kj[:len(kernels[j])] = kernels[j]
            overlap = float(np.sum(np.minimum(ki, kj)))
            max_overlap = max(max_overlap, overlap)
    return max_overlap


def gaussian_kernel_constraints_satisfied(
    kernel_params: np.ndarray,
    min_mean_lag_separation: float = 0.0,
    max_overlap: float | None = None,
    support_sigma_multiple: float | None = None,
    support_lag_range: tuple[float, float] | None = None,
) -> bool:
    """Check the Gaussian-kernel geometry used in simulation and fitting."""
    kernel_params = sort_kernel_params(kernel_params, "gaussian")
    centers = np.exp(kernel_params[:, 0])
    spreads = np.exp(kernel_params[:, 1])

    if np.any(spreads <= 0.0):
        return False

    if support_sigma_multiple is not None and support_lag_range is not None:
        support_min, support_max = support_lag_range
        left = centers - support_sigma_multiple * spreads
        right = centers + support_sigma_multiple * spreads
        if np.any(left < support_min) or np.any(right > support_max):
            return False

    if len(centers) > 1 and min_mean_lag_separation > 0.0:
        if np.any(np.diff(centers) < min_mean_lag_separation):
            return False

    if max_overlap is not None and max_pairwise_kernel_overlap(kernel_params, "gaussian") > max_overlap:
        return False

    return True


def sample_constrained_gaussian_kernel_params(
    rng: np.random.Generator,
    K: int,
    spread_range: tuple[float, float] = (2.0, 4.0),
    min_mean_lag_separation: float = 10.0,
    max_overlap: float = 0.05,
    support_sigma_multiple: float = 3.0,
    support_lag_range: tuple[float, float] = (0.0, 90.0),
    max_attempts: int = 2000,
) -> np.ndarray:
    """Sample ordered Gaussian kernels satisfying the shared geometry constraints."""
    support_min, support_max = support_lag_range

    for _ in range(max_attempts):
        spreads = rng.uniform(spread_range[0], spread_range[1], size=K)
        lo = np.maximum(support_min + support_sigma_multiple * spreads, 4.0)
        hi = support_max - support_sigma_multiple * spreads
        if np.any(lo >= hi):
            continue
        centers = np.sort(rng.uniform(lo, hi))
        kernel_params = np.column_stack([np.log(centers), np.log(spreads)])
        if gaussian_kernel_constraints_satisfied(
            kernel_params,
            min_mean_lag_separation=min_mean_lag_separation,
            max_overlap=max_overlap,
            support_sigma_multiple=support_sigma_multiple,
            support_lag_range=support_lag_range,
        ):
            return kernel_params

    raise ValueError(
        f"Could not sample {K} constrained Gaussian kernels after {max_attempts} attempts"
    )


def kernel_mean_lag(log_delta: float, log_sigma: float, 
                    kernel_type: str = "gamma") -> float:
    """Mean lag (days) for kernel."""
    delta = np.exp(log_delta)
    sigma = np.exp(log_sigma)
    if kernel_type == "gaussian":
        return delta
    return delta * sigma


def kernel_spread(log_delta: float, log_sigma: float,
                  kernel_type: str = "gamma") -> float:
    """Spread (std dev in days) for kernel."""
    delta = np.exp(log_delta)
    sigma = np.exp(log_sigma)
    if kernel_type == "gaussian":
        return sigma
    return np.sqrt(delta) * sigma
