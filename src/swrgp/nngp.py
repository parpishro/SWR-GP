"""
NNGP module - Optimized with Numba JIT.
Uses fixed-size arrays to avoid Numba parallel issues.
"""

import numpy as np
from typing import Tuple
import numba
from numba import jit


# =============================================================================
# Matérn Covariance (Closed-form for common nu values)
# =============================================================================

@jit(nopython=True, cache=True)
def matern_cov_nu15(h: float, sigma_sq: float, phi: float) -> float:
    """Matérn with ν=1.5: C(h) = σ² (1 + √3 h/φ) exp(-√3 h/φ)"""
    if h < 1e-10:
        return sigma_sq
    sqrt3 = 1.7320508075688772
    scaled = sqrt3 * h / phi
    return sigma_sq * (1.0 + scaled) * np.exp(-scaled)


@jit(nopython=True, cache=True)
def matern_cov_nu05(h: float, sigma_sq: float, phi: float) -> float:
    """Matérn with ν=0.5 (exponential)."""
    if h < 1e-10:
        return sigma_sq
    return sigma_sq * np.exp(-h / phi)


@jit(nopython=True, cache=True)
def matern_cov_nu25(h: float, sigma_sq: float, phi: float) -> float:
    """Matérn with ν=2.5."""
    if h < 1e-10:
        return sigma_sq
    sqrt5 = 2.23606797749979
    scaled = sqrt5 * h / phi
    return sigma_sq * (1.0 + scaled + scaled*scaled/3.0) * np.exp(-scaled)


# =============================================================================
# Cholesky solve for small systems (Numba-compatible)
# =============================================================================

@jit(nopython=True, cache=True)
def solve_chol(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A @ x = b using Cholesky for positive definite A."""
    n = len(b)
    
    # Cholesky decomposition: A = L @ L.T
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = np.sqrt(max(A[i, i] - s, 1e-10))
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    
    # Forward solve: L @ y = b
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]
    
    # Backward solve: L.T @ x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += L[j, i] * x[j]
        x[i] = (y[i] - s) / L[i, i]
    
    return x


# =============================================================================
# NNGP Matrix Build (Serial but JIT-optimized)
# =============================================================================

@jit(nopython=True, cache=True)
def build_nngp_matrices_jit(n: int, m: int, sigma_sq: float, phi: float, 
                             tau_sq: float, nu_flag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build NNGP matrices B and F for temporal data.
    
    nu_flag: 0=0.5, 1=1.5, 2=2.5
    Returns B as (n, m) and F as (n,)
    """
    B = np.zeros((n, m))
    F = np.zeros(n)
    
    # Pre-allocate workspace for max-size system
    C_nn = np.zeros((m, m))
    c_tn = np.zeros(m)
    
    for t in range(n):
        nn = min(t, m)  # number of neighbors
        
        if nn == 0:
            F[t] = sigma_sq + tau_sq
        else:
            # Fill covariance matrix (only use first nn x nn)
            for i in range(nn):
                ni = t - nn + i
                # Cross-covariance
                h_tn = float(t - ni)
                if nu_flag == 1:
                    c_tn[i] = matern_cov_nu15(h_tn, sigma_sq, phi)
                elif nu_flag == 0:
                    c_tn[i] = matern_cov_nu05(h_tn, sigma_sq, phi)
                else:
                    c_tn[i] = matern_cov_nu25(h_tn, sigma_sq, phi)
                
                for j in range(nn):
                    nj = t - nn + j
                    h = float(abs(ni - nj))
                    if nu_flag == 1:
                        C_nn[i, j] = matern_cov_nu15(h, sigma_sq, phi)
                    elif nu_flag == 0:
                        C_nn[i, j] = matern_cov_nu05(h, sigma_sq, phi)
                    else:
                        C_nn[i, j] = matern_cov_nu25(h, sigma_sq, phi)
                    if i == j:
                        C_nn[i, j] += tau_sq
            
            # Extract submatrix and solve
            C_sub = C_nn[:nn, :nn].copy()
            c_sub = c_tn[:nn].copy()
            b = solve_chol(C_sub, c_sub)
            
            # Store in B (aligned to end)
            for i in range(nn):
                B[t, m - nn + i] = b[i]
            
            # Conditional variance
            var_t = sigma_sq + tau_sq
            dot_prod = 0.0
            for i in range(nn):
                dot_prod += c_sub[i] * b[i]
            F[t] = max(var_t - dot_prod, 1e-10)
    
    return B, F


def build_nngp_matrices(n: int, m: int, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper for JIT function."""
    sigma_sq, phi, nu, tau_sq = theta
    
    # Convert nu to flag
    if abs(nu - 0.5) < 0.1:
        nu_flag = 0
    elif abs(nu - 1.5) < 0.1:
        nu_flag = 1
    else:
        nu_flag = 2
    
    return build_nngp_matrices_jit(n, m, sigma_sq, phi, tau_sq, nu_flag)


# =============================================================================
# Decorrelation (JIT-optimized)
# =============================================================================

@jit(nopython=True, cache=True)
def decorrelate_jit(Y: np.ndarray, B: np.ndarray, F: np.ndarray, m: int) -> np.ndarray:
    """Decorrelate Y using NNGP: Y* = F^{-1/2} (Y - B @ Y_neighbors)"""
    n = len(Y)
    Y_star = np.zeros(n)
    
    for t in range(n):
        val = Y[t]
        nn = min(t, m)
        for j in range(nn):
            neighbor_idx = t - nn + j
            val -= B[t, m - nn + j] * Y[neighbor_idx]
        Y_star[t] = val / np.sqrt(F[t])
    
    return Y_star


@jit(nopython=True, cache=True)
def decorrelate_matrix_jit(X: np.ndarray, B: np.ndarray, F: np.ndarray, m: int) -> np.ndarray:
    """Decorrelate each column of X."""
    n, p = X.shape
    X_star = np.zeros((n, p))
    for k in range(p):
        X_star[:, k] = decorrelate_jit(X[:, k], B, F, m)
    return X_star


def decorrelate(Y: np.ndarray, B: np.ndarray, F: np.ndarray, m: int) -> np.ndarray:
    return decorrelate_jit(Y, B, F, m)


def decorrelate_matrix(X: np.ndarray, B: np.ndarray, F: np.ndarray, m: int) -> np.ndarray:
    return decorrelate_matrix_jit(X, B, F, m)


# =============================================================================
# Theta Estimation
# =============================================================================

def estimate_theta_mom(residuals: np.ndarray, nu: float = 1.5) -> np.ndarray:
    """Estimate covariance parameters via method of moments."""
    var_total = np.var(residuals)
    acf1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    acf1 = max(0.1, min(0.99, acf1))
    phi = max(1.0, -1.0 / np.log(acf1 + 1e-10))
    tau_sq = 0.1 * var_total
    sigma_sq = 0.9 * var_total
    return np.array([sigma_sq, phi, nu, tau_sq])


def estimate_theta_mle(residuals: np.ndarray, nu: float = 1.5, m: int = 10) -> np.ndarray:
    """Estimate covariance parameters via full NNGP marginal log-likelihood.

    Optimizes over (sigma_sq, phi, tau_sq) in log-space using L-BFGS-B,
    holding the NN mean fixed (residuals = y - mu_nn already computed).
    Falls back to MOM estimate if optimization fails.
    """
    from scipy.optimize import minimize

    n = len(residuals)
    zeros = np.zeros(n)

    def neg_loglik(log_params: np.ndarray) -> float:
        sigma_sq = np.exp(np.clip(log_params[0], -10.0, 15.0))
        phi = np.exp(np.clip(log_params[1], -2.0, 12.0))
        tau_sq = np.exp(np.clip(log_params[2], -15.0, 10.0))
        theta = np.array([sigma_sq, phi, nu, tau_sq])
        try:
            B, F = build_nngp_matrices(n, m, theta)
            return -gls_log_likelihood(residuals, zeros, B, F, m)
        except Exception:
            return 1e10

    # Warm-start from MOM estimate
    theta0 = estimate_theta_mom(residuals, nu)
    x0 = np.array([
        np.log(max(theta0[0], 1e-8)),
        np.log(max(theta0[1], 1e-8)),
        np.log(max(theta0[3], 1e-8)),
    ])

    result = minimize(
        neg_loglik, x0, method="L-BFGS-B",
        bounds=[(-10.0, 15.0), (-2.0, 12.0), (-15.0, 10.0)],
        options={"maxiter": 200, "ftol": 1e-8, "gtol": 1e-6},
    )

    if result.success or result.fun < neg_loglik(x0):
        sigma_sq = np.exp(np.clip(result.x[0], -10.0, 15.0))
        phi = np.exp(np.clip(result.x[1], -2.0, 12.0))
        tau_sq = np.exp(np.clip(result.x[2], -15.0, 10.0))
        return np.array([sigma_sq, phi, nu, tau_sq])
    else:
        return theta0


# =============================================================================
# GLS Log-Likelihood
# =============================================================================

def gls_log_likelihood(Y: np.ndarray, pred: np.ndarray, 
                       B: np.ndarray, F: np.ndarray, m: int) -> float:
    """Compute GLS log-likelihood."""
    n = len(Y)
    Y_star = decorrelate_jit(Y, B, F, m)
    pred_star = decorrelate_jit(pred, B, F, m)
    log_det = np.sum(np.log(F))
    rss = np.sum((Y_star - pred_star) ** 2)
    return -0.5 * (n * np.log(2 * np.pi) + log_det + rss)


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("NNGP Module Benchmark (Numba JIT)")
    print("=" * 50)
    
    # Warm up JIT
    print("Warming up JIT compilation...")
    theta = np.array([1.0, 10.0, 1.5, 0.1])
    B, F = build_nngp_matrices(100, 20, theta)
    Y = np.random.randn(100)
    _ = decorrelate(Y, B, F, 20)
    print("JIT warm-up complete\n")
    
    # Benchmark
    print(f"{'n':>6} {'m':>4} {'build (s)':>12} {'decor (s)':>12}")
    print("-" * 40)
    
    for n in [500, 1000, 5000, 10000]:
        m = 50
        theta = np.array([1.0, 10.0, 1.5, 0.1])
        
        t0 = time.time()
        B, F = build_nngp_matrices(n, m, theta)
        t1 = time.time()
        
        Y = np.random.randn(n)
        t2 = time.time()
        Y_star = decorrelate(Y, B, F, m)
        t3 = time.time()
        
        print(f"{n:>6} {m:>4} {t1-t0:>12.4f} {t3-t2:>12.6f}")
    
    print("-" * 40)
    print("Benchmark complete")
