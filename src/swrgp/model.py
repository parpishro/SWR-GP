"""
GP-SWR Model - Optimized with CMA-ES and proper warm-up handling.
"""

import numpy as np
from scipy.optimize import minimize, nnls
from typing import Tuple, Optional, List, Dict
import time

from .kernels import (
    build_design_matrix,
    gaussian_kernel_constraints_satisfied,
    kernel_mean_lag,
    kernel_spread,
    sample_constrained_gaussian_kernel_params,
    sort_kernel_params,
)
from .nngp import (build_nngp_matrices, decorrelate, decorrelate_matrix,
                   gls_log_likelihood, estimate_theta_mom, estimate_theta_mle)
from .metrics import compute_metrics, compute_aic, compute_bic, compute_aic_eff, compute_bic_eff, estimate_effective_sample_size

class GPSWR:
    """
    Gaussian Process Sliding Windows Regression.
    
    Uses CMA-ES for global optimization (faster than DE).
    """
    
    def __init__(self,
                 K: int = 2,
                 m: int = 50,
                 nu: float = 1.5,
                 kernel_type: str = "gamma",
                 max_lag: int = 365,
                 maxiter: int = 200,
                 n_restarts: int = 1,
                 log_transform: bool = False,
                 seed: int = 42,
                 verbose: bool = True,
                 sigma_sq_max: Optional[float] = None,
                 min_mean_lag_separation: Optional[float] = None,
                 max_kernel_overlap: Optional[float] = None,
                 enforce_max_kernel_overlap_in_estimation: bool = True,
                 gaussian_kernel_log_bounds: Optional[
                     Tuple[Tuple[float, float], Tuple[float, float]]
                 ] = None,
                 support_sigma_multiple: Optional[float] = None,
                 support_lag_range: Optional[Tuple[float, float]] = None,
                 restart_spread_range: Optional[Tuple[float, float]] = None):
        """
        Parameters
        ----------
        K : int
            Number of kernels
        m : int
            NNGP neighbor size
        nu : float
            Matérn smoothness
        kernel_type : str
            Kernel type
        max_lag : int
            Maximum lag (days) used to build each kernel
        maxiter : int
            CMA-ES max iterations
        n_restarts : int
            Number of optimizer restarts. Best fit is kept.
        log_transform : bool
            If True, fit on log(y + 0.01). Ensures positivity.
        seed : int
            Random seed
        verbose : bool
            Print progress
        """
        self.K = K
        self.m = m
        self.nu = nu
        self.kernel_type = kernel_type
        self.max_lag = max_lag
        self.maxiter = maxiter
        self.n_restarts = max(1, n_restarts)
        self.log_transform = log_transform
        self.seed = seed
        self.verbose = verbose
        self.sigma_sq_max = sigma_sq_max  # if set, caps log_sigma_sq upper bound
        self.epsilon_target = 0.01 # Flow offset for log
        self.min_mean_lag_separation = min_mean_lag_separation
        self.max_kernel_overlap = max_kernel_overlap
        self.enforce_max_kernel_overlap_in_estimation = (
            enforce_max_kernel_overlap_in_estimation
        )
        self.gaussian_kernel_log_bounds = gaussian_kernel_log_bounds
        self.support_sigma_multiple = support_sigma_multiple
        self.support_lag_range = support_lag_range
        self.restart_spread_range = restart_spread_range
        
        # ... (rest of fitted params)
        self.kernel_params_ = None
        self.beta_ = None
        self.theta_ = None
        self.B_ = None
        self.F_ = None
        self.warmup_ = 0
        self.restart_summaries_ = []

    def _active_max_kernel_overlap(self) -> Optional[float]:
        """Return the fit-time overlap cap if this estimation mode uses it."""
        if self.enforce_max_kernel_overlap_in_estimation:
            return self.max_kernel_overlap
        return None
        
    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds."""
        n_params = 2 * self.K + 3
        lower = np.zeros(n_params)
        upper = np.zeros(n_params)
        if self.kernel_type == "gaussian":
            if self.gaussian_kernel_log_bounds is None:
                # Match the SWR application-study search box so both models use
                # the same kernel geometry setup unless a caller opts out.
                kernel_lower = (-2.0, -1.0)
                kernel_upper = (3.0, 4.0)
            else:
                kernel_lower, kernel_upper = self.gaussian_kernel_log_bounds
        else:
            kernel_lower = (-2.0, -1.0)
            kernel_upper = (3.0, 4.0)
        for k in range(self.K):
            lower[2*k] = kernel_lower[0]
            upper[2*k] = kernel_upper[0]
            lower[2*k + 1] = kernel_lower[1]
            upper[2*k + 1] = kernel_upper[1]
        idx = 2 * self.K
        lower[idx] = -3.0          # log_sigma_sq
        upper[idx] = (np.log(self.sigma_sq_max)
                      if self.sigma_sq_max is not None else 4.0)
        lower[idx + 1] = 0.0       # log_phi
        upper[idx + 1] = 5.0
        lower[idx + 2] = -5.0      # log_tau_sq
        upper[idx + 2] = 2.0
        return lower, upper
    
    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack flat params."""
        n_kernel = 2 * self.K
        kernel_params = params[:n_kernel].reshape(self.K, 2)
        if self.kernel_type == "gaussian":
            kernel_params = sort_kernel_params(kernel_params, self.kernel_type)
        theta = np.array([
            np.exp(params[n_kernel]),      # sigma_sq
            np.exp(params[n_kernel + 1]),  # phi
            self.nu,
            np.exp(params[n_kernel + 2])   # tau_sq
        ])
        return kernel_params, theta

    def _kernel_constraints_satisfied(self, kernel_params: np.ndarray) -> bool:
        """Return whether the current kernel geometry is feasible."""
        if self.kernel_type != "gaussian":
            return True
        active_max_overlap = self._active_max_kernel_overlap()
        if (
            self.min_mean_lag_separation is None
            and active_max_overlap is None
            and self.support_sigma_multiple is None
            and self.support_lag_range is None
        ):
            return True
        return gaussian_kernel_constraints_satisfied(
            kernel_params,
            min_mean_lag_separation=self.min_mean_lag_separation or 0.0,
            max_overlap=active_max_overlap,
            support_sigma_multiple=self.support_sigma_multiple,
            support_lag_range=self.support_lag_range,
        )

    def _make_restart_initial_guess(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        restart: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Construct a feasible restart initial guess when constraints are active."""
        x0 = (lower + upper) / 2
        if restart == 0 and self.kernel_type != "gaussian":
            return self._move_strictly_inside_bounds(x0, lower, upper)

        n_kernel = 2 * self.K
        if self.kernel_type == "gaussian" and self._kernel_constraints_satisfied(
            x0[:n_kernel].reshape(self.K, 2)
        ):
            return self._move_strictly_inside_bounds(x0, lower, upper)

        x0 = lower + rng.random(len(lower)) * (upper - lower)
        active_max_overlap = self._active_max_kernel_overlap()
        if self.kernel_type != "gaussian" or not (
            self.min_mean_lag_separation is not None
            or active_max_overlap is not None
            or self.support_sigma_multiple is not None
            or self.support_lag_range is not None
        ):
            return self._move_strictly_inside_bounds(x0, lower, upper)

        if self.support_lag_range is None or self.support_sigma_multiple is None:
            return self._move_strictly_inside_bounds(x0, lower, upper)

        spread_range = self.restart_spread_range
        if spread_range is None:
            spread_range = (
                float(np.exp(lower[1])),
                float(np.exp(upper[1])),
            )

        kernel_params = sample_constrained_gaussian_kernel_params(
            rng=rng,
            K=self.K,
            spread_range=spread_range,
            min_mean_lag_separation=self.min_mean_lag_separation or 0.0,
            max_overlap=active_max_overlap if active_max_overlap is not None else 1.0,
            support_sigma_multiple=self.support_sigma_multiple,
            support_lag_range=self.support_lag_range,
        )
        x0[:n_kernel] = kernel_params.reshape(-1)
        return self._move_strictly_inside_bounds(x0, lower, upper)

    def _move_strictly_inside_bounds(
        self,
        x: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Move a vector strictly inside box bounds for CMA-ES initialization."""
        span = upper - lower
        eps = np.maximum(1e-8 * span, 1e-10)
        return np.minimum(np.maximum(x, lower + eps), upper - eps)

    def _sample_feasible_cma_solution(
        self,
        es,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
        max_attempts: int = 200,
    ) -> np.ndarray:
        """Sample a feasible CMA-ES proposal before objective evaluation."""
        for _ in range(max_attempts):
            proposal = self._move_strictly_inside_bounds(es.ask(1)[0], lower, upper)
            kernel_params, _ = self._unpack_params(proposal)
            if self._kernel_constraints_satisfied(kernel_params):
                return proposal
        return self._make_restart_initial_guess(lower, upper, restart=1, rng=rng)

    def _objective(self, params: np.ndarray, 
                   rainfall: np.ndarray, 
                   streamflow: np.ndarray) -> float:
        """Objective function (minimize negative log-lik)."""
        self.n_evals_ += 1
        
        y_target = streamflow
        if self.log_transform:
            y_target = np.log(streamflow + self.epsilon_target)
            
        try:
            kernel_params, theta = self._unpack_params(params)
            if not self._kernel_constraints_satisfied(kernel_params):
                return 1e10
            
            # Build design matrix
            X = build_design_matrix(
                rainfall,
                kernel_params,
                self.kernel_type,
                max_length=self.max_lag,
            )
            
            # Build NNGP
            B, F = build_nngp_matrices(len(y_target), self.m, theta)
            
            # Keep the objective consistent with the final fitted model.
            Y_star = decorrelate(y_target, B, F, self.m)
            X_star = decorrelate_matrix(X, B, F, self.m)
            beta, _ = nnls(X_star, Y_star)
            
            # Log-likelihood
            pred = X @ beta
            log_lik = gls_log_likelihood(y_target, pred, B, F, self.m)
            
            return -log_lik
            
        except Exception:
            return 1e10
    
    def fit(self, rainfall: np.ndarray, streamflow: np.ndarray) -> 'GPSWR':
        """Fit GP-SWR model using CMA-ES."""
        t0 = time.time()
        n = len(rainfall)
        
        self.rainfall_ = rainfall.copy()
        self.streamflow_ = streamflow.copy()
        self.n_evals_ = 0
        
        y_target = streamflow
        if self.log_transform:
            y_target = np.log(streamflow + self.epsilon_target)
            
        n_params = 2 * self.K + 3
        self.n_params_ = n_params
        
        if self.verbose:
            print(f"Fitting GP-SWR: K={self.K}, n={n}, m={self.m}")
        
        lower, upper = self._get_bounds()
        
        # Try CMA-ES first
        try:
            import cma
            
            if self.verbose:
                print(f"Using CMA-ES optimizer (maxiter={self.maxiter})")
            
            sigma0 = np.mean(upper - lower) / 4
            best_neg_ll = np.inf
            best_params = None
            self.restart_summaries_ = []
            for restart in range(self.n_restarts):
                rng = np.random.default_rng(self.seed + restart)
                x0 = self._make_restart_initial_guess(lower, upper, restart, rng)
                opts = {
                    'bounds': [lower.tolist(), upper.tolist()],
                    'maxiter': self.maxiter,
                    'seed': self.seed + restart,
                    'verbose': -9,
                    'tolfun': 1e-6,
                    'popsize': max(10, 4 + int(3 * np.log(n_params)))
                }
                es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
                while not es.stop():
                    solutions = [
                        self._sample_feasible_cma_solution(es, lower, upper, rng)
                        for _ in range(es.popsize)
                    ]
                    fitness = [self._objective(x, rainfall, streamflow) for x in solutions]
                    es.tell(solutions, fitness)
                if es.result.fbest < best_neg_ll:
                    best_neg_ll = es.result.fbest
                    best_params = es.result.xbest
                self.restart_summaries_.append({
                    "restart": restart + 1,
                    "seed": self.seed + restart,
                    "optimizer": "cma",
                    "iters": int(es.countiter),
                    "best_objective": float(es.result.fbest),
                    "best_loglik": float(-es.result.fbest),
                })

        except ImportError:
            # Fall back to L-BFGS-B with multiple restarts
            if self.verbose:
                print("CMA-ES not available, using L-BFGS-B with restarts")
            
            best_neg_ll = np.inf
            best_params = None
            self.restart_summaries_ = []
            
            for restart in range(self.n_restarts):
                rng = np.random.default_rng(self.seed + restart)
                x0 = self._make_restart_initial_guess(lower, upper, restart, rng)
                
                result = minimize(
                    lambda p: self._objective(p, rainfall, streamflow),
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lower, upper)),
                    options={'maxiter': self.maxiter // 5}
                )
                
                if result.fun < best_neg_ll:
                    best_neg_ll = result.fun
                    best_params = result.x
                self.restart_summaries_.append({
                    "restart": restart + 1,
                    "seed": self.seed + restart,
                    "optimizer": "lbfgsb",
                    "iters": int(getattr(result, "nit", 0)),
                    "best_objective": float(result.fun),
                    "best_loglik": float(-result.fun),
                })
        
        # Extract best parameters
        self.kernel_params_, self.theta_ = self._unpack_params(best_params)
        
        # Build final NNGP and estimate beta
        X = build_design_matrix(
            rainfall,
            self.kernel_params_,
            self.kernel_type,
            max_length=self.max_lag,
        )
        self.B_, self.F_ = build_nngp_matrices(n, self.m, self.theta_)
        
        Y_star = decorrelate(y_target, self.B_, self.F_, self.m)
        X_star = decorrelate_matrix(X, self.B_, self.F_, self.m)
        self.beta_, _ = nnls(X_star, Y_star)
        
        # Compute max kernel length for warm-up
        max_lag = 0
        for k in range(self.K):
            lag = kernel_mean_lag(self.kernel_params_[k, 0], 
                                  self.kernel_params_[k, 1],
                                  self.kernel_type)
            spread = kernel_spread(self.kernel_params_[k, 0],
                                   self.kernel_params_[k, 1],
                                   self.kernel_type)
            max_lag = max(max_lag, lag + 3 * spread)
        self.warmup_ = int(np.ceil(max_lag))
        
        # Log-likelihood
        pred = X @ self.beta_
        self.log_lik_ = gls_log_likelihood(y_target, pred, self.B_, self.F_, self.m)
        
        # Information criteria
        n_effective = 3 * self.K + 3
        self.aic_ = compute_aic(self.log_lik_, n_effective)
        residuals = y_target - pred
        self.n_eff_ = estimate_effective_sample_size(residuals)
        self.aic_eff_ = compute_aic_eff(self.log_lik_, n_effective, residuals)
        self.bic_eff_ = compute_bic_eff(self.log_lik_, n_effective, residuals)
        self.bic_ = compute_bic(self.log_lik_, n_effective, n)
        
        self.fit_time_ = time.time() - t0
        
        if self.verbose:
            print(f"\nFit complete in {self.fit_time_:.1f}s ({self.n_evals_} evals)")
            print(f"Log-lik: {self.log_lik_:.2f}, AIC: {self.aic_:.2f}, AIC_eff: {self.aic_eff_:.2f}, BIC_eff: {self.bic_eff_:.2f}")
            print(f"Warm-up period: {self.warmup_} days")
        
        return self
    
    def predict(self, rainfall: np.ndarray, return_warmup_mask: bool = False) -> np.ndarray:
        """
        Predict streamflow (point forecast = conditional median).
        
        NOTE: First warmup_ values are unreliable due to kernel convolution.
        """
        X = build_design_matrix(
            rainfall,
            self.kernel_params_,
            self.kernel_type,
            max_length=self.max_lag,
        )
        pred = X @ self.beta_
        
        if self.log_transform:
            # Conditional median: exp(mu) - epsilon_target
            # Using median instead of lognormal mean exp(mu + var/2) avoids
            # systematic positive bias from the variance correction.
            pred = np.exp(pred) - self.epsilon_target
        
        if return_warmup_mask:
            mask = np.ones(len(pred), dtype=bool)
            mask[:self.warmup_] = False
            return pred, mask
        
        return pred
    
    def predict_with_history(self, train_rain: np.ndarray, test_rain: np.ndarray) -> np.ndarray:
        """
        Predict on test data using train data as warm-up history.
        
        This is the proper way to predict on test data - avoids warm-up issues.
        """
        # Concatenate for proper convolution
        full_rain = np.concatenate([train_rain, test_rain])
        full_pred = self.predict(full_rain)
        
        # Return only test portion
        return full_pred[len(train_rain):]
    
    def get_kernel_summary(self) -> List[Dict]:
        """Get fitted kernel parameters."""
        summary = []
        for k in range(self.K):
            summary.append({
                'kernel': k + 1,
                'beta': self.beta_[k],
                'delta': np.exp(self.kernel_params_[k, 0]),
                'sigma': np.exp(self.kernel_params_[k, 1]),
                'mean_lag': kernel_mean_lag(self.kernel_params_[k, 0], 
                                            self.kernel_params_[k, 1],
                                            self.kernel_type),
                'spread': kernel_spread(self.kernel_params_[k, 0],
                                        self.kernel_params_[k, 1],
                                        self.kernel_type)
            })
        return summary
    
    def forecast(self, rainfall: np.ndarray, observed_streamflow: np.ndarray) -> np.ndarray:
        """
        One-step-ahead forecast using Kriging (Conditional Mean).
        """
        n = len(rainfall)
        y_target = observed_streamflow
        if self.log_transform:
            y_target = np.log(observed_streamflow + self.epsilon_target)
            
        X = build_design_matrix(
            rainfall,
            self.kernel_params_,
            self.kernel_type,
            max_length=self.max_lag,
        )
        mu = X @ self.beta_
        epsilon = y_target - mu
        
        # Build B, F for the input size specifically
        B, F = build_nngp_matrices(n, self.m, self.theta_)
        
        correction = np.zeros(n)
        for t in range(n):
            nn = min(t, self.m)
            for j in range(nn):
                neighbor_idx = t - nn + j
                correction[t] += B[t, self.m - nn + j] * epsilon[neighbor_idx]
        
        forecast_val = mu + correction

        if self.log_transform:
            # Conditional median: exp(mu) - epsilon_target
            # Using median instead of lognormal mean exp(mu + var/2) avoids
            # systematic positive bias from the variance correction (see Gotcha #15).
            forecast_val = np.exp(forecast_val) - self.epsilon_target
            
        return forecast_val

    def summary(self) -> str:
        """Print model summary."""
        lines = [
            f"GP-SWR Model Summary",
            f"=" * 50,
            f"Kernels: {self.K}",
            f"NNGP neighbors: {self.m}",
            f"Matérn ν: {self.nu}",
            f"Warm-up period: {self.warmup_} days",
            f"",
            f"Fit Statistics:",
            f"  Log-likelihood: {self.log_lik_:.4f}",
            f"  AIC: {self.aic_:.4f}",
            f"  AIC_eff: {self.aic_eff_:.4f}",
            f"  BIC: {self.bic_:.4f}",
            f"  BIC_eff: {self.bic_eff_:.4f}",
            f"  n_eff: {self.n_eff_:.2f}",
            f"  Fit time: {self.fit_time_:.1f}s ({self.n_evals_} evals)",
            f"",
            f"Kernel Parameters:",
        ]
        
        for ks in self.get_kernel_summary():
            lines.append(f"  K{ks['kernel']}: β={ks['beta']:.4f}, "
                        f"δ={ks['delta']:.2f}, σ={ks['sigma']:.2f}, "
                        f"mean_lag={ks['mean_lag']:.1f}d")
        
        lines.extend([
            f"",
            f"Covariance Parameters:",
            f"  σ² (signal): {self.theta_[0]:.4f}",
            f"  φ  (range):  {self.theta_[1]:.1f} days",
            f"  ν  (smooth): {self.theta_[2]:.1f}",
            f"  τ² (nugget): {self.theta_[3]:.4f}",
        ])
        
        return "\n".join(lines)
