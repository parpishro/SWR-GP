"""
GP-NN Model (NN-GLS) - Neural Network Mean Function with NNGP Covariance.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import boxcox as scipy_boxcox, inv_boxcox
from typing import Tuple, List, Optional
import time
from .nngp import build_nngp_matrices, decorrelate


def _bc_back_transform(
    mu_t: np.ndarray,
    sigma_t: "float | np.ndarray",
    lam: float,
    shift: float,
) -> np.ndarray:
    """Delta-method back-transform for Box-Cox: approximate E[g^{-1}(Z)] for Z~N(mu_t, sigma_t^2)."""
    mu_t    = np.asarray(mu_t, dtype=float)
    sigma_t = np.maximum(np.asarray(sigma_t, dtype=float), 1e-8)
    sigma2  = sigma_t ** 2
    base    = inv_boxcox(mu_t, lam) - shift
    inner   = np.maximum(lam * mu_t + 1.0, 1e-8)
    second  = (1.0 / lam - 1.0) * lam * np.power(inner, 1.0 / lam - 2.0)
    return np.maximum(base + 0.5 * second * sigma2, 0.0)


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible NN runs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            pass

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 24, dropout: float = 0.0):
        super(MLP, self).__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GPNN:
    def __init__(self,
                 lag_window: int = 60,
                 m: int = 50,
                 nu: float = 1.5,
                 hidden_dim: int = 24,
                 dropout: float = 0.0,
                 epochs: int = 100,
                 lr: float = 0.001,
                 n_gls_iterations: int = 3,
                 weight_decay: float = 1e-3,
                 early_stopping: bool = True,
                 patience: int = 30,
                 min_delta: float = 1e-5,
                 min_epochs: int = 20,
                 max_grad_norm: Optional[float] = 5.0,
                 seed: Optional[int] = None,
                 deterministic_torch: bool = True,
                 log_transform: bool = False,
                 boxcox_lambda: Optional[float] = None,
                 verbose: bool = True):
        self.lag_window = lag_window
        self.m = m
        self.nu = nu
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.n_gls_iterations = n_gls_iterations
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.max_grad_norm = max_grad_norm
        self.seed = int(seed) if seed is not None else None
        self.deterministic_torch = deterministic_torch
        self.log_transform = log_transform
        self.boxcox_lambda = boxcox_lambda
        self.verbose = verbose
        self.epsilon_target = 0.01
        self.epochs_per_iteration_ = max(1, self.epochs // max(1, self.n_gls_iterations))

        if self.seed is not None:
            set_global_seed(self.seed, deterministic_torch=self.deterministic_torch)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = MLP(lag_window, hidden_dim, dropout=dropout).to(self.device)

        # Set at fit time
        self.theta_ = None          # [sigma_sq, phi, nu, tau_sq]
        self.boxcox_shift_: Optional[float] = None
        self.B_ = None
        self.F_ = None
        # Set after predict / forecast calls — raw output in fitted (transformed) space
        self.predict_raw_: Optional[np.ndarray] = None
        self.sigma_marginal_: Optional[float] = None
        self.forecast_raw_: Optional[np.ndarray] = None
        self.F_krig_: Optional[np.ndarray] = None
        
    def _prepare_inputs(self, rainfall: np.ndarray) -> torch.Tensor:
        """Create lag window matrix."""
        n = len(rainfall)
        X = np.zeros((n, self.lag_window))
        for i in range(self.lag_window):
            X[i:, i] = rainfall[:n-i]
        return torch.FloatTensor(X).to(self.device)

    def fit(self, rainfall: np.ndarray, streamflow: np.ndarray):
        if self.seed is not None:
            set_global_seed(self.seed, deterministic_torch=self.deterministic_torch)

        n = len(rainfall)
        X = self._prepare_inputs(rainfall)

        if self.boxcox_lambda is not None:
            y_min = float(np.nanmin(streamflow))
            self.boxcox_shift_ = max(0.0, -y_min + 1e-3) if y_min <= 0.0 else 0.0
            y = scipy_boxcox(streamflow + self.boxcox_shift_, self.boxcox_lambda)
        elif self.log_transform:
            y = np.log(streamflow + self.epsilon_target)
        else:
            y = streamflow

        y_arr = np.array(y, dtype=np.float32, copy=True)
        y_tensor = torch.from_numpy(y_arr).to(self.device).view(-1, 1)
        
        # 1. Initialize theta (start with OLS assumption)
        self.theta_ = np.array([np.var(y), 10.0, self.nu, 0.1 * np.var(y)])
        
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        for iteration in range(self.n_gls_iterations):
            if self.verbose:
                print(f"Iteration {iteration+1}/{self.n_gls_iterations}")
                
            # A. Update NNGP matrices
            self.B_, self.F_ = build_nngp_matrices(n, self.m, self.theta_)
            B_torch = torch.FloatTensor(self.B_).to(self.device)
            F_sqrt_torch = torch.FloatTensor(np.sqrt(self.F_)).to(self.device).view(-1, 1)
            
            # B. Train NN with GLS Loss
            self.net.train()
            
            # Pre-calculate B-weights for vectorized shift
            # B is (n, m). We want to shift residual by 1, 2, ..., m
            B_shifts = []
            for j in range(self.m):
                # Weight for lag (j+1) is B[:, m-1-j]
                B_shifts.append(B_torch[:, self.m - 1 - j].view(-1, 1))
            
            best_loss = float("inf")
            stale_epochs = 0
            best_state_dict = None

            for epoch in range(self.epochs_per_iteration_):
                optimizer.zero_grad()
                
                # Get NN predictions
                mu = self.net(X)
                residual = y_tensor - mu
                
                # Vectorized correction: sum_{j=1}^m B_{t, m-j} * res_{t-j}
                correction = torch.zeros_like(residual)
                for j in range(self.m):
                    # Shift residual down by j+1
                    lag = j + 1
                    shifted_res = torch.zeros_like(residual)
                    shifted_res[lag:] = residual[:-lag]
                    correction += B_shifts[j] * shifted_res
                
                decorr_resid = (residual - correction) / F_sqrt_torch
                loss = torch.mean(decorr_resid**2)
                
                loss.backward()

                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

                optimizer.step()

                loss_val = float(loss.item())
                if loss_val < (best_loss - self.min_delta):
                    best_loss = loss_val
                    stale_epochs = 0
                    if self.early_stopping:
                        best_state_dict = {
                            k: v.detach().cpu().clone()
                            for k, v in self.net.state_dict().items()
                        }
                else:
                    stale_epochs += 1

                if (
                    self.early_stopping
                    and (epoch + 1) >= self.min_epochs
                    and stale_epochs >= self.patience
                ):
                    if self.verbose:
                        print(
                            f"  Early stop at epoch {epoch + 1} "
                            f"(best GLS loss={best_loss:.6f})"
                        )
                    break
                
                if self.verbose and epoch % 20 == 0:
                    print(f"  Epoch {epoch}: GLS Loss = {loss.item():.6f}")

            if self.early_stopping and best_state_dict is not None:
                self.net.load_state_dict(best_state_dict)
            
            # C. Update GP Parameters via full NNGP MLE (Block B in Zhan-Datta)
            with torch.no_grad():
                mu = self.net(X).cpu().numpy().flatten()
                residuals = y - mu

                from .nngp import estimate_theta_mle
                self.theta_ = estimate_theta_mle(residuals, self.nu, self.m)
                if self.verbose:
                    print(f"  Updated Theta (MLE): {self.theta_}")
        
        return self

    def predict(self, rainfall: np.ndarray) -> np.ndarray:
        """Return mean predictions in the original response scale.

        Also sets ``predict_raw_`` (NN output in the fitted/transformed scale)
        and ``sigma_marginal_`` (marginal predictive std in fitted scale).
        """
        self.net.eval()
        with torch.no_grad():
            X = self._prepare_inputs(rainfall)
            mu = self.net(X).cpu().numpy().flatten()

        self.predict_raw_ = mu.copy()
        self.sigma_marginal_ = float(
            np.sqrt(max(float(self.theta_[0]) + float(self.theta_[3]), 1e-12))
        ) if self.theta_ is not None else None

        if self.boxcox_lambda is not None:
            return _bc_back_transform(
                mu, self.sigma_marginal_, self.boxcox_lambda, self.boxcox_shift_
            )
        if self.log_transform:
            return np.exp(mu) - self.epsilon_target
        return mu

    def forecast(self, rainfall: np.ndarray, observed_streamflow: np.ndarray) -> np.ndarray:
        """Return kriging-corrected predictions in the original response scale.

        Also sets ``forecast_raw_`` (kriging output in the fitted/transformed scale)
        and ``F_krig_`` (per-observation conditional variance in fitted scale).
        """
        n = len(rainfall)

        # Raw NN output in fitted (transformed) space
        self.net.eval()
        with torch.no_grad():
            X = self._prepare_inputs(rainfall)
            mu_t = self.net(X).cpu().numpy().flatten()

        # Transform observed values to fitted space for residual computation
        if self.boxcox_lambda is not None:
            y_t = scipy_boxcox(observed_streamflow + self.boxcox_shift_, self.boxcox_lambda)
        elif self.log_transform:
            y_t = np.log(observed_streamflow + self.epsilon_target)
        else:
            y_t = observed_streamflow

        epsilon = y_t - mu_t

        B, F = build_nngp_matrices(n, self.m, self.theta_)
        self.F_krig_ = F.copy()

        correction = np.zeros(n)
        for t in range(n):
            nn_count = min(t, self.m)
            for j in range(nn_count):
                neighbor_idx = t - nn_count + j
                correction[t] += B[t, self.m - nn_count + j] * epsilon[neighbor_idx]

        krig_t = mu_t + correction          # in fitted (transformed) space
        self.forecast_raw_ = krig_t.copy()

        if self.boxcox_lambda is not None:
            sigma_krig = np.sqrt(np.maximum(F, 1e-12))
            return _bc_back_transform(
                krig_t, sigma_krig, self.boxcox_lambda, self.boxcox_shift_
            )
        if self.log_transform:
            return np.exp(krig_t) - self.epsilon_target
        return krig_t
