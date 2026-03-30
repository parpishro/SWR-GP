"""Generate final paper figures: time series plot and kernel shape visualization."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.special import inv_boxcox
from scipy.stats import gamma

# ── paths ──
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from swrgp.bigsur_data import load_bigsur_train_test
from swrgp.kernels import build_design_matrix, build_kernel, kernel_mean_lag, kernel_spread
from swrgp.nngp import build_nngp_matrices

# Needed for unpickling SWR models (run_comprehensive defines SWRKernelOLS)
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from run_comprehensive import SWRKernelOLS  # noqa: E402

DATA_DIR = PROJECT / "data"
MODELS_DIR = PROJECT / "experiments" / "exp_swr_boxcox" / "models"
PLOTS_DIR = PROJECT / "experiments" / "exp_swr_boxcox" / "plots"

# ── config ──
LAM = 0.2
M_NNGP = 20
NU = 1.5
MAX_LAG = 100


def _boxcox_shift(y: np.ndarray) -> float:
    y_min = float(np.nanmin(y))
    return 0.0 if y_min > 0 else abs(y_min) + 1e-3


def _inverse_boxcox_median(mu_t: np.ndarray, lam: float, shift: float) -> np.ndarray:
    """Conditional median back-transform."""
    return np.maximum(inv_boxcox(mu_t, lam) - shift, 0.0)


def _fit_progressive_ar(train_resid: np.ndarray, p_max: int = 8) -> dict:
    """Fit AR correction progressively."""
    def _dw(r):
        if r.size < 2:
            return 2.0
        return float(np.sum((r[1:] - r[:-1]) ** 2) / max(np.sum(r ** 2), 1e-12))

    best = {"p": 0, "phi": np.array([], dtype=float), "dw": _dw(train_resid)}
    for p in range(1, p_max + 1):
        if train_resid.size <= p + 5:
            break
        y = train_resid[p:]
        X = np.column_stack([train_resid[p - j - 1: -j - 1] for j in range(p)])
        phi, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ phi
        dw = _dw(e)
        if abs(dw - 2.0) < abs(best["dw"] - 2.0) - 0.03:
            best = {"p": p, "phi": np.asarray(phi, dtype=float), "dw": float(dw)}
        else:
            break
    return best


def _apply_ar_correction(base_pred: np.ndarray, y_obs: np.ndarray,
                          start_index: int, phi: np.ndarray) -> np.ndarray:
    out = base_pred.copy()
    p = phi.size
    if p == 0:
        return out
    resid = y_obs - base_pred
    start = max(int(start_index), int(p))
    for t in range(start, out.size):
        hist = resid[t - p: t][::-1]
        out[t] += float(np.dot(phi, hist))
    return out


def generate_timeseries_plot():
    """Observed vs predicted (sim + kriging) for SWR-BC on test set."""
    # Load data
    train_df, test_df = load_bigsur_train_test(DATA_DIR)
    rain_train = train_df["rain"].to_numpy(dtype=float)
    y_train = train_df["gauge"].to_numpy(dtype=float)
    rain_test = test_df["rain"].to_numpy(dtype=float)
    y_test = test_df["gauge"].to_numpy(dtype=float)
    dates_test = pd.to_datetime(test_df["date"].values)

    full_rain = np.concatenate([rain_train, rain_test])
    full_y = np.concatenate([y_train, y_test])

    shift = _boxcox_shift(y_train)
    from scipy.special import boxcox as scipy_boxcox
    y_train_t = scipy_boxcox(y_train + shift, LAM)
    full_y_t = scipy_boxcox(full_y + shift, LAM)

    # Load SWR-BC model
    with open(MODELS_DIR / "best.pkl", "rb") as f:
        model = pickle.load(f)

    # Simulation predictions (transformed scale)
    sim_full_t = model.predict(full_rain)
    sim_train_t = sim_full_t[: len(y_train)]
    sim_test_t = sim_full_t[len(y_train):]

    # AR correction in transformed space
    ar_fit = _fit_progressive_ar(y_train_t - sim_train_t, p_max=8)
    krig_full_t = _apply_ar_correction(
        sim_full_t, full_y_t, start_index=0, phi=ar_fit["phi"]
    )
    krig_test_t = krig_full_t[len(y_train):]

    # Back-transform to original scale
    sim_test = _inverse_boxcox_median(sim_test_t, LAM, shift)
    krig_test = _inverse_boxcox_median(krig_test_t, LAM, shift)

    # ── Plot: full test period ──
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(dates_test, y_test, color="black", linewidth=0.6, alpha=0.85, label="Observed")
    ax1.plot(dates_test, krig_test, color="#2166ac", linewidth=0.6, alpha=0.8, label="SWR-BC kriging")
    ax1.plot(dates_test, sim_test, color="#b2182b", linewidth=0.5, alpha=0.5, label="SWR-BC simulation")
    ax1.set_ylabel("Streamflow (cfs)")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.set_title("SWR-BC: Observed vs. Predicted Streamflow (Held-Out Test Period)", fontsize=10)

    # Residual panel
    ax2 = axes[1]
    resid_krig = y_test - krig_test
    ax2.fill_between(dates_test, resid_krig, 0, color="#2166ac", alpha=0.4, linewidth=0)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Residual (cfs)")
    ax2.set_xlabel("Date")

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out_path = PLOTS_DIR / "timeseries__swr__boxcox__krig_test.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── Plot: zoomed 2-year window ──
    # Pick a representative 2-year window with both peaks and recession
    zoom_start = pd.Timestamp("2010-10-01")
    zoom_end = pd.Timestamp("2012-09-30")
    mask = (dates_test >= zoom_start) & (dates_test <= zoom_end)

    fig2, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(dates_test[mask], y_test[mask], color="black", linewidth=0.8, alpha=0.85, label="Observed")
    ax.plot(dates_test[mask], krig_test[mask], color="#2166ac", linewidth=0.8, alpha=0.8, label="SWR-BC kriging")
    ax.plot(dates_test[mask], sim_test[mask], color="#b2182b", linewidth=0.6, alpha=0.5, label="SWR-BC simulation")
    ax.set_ylabel("Streamflow (cfs)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("SWR-BC: Two-Year Detail (Water Years 2011\u20132012)", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig2.tight_layout()
    out_path2 = PLOTS_DIR / "timeseries_zoom__swr__boxcox__krig_test.png"
    fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path2}")


def generate_kernel_plot():
    """Fitted Gaussian lag kernels for SWR-BC (best K)."""
    with open(MODELS_DIR / "best.pkl", "rb") as f:
        model = pickle.load(f)

    K = model.K
    kernel_type = model.kernel_type

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#2166ac", "#b2182b", "#4dac26", "#7570b3"]

    for k in range(K):
        log_delta, log_sigma = model.kernel_params_[k]
        kernel = build_kernel(log_delta, log_sigma, kernel_type, max_length=MAX_LAG)
        lags = np.arange(len(kernel))
        mean_lag = kernel_mean_lag(log_delta, log_sigma, kernel_type)
        spread = kernel_spread(log_delta, log_sigma, kernel_type)
        beta = model.beta_[k]

        label = (f"$k={k+1}$: $\\beta={beta:.3f}$, "
                 f"$\\bar{{\\ell}}={mean_lag:.1f}$d, "
                 f"$s={spread:.1f}$d")
        ax.fill_between(lags, kernel * abs(beta), alpha=0.25, color=colors[k % len(colors)])
        ax.plot(lags, kernel * abs(beta), color=colors[k % len(colors)],
                linewidth=1.5, label=label)

    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Weighted kernel density ($|\\beta_k| \\cdot \\kappa_{k,s}$)")
    ax.set_title("Fitted Lag Kernels (SWR-BC, $K=4$)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.set_xlim(0, MAX_LAG)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out_path = PLOTS_DIR / "kernel_shapes__swr__boxcox__best.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_timeseries_plot()
    generate_kernel_plot()
    print("Done.")
