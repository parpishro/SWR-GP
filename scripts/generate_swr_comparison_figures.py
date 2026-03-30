"""Generate Section 5.2 comparison figures: SWR vs SWR-GP (untransformed).

Produces:
  experiments/exp_swr_none/plots/timeseries_swr_vs_gpkr.png
      One-year observed vs corrected-prediction comparison for SWR (AR) and
      SWR-GP (NNGP kriging), with the raw simulation prediction shown as a
      reference to make the correction step visible.

  experiments/exp_swr_none/plots/kernel_comparison_swr_vs_gpkr.png
      Side-by-side fitted lag kernels for SWR and SWR-GP (selected K),
      weighted by |β_k|.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from swrgp.bigsur_data import load_bigsur_train_test
from swrgp.kernels import build_design_matrix, build_kernel, kernel_mean_lag, kernel_spread
from swrgp.paths import DATA_DIR

# Need SWRKernelOLS registered so pickle can reconstruct the class
from run_comprehensive import SWRKernelOLS, _fit_progressive_ar, _apply_ar_correction_from  # noqa: F401

SWR_PKL = PROJECT / "experiments/exp_swr_none/models/best.pkl"
GP_PKL  = PROJECT / "experiments/exp_swr_gp_none/models/best.pkl"
PLOTS   = PROJECT / "experiments/exp_swr_none/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Representative one-year window: water year 2011 (Oct 2010 – Sep 2011)
# Captures the main wet season and summer recession typical of the Big Sur record.
ZOOM_START = pd.Timestamp("2010-10-01")
ZOOM_END   = pd.Timestamp("2011-09-30")

MAX_LAG = 100


# ── Data & predictions ──────────────────────────────────────────────────

def _load_models():
    with open(SWR_PKL, "rb") as f:
        m_swr = pickle.load(f)
    with open(GP_PKL, "rb") as f:
        m_gp = pickle.load(f)
    return m_swr, m_gp


def _get_predictions(m_swr, m_gp, train_df, test_df):
    rain_train = train_df["rain"].to_numpy(float)
    y_train    = train_df["gauge"].to_numpy(float)
    rain_test  = test_df["rain"].to_numpy(float)
    y_test     = test_df["gauge"].to_numpy(float)

    full_rain = np.concatenate([rain_train, rain_test])
    full_y    = np.concatenate([y_train, y_test])
    n_train   = len(y_train)

    # ── SWR: simulation (kernel mean only) ──
    swr_sim_full = m_swr.predict(full_rain)
    swr_sim_test = swr_sim_full[n_train:]

    # ── SWR: AR-corrected prediction (Cochrane-Orcutt) ──
    train_resid = y_train - swr_sim_full[:n_train]
    ar_fit = _fit_progressive_ar(train_resid, p_max=8)
    swr_krig_full = _apply_ar_correction_from(
        swr_sim_full, full_y, start_index=n_train, phi=ar_fit["phi"]
    )
    swr_krig_test = swr_krig_full[n_train:]
    ar_order = ar_fit["p"]

    # ── SWR-GP: simulation (kernel mean only) ──
    gp_sim_full  = m_gp.predict(full_rain)
    gp_krig_full = m_gp.forecast(full_rain, full_y)
    gp_krig_test = gp_krig_full[n_train:]

    return {
        "y_test":       y_test,
        "swr_sim":      swr_sim_test,
        "swr_krig":     swr_krig_test,
        "gp_krig":      gp_krig_test,
        "ar_order":     ar_order,
    }


# ── Figure 1: one-year timeseries comparison ────────────────────────────

def generate_timeseries(preds, dates_test):
    mask = (dates_test >= ZOOM_START) & (dates_test <= ZOOM_END)
    t    = dates_test[mask]

    obs      = preds["y_test"][mask]
    swr_sim  = preds["swr_sim"][mask]
    swr_krig = preds["swr_krig"][mask]
    gp_krig  = preds["gp_krig"][mask]

    fig, ax = plt.subplots(figsize=(10, 3.8))

    # Simulation as a thin background reference
    ax.plot(t, swr_sim,  color="#aaaaaa", lw=0.7, alpha=0.7, zorder=1,
            label="Simulation (mean only)")

    # Corrected predictions
    ax.plot(t, swr_krig, color="#2166ac", lw=1.0, zorder=3,
            label="SWR — AR corrected")
    ax.plot(t, gp_krig,  color="#d6604d", lw=1.0, zorder=3,
            label="SWR-GP — kriging")

    # Observed on top
    ax.plot(t, obs, color="black", lw=0.85, alpha=0.9, zorder=4,
            label="Observed")

    ax.set_ylabel("Streamflow (cfs)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = PLOTS / "timeseries_swr_vs_gpkr.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Figure 2: side-by-side kernel shapes ───────────────────────────────

def generate_kernel_comparison(m_swr, m_gp):
    colors = ["#2166ac", "#b2182b", "#4dac26", "#7570b3"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)

    for ax, model, title in [(axes[0], m_swr, "SWR"), (axes[1], m_gp, "SWR-GP")]:
        for k in range(model.K):
            log_delta, log_sigma = model.kernel_params_[k]
            kern     = build_kernel(log_delta, log_sigma, model.kernel_type,
                                    max_length=MAX_LAG)
            beta     = model.beta_[k]
            lags     = np.arange(len(kern))
            mean_lag = kernel_mean_lag(log_delta, log_sigma, model.kernel_type)
            label    = (f"$k={k+1}$:  $\\beta={beta:+.2f}$,"
                        f"  $\\bar{{\\ell}}={mean_lag:.0f}$d")
            ax.fill_between(lags, kern * abs(beta), alpha=0.22, color=colors[k])
            ax.plot(lags, kern * abs(beta), color=colors[k], lw=1.5, label=label)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Lag (days)")
        ax.set_xlim(0, MAX_LAG)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel(r"Weighted kernel  $|\beta_k|\,\kappa_{k,s}$")
    fig.tight_layout()

    out = PLOTS / "kernel_comparison_swr_vs_gpkr.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import matplotlib
    matplotlib.use("Agg")

    train_df, test_df = load_bigsur_train_test(DATA_DIR)
    m_swr, m_gp = _load_models()

    preds      = _get_predictions(m_swr, m_gp, train_df, test_df)
    dates_test = pd.to_datetime(test_df["date"].values)

    print(f"AR order fitted: p={preds['ar_order']}")
    print(f"SWR-GP covariance: σ²={m_gp.theta_[0]:.3f}, φ={m_gp.theta_[1]:.1f}d, "
          f"τ²={m_gp.theta_[3]:.3f}, τ²/(σ²+τ²)={m_gp.theta_[3]/(m_gp.theta_[0]+m_gp.theta_[3]):.3f}")

    generate_timeseries(preds, dates_test)
    generate_kernel_comparison(m_swr, m_gp)
    print("Done.")


if __name__ == "__main__":
    main()
