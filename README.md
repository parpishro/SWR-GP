# SWR-GP

**Gaussian Process Kernel Regression for Hydrological Modeling** — a temporal adaptation of the NN-GLS framework (Zhan & Datta, 2024) for rainfall-runoff modeling at the Big Sur catchment.

The central idea is to replace the ad hoc autoregressive residual correction common in hydrological models with an explicit temporal Gaussian process covariance, separating the physical mean function from residual dependence.

Paper: *Gaussian Process Kernel Regression for Hydrological Modeling: A Temporal Adaptation of the NN-GLS Framework* — Par Pishrobat (2026). Source at `docs/final_paper.qmd`.

---

## Models

**GP-SWR** (`GPSWR`) — Sliding-Window Regression mean with GP residual covariance. The mean decomposes streamflow into $K$ kernel-convolved rainfall signals, each representing a runoff pathway (surface flow, interflow, baseflow) with interpretable lag-timing parameters. Residual autocorrelation is modeled via a temporal NNGP approximation to a Matérn-3/2 covariance. Variants: `SWR-N` (untransformed), `SWR-BC` (Box-Cox $\lambda=0.2$), `SWR-GP-N`, `SWR-GP-BC`.

**GP-NN** (`GPNN`) — Two-hidden-layer MLP mean under the same GP covariance structure. Used as a nonlinearity benchmark; the SWR kernel-convolution provides sufficient mean-function flexibility at this site.

Full benchmark results are in `docs/final_paper.qmd` (Section 5) and per-experiment metrics in `experiments/exp_*/output/results.json`.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt
```

Dependencies: `numpy`, `scipy`, `pandas`, `numba`, `cma`, `torch`, `matplotlib`, `scikit-learn`, `statsmodels`, `tqdm`.

---

## Usage

**Quick smoke test** on a small data subset:

```bash
.venv/bin/python scripts/quick_run.py --maxiter 5 --train-size 500 --test-size 100
```

**Run all six model variants** (SWR-N, SWR-BC, SWR-GP-N, SWR-GP-BC, NN-GP-N, NN-GP-BC):

```bash
.venv/bin/python scripts/run_comprehensive.py --n-jobs 1
```

**Minimal Python example** — fit GP-SWR and generate predictions:

```python
import numpy as np
from swrgp.model import GPSWR
from swrgp.metrics import compute_metrics
from swrgp.bigsur_data import load_bigsur_train_test
from swrgp.paths import DATA_DIR

train, test = load_bigsur_train_test(DATA_DIR)

# K=3 kernels, m=10 NNGP neighbors, Matern-1.5
model = GPSWR(K=3, m=10, nu=1.5, maxiter=200, seed=42)
model.fit(train["rain"].values, train["gauge"].values)
print(model.summary())  # kernel shapes, covariance params, log-likelihood

# Simulation predictions (rainfall input only)
sim_pred = model.predict_with_history(train["rain"].values, test["rain"].values)

# Kriging predictions (conditions on observed streamflow history)
rain_full = np.concatenate([train["rain"].values, test["rain"].values])
y_full    = np.concatenate([train["gauge"].values, test["gauge"].values])
krig_pred = model.forecast(rain_full, y_full)[len(train):]

print(compute_metrics(test["gauge"].values, krig_pred))
```

---

## Repository Layout

```
SWR-GP/
├── src/swrgp/           # Core package
│   ├── model.py         # GPSWR: SWR mean + GP covariance; CMA-ES optimization
│   ├── nn_model.py      # GPNN: MLP mean + GP covariance; iterative GLS training
│   ├── nngp.py          # NNGP precision, decorrelation, kriging
│   ├── kernels.py       # Gamma/Gaussian lag kernels; kernel convolution
│   ├── metrics.py       # NSE, KGE, CRPS, RMSE, BIC_eff, PIT
│   ├── bigsur_data.py   # Data loading and train/test split
│   ├── diagnostics.py   # PIT histograms, QQ plots, time series plots
│   └── paths.py         # Experiment artifact path conventions
├── experiments/         # One directory per experiment (see below)
├── scripts/             # run_comprehensive.py, quick_run.py, generate_paper_figures.py
├── data/bigsur.csv      # 14,154 daily observations, Big Sur catchment 1980–2018
└── docs/                # final_paper.qmd, figures, references.bib
```

### Experiments

| Directory | Description |
|:----------|:------------|
| `exp_swr_none/` | SWR-N benchmark |
| `exp_swr_boxcox/` | SWR-BC benchmark |
| `exp_swr_gp_none/` | SWR-GP-N benchmark |
| `exp_swr_gp_boxcox/` | SWR-GP-BC benchmark |
| `exp_nn_gp_none/` | NN-GP-N neural benchmark |
| `exp_nn_gp_boxcox/` | NN-GP-BC neural benchmark |
| `exp_simulation_recovery/` | Oracle-K simulation recovery grid (4×3 SNR grid, 5 reps/cell) |
| `exp_multiscale_residual_covariance/` | Single vs. multi-scale Matérn NNGP comparison |
| `exp_gamma_likelihood_latent_gp/` | Gamma likelihood on GPSWR latent path |

Each experiment owns: `script/` (runnable entrypoint), `output/` (results.json, best.pkl, figures), `run.log`.

---

## Key Parameters

| Parameter | Default | Notes |
|:----------|:-------:|:------|
| `K` | 3 | Number of rainfall kernels; 4 used in some benchmarks |
| `m` | 10 | NNGP neighbor depth (20 for primary benchmark) |
| `nu` | 1.5 | Matérn smoothness, fixed |
| Box-Cox `lambda` | 0.2 | Fixed for all BC variants |

Model selection within a run uses `BIC_eff` (SWR/SWR-GP) or held-out validation CRPS (NN-GP). Cross-model comparison uses test kriging CRPS.

Point predictions on the Box-Cox scale use the conditional median (`exp(mu)` for the log-transform), not the lognormal mean.

---

## Reference

Zhan, M. & Datta, A. (2024). Neural networks for geospatial data. *Journal of the American Statistical Association*. arXiv:2304.09157
