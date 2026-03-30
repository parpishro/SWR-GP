# exp_swr_gp_boxcox — SWR-GP, Box-Cox Transform

SWR with GP residual structure (SWR-GP): Gaussian kernel mean + NNGP temporal covariance, Box-Cox response transform (λ=0.2).
K-sweep over K∈{1,2,3,4}, model selected by BIC_eff on training data.
This is our primary model and the main contribution of the paper.

## Setup

| Parameter | Value |
|---|---|
| Method | SWR-GP (kernel mean + NNGP temporal GP covariance) |
| Transform | Box-Cox (λ=0.2) |
| Kernel | Gaussian |
| K values | 1, 2, 3, 4 |
| NNGP neighbors | m=20 |
| Matérn smoothness | ν=1.5 |
| GP optimizer maxiter | 202 |
| Seed | 42 |
| Train split | hydr_year < 30 |
| Test split | hydr_year ≥ 30 |

## Results (best K=4)

| Mode | CRPS | NSE | RMSE |
|---|---|---|---|
| sim (test) | 0.8985 | 0.6023 | 2.6629 |
| krig (test) | 0.3719 | 0.8871 | 1.4191 |

## Best model parameters

θ = (σ²=0.727, φ=14.548, ν=1.5, τ²=0.018)

## Reproduce

```bash
cd experiments/exp_swr_gp_boxcox
../../.venv/bin/python script/run.py
```
