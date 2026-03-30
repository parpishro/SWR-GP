# exp_swr_boxcox — SWR, Box-Cox Transform

Sliding Window Regression (SWR) with Gaussian kernels and Box-Cox response transform (λ=0.2).
K-sweep over K∈{1,2,3,4}, model selected by BIC_eff on training data.

## Setup

| Parameter | Value |
|---|---|
| Method | SWR (kernel-convolution mean, no GP residual) |
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
| sim (test) | 0.5260 | 0.7681 | 2.0333 |
| krig (test) | 0.3092 | 0.8925 | 1.3845 |

## Reproduce

```bash
cd experiments/exp_swr_boxcox
../../.venv/bin/python script/run.py
```
