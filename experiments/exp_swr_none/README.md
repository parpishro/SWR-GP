# exp_swr_none — SWR, Identity Transform

Sliding Window Regression (SWR) with Gaussian kernels, no response transform.
K-sweep over K∈{1,2,3,4}, model selected by BIC_eff on training data.

## Setup

| Parameter | Value |
|---|---|
| Method | SWR (kernel-convolution mean, no GP residual) |
| Transform | Identity (none) |
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
| sim (test) | 1.1309 | 0.6496 | 2.4994 |
| krig (test) | 0.7943 | 0.8815 | 1.4533 |

## Reproduce

```bash
cd experiments/exp_swr_none
../../.venv/bin/python script/run.py
```
