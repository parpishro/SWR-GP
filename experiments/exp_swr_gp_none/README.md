# exp_swr_gp_none — SWR-GP, Identity Transform

SWR with GP residual structure (SWR-GP): Gaussian kernel mean + NNGP temporal covariance, no response transform.
K-sweep over K∈{1,2,3,4}, model selected by BIC_eff on training data.

## Setup

| Parameter | Value |
|---|---|
| Method | SWR-GP (kernel mean + NNGP temporal GP covariance) |
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
| sim (test) | 1.1879 | 0.6170 | 2.6132 |
| krig (test) | 0.7987 | 0.8783 | 1.4729 |

## Reproduce

```bash
cd experiments/exp_swr_gp_none
../../.venv/bin/python script/run.py
```
