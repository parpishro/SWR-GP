# exp_nn_gp_none — NN-GP, Identity Transform

Neural network mean function with GP residual covariance (NN-GP), no response transform.
Fixed architecture (hidden_dim=24); no complexity sweep. Model selected by kriging CRPS on
the held-out tail of the training period.

## Setup

| Parameter | Value |
|---|---|
| Method | NN-GP (NN mean + NNGP temporal GP covariance) |
| Transform | Identity (none) |
| hidden_dim | 24 |
| NNGP neighbors | m=20 |
| Matérn smoothness | ν=1.5 |
| NN epochs | 400 |
| GLS iterations | 6 |
| Weight decay | 1e-3 |
| Seed | 42 |
| Train split | hydr_year < 30 (n=10,501) |
| Test split | hydr_year ≥ 30 (n=3,652) |
| Selection criterion | val kriging CRPS (tail-of-training held-out) |

## Results

| Mode | CRPS | NSE | KGE | RMSE |
|---|---|---|---|---|
| sim (train) | 0.984 | 0.785 | 0.688 | 2.228 |
| sim (test) | 1.064 | 0.575 | 0.535 | 2.753 |
| krig (train) | 0.125 | 0.996 | 0.989 | 0.296 |
| krig (test) | 0.533 | 0.748 | 0.858 | 2.118 |

θ = (σ²=4.667, φ=9.0 days, ν=1.5, τ²=0.0065)

## Reproduce

```bash
.venv/bin/python experiments/exp_nn_gp_none/script/run.py
```
