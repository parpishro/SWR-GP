# exp_nn_gp_boxcox — NN-GP, Box-Cox λ=0.2

Neural network mean function with GP residual covariance (NN-GP), Box-Cox response transform
(λ=0.2). Fixed architecture (hidden_dim=24); no complexity sweep. Model selected by
kriging CRPS on the held-out tail of the training period (Box-Cox–aware CRPS).

Back-transformation uses the delta-method approximation to the conditional mean
(identical to the SWR-GP Box-Cox procedure).

## Setup

| Parameter | Value |
|---|---|
| Method | NN-GP (NN mean + NNGP temporal GP covariance) |
| Transform | Box-Cox λ=0.2 |
| hidden_dim | 24 |
| NNGP neighbors | m=20 |
| Matérn smoothness | ν=1.5 |
| NN epochs | 400 |
| GLS iterations | 6 |
| Weight decay | 1e-3 |
| Seed | 42 |
| Train split | hydr_year < 30 (n=10,501) |
| Test split | hydr_year ≥ 30 (n=3,652) |
| Selection criterion | val kriging CRPS (tail-of-training held-out, Box-Cox) |

## Results

| Mode | CRPS | NSE | KGE | RMSE |
|---|---|---|---|---|
| sim (train) | 0.830 | 0.684 | 0.574 | 2.698 |
| sim (test) | 0.798 | 0.553 | 0.613 | 2.824 |
| krig (train) | 0.068 | 0.996 | 0.982 | 0.291 |
| krig (test) | 0.504 | 0.257 | 0.543 | 3.639 |

θ = (σ²=0.431, φ=13.5 days, ν=1.5, τ²=0.00094) — fit in BC space

**Note:** Krig NSE collapses to 0.257 despite NSE=0.937 in BC space. The quintic inverse
Box-Cox g⁻¹(x)=(0.2x+1)⁵ amplifies peak-flow prediction errors on the original scale.
CRPS improves (0.504 vs 0.533 for NN-GP-N) because the full predictive distribution
is correctly propagated through the back-transform.

## Reproduce

```bash
.venv/bin/python experiments/exp_nn_gp_boxcox/script/run.py
```
