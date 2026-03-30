# exp_multiscale_residual_covariance — Multi-Scale Residual Covariance

Compares single-scale vs multi-scale Matérn NNGP residual covariance on Big Sur streamflow.
The multi-scale variant uses two sequential Matérn scales (short + long) in a stacked NNGP whitening objective.

## Setup

| Parameter | Value |
|---|---|
| Kernel | Gaussian, K=4 |
| NNGP neighbors | m=20 |
| Matérn smoothness | ν=1.5 |
| Optimizer maxiter | 202 |
| Seed | 42 |
| Train split | hydr_year < 30 |
| Test split | hydr_year ≥ 30 |

## Results

| Model | Test sim CRPS | Test krig CRPS | Test sim NSE | Test krig NSE |
|---|---|---|---|---|
| Single-scale (SWR-GP) | 0.644 | 0.394 | 0.660 | 0.823 |
| Multi-scale | 0.831 | 0.533 | 0.424 | 0.774 |

Multi-scale performs worse on this dataset. The NNGP residual covariance does not benefit from two-scale decomposition here.

## Reproduce

```bash
cd experiments/exp_multiscale_residual_covariance
../../.venv/bin/python script/multiscale_residual_covariance.py --K 4 --m 20 --nu 1.5 --maxiter 202 --seed 42
```

## Quick smoke run

```bash
../../.venv/bin/python script/multiscale_residual_covariance.py --test --quiet --maxiter 20 --seed 42
```

## Notes

- Experiment-level extension; does not modify `GPSWR` core behavior.
- Improvement reported as multi-scale minus single-scale deltas for test NSE and CRPS.
