# exp_gamma_likelihood_latent_gp — Gamma Likelihood Latent GP

Prototype replacing the Gaussian/Box-Cox observation model with a Gamma likelihood.
The latent GP mean is the SWR kernel-convolution structure (K=4, Gaussian kernels).
Comparison baseline: log-Gaussian SWR-GP kriging fitted with matched parameters.

## Setup

| Parameter | Value |
|---|---|
| Method | Gamma likelihood latent GP (prototype) |
| Kernel | Gaussian, K=4 |
| NNGP neighbors | m=20 |
| Matérn smoothness | ν=1.5 |
| Optimizer maxiter | 202 |
| Posterior samples | 1000 |
| Seed | 42 |
| Train split | hydr_year < 30 |
| Test split | hydr_year ≥ 30 |

## Results

| Model | RMSE | NSE | CRPS | Coverage 95% |
|---|---|---|---|---|
| SWR-GP log-Gaussian (baseline) | 1.776 | 0.823 | — | — |
| Gamma latent GP | 1.776 | 0.823 | 0.271 | 98.2% |

CRPS = 0.271 beats SWR-GP-BC krig CRPS (0.372) but coverage is slightly over-dispersed.

## Reproduce

```bash
cd experiments/exp_gamma_likelihood_latent_gp
../../.venv/bin/python script/gamma_latent_gp.py --K 4 --m 20 --nu 1.5 --maxiter 202 --seed 42
```

## Quick smoke run

```bash
../../.venv/bin/python script/gamma_latent_gp.py --test --quiet --maxiter 20 --n-samples 400 --seed 42
```

## Notes

- Prototype only; does not change `GPSWR` core objective.
- CRPS computed from Monte Carlo samples (empirical CRPS); `compute_crps` supports Gaussian family only.
