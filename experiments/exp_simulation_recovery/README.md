# exp_simulation_recovery

Oracle-$K$ simulation recovery experiment for GP-SWR.

## Directory Structure

```text
exp_simulation_recovery/
├── README.md
├── report.qmd
├── run.log
├── output/
│   ├── simulation_recovery_grid.json
│   ├── simulation_recovery_summary.csv
│   ├── recovery_nse_by_k.png
│   ├── recovery_nrmse_heatmap.png
│   ├── recovery_parameter_heatmaps.png
│   └── report.pdf
└── script/
    └── simulation_recovery_grid.py
```

## Files

- `run.log`
  - Console log from the latest run only.
  - This file lives in the experiment root; there is no `logs/` directory.

- `output/simulation_recovery_grid.json`
  - Canonical full results file.
  - Contains experiment config, summary table, cell-level summaries, and per-replication records.
  - Per-replication records include:
    - true and estimated kernel summaries
    - matched true/estimated kernel pairs
    - true and estimated covariance parameters
    - mean-recovery metrics
    - parameter-recovery metrics

- `output/simulation_recovery_summary.csv`
  - Flat summary table for the grid.
  - Intended for quick inspection, spreadsheets, and downstream reporting.

- `output/recovery_nse_by_k.png`
  - NSE figure with one panel per true `K`.

- `output/recovery_nrmse_heatmap.png`
  - NRMSE heatmap across the simulation grid.

- `output/recovery_parameter_heatmaps.png`
  - `2 x 3` parameter-recovery heatmaps.

- `report.qmd`
  - Minimal report source for the experiment.
  - Reads directly from `output/simulation_recovery_grid.json`.

- `output/report.pdf`
  - Rendered PDF version of `report.qmd`.

- `script/simulation_recovery_grid.py`
  - Main experiment script.
  - Runs the simulation grid and writes all generated artifacts to `output/`.

## Run

From the repository root:

```bash
.venv/bin/python experiments/exp_simulation_recovery/script/simulation_recovery_grid.py \
  --reps 1 --n-restarts 3 --maxiter 202 \
  > experiments/exp_simulation_recovery/run.log 2>&1
```

Render the experiment report:

```bash
quarto render experiments/exp_simulation_recovery/report.qmd \
  --output-dir experiments/exp_simulation_recovery/output
```

## Notes

- The `output/` directory is treated as the canonical location for all generated artifacts.
- The root `run.log` should correspond to the latest run only.
- If a new metric or visualization is needed later, first check whether the required raw quantities are already stored in `simulation_recovery_grid.json`.
