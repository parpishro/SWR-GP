"""Shared project-relative paths for experiment scripts."""

from datetime import datetime
from pathlib import Path
import shutil
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

LEGACY_RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR = EXPERIMENTS_DIR

EXPERIMENT_DIR_NAMES = {
    "bigsur": "exp_bigsur_k_sweep",
    "bigsur_boxcox": "exp_bigsur_boxcox_k_sweep",
    "nn": "exp_gpnn_bigsur_benchmark",
    "kriging": "exp_kriging_comparison",
    "coverage": "exp_coverage_analysis",
    "diagnostics": "exp_residual_diagnostics",
    "sim_recovery": "exp_simulation_recovery",
    "transform_study": "exp_response_transform_study",
    "gamma_latent_gp": "exp_gamma_likelihood_latent_gp",
    "qq_kriging": "exp_qq_kriging",
    "fair_benchmark": "exp_fair_benchmark",
    "assumption_stress": "exp_assumption_stress",
    "multiscale_covariance": "exp_multiscale_residual_covariance",
    "comprehensive": "exp_comprehensive",
}


def _resolve_experiment_dir_name(experiment_key: Optional[str]) -> str:
    if not experiment_key:
        raise ValueError("experiment_key is required. Outputs must be written to a specific experiment directory.")
    return EXPERIMENT_DIR_NAMES.get(experiment_key, experiment_key.strip().lower().replace("-", "_"))


def _snapshot_script(script_path: Optional[str], script_dir: Path) -> None:
    if not script_path:
        return

    source = Path(script_path).resolve()
    if not source.exists() or source.suffix != ".py":
        return

    target = script_dir / source.name
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return
    shutil.copy2(source, target)


def ensure_output_dirs() -> None:
    """Ensure standardized artifact directories exist."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_results_dirs(
    results_root: Optional[str] = None,
    experiment_key: Optional[str] = None,
) -> Tuple[Path, Path, Path, Path, Path]:
    """Resolve experiment artifact directories under merged experiments root.

    Parameters
    ----------
    results_root : Optional[str]
        Optional override root. If omitted, defaults to ``experiments``.
    experiment_key : Optional[str]
        Experiment identifier used to create informative subdirectories.
    """
    root = Path(results_root).expanduser().resolve() if results_root else EXPERIMENTS_DIR
    exp_dir = root / _resolve_experiment_dir_name(experiment_key)
    logs_dir = exp_dir / "logs"
    metrics_dir = exp_dir / "metrics"
    models_dir = exp_dir / "models"
    plots_dir = exp_dir / "plots"
    return exp_dir, logs_dir, metrics_dir, models_dir, plots_dir


def ensure_output_dirs_for_root(
    results_root: Optional[str] = None,
    experiment_key: Optional[str] = None,
    script_path: Optional[str] = None,
) -> Tuple[Path, Path, Path, Path, Path]:
    """Create standardized experiment output directories and return resolved paths.

    Returns ``(experiment_dir, logs_dir, metrics_dir, models_dir, plots_dir)``.
    """
    exp_dir, logs_dir, metrics_dir, models_dir, plots_dir = resolve_results_dirs(
        results_root=results_root,
        experiment_key=experiment_key,
    )
    script_dir = exp_dir / "script"
    for directory in (exp_dir, script_dir, logs_dir, metrics_dir, models_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _snapshot_script(script_path=script_path, script_dir=script_dir)
    return exp_dir, logs_dir, metrics_dir, models_dir, plots_dir


def timestamp() -> str:
    """Timestamp string for artifact naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
