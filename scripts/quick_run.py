"""
Big Sur Experiment - Quick Test Run

Validates GP-SWR on small subset before full run.
"""

import numpy as np
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swrgp.model import GPSWR
from swrgp.metrics import compute_metrics
from swrgp.paths import DATA_DIR, ensure_output_dirs_for_root, timestamp
from swrgp.bigsur_data import load_bigsur_train_test


def run_test(maxiter: int = 5,
             m: int = 20,
             nu: float = 1.5,
             seed: int = 42,
             n_train_test: int = 500,
             n_test_test: int = 100,
             results_root: str = None,
             verbose: bool = True):
    """Quick test run with limited data and iterations."""
    _, _, metrics_dir, _, _ = ensure_output_dirs_for_root(
        results_root,
        experiment_key="quick_run",
        script_path=__file__,
    )
    run_timestamp = timestamp()
    
    print("=" * 70)
    print("GP-SWR TEST RUN - Bug Validation")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()
    
    # Load data
    print("Loading data...")
    train_full, test_full = load_bigsur_train_test(DATA_DIR)
    
    train = train_full.iloc[:n_train_test]
    test = test_full.iloc[:n_test_test]
    
    print(f"Train: {len(train)} obs (subset of {len(train_full)})")
    print(f"Test: {len(test)} obs (subset of {len(test_full)})")
    print()
    
    # Quick test with K=1
    print("=" * 50)
    print("Testing K=1 (quick validation)")
    print("=" * 50)
    
    model = GPSWR(
        K=1,
        m=m,
        nu=nu,
        maxiter=maxiter,
        seed=seed,
        verbose=verbose
    )
    
    try:
        model.fit(
            train['rain'].values,
            train['gauge'].values
        )
        
        print("\n" + model.summary())
        
        # Predict on test with train history for warm-up
        test_pred = model.predict_with_history(
            train['rain'].values,
            test['rain'].values
        )
        test_obs = test['gauge'].values
        
        print("\nTest Metrics:")
        metrics = compute_metrics(test_obs, test_pred)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        # Save test results
        results = {
            'test_run': True,
            'n_train': len(train),
            'n_test': len(test),
            'K': model.K,
            'm': model.m,
            'nu': nu,
            'seed': seed,
            'maxiter': maxiter,
            'log_lik': model.log_lik_,
            'aic': model.aic_,
            'bic': model.bic_,
            'kernel_summary': model.get_kernel_summary(),
            'theta': model.theta_.tolist(),
            'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                           for k, v in metrics.items()},
            'fit_time': model.fit_time_
        }
        
        results_file = metrics_dir / 'test_results.json'
        archive_results_file = metrics_dir / f'test_results_{run_timestamp}.json'

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        with open(archive_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest results saved to {results_file}")
        print(f"Archived test results saved to {archive_results_file}")
        print("\n" + "=" * 70)
        print("TEST PASSED - No bugs detected")
        print("=" * 70)
        
        return True, model
        
    except Exception as e:
        print(f"\n*** TEST FAILED ***")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GP-SWR quick test run')
    parser.add_argument('--maxiter', type=int, default=5,
                        help='CMA-ES max iterations for quick test (default: 5)')
    parser.add_argument('--m', type=int, default=20,
                        help='NNGP neighbors for quick test (default: 20)')
    parser.add_argument('--nu', type=float, default=1.5,
                        help='Matérn smoothness (default: 1.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--train-size', type=int, default=500,
                        help='Train subset size for quick test (default: 500)')
    parser.add_argument('--test-size', type=int, default=100,
                        help='Test subset size for quick test (default: 100)')
    parser.add_argument('--results-root', type=str, default=None,
                        help='Optional output root directory (default: tests/artifacts/)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce model-level verbosity')
    args = parser.parse_args()

    success, model = run_test(
        maxiter=args.maxiter,
        m=args.m,
        nu=args.nu,
        seed=args.seed,
        n_train_test=args.train_size,
        n_test_test=args.test_size,
        results_root=args.results_root,
        verbose=not args.quiet,
    )
    sys.exit(0 if success else 1)
