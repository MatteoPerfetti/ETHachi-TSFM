"""Benchmark entry point: run the shared rolling-window backtest.

Everything is configured in config.py. The Streamlit dashboard is app.py.

    python main.py
    python main.py --reuse
"""
import argparse
from pathlib import Path

import config
from logger import logger
from src.backtest import BacktestSettings, load_backtest, run_backtest, save_backtest
from src.dataset import load_dataset
from src.metrics import print_summary
from src.plotter import plot_backtest
from src.raft import RaftSettings
from src.validate_config import validate_config


def run(reuse: bool = False, raft: RaftSettings | None = None, *,
        validation_start: str | None = None, raft_trials: int = 50, raft_seed: int = 42) -> None:
    validate_config()
    settings = BacktestSettings(raft=raft)
    dataset = None
    tuning_metadata = None
    if validation_start is not None:
        from src.raft import RAFT_NAME
        from src.raft_tuning import tune_raft

        dataset = load_dataset()
        tsfm_keys = [key for key in config.BACKTEST_MODELS if key in config.TSFM_MODELS]
        tuning = tune_raft(dataset, tsfm_keys, settings, validation_start, raft_trials, raft_seed)
        settings.raft = tuning.settings
        tuning_metadata = tuning.metadata
        logger.info("RAFT Optuna best parameters %s, validation RMSE %.2f MW",
                    tuning.metadata["best_params"], tuning.metadata["validation_rmse_mw"])
    results_path = config.BACKTEST_RESULTS_DIR / f"{settings.run_name}.parquet"

    if reuse and tuning_metadata is None and results_path.exists():
        logger.info("Reusing saved backtest %s", results_path)
        predictions, _ = load_backtest(settings.run_name)
    else:
        if reuse:
            logger.info("No saved backtest %s, running it", results_path)
        result = run_backtest(dataset if dataset is not None else load_dataset(), config.BACKTEST_MODELS, settings)
        if tuning_metadata is not None:
            result.model_configs[RAFT_NAME]["tuning"] = tuning_metadata
        save_backtest(result)
        predictions = result.predictions

    print()
    print_summary(predictions, settings.run_name)
    print()
    plot_backtest(predictions, settings.run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="use saved results for the current settings instead of recomputing them",
    )
    parser.add_argument("--raft-weather", type=Path, help="hourly weather archive indexed by forecast issue time")
    parser.add_argument("--raft-capacity-mw", type=float, help="positive nominal capacity for the selected target")
    parser.add_argument("--raft-gamma", type=float, default=0.5, help="TSFM weight in the blend (default: 0.5)")
    parser.add_argument("--raft-validation-start", help="UTC start of Optuna validation; enables RAFT tuning")
    parser.add_argument("--raft-trials", type=int, default=50, help="number of Optuna trials (default: 50)")
    parser.add_argument("--raft-seed", type=int, default=42, help="Optuna sampler seed (default: 42)")
    args = parser.parse_args()
    if (args.raft_weather is None) != (args.raft_capacity_mw is None):
        parser.error("--raft-weather and --raft-capacity-mw are required together")
    if args.raft_validation_start and (args.raft_weather is None or args.raft_trials < 1 or args.reuse):
        parser.error("RAFT tuning requires a weather archive, a positive trial count and a fresh run")
    try:
        raft = (None if args.raft_weather is None else
                RaftSettings(str(args.raft_weather.resolve()), args.raft_capacity_mw, gamma=args.raft_gamma))
        run(reuse=args.reuse, raft=raft, validation_start=args.raft_validation_start,
            raft_trials=args.raft_trials, raft_seed=args.raft_seed)
    except (OSError, ValueError) as error:
        parser.error(str(error))
