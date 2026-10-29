"""Benchmark entry point: load the data, run the rolling-window backtest, print the metrics, save the forecast plot.

Everything is set in config.py (models, issue time, horizon, context, latency, covariates, plot period).
    python main.py           run the backtest and save it to results/
    python main.py --reuse   skip the backtest and use the saved results of the same settings, if any
"""
import argparse

import config
from logger import logger
from src.backtest import BacktestSettings, load_backtest, run_backtest, save_backtest
from src.dataset import load_dataset
from src.metrics import print_summary
from src.plotter import plot_backtest
from src.validate_config import validate_config


def backtest_models() -> list[str]:
    """BACKTEST_MODELS without the traditional models while RUN_TRADITIONAL_MODELS is False."""
    models = []
    for key in config.BACKTEST_MODELS:
        if config.MODELS_REGISTRY[key]["type"] == "traditional" and not config.RUN_TRADITIONAL_MODELS:
            logger.info("Skipping %s: RUN_TRADITIONAL_MODELS is False", key)
            continue
        models.append(key)
    return models


def run(reuse: bool = False) -> None:
    validate_config()
    settings = BacktestSettings()
    results_path = config.BACKTEST_RESULTS_DIR / f"{settings.run_name}.parquet"

    if reuse and results_path.exists():
        logger.info("Reusing saved backtest %s", results_path)
        predictions, _ = load_backtest(settings.run_name)
    else:
        if reuse:
            logger.info("No saved backtest %s, running it", results_path)
        result = run_backtest(load_dataset(), backtest_models(), settings)
        save_backtest(result)
        predictions = result.predictions

    print()
    print_summary(predictions, settings.run_name)
    print()

    plot_backtest(predictions, settings.run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true",
                        help="use the saved backtest results of the current settings instead of recomputing them")
    run(reuse=parser.parse_args().reuse)
