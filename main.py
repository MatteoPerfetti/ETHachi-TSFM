"""Benchmark entry point: run the shared rolling-window backtest.

Everything is configured in config.py. The Streamlit dashboard is app.py.

    python main.py
    python main.py --reuse
"""
import argparse

import config
from logger import logger
from src.backtest import BacktestSettings, load_backtest, run_backtest, save_backtest
from src.dataset import load_dataset
from src.metrics import print_summary
from src.plotter import plot_backtest
from src.validate_config import validate_config


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
        result = run_backtest(load_dataset(), config.BACKTEST_MODELS, settings)
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
    run(reuse=parser.parse_args().reuse)
