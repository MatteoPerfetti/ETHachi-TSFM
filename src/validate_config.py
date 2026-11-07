from datetime import datetime

import pandas as pd

import config


def _timestamps() -> dict[str, pd.Timestamp]:
    names = ["DATA_START", "DATA_END", "TRAIN_START", "TRAIN_END", "TEST_START", "TEST_END"]
    return {name: pd.Timestamp(getattr(config, name)) for name in names}


def _is_hhmm(value: str) -> bool:
    try:
        datetime.strptime(value, "%H:%M")
        return True
    except (TypeError, ValueError):
        return False


def _dates_ordered(start: str, end: str) -> bool:
    try:
        return pd.Timestamp(start) < pd.Timestamp(end)
    except (TypeError, ValueError):
        return False


def validate_config() -> None:
    """Check the time ranges and model settings in config.py. Raises ValueError listing every violated rule."""
    try:
        t = _timestamps()
        step = pd.Timedelta(config.RESOLUTIONCODE)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid config: cannot parse timestamps or RESOLUTIONCODE ({e})") from e

    # Leave enough source history for either traditional feature mode. The
    # backtest raises its request context to this minimum when fixed_gap runs.
    history_steps = config.PREDICTION_LENGTH + config.TRADITIONAL_CONTEXT_STEPS - 1
    rules = [
        (isinstance(config.CONTEXT_LENGTH, int) and config.CONTEXT_LENGTH >= 1,
         "CONTEXT_LENGTH must be an integer >= 1"),
        (isinstance(config.PREDICTION_LENGTH, int) and config.PREDICTION_LENGTH >= 1,
         "PREDICTION_LENGTH must be an integer >= 1"),
        (t["DATA_START"] < t["DATA_END"], "DATA_START must be before DATA_END"),
        (t["DATA_START"] + history_steps * step <= t["TRAIN_START"],
         f"TRAIN_START must leave at least {history_steps} history steps after DATA_START"),
        (t["TRAIN_START"] < t["TRAIN_END"], "TRAIN_START must be before TRAIN_END"),
        (t["TRAIN_END"] < t["TEST_START"], "TRAIN_END must be before TEST_START"),
        (t["TEST_START"] < t["TEST_END"], "TEST_START must be before TEST_END"),
        (t["TEST_END"] <= t["DATA_END"], "TEST_END must not be after DATA_END"),
        ((t["TEST_END"] - t["TEST_START"]) / step >= config.PREDICTION_LENGTH,
         f"test span must cover at least one forecast horizon, PREDICTION_LENGTH = {config.PREDICTION_LENGTH} steps"),
        (config.CV_N_SPLITS >= 2, "CV_N_SPLITS must be at least 2"),
        (config.TRADITIONAL_STRATEGY in config.TRADITIONAL_STRATEGIES,
         f"TRADITIONAL_STRATEGY must be one of {config.TRADITIONAL_STRATEGIES}"),
        (set(config.TRADITIONAL_MODEL_PARAMETERS) == set(config.TRADITIONAL_MODELS),
         "TRADITIONAL_MODEL_PARAMETERS must contain exactly the registered traditional models"),
        (not config.TSFM_USE_COVARIATES or config.TARGET_SOURCE in config.TSFM_COVARIATES,
         f"TSFM_COVARIATES has no covariate list for TARGET_SOURCE = {config.TARGET_SOURCE!r}"),
        (not (config.TSFM_USE_COVARIATES and config.TARGET_SOURCE == "wind") or config.DATA_USE_DWD,
         "wind TSFM_COVARIATES come from the DWD file, set DATA_USE_DWD = True"),
        (all(key in config.MODELS_REGISTRY for key in config.BACKTEST_MODELS),
         f"BACKTEST_MODELS must be keys of MODELS_REGISTRY, got {config.BACKTEST_MODELS}"),
        (_is_hhmm(config.BACKTEST_ISSUE_TIME), "BACKTEST_ISSUE_TIME must be HH:MM, e.g. '09:00'"),
        (isinstance(config.BACKTEST_HORIZON_STEPS, int) and config.BACKTEST_HORIZON_STEPS >= 1,
         "BACKTEST_HORIZON_STEPS must be an integer >= 1"),
        (isinstance(config.BACKTEST_CONTEXT_STEPS, int) and config.BACKTEST_CONTEXT_STEPS >= 1,
         "BACKTEST_CONTEXT_STEPS must be an integer >= 1"),
        (isinstance(config.BACKTEST_LATENCY_STEPS, int) and config.BACKTEST_LATENCY_STEPS >= 0,
         "BACKTEST_LATENCY_STEPS must be an integer >= 0"),
        (config.BACKTEST_EVAL_WINDOW in ("all", "d+1"), "BACKTEST_EVAL_WINDOW must be 'all' or 'd+1'"),
        (_dates_ordered(config.BACKTEST_PLOT_START, config.BACKTEST_PLOT_END),
         "BACKTEST_PLOT_START and BACKTEST_PLOT_END must be YYYY-MM-DD dates with START before END"),
    ]
    errors = [message for ok, message in rules if not ok]
    if errors:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))


if __name__ == "__main__":
    validate_config()
    print("Config OK")
