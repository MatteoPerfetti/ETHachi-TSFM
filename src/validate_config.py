import pandas as pd

import config


def _timestamps() -> dict[str, pd.Timestamp]:
    names = ["DATA_START", "DATA_END", "TRAIN_START", "TRAIN_END", "TEST_START", "TEST_END"]
    return {name: pd.Timestamp(getattr(config, name)) for name in names}


def validate_config() -> None:
    """Check the time ranges and model settings in config.py. Raises ValueError listing every violated rule."""
    try:
        t = _timestamps()
        step = pd.Timedelta(config.RESOLUTIONCODE)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid config: cannot parse timestamps or RESOLUTIONCODE ({e})") from e

    history_steps = (
        config.CONTEXT_LENGTH
        if config.ML_FORECAST_STRATEGY == "origin_aligned"
        else config.PREDICTION_LENGTH + config.CONTEXT_LENGTH - 1
    )
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
    ]
    errors = [message for ok, message in rules if not ok]
    if errors:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))


if __name__ == "__main__":
    validate_config()
    print("Config OK")
