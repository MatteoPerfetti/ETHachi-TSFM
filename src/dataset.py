
"""Adapter between the data extraction pipeline and the forecasting code.

Only calls the extraction functions (datasource, merge helpers in features), it never changes them.
Everything downstream (models, backtest, metrics) uses `load_dataset()` and the `Dataset` it returns,
so a new extraction strategy only needs changes here.
"""
from dataclasses import dataclass

import pandas as pd

import config
from logger import logger
from src.datasource import load_elia, load_openmeteo
from src.features import encode_directions, interpolate_to_15min, merge, merge_dwd_weather
from src.http_client import HttpClient


@dataclass
class Dataset:
    """Target and weather on the same regular 15 min UTC grid; missing values are explicit NaN, never dropped rows."""
    target: pd.Series        # generation in MW, name "<TARGET_SOURCE>_MW"
    weather: pd.DataFrame    # weather covariates, directions sin/cos encoded


def load_sources() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load the raw Elia and Open-Meteo data from the local parquet cache or from the APIs."""
    features_data = {}
    targets_data = {}
    for name, src in config.ELIA_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            targets_data[name] = load_elia(name, client, src["region"], config.DATA_START, config.DATA_END,
                                           config.RESOLUTIONCODE)
    for name, src in config.OPENMETEO_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            features_data[name] = load_openmeteo(name, client, src["latitude"], src["longitude"], src["hourly"],
                                                 config.DATA_START, config.DATA_END)
    return features_data, targets_data


def load_targets(targets_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Target frame of TARGET_SOURCE: one column per region if MULTILE_TARGETS, else the sum over regions."""
    target = targets_data[config.TARGET_SOURCE]
    if config.MULTILE_TARGETS:
        return target.add_prefix(f"{config.TARGET_SOURCE}_").add_suffix("_MW")
    # min_count=1 keeps NaN where every region is missing, instead of turning it into 0
    return target.sum(axis=1, min_count=1).to_frame(name=f"{config.TARGET_SOURCE}_MW")


def load_weather(features_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align the weather to 15 min (Open-Meteo, plus DWD if configured) and encode directions as sin/cos."""
    if config.DATA_USE_DWD:
        weather = merge_dwd_weather(features_data["weather"], config.DATA_DWD_PATH)
    else:
        weather = merge({name: interpolate_to_15min(df) for name, df in features_data.items()})
    return encode_directions(weather)


def regular_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Complete 15 min UTC grid from the first to the last timestamp of `index`."""
    index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return pd.date_range(index.min(), index.max(), freq=pd.Timedelta(config.RESOLUTIONCODE), name="timestamp")


def load_dataset() -> Dataset:
    """Load target and weather and put both on the regular grid spanned by the target."""
    if config.MULTILE_TARGETS:
        raise ValueError("load_dataset supports a single target only, set MULTILE_TARGETS = False")

    features_data, targets_data = load_sources()
    target = load_targets(targets_data).iloc[:, 0]
    weather = load_weather(features_data)

    grid = regular_grid(target.index)
    target = target.reindex(grid)
    weather = weather.reindex(grid)

    missing_target = int(target.isna().sum())
    missing_weather = int(weather.isna().any(axis=1).sum())
    if missing_target or missing_weather:
        logger.warning("Dataset %s..%s: %d of %d timestamps without target, %d without complete weather",
                       grid[0], grid[-1], missing_target, len(grid), missing_weather)
    logger.info("Dataset loaded: %d timestamps, target %s, %d weather columns",
                len(grid), target.name, weather.shape[1])
    return Dataset(target=target, weather=weather)
