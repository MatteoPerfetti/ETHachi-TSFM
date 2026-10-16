import pandas as pd

from . import config
from src.data_ingest.datasource import load_elia, load_openmeteo
from src.data_ingest.http_client import HttpClient
from src.logger import logger
from src.features.features import cross_validation_splits, interpolate_to_15min, merge, merge_dwd_weather, split_train_test


def load_sources() -> dict[str, pd.DataFrame]:
    """Load every configured Elia and Open-Meteo source: from data/ if cached, otherwise from the API."""
    features = {}
    targets = {}
    for name, src in config.ELIA_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            targets[name] = load_elia(name, client, src["region"], config.START, config.END, config.RESOLUTIONCODE)
    for name, src in config.OPENMETEO_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            features[name] = load_openmeteo(name, client, src["latitude"], src["longitude"], src["hourly"],
                                           config.START, config.END)
    return features, targets


def load_features(features: dict[str, pd.DataFrame]):
    """Align all sources to 15min steps, merge them, and split into train/test and cross validation folds."""
    # _____________
    # Use temporary workaround to merge DWD weather data with Open-Meteo weather data
    if config.DATA_USE_DWD:
        data = merge_dwd_weather(features["weather"], config.DATA_DWD_PATH)
    else:
        features = {**features, "weather": interpolate_to_15min(features["weather"])}
        data = merge(features)
    # _____________
    train, test = split_train_test(data, config.TEST_SIZE, config.SHUFFLE, config.RANDOM_STATE)
    logger.info("Prepared %d train rows, %d test rows", len(train), len(test))
    folds = cross_validation_splits(train, config.CV_N_SPLITS, config.CV_TEST_SIZE, config.CV_GAP,
                                    config.CV_MAX_TRAIN_SIZE)
    return train, test, folds


def run() -> None:
    # 1. Load all sources (from cache or API)
    features, targets = load_sources()
    # 2. Prepare features: align, merge, split into train/test and folds
    # train, test, folds = load_features(frames)
