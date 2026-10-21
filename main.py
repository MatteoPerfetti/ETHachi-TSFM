import pandas as pd
import config
from logger import logger
from src.datasource import load_elia, load_openmeteo
from src.features import interpolate_to_15min, merge, merge_dwd_weather, encode_directions, add_lag_features, add_calendar_features
from src.http_client import HttpClient
from src.models import split_train_test, cross_validation_splits, run_model
from src.plotter import plot_forecast
from src.validate_config import validate_config

import torch
from chronos import BaseChronosPipeline
import numpy as np


def load_sources() -> dict[str, pd.DataFrame]:
    """Load every configured Elia and Open-Meteo source: from data/ if cached, otherwise from the API."""
    features_data = {}
    targets_data = {}
    for name, src in config.ELIA_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            targets_data[name] = load_elia(name, client, src["region"], config.DATA_START, config.DATA_END, config.RESOLUTIONCODE)
    for name, src in config.OPENMETEO_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            features_data[name] = load_openmeteo(name, client, src["latitude"], src["longitude"], src["hourly"],
                                           config.DATA_START, config.DATA_END)
    return features_data, targets_data



def load_features(features_data: dict[str, pd.DataFrame], target: pd.DataFrame) -> pd.DataFrame:
    """Align all sources to 15min steps, merge them, and split into train/test and cross validation folds."""
    # _____________
    # Use temporary workaround to merge DWD weather data with Open-Meteo weather data
    if config.DATA_USE_DWD:
        features = merge_dwd_weather(features_data["weather"], config.DATA_DWD_PATH)
    else:
        for name, df in features_data.items():
            features_data[name] = interpolate_to_15min(df)
        features = merge(features_data)
    # _____________
    # Convert wind direction features (sin/cos) to the merged data
    features = encode_directions(features)
    # Add lags and calendar features to the merged data
    lag_features = add_lag_features(target, config.CONTEXT_LENGTH, config.PREDICTION_LENGTH)
    features = features.join(lag_features, how="inner")
    features = add_calendar_features(features, config.TIMEZONE, config.COUNTRY)

    return features

def load_targets(targets_data: dict[str, pd.DataFrame]):
    """Align all target sources to 15min steps, merge them, and split into train/test and cross validation folds."""
    target = targets_data[config.TARGET_SOURCE]
    if config.MULTILE_TARGETS:
        # Add TARGET_SOURCE str to column names
        target = target.add_prefix(f"{config.TARGET_SOURCE}_")
        target = target.add_suffix("_MW")
        return target
    else:
        # sum over all columns to get a single target column
        # and name that column <TARGET_SOURCE>_MW
        target = target.sum(axis=1).to_frame(name=f"{config.TARGET_SOURCE}_MW")
    return target

def split(X: pd.DataFrame, y: pd.Series):
    """Load the models from the models/ directory."""
    if config.USE_CROSS_VALIDATION:
        # Create cross-validation folds
        X_folds = cross_validation_splits(X, config.CV_N_SPLITS, config.CV_TEST_SIZE, config.CV_GAP,
                                            config.CV_MAX_TRAIN_SIZE)
        y_folds = cross_validation_splits(y, config.CV_N_SPLITS, config.CV_TEST_SIZE, config.CV_GAP,
                                            config.CV_MAX_TRAIN_SIZE)
        logger.info("Using cross-validation with %d folds, test_size=%s, gap=%s, max_train_size=%s",
                            config.CV_N_SPLITS, config.CV_TEST_SIZE, config.CV_GAP, config.CV_MAX_TRAIN_SIZE)
        return X_folds, y_folds
    else:
        X_train, X_test = split_train_test(X, config.TRAIN_START, config.TRAIN_END, config.TEST_START, config.TEST_END)
        y_train, y_test = split_train_test(y, config.TRAIN_START, config.TRAIN_END, config.TEST_START, config.TEST_END)
        logger.info("Using train/test split: train=%s..%s, test=%s..%s",
                    config.TRAIN_START, config.TRAIN_END, config.TEST_START, config.TEST_END)
        return X_train, X_test, y_train, y_test


def run() -> None:
    # 0. Fail fast on inconsistent config values
    validate_config()
    # 1. Load all sources (from cache or API)
    features_data, targets_data = load_sources()
    # 2. Prepare targets & features: align, merge, add lags & calendar features
    target = load_targets(targets_data)
    features = load_features(features_data, target)
    # Keep only target rows that have a complete feature row (lags, merges may drop rows)
    target = target.loc[features.index]

    ######################################################
    context_df = target.reset_index()
    context_df.columns = ["timestamp", "target"]
    context_df["item_id"] = "wind_belgium"
    context_df = context_df[["item_id", "timestamp", "target"]]

    context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])
    context_df["timestamp"] = context_df["timestamp"].dt.tz_localize(None)

    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
    )

    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=39,
        quantile_levels = [0.1, 0.5, 0.9]
    )

    print(pred_df.shape)
    print(pred_df.head())
    print(pred_df.tail())
    '''
    X_train, X_test, y_train, y_test = split(features, target)
    rmse, y_pred = run_model(config.MODEL_TYPE, X_train, X_test, y_train, y_test)
    logger.info("Test RMSE: %.2f MW", rmse)

    # 4. Plot the forecast & save it to a file
    plot_forecast(y_test, y_pred, rmse, config.PLOT_DIRECTORY / f"forecast_{config.TARGET_SOURCE}_{config.MODEL_TYPE}.png")
    '''

if __name__ == "__main__":
    run()
