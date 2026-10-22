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
from timesfm import TimesFM3Forecaster
from chronos import BaseChronosPipeline
from timestral import TimestralForecaster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error


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
    # Common settings for TSFM zero-shot forecasting
    horizon = 4 * 39  # 156 timesteps = 39 hours
    # Context: only data up to end of 2024, so predictions start at 2025-01-01 00:00
    context_target = target.loc[target.index < "2025-01-01T00:00:00Z"]

    # ==================== 1. Chronos-2 Zero-Shot ====================
    logger.info("Running Chronos-2 Zero-Shot...")
    context_df = context_target.reset_index()
    context_df.columns = ["timestamp", "target"]
    context_df["item_id"] = "wind_belgium"
    context_df = context_df[["item_id", "timestamp", "target"]]

    context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])
    context_df["timestamp"] = context_df["timestamp"].dt.tz_localize(None)

    chronos_pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
    )

    pred_chronos = chronos_pipeline.predict_df(
        context_df,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9]
    )

    y_chronos_median = pred_chronos["0.5"].values
    y_chronos_q10 = pred_chronos["0.1"].values
    y_chronos_q90 = pred_chronos["0.9"].values

    # ==================== 2. Timestral-209M Zero-Shot ====================
    logger.info("Running Timestral-209M Zero-Shot...")
    timestral_forecaster = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-209M",
        device="cpu"
    )

    # Input shape: (series=1, variates=1, time)
    context_timestral = context_target.values.squeeze().reshape(1, 1, -1)

    forecast_timestral = timestral_forecaster.forecast(
        context_timestral,
        prediction_length=horizon
    )

    y_timestral_median = forecast_timestral.median.squeeze()
    y_timestral_q10 = forecast_timestral.quantile(0.1).squeeze()
    y_timestral_q90 = forecast_timestral.quantile(0.9).squeeze()

    # ==================== 2. TimesFM-3-330M Zero-Shot ====================

    logger.info("Loading TimesFM-3-330M model...")
    timesfm_forecaster = TimesFM3Forecaster.from_pretrained(
        "google/timesfm-3.0-pytorch",
        device="cpu",  # Use "cuda" if a compatible GPU is available
    )
    # 1. Format input context from shared context_target
    # TimesFM expects a 1D NumPy float32 array for single time-series forecasting
    context_timesfm = context_target.values.squeeze().astype(np.float32)
    logger.info("Generating zero-shot forecast with TimesFM-3-330M...")
    output_timesfm = timesfm_forecaster.predict(
        context_timesfm,
        horizon=horizon,  # Shared variable: 156 timesteps (39 hours)
        return_quantiles=True,
    )
    # 2. Extract median and confidence intervals
    # output.forecast returns the median prediction (index 4 in the 9 default quantiles)
    y_timesfm_median = output_timesfm.forecast
    y_timesfm_q10 = output_timesfm.quantiles[
        :, 0
    ]  # 10th percentile (lower bound)
    y_timesfm_q90 = output_timesfm.quantiles[
        :, 8
    ]  # 90th percentile (upper bound)

    # ==================== Ground Truth & Metrics ====================
    # Real data for the first 39 hours of 2025
    y_true_39h = target.loc[target.index >= "2025-01-01T00:00:00Z"].iloc[:horizon]
    y_true_values = y_true_39h.values.squeeze()

    rmse_chronos = root_mean_squared_error(y_true_values, y_chronos_median)
    rmse_timestral = root_mean_squared_error(y_true_values, y_timestral_median)
    rmse_timesfm = root_mean_squared_error(y_true_values, y_timesfm_median)

    # ==================== Unified Comparison: Real vs Chronos vs Timestral vs TimesFM ====================
    comparison_df = pd.DataFrame({
        "Actual_MW": y_true_values,
        "Chronos2_MW": y_chronos_median,
        "Timestral209M_MW": y_timestral_median,
        "TimesFM3_MW": y_timesfm_median,
    }, index=y_true_39h.index)

    print("\n" + "=" * 80)
    print("VALUES COMPARISON: REAL (ELIA) vs CHRONOS-2 vs TIMESTRAL-209M vs TIMESFM-3")
    print(f"-> 39h RMSE Chronos-2:      {rmse_chronos:.2f} MW")
    print(f"-> 39h RMSE Timestral-209M: {rmse_timestral:.2f} MW")
    print(f"-> 39h RMSE TimesFM-3:      {rmse_timesfm:.2f} MW")
    print("=" * 80)
    print("First 8 observations:")
    print(comparison_df.head(8))
    print("\nLast 8 observations:")
    print(comparison_df.tail(8))
    print("=" * 80 + "\n")

    logger.info("39h Chronos-2      RMSE: %.2f MW", rmse_chronos)
    logger.info("39h Timestral-209M  RMSE: %.2f MW", rmse_timestral)
    logger.info("39h TimesFM-3       RMSE: %.2f MW", rmse_timesfm)

    # ==================== Comparison Plot ====================
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(y_true_39h.index, y_true_values, label="Actual (Elia)", color="#1f77b4", linewidth=2.5)
    ax.plot(y_true_39h.index, y_chronos_median, label=f"Chronos-2 (RMSE: {rmse_chronos:.1f} MW)", color="#2ca02c", linewidth=2)
    ax.fill_between(y_true_39h.index, y_chronos_q10, y_chronos_q90, color="#2ca02c", alpha=0.10, label="Chronos 80% CI")
    ax.plot(y_true_39h.index, y_timestral_median, label=f"Timestral-209M (RMSE: {rmse_timestral:.1f} MW)", color="#d62728", linestyle="--", linewidth=2)
    ax.fill_between(y_true_39h.index, y_timestral_q10, y_timestral_q90, color="#d62728", alpha=0.10, label="Timestral 80% CI")
    ax.plot(y_true_39h.index, y_timesfm_median, label=f"TimesFM-3 (RMSE: {rmse_timesfm:.1f} MW)", color="#9467bd", linestyle="-.", linewidth=2)
    ax.fill_between(y_true_39h.index, y_timesfm_q10, y_timesfm_q90, color="#9467bd", alpha=0.10, label="TimesFM 80% CI")

    ax.set_title("First 39 Hours of 2025: Actual vs Chronos-2 vs Timestral-209M vs TimesFM-3 Zero-Shot")
    ax.set_xlabel("Date & Time")
    ax.set_ylabel("Wind Power (MW)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = config.PLOT_DIRECTORY / "forecast_wind_chronos_39h.png"
    fig.savefig(plot_path, dpi=150)
    fig.savefig(config.PLOT_DIRECTORY / "forecast_wind_tsfm_comparison_39h.png", dpi=150)
    logger.info("Plot saved to %s", plot_path)
    plt.show()

    '''
    X_train, X_test, y_train, y_test = split(features, target)
    rmse, y_pred = run_model(config.MODEL_TYPE, X_train, X_test, y_train, y_test)
    logger.info("Test RMSE: %.2f MW", rmse)

    # 4. Plot the forecast & save it to a file
    plot_forecast(y_test, y_pred, rmse, config.PLOT_DIRECTORY / f"forecast_{config.TARGET_SOURCE}_{config.MODEL_TYPE}.png")
    '''

if __name__ == "__main__":
    run()
