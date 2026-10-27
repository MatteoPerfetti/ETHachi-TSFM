from pathlib import Path

import numpy as np
import pandas as pd

import config
from logger import logger
from src.datasource import load_elia, load_openmeteo
from src.features import (
    add_calendar_features,
    add_lag_features,
    encode_directions,
    interpolate_to_15min,
    merge,
    merge_dwd_weather,
)
from src.http_client import HttpClient
from src.metrics import calculate_metrics, metrics_summary_table
from src.models import cross_validation_splits, run_model, split_train_test
from src.models_tsfm import TSFMRunner
from src.plotter import plot_benchmark_comparison
from src.validate_config import validate_config


def load_sources() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Carica i dati grezzi Elia e Open-Meteo da cache parquet locale o tramite API."""
    features_data = {}
    targets_data = {}
    for name, src in config.ELIA_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            targets_data[name] = load_elia(name, client, src["region"], config.DATA_START, config.DATA_END, config.RESOLUTIONCODE)
    for name, src in config.OPENMETEO_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            features_data[name] = load_openmeteo(
                name, client, src["latitude"], src["longitude"], src["hourly"],
                config.DATA_START, config.DATA_END
            )
    return features_data, targets_data


def load_features(features_data: dict[str, pd.DataFrame], target: pd.DataFrame) -> pd.DataFrame:
    """Allinea a 15 min, calcola direzioni, lag tabulari e feature di calendario."""
    if config.DATA_USE_DWD:
        features = merge_dwd_weather(features_data["weather"], config.DATA_DWD_PATH)
    else:
        for name, df in features_data.items():
            features_data[name] = interpolate_to_15min(df)
        features = merge(features_data)

    features = encode_directions(features)
    lag_features = add_lag_features(target, config.CONTEXT_LENGTH, config.PREDICTION_LENGTH)
    features = features.join(lag_features, how="inner")
    features = add_calendar_features(features, config.TIMEZONE, config.COUNTRY)

    return features


def load_targets(targets_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prepara il DataFrame del target aggregando le colonne se single-target."""
    target = targets_data[config.TARGET_SOURCE]
    if config.MULTILE_TARGETS:
        target = target.add_prefix(f"{config.TARGET_SOURCE}_")
        target = target.add_suffix("_MW")
        return target
    else:
        target = target.sum(axis=1).to_frame(name=f"{config.TARGET_SOURCE}_MW")
    return target


def split(X: pd.DataFrame, y: pd.DataFrame):
    """Esegue train/test split o cross-validation per i modelli tradizionali."""
    if config.USE_CROSS_VALIDATION:
        X_folds = cross_validation_splits(X, config.CV_N_SPLITS, config.CV_TEST_SIZE, config.CV_GAP, config.CV_MAX_TRAIN_SIZE)
        y_folds = cross_validation_splits(y, config.CV_N_SPLITS, config.CV_TEST_SIZE, config.CV_GAP, config.CV_MAX_TRAIN_SIZE)
        logger.info("Using cross-validation with %d folds", config.CV_N_SPLITS)
        return X_folds, y_folds
    else:
        X_train, X_test = split_train_test(X, config.TRAIN_START, config.TRAIN_END, config.TEST_START, config.TEST_END)
        y_train, y_test = split_train_test(y, config.TRAIN_START, config.TRAIN_END, config.TEST_START, config.TEST_END)
        logger.info(
            "Using train/test split: train=%s..%s, test=%s..%s",
            config.TRAIN_START, config.TRAIN_END, config.TEST_START, config.TEST_END,
        )
        return X_train, X_test, y_train, y_test


def run() -> None:
    # 0. Validazione preliminare dei parametri
    validate_config()

    # 1. Caricamento dati grezzi
    features_data, targets_data = load_sources()

    # 2. Preparazione target e feature
    target = load_targets(targets_data)
    features = load_features(features_data, target)
    target = target.loc[features.index]

    # 3. Definizione orizzonte e contesto
    horizon = config.HORIZON_STEPS  # 156 step = 39 ore
    context_target_all = target.loc[target.index < config.TEST_START]
    
    # Seleziona la finestra di contesto standard configurata
    context_steps = config.DEFAULT_CONTEXT  # 1344 step = 14 giorni
    if len(context_target_all) > context_steps:
        context_window = context_target_all.iloc[-context_steps:]
    else:
        context_window = context_target_all

    logger.info("Context target timesteps: %d (Window: %s)", len(context_window), "14D")

    # Ground truth per le 39 ore
    y_true_39h = target.loc[target.index >= config.TEST_START].iloc[:horizon]
    y_true_vals = y_true_39h.values.squeeze()

    forecasts = {}
    all_metrics = {}

    # ==================== 1. Chronos-2 Zero-Shot ====================
    logger.info("Running Chronos-2 Zero-Shot...")
    res_chronos = TSFMRunner.run_chronos(context_window, horizon=horizon)
    forecasts["Chronos-2"] = {
        "median": res_chronos.median,
        "q10": res_chronos.q10,
        "q90": res_chronos.q90,
    }
    all_metrics["Chronos-2"] = calculate_metrics(
        y_true_vals, res_chronos.median, res_chronos.q10, res_chronos.q90
    )

    # ==================== 2. Timestral-209M Zero-Shot ====================
    logger.info("Running Timestral-209M Zero-Shot...")
    res_timestral = TSFMRunner.run_timestral(context_window, horizon=horizon)
    forecasts["Timestral-209M"] = {
        "median": res_timestral.median,
        "q10": res_timestral.q10,
        "q90": res_timestral.q90,
    }
    all_metrics["Timestral-209M"] = calculate_metrics(
        y_true_vals, res_timestral.median, res_timestral.q10, res_timestral.q90
    )

    # ==================== 3. TimesFM-3 Zero-Shot ====================
    logger.info("Running TimesFM-3 Zero-Shot...")
    res_timesfm = TSFMRunner.run_timesfm(context_window, horizon=horizon)
    forecasts["TimesFM-3"] = {
        "median": res_timesfm.median,
        "q10": res_timesfm.q10,
        "q90": res_timesfm.q90,
    }
    all_metrics["TimesFM-3"] = calculate_metrics(
        y_true_vals, res_timesfm.median, res_timesfm.q10, res_timesfm.q90
    )

    # ==================== 4. XGBoost Baseline (Opzionale) ====================
    run_xgboost = "xgboost" in config.TRADITIONAL_MODELS
    if run_xgboost:
        logger.info("Running XGBoost Baseline...")
        try:
            X_train, X_test, y_train, y_test = split(features, target)
            _, y_pred_xgb_full = run_model("xgboost", X_train, X_test, y_train, y_test)
            y_xgb_39h = np.asarray(y_pred_xgb_full[:horizon]).squeeze()
            forecasts["XGBoost"] = {"median": y_xgb_39h}
            all_metrics["XGBoost"] = calculate_metrics(y_true_vals, y_xgb_39h)
        except Exception as e:
            logger.warning("Could not execute XGBoost baseline: %s", e)

    # ==================== Report Metriche ====================
    summary_df = metrics_summary_table(all_metrics)
    print("\n" + "=" * 80)
    print(f"BENCHMARK METRICS SUMMARY: {config.TARGET_SOURCE.upper()} (Horizon: {horizon} steps / 39h)")
    print("=" * 80)
    print(summary_df.to_string())
    print("=" * 80 + "\n")

    # ==================== Salvataggio Cache Predizioni ====================
    cache_dict = {"Actual_MW": y_true_vals}
    for model_name, preds in forecasts.items():
        cache_dict[f"{model_name}_median"] = preds["median"]
        if preds.get("q10") is not None and preds.get("q90") is not None:
            cache_dict[f"{model_name}_q10"] = preds["q10"]
            cache_dict[f"{model_name}_q90"] = preds["q90"]

    cache_df = pd.DataFrame(cache_dict, index=y_true_39h.index)
    cache_path = config.DATA_DIR / "predictions_cache.parquet"
    cache_df.to_parquet(cache_path)
    logger.info("Predictions saved to cache at %s", cache_path)

    # ==================== Salvataggio Grafico Comparativo ====================
    plot_path = config.PLOT_DIRECTORY / "forecast_wind_tsfm_comparison_39h.png"
    plot_benchmark_comparison(
        y_true_39h,
        forecasts,
        metrics=all_metrics,
        path=plot_path,
        show=False,
    )
    logger.info("Benchmark comparison plot saved to %s", plot_path)


if __name__ == "__main__":
    run()
