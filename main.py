import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

import config
from logger import logger
from src.datasource import load_elia, load_openmeteo
from src.features import (
    add_calendar_features,
    add_lag_features,
    encode_directions,
    interpolate_to_15min,
    load_dwd_weather,
    merge,
)
from src.http_client import HttpClient
from src.MLtrad import (
    build_origin_aligned_dataset,
    cross_validate_traditional_model,
    fit_traditional_model,
    predict_quantiles,
    quantile_column,
    score_quantile_forecast,
    tune_traditional_model,
)
from src.models import split_train_test
from src.plotter import plot_forecast, plot_rolling_forecasts
from src.statistical_models import rolling_origin_sarimax
from src.validate_config import validate_config

def load_sources() -> dict[str, pd.DataFrame]:
    """Load configured targets and the weather source selected in config."""
    features_data = {}
    targets_data = {}
    for name, src in config.ELIA_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            targets_data[name] = load_elia(name, client, src["region"], config.DATA_START, config.DATA_END, config.RESOLUTIONCODE)
    if not config.DATA_USE_DWD:
        for name, src in config.OPENMETEO_SOURCES.items():
            with HttpClient(src["base_url"]) as client:
                features_data[name] = load_openmeteo(name, client, src["latitude"], src["longitude"], src["hourly"],
                                               config.DATA_START, config.DATA_END)
    return features_data, targets_data



def load_covariates(features_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build aligned 15-minute weather and calendar covariates."""
    # Use the DWD/ICON dataset as the single weather provider.
    if config.DATA_USE_DWD:
        features = load_dwd_weather(config.DATA_DWD_PATH)
    else:
        for name, df in features_data.items():
            features_data[name] = interpolate_to_15min(df)
        features = merge(features_data)
    # Convert wind direction features (sin/cos) to the merged data
    features = encode_directions(features)
    features = add_calendar_features(features, config.TIMEZONE, config.COUNTRY)
    return features


def build_fixed_gap_features(
    covariates: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Combine covariates with the original horizon-wide safe lag window."""
    lag_features = add_lag_features(
        target,
        config.CONTEXT_LENGTH,
        config.PREDICTION_LENGTH,
    )
    covariates.index = covariates.index.astype(target.index.dtype)
    return covariates.join(lag_features, how="inner")

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

def run() -> None:
    # 0. Fail fast on inconsistent config values
    validate_config()
    # 1. Load all sources (from cache or API)
    features_data, targets_data = load_sources()
    # 2. Prepare targets & features: align, merge, add lags & calendar features
    target = load_targets(targets_data)

    if config.MODEL_TYPE == "sarimax":

        y = (
            target.iloc[:, 0]
            .astype(float)
            .sort_index()
        )

        test_start = pd.Timestamp(config.TEST_START)
        test_end = pd.Timestamp(config.TEST_END)

        # DWD weather observations do not preserve the forecast vintage
        # available at each origin, so this evaluation is a proxy.
        logger.warning(
            "SARIMAX evaluation uses historical DWD weather covariates, not "
            "weather forecasts archived by issue time; results may be optimistic."
        )
        dwd_weather = pd.read_parquet(config.DATA_DWD_PATH)
        if "timestamp" not in dwd_weather.columns:
            raise ValueError(
                f"DWD weather data at {config.DATA_DWD_PATH} has no timestamp column"
            )
        dwd_weather.index = pd.DatetimeIndex(
            pd.to_datetime(dwd_weather.pop("timestamp"), utc=True),
            name="timestamp",
        )
        exog_columns = config.SARIMAX_COVARIATES
        missing_exog_columns = set(exog_columns).difference(dwd_weather.columns)
        if missing_exog_columns:
            raise ValueError(
                "DWD weather data is missing required SARIMAX covariates: "
                f"{sorted(missing_exog_columns)}"
            )
        weather_exog = dwd_weather[list(exog_columns)].copy()
        direction_mean_columns = [
            column
            for column in weather_exog.columns
            if column.startswith("wind_direction_") and column.endswith("_mean")
        ]
        encoded_directions = encode_directions(
            weather_exog[direction_mean_columns]
        )
        weather_exog = weather_exog.drop(
            columns=direction_mean_columns
        ).join(encoded_directions)
        local_weather_index = weather_exog.index.tz_convert(config.TIMEZONE)
        hour_of_day = (
            local_weather_index.hour
            + local_weather_index.minute / 60
        )
        weather_exog["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
        weather_exog["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)

        # ============================================================
        # Final rolling test evaluation
        # ============================================================

        evaluation_index = y.index[
            (y.index >= test_start)
            & (y.index < test_end)
        ]

        results = rolling_origin_sarimax(
            y=y,
            exog=weather_exog,
            evaluation_index=evaluation_index,
            prediction_length=config.PREDICTION_LENGTH,
            forecast_time=config.FORECAST_DAYTIME,
            timezone=config.TIMEZONE,
            order=config.SARIMAX_ORDER,
            seasonal_order=config.SARIMAX_SEASONAL_ORDER,
            test_start=config.TEST_START,
            test_end=config.TEST_END,
        )

        # ============================================================
        # Compare SARIMAX against persistence
        # ============================================================

        valid_results = results.dropna(
            subset=["actual", "predicted"]
        )

        comparison = valid_results.reset_index().copy()

        last_observed_by_origin = {}

        for origin in comparison["forecast_origin"].unique():
            available_history = y.loc[
                y.index < origin
            ]

            last_observed_by_origin[origin] = (
                available_history.iloc[-1]
            )

        comparison["persistence"] = (
            comparison["forecast_origin"]
            .map(last_observed_by_origin)
        )

        valid_comparison = comparison.dropna(
            subset=[
                "actual",
                "predicted",
                "persistence",
            ]
        )

        sarimax_rmse = root_mean_squared_error(
            valid_comparison["actual"],
            valid_comparison["predicted"],
        )

        persistence_rmse = root_mean_squared_error(
            valid_comparison["actual"],
            valid_comparison["persistence"],
        )

        logger.info(
            "SARIMAX rolling-origin RMSE: %.2f MW",
            sarimax_rmse,
        )

        logger.info(
            "Persistence rolling-origin RMSE: %.2f MW",
            persistence_rmse,
        )

        print(f"SARIMAX RMSE:     {sarimax_rmse:.2f} MW")
        print(f"Persistence RMSE: {persistence_rmse:.2f} MW")

        # ============================================================
        # Forecast diagnostics
        # ============================================================

        diagnostics = (
            results.reset_index()
            .groupby("forecast_origin")
            .agg(
                number_of_predictions=("predicted", "size"),
                first_target=("target_time", "min"),
                last_target=("target_time", "max"),
                minimum_prediction=("predicted", "min"),
                maximum_prediction=("predicted", "max"),
            )
        )

        print("Rolling-origin diagnostics:")
        print(diagnostics)

        # ============================================================
        # Plot all rolling forecast horizons
        # ============================================================

        plot_rolling_forecasts(
            results=results,
            rmse=sarimax_rmse,
            timezone=config.TIMEZONE,
            path=(
                config.PLOT_DIRECTORY
                / f"rolling_{config.TARGET_SOURCE}_sarimax.png"
            ),
        )

        # Do not continue into the feature-based ML workflow.
        return

    # ============================================================
    # Feature-based models
    # ============================================================

    if config.MODEL_TYPE in config.ML_MODELS:
        covariates = load_covariates(features_data)

        if config.ML_FORECAST_STRATEGY == "origin_aligned":
            X_train, y_train = build_origin_aligned_dataset(
                target=target,
                covariates=covariates,
                start=config.TRAIN_START,
                end=config.TRAIN_END,
                context_length=config.CONTEXT_LENGTH,
                prediction_length=config.PREDICTION_LENGTH,
                forecast_time=config.FORECAST_DAYTIME,
                timezone=config.TIMEZONE,
                frequency=config.RESOLUTIONCODE,
            )
            X_test, y_test = build_origin_aligned_dataset(
                target=target,
                covariates=covariates,
                start=config.TEST_START,
                end=config.TEST_END,
                context_length=config.CONTEXT_LENGTH,
                prediction_length=config.PREDICTION_LENGTH,
                forecast_time=config.FORECAST_DAYTIME,
                timezone=config.TIMEZONE,
                frequency=config.RESOLUTIONCODE,
            )
        else:
            features = build_fixed_gap_features(covariates, target)
            target = target.loc[features.index]
            X_train, X_test = split_train_test(
                features,
                config.TRAIN_START,
                config.TRAIN_END,
                config.TEST_START,
                config.TEST_END,
            )
            y_train, y_test = split_train_test(
                target,
                config.TRAIN_START,
                config.TRAIN_END,
                config.TEST_START,
                config.TEST_END,
            )
        logger.info(
            "Using %s ML dataset: train=%s..%s, test=%s..%s",
            config.ML_FORECAST_STRATEGY,
            config.TRAIN_START,
            config.TRAIN_END,
            config.TEST_START,
            config.TEST_END,
        )
        logger.info(
            "ML matrices: train=%d rows x %d features, test=%d rows x %d features",
            *X_train.shape,
            *X_test.shape,
        )

        if config.UNTUNED:
            model_params = None
            cv_pinball, _ = cross_validate_traditional_model(
                config.MODEL_TYPE,
                X_train,
                y_train,
                config.QUANTILES,
            )
        else:
            model_params, cv_pinball = tune_traditional_model(
                config.MODEL_TYPE,
                X_train,
                y_train,
                config.QUANTILES,
                config.OPTUNA_N_TRIALS,
            )

        logger.info(
            "%s validation mean pinball loss: %.2f MW",
            config.MODEL_TYPE,
            cv_pinball,
        )
        logger.info("Selected parameters: %s", model_params or "defaults")

        fitted_model = fit_traditional_model(
            config.MODEL_TYPE,
            X_train,
            y_train,
            config.QUANTILES,
            model_params,
        )
        predictions = predict_quantiles(fitted_model, X_test)
        metrics = score_quantile_forecast(
            y_test,
            predictions,
            config.QUANTILES,
        )

        logger.info("Test RMSE (Q50): %.2f MW", metrics["rmse"])
        logger.info(
            "Test mean pinball loss: %.2f MW",
            metrics["mean_pinball_loss"],
        )
        logger.info(
            "Test mean Winkler interval: %.2f MW",
            metrics["mean_winkler_interval"],
        )
        logger.info(
            "Test Q10-Q90 coverage: %.1f%%",
            100 * metrics["interval_coverage"],
        )

        if config.ML_FORECAST_STRATEGY == "origin_aligned":
            results = predictions.copy()
            results["lead_step"] = X_test["lead_step"]
            results["actual"] = y_test
            results["predicted"] = predictions[quantile_column(0.5)]
            plot_rolling_forecasts(
                results=results,
                rmse=metrics["rmse"],
                timezone=config.TIMEZONE,
                path=(
                    config.PLOT_DIRECTORY
                    / f"rolling_{config.TARGET_SOURCE}_{config.MODEL_TYPE}.png"
                ),
                forecast_name=config.MODEL_TYPE.upper(),
                lower_column=quantile_column(0.1),
                upper_column=quantile_column(0.9),
                title_note="historical weather proxy; not vintage-verified",
            )
        else:
            plot_forecast(
                y_test,
                predictions[quantile_column(0.5)],
                metrics["rmse"],
                config.PLOT_DIRECTORY
                / f"forecast_{config.TARGET_SOURCE}_{config.MODEL_TYPE}.png",
                lower=predictions[quantile_column(0.1)],
                upper=predictions[quantile_column(0.9)],
            )

if __name__ == "__main__":
    run()
