"""SARIMAX wind-generation benchmark.

Implementation:
- Fit a non-seasonal ARIMA(1, 1, 1) with DWD wind-speed, gust, encoded
  wind-direction, temperature, humidity, pressure, and local time-of-day
  covariates.
- Standardize covariates with training-fold means and standard deviations.
- Refit on an expanding target history at each daily 09:00 Brussels origin and
  forecast 156 15-minute timestamps (39 hours, including the origin timestamp).

Assumptions and limitations:
- Future DWD covariates are historical weather observations, not weather forecasts archived by issue time. 
  This is an oracle-weather backtest and may overstate operational forecast skill.
- Exploratory 2024 comparison over 12 monthly origins: the wind-only baseline
  scored 905.6 MW aggregate RMSE; adding temperature, humidity, and pressure
  means scored 778.7 MW. Adding wind-speed and gust standard deviations on top
  of those weather means changed aggregate RMSE only slightly (775.2 MW).
- These results are exploratory and use historical future-valid weather values;
  they are not a leakage-free estimate of operational forecast performance.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from config import (
    FORECAST_DAYTIME,
    PREDICTION_LENGTH,
    SARIMAX_ENFORCE_INVERTIBILITY,
    SARIMAX_ENFORCE_STATIONARITY,
    SARIMAX_FIT_METHOD,
    SARIMAX_LOW_MEMORY,
    SARIMAX_MAXITER,
    SARIMAX_ORDER,
    SARIMAX_SEASONAL_ORDER,
    TEST_END,
    TEST_START,
    TIMEZONE,
)

def fit_sarimax(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    X_future: pd.DataFrame,
    order: tuple[int, int, int] = SARIMAX_ORDER,
    seasonal_order: tuple[int, int, int, int] = SARIMAX_SEASONAL_ORDER,
) -> tuple[SARIMAXResultsWrapper, pd.Series]:
    """Fit SARIMAX with covariates and predict using future covariates."""

    X_train = X_train.reindex(y_train.index)
    X_future = X_future.reindex(
        columns=X_train.columns
    )

    y_values = y_train.astype(float)
    X_train = X_train.astype(float)
    X_future = X_future.astype(float)

    exog_mean = X_train.mean()
    exog_scale = X_train.std().replace(0, 1)
    X_train = (X_train - exog_mean) / exog_scale # scale the whole exogeneous timeseries (both past and forecast)
    X_future = (X_future - exog_mean) / exog_scale

    model = sm.tsa.statespace.SARIMAX(
        endog=y_values,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=SARIMAX_ENFORCE_STATIONARITY,
        enforce_invertibility=SARIMAX_ENFORCE_INVERTIBILITY,
    )

    fitted_model = model.fit(
        method=SARIMAX_FIT_METHOD,
        maxiter=SARIMAX_MAXITER,
        disp=False,
        low_memory=SARIMAX_LOW_MEMORY,
    )

    forecast_result = fitted_model.get_forecast(
        steps=len(X_future),
        exog=X_future,
    )

    forecast = pd.Series(
        np.asarray(
            forecast_result.predicted_mean
        ),
        index=X_future.index,
        name="predicted",
    )

    return fitted_model, forecast


def rolling_origin_sarimax(
    y: pd.Series,
    exog: pd.DataFrame,
    evaluation_index: pd.DatetimeIndex,
    prediction_length: int = PREDICTION_LENGTH,
    forecast_time: str = FORECAST_DAYTIME,
    timezone: str = TIMEZONE,
    order: tuple[int, int, int] = SARIMAX_ORDER,
    seasonal_order: tuple[int, int, int, int] = SARIMAX_SEASONAL_ORDER,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> pd.DataFrame:
    """Evaluate SARIMAX at daily origins using only prior targets and aligned covariates."""

    y = y.sort_index()
    exog = exog.sort_index()

    test_start_ts = pd.Timestamp(test_start).tz_convert(timezone)
    test_end_ts = pd.Timestamp(test_end).tz_convert(timezone)
    hour, minute = map(int, forecast_time.split(":"))
    local_days = pd.date_range(
        start=test_start_ts.tz_localize(None).normalize(),
        end=test_end_ts.tz_localize(None).normalize(),
        freq="1D",
    )
    local_origins = (
        local_days + pd.Timedelta(hours=hour, minutes=minute)
    ).tz_localize(timezone)
    forecast_origins = local_origins.tz_convert(y.index.tz)
    test_start_y_tz = test_start_ts.tz_convert(y.index.tz)
    test_end_y_tz = test_end_ts.tz_convert(y.index.tz)
    evaluation_index = evaluation_index.tz_convert(y.index.tz)
    horizon_duration = pd.Timedelta(minutes=15 * (prediction_length - 1))

    forecast_origins = forecast_origins[
        (forecast_origins >= test_start_y_tz)
        & (forecast_origins + horizon_duration <= test_end_y_tz)
        & forecast_origins.isin(evaluation_index)
    ]

    results = []
    for forecast_origin in forecast_origins:
        forecast_index = pd.date_range(
            start=forecast_origin,
            periods=prediction_length,
            freq="15min",
        )
        y_train = y.loc[y.index < forecast_origin].dropna()
        X_train = exog.loc[exog.index < forecast_origin].reindex(y_train.index)
        X_future = exog.reindex(forecast_index)

        fitted_model, forecast = fit_sarimax(
            y_train=y_train,
            X_train=X_train,
            X_future=X_future,
            order=order,
            seasonal_order=seasonal_order,
        )
        converged = fitted_model.mle_retvals.get("converged", False)

        actual = y.reindex(forecast_index)
        results.append(
            pd.DataFrame(
                {
                    "forecast_origin": forecast_origin,
                    "target_time": forecast_index,
                    "lead_step": np.arange(1, prediction_length + 1),
                    "actual": actual.to_numpy(),
                    "predicted": forecast.to_numpy(),
                    "converged": converged,
                }
            )
        )

    if not results:
        raise ValueError("No eligible rolling forecast origins were found")

    return (
        pd.concat(results, ignore_index=True)
        .set_index(["forecast_origin", "target_time"])
        .sort_index()
    )