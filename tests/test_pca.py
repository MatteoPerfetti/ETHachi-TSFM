import numpy as np
import pandas as pd

from src.forecast import ForecastRequest
from src.models_traditional import TraditionalMLAdapter
from src.pca import CovariatePCA


def _weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    base = np.sin(np.arange(len(index)) / 20)
    return pd.DataFrame({"speed": base, "gust": 2 * base + 0.01 * np.cos(np.arange(len(index))),
                         "pressure": np.cos(np.arange(len(index)) / 50)}, index=index)


def test_pca_replaces_only_the_covariate_columns():
    index = pd.date_range("2024-01-01", periods=500, freq="15min", tz="UTC")
    weather = _weather(index)
    frame = weather.assign(lead_step=1)

    pca = CovariatePCA(2, list(weather.columns)).fit(frame)
    reduced = pca.transform(frame)

    assert reduced.columns.tolist() == ["lead_step", "pc_1", "pc_2"]
    assert reduced.index.equals(frame.index)


def test_traditional_models_get_pca_after_the_origin_aligned_build():
    index = pd.date_range("2024-01-01", periods=12 * 96, freq="15min", tz="UTC")
    target = pd.Series(1000 + 200 * np.sin(np.arange(len(index)) / 20), index=index, name="wind_MW")
    adapter = TraditionalMLAdapter("xgboost", "origin_aligned", horizon_steps=4, latency_steps=0, context_steps=3,
                                   parameters={"n_estimators": 5, "max_depth": 2}, pca_components=2)

    adapter.fit(target, _weather(index))

    assert {"pc_1", "pc_2", "lead_step", "wind_MW_lag_1", "is_holiday"} <= set(adapter.feature_columns)
    assert not {"speed", "gust", "pressure"} & set(adapter.feature_columns)


def test_pca_transforms_request_covariates():
    index = pd.date_range("2024-01-01", periods=500, freq="15min", tz="UTC", name="timestamp")
    weather = _weather(index)
    pca = CovariatePCA(2, list(weather.columns)).fit(weather)
    issue = index[100]
    context = pd.Series(1.0, index=index[:100], name="wind_MW")
    forecast_index = index[100:104]
    request = ForecastRequest(issue, forecast_index, context,
                              past_covariates=weather.iloc[:100], future_covariates=weather.iloc[100:104])

    reduced = pca.transform_request(request)

    assert reduced.past_covariates.columns.tolist() == ["pc_1", "pc_2"]
    assert reduced.future_covariates.columns.tolist() == ["pc_1", "pc_2"]
