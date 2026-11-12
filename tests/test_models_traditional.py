import numpy as np
import pandas as pd

import config
from src.backtest import BacktestSettings, run_backtest
from src.dataset import Dataset
from src.forecast import ForecastRequest
from src.models_traditional import TraditionalMLAdapter


def _request(context_length: int = 12, horizon: int = 4) -> ForecastRequest:
    issue = pd.Timestamp("2025-01-03 09:00", tz="UTC")
    context_index = pd.date_range(
        end=issue - pd.Timedelta("15min"),
        periods=context_length,
        freq="15min",
        name="timestamp",
    )
    forecast_index = pd.date_range(issue, periods=horizon, freq="15min", name="timestamp")
    model_index = pd.date_range(
        context_index[-1] + pd.Timedelta("15min"),
        forecast_index[-1],
        freq="15min",
        name="timestamp",
    )
    context = pd.Series(np.arange(context_length, dtype=float), index=context_index, name="wind_MW")
    past = pd.DataFrame({"weather": np.arange(context_length)}, index=context_index)
    future = pd.DataFrame({"weather": np.arange(len(model_index))}, index=model_index)
    return ForecastRequest(issue, forecast_index, context, past, future)


def test_origin_aligned_repeats_the_latest_context_for_every_lead():
    request = _request()
    adapter = TraditionalMLAdapter("xgboost", "origin_aligned", horizon_steps=4,
                                   latency_steps=0, context_steps=3)

    features = adapter._origin_aligned_features(request)

    assert features["lead_step"].tolist() == [1, 2, 3, 4]
    assert features["wind_MW_lag_1"].tolist() == [11.0] * 4
    assert features["wind_MW_lag_3"].tolist() == [9.0] * 4


def test_fixed_gap_moves_safe_lags_with_each_target_timestamp():
    request = _request()
    adapter = TraditionalMLAdapter("xgboost", "fixed_gap", horizon_steps=4,
                                   latency_steps=0, context_steps=3)

    features = adapter._fixed_gap_features(request)

    assert features["wind_MW_lag_4"].tolist() == [8.0, 9.0, 10.0, 11.0]
    assert features["wind_MW_lag_6"].tolist() == [6.0, 7.0, 8.0, 9.0]
    assert adapter.min_context == 6


def test_xgboost_adapter_fits_and_returns_shared_quantile_result():
    index = pd.date_range("2024-01-01", periods=12 * 96, freq="15min", tz="UTC")
    target = pd.Series(
        1000 + 200 * np.sin(np.arange(len(index)) / 20),
        index=index,
        name="wind_MW",
    )
    covariates = pd.DataFrame(
        {"weather": np.cos(np.arange(len(index)) / 15)},
        index=index,
    )
    adapter = TraditionalMLAdapter(
        "xgboost",
        "origin_aligned",
        horizon_steps=4,
        latency_steps=0,
        context_steps=3,
        parameters={
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
    )
    adapter.fit(target, covariates)

    result = adapter.predict([_request()])[0]

    assert result.model == "XGBoost [origin_aligned]"
    assert result.quantiles.columns.tolist() == ["q10", "q50", "q90"]
    assert len(result.quantiles) == 4
    assert result.quantiles.notna().all().all()


def test_shared_backtest_runs_both_traditional_feature_modes(monkeypatch):
    index = pd.date_range("2024-01-01", "2024-01-07", freq="15min", tz="UTC")
    values = np.arange(len(index))
    dataset = Dataset(
        target=pd.Series(1000 + 100 * np.sin(values / 25), index=index, name="wind_MW"),
        weather=pd.DataFrame({"weather": np.cos(values / 18)}, index=index),
    )
    monkeypatch.setitem(
        config.TRADITIONAL_MODEL_PARAMETERS,
        "xgboost",
        {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
    )

    for strategy in config.TRADITIONAL_STRATEGIES:
        settings = BacktestSettings(
            train_start="2024-01-01T00:00:00Z",
            train_end="2024-01-04T00:00:00Z",
            test_start="2024-01-05T00:00:00Z",
            test_end="2024-01-06T00:00:00Z",
            horizon_steps=4,
            context_steps=8,
            covariates=["weather"],
            traditional_strategy=strategy,
        )
        result = run_backtest(dataset, ["xgboost"], settings)

        assert len(result.predictions) == 4
        assert result.predictions["model"].unique().tolist() == [f"XGBoost [{strategy}]"]
        assert result.predictions[["q10", "q50", "q90"]].notna().all().all()
        assert result.model_configs[f"XGBoost [{strategy}]"]["strategy"] == strategy
