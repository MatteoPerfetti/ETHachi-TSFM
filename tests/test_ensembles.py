import numpy as np
import pandas as pd

import config
from src.backtest import BacktestSettings, run_backtest
from src.dataset import Dataset
from src.ensembles import equal_mean_ensemble, top_k_exponential_ensemble


def _predictions(
    issues: list[pd.Timestamp],
    forecasts: dict[str, list[float]],
    actuals: list[float],
    horizon_steps: int = 1,
) -> pd.DataFrame:
    rows = []
    for issue_number, issue in enumerate(issues):
        for model, values in forecasts.items():
            for lead_step in range(horizon_steps):
                point = values[issue_number]
                rows.append(
                    {
                        "issue_time": issue,
                        "timestamp": issue + lead_step * pd.Timedelta("15min"),
                        "lead_step": lead_step,
                        "model": model,
                        "q10": point - 10,
                        "q50": point,
                        "q90": point + 10,
                        "actual": actuals[issue_number],
                    }
                )
    return pd.DataFrame(rows)


def test_equal_mean_combines_all_quantiles_and_records_weights():
    issue = pd.Timestamp("2025-01-01 09:00", tz="UTC")
    predictions = _predictions(
        [issue],
        {"A": [100.0], "B": [200.0]},
        [150.0],
    )

    result = equal_mean_ensemble(predictions, models=["A", "B"])

    assert result.predictions.loc[0, ["q10", "q50", "q90"]].tolist() == [140.0, 150.0, 160.0]
    assert result.weights["weight"].tolist() == [0.5, 0.5]


def test_equal_mean_leaves_interval_empty_if_one_model_has_no_quantiles():
    issue = pd.Timestamp("2025-01-01 09:00", tz="UTC")
    predictions = _predictions(
        [issue],
        {"A": [100.0], "Point only": [200.0]},
        [150.0],
    )
    predictions.loc[predictions["model"] == "Point only", ["q10", "q90"]] = np.nan

    result = equal_mean_ensemble(predictions)

    assert result.predictions.loc[0, "q50"] == 150.0
    assert result.predictions.loc[0, ["q10", "q90"]].isna().all()


def test_top_k_weights_match_the_report_three_day_example():
    issues = list(pd.date_range("2025-01-01 09:00", periods=4, freq="D", tz="UTC"))
    predictions = _predictions(
        issues,
        {
            "A": [0.0, 10.0, 0.0, 100.0],
            "B": [10.0, 0.0, 10.0, 200.0],
        },
        [0.0, 0.0, 0.0, 150.0],
    )

    result = top_k_exponential_ensemble(
        predictions,
        lookback_days=3,
        top_k=1,
        decay=1.0,
    )

    current = result.weights[result.weights["issue_time"] == issues[-1]].set_index("model")
    normalizer = 1 + np.exp(-1) + np.exp(-2)
    assert np.isclose(current.loc["A", "weight"], (1 + np.exp(-2)) / normalizer)
    assert np.isclose(current.loc["B", "weight"], np.exp(-1) / normalizer)


def test_top_k_does_not_score_an_overlapping_unfinished_horizon():
    issues = list(pd.date_range("2025-01-01 09:00", periods=3, freq="D", tz="UTC"))
    predictions = _predictions(
        issues,
        {
            "A": [0.0, 10.0, 100.0],
            "B": [10.0, 0.0, 200.0],
        },
        [0.0, 0.0, 150.0],
        horizon_steps=100,
    )

    result = top_k_exponential_ensemble(
        predictions,
        lookback_days=2,
        top_k=1,
    )

    current = result.weights[result.weights["issue_time"] == issues[-1]]
    assert current[["model", "weight"]].to_dict("records") == [{"model": "A", "weight": 1.0}]
    assert current.iloc[0]["source_days"] == 1


def test_backtest_runs_ensemble_dependencies_without_exposing_them(monkeypatch):
    index = pd.date_range("2024-01-01", "2024-01-08", freq="15min", tz="UTC")
    values = np.arange(len(index))
    dataset = Dataset(
        target=pd.Series(1000 + 100 * np.sin(values / 25), index=index, name="wind_MW"),
        weather=pd.DataFrame({"weather": np.cos(values / 18)}, index=index),
    )
    monkeypatch.setattr(config, "ENSEMBLE_BASE_MODELS", ["xgboost", "lightgbm"])
    monkeypatch.setitem(
        config.TRADITIONAL_MODEL_PARAMETERS,
        "xgboost",
        {
            "n_estimators": 2,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
    )
    monkeypatch.setitem(
        config.TRADITIONAL_MODEL_PARAMETERS,
        "lightgbm",
        {
            "n_estimators": 2,
            "num_leaves": 5,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
    )
    settings = BacktestSettings(
        train_start="2024-01-01T00:00:00Z",
        train_end="2024-01-04T00:00:00Z",
        test_start="2024-01-05T00:00:00Z",
        test_end="2024-01-08T00:00:00Z",
        horizon_steps=4,
        context_steps=8,
        covariates=["weather"],
        traditional_strategy="fixed_gap",
    )

    result = run_backtest(
        dataset,
        ["equal_mean", "top_k_exponential"],
        settings,
    )

    assert sorted(result.predictions["model"].unique()) == [
        "Equal mean ensemble [fixed_gap]",
        "Top-K exponential ensemble [fixed_gap]",
    ]
    assert result.predictions[["q10", "q50", "q90"]].notna().all().all()
    assert set(result.timings) == set(result.predictions["model"].unique())
    assert all(
        values["strategy"] == "fixed_gap"
        for values in result.model_configs.values()
    )
