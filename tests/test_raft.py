from dataclasses import replace

import numpy as np
import pandas as pd

import config
from src.backtest import BacktestSettings, run_backtest
from src.dataset import Dataset
from src.forecast import Forecaster, ForecastResult
from src.metrics import daily_scores
from src.models_tsfm import blend_tsfm_with_raft
from src.raft import HORIZON_HOURS, WEATHER_WEIGHTS, RaftRetriever, RaftSettings, production_distance
from src.raft_tuning import tune_raft


def _fixture(tmp_path):
    grid = pd.date_range("2024-01-01", periods=40 * 96, freq="15min", tz="UTC")
    times = np.arange(len(grid))
    target = pd.Series(200 + 40 * np.sin(times / 50) + times % 4, index=grid)
    columns = list(WEATHER_WEIGHTS["solar"])
    frames = []
    for issue in grid[(grid.minute == 0) & (grid.hour % 6 == 3)]:
        index = pd.date_range(issue, periods=HORIZON_HOURS, freq="h")
        frame = pd.DataFrame({
            name: np.arange(HORIZON_HOURS) * (i + 1) + issue.day
            for i, name in enumerate(columns)
        })
        frames.append(frame.assign(issue_time=issue, timestamp=index))
    archive = tmp_path / "weather.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(archive)
    settings = RaftSettings(str(archive), 400, neighbors=10)
    return target, columns, settings


def test_constant_production_has_finite_distance():
    q = np.zeros(72)
    keys = np.array([q, np.ones(72), np.arange(72)])
    np.testing.assert_allclose(production_distance(q, keys, 100, alpha=0.5),
                               [0, 0.25 + 0.005, 0.25 + 0.1775])
    np.testing.assert_allclose(production_distance(q + 0.1, np.array([q + 0.1, q + 0.3]), 100, alpha=1),
                               [0, 0.5])


def test_raft_never_uses_future_target_or_forecast_runs(tmp_path, monkeypatch):
    target, columns, settings = _fixture(tmp_path)
    start = target.index[0]
    split = pd.Timestamp("2024-01-25", tz="UTC")
    issue = split + pd.Timedelta(hours=9)
    original = RaftRetriever(settings, target, "solar", 9, start, split)
    distance = original._production_distance

    def check_production_resolution(query, candidates):
        expected = target.loc[issue - pd.Timedelta(hours=72):issue - pd.Timedelta(minutes=15)]
        np.testing.assert_array_equal(query, expected.to_numpy())
        assert original.keys.shape[1] == 288
        return distance(query, candidates)

    monkeypatch.setattr(original, "_production_distance", check_production_resolution)
    assert original.weather.shape[1:] == (39, len(columns))

    altered = target.copy()
    altered.loc[issue:] += 500
    archive = pd.read_parquet(settings.weather_archive)
    archive.loc[archive["issue_time"] > issue, columns] += 10_000
    archive.to_parquet(settings.weather_archive)
    changed = RaftRetriever(settings, altered, "solar", 9, start, split)
    pd.testing.assert_frame_equal(original.predict(issue).quantiles, changed.predict(issue).quantiles)
    assert changed.predict(issue - pd.Timedelta(hours=1)) is None


def test_backtest_adds_one_blend_per_selected_tsfm(tmp_path, monkeypatch):
    target, _, raft_settings = _fixture(tmp_path)
    model_weather = pd.DataFrame({"tsfm_weather": np.arange(len(target), dtype=float)}, index=target.index)
    dataset = Dataset(target, model_weather)

    class StubTSFM(Forecaster):
        uses_covariates = True

        def __init__(self, name):
            self.name = name

        def predict(self, requests):
            results = []
            for request in requests:
                pd.testing.assert_frame_equal(request.past_covariates, model_weather.loc[request.context.index])
                pd.testing.assert_frame_equal(request.future_covariates, model_weather.loc[request.forecast_index])
                point = np.full(len(request.forecast_index), 100.0)
                results.append(ForecastResult.from_model_output(
                    self.name, request, {0.1: point - 20, 0.5: point, 0.9: point + 20}
                ))
            return results

    monkeypatch.setattr("src.backtest.load_forecaster", lambda key, settings: StubTSFM(key))
    settings = BacktestSettings(
        train_start="2024-01-01T00:00Z", train_end="2024-01-25T00:00Z",
        test_start="2024-01-25T00:00Z", test_end="2024-01-28T00:00Z",
        target_source="solar", covariates=["tsfm_weather"], raft=raft_settings,
    )
    output = run_backtest(dataset, config.TSFM_MODELS, settings).predictions
    assert set(output["model"]) == {"RAFT", *config.TSFM_MODELS, *(f"{key} + RAFT" for key in config.TSFM_MODELS)}
    points = output.pivot(index=["issue_time", "timestamp"], columns="model", values="q50")
    for key in config.TSFM_MODELS:
        combined = f"{key} + RAFT"
        np.testing.assert_allclose(points[combined], 0.5 * points[key] + 0.5 * points["RAFT"])
        assert output.loc[output["model"] == combined, ["q10", "q90"]].isna().all().all()
    tsfm = output[output["model"] == config.TSFM_MODELS[0]]
    raft = output[output["model"] == "RAFT"]
    endpoint = blend_tsfm_with_raft(tsfm, raft, gamma=1)
    np.testing.assert_array_equal(endpoint[["q10", "q50", "q90"]], tsfm[["q10", "q50", "q90"]])

    base_settings = replace(settings, raft=None)
    assert base_settings.run_name != settings.run_name
    assert set(run_backtest(dataset, config.TSFM_MODELS, base_settings).predictions["model"]) == set(config.TSFM_MODELS)

    calls = []

    def validate_once(data, models, validation):
        calls.append(validation)
        return run_backtest(data, models, validation)

    monkeypatch.setattr("src.raft_tuning.run_backtest", validate_once)
    tuned = tune_raft(dataset, config.TSFM_MODELS, settings, "2024-01-18T00:00Z", trials=4)
    assert len(calls) == 1
    assert calls[0].train_end == "2024-01-18T00:00:00+00:00"
    assert pd.Timestamp(tuned.metadata["validation_last_issue"]) + pd.Timedelta(hours=39) <= pd.Timestamp(
        settings.test_start
    )
    assert tuned.metadata["trials"] == 4
    assert np.isfinite(tuned.metadata["validation_rmse_mw"])
    assert tuned.settings.alpha == tuned.metadata["best_params"]["alpha"]
    replay = run_backtest(dataset, config.TSFM_MODELS, replace(calls[0], raft=tuned.settings)).predictions
    combined = replay.loc[replay["model"].str.endswith(" + RAFT")]
    np.testing.assert_allclose(daily_scores(combined, window="all")["RMSE"].mean(),
                               tuned.metadata["validation_rmse_mw"])
