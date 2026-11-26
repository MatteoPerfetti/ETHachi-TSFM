import numpy as np
import pandas as pd
import pytest

from src import backtest, context_sweep
from src.dataset import Dataset, RollingHorizon
from src.forecast import ForecastResult

TEST_PERIOD = {"test_start": "2024-12-02T00:00:00Z", "test_end": "2024-12-06T00:00:00Z", "covariates": []}


def issue_times(settings):
    horizons = RollingHorizon.daily(settings.test_start, settings.test_end, settings.issue_time,
                                    settings.horizon_steps, settings.latency_steps, settings.issue_every_days)
    return [h.issue_time for h in horizons]


class FakeModel:
    """q50 = last context value, q10/q90 = q50 -/+ 1; records the context length it receives."""
    max_context = None
    uses_covariates = False

    def __init__(self, name, min_context=1):
        self.name = name
        self.min_context = min_context
        self.context_lengths = []

    def fit(self, *_args):
        pass

    def predict(self, requests):
        results = []
        for r in requests:
            self.context_lengths.append(len(r.context))
            last = float(r.context.iloc[-1])
            values = {0.1: [last - 1] * r.model_horizon, 0.5: [last] * r.model_horizon,
                      0.9: [last + 1] * r.model_horizon}
            results.append(ForecastResult.from_model_output(self.name, r, values))
        return results


@pytest.fixture
def fake_models(monkeypatch):
    created = []

    def load(key, settings, *_args):
        if key == "persistence":
            model = FakeModel("Persistence", min_context=settings.horizon_steps)
        else:
            model = FakeModel(context_sweep.model_name(key))
        created.append(model)
        return model
    monkeypatch.setattr(backtest, "load_forecaster", load)
    return created


def dataset(gap_at=None):
    index = pd.date_range("2024-11-20", "2024-12-10", freq="15min", tz="UTC", name="timestamp")
    target = pd.Series(np.sin(np.arange(len(index)) / 20) * 100 + 500, index=index, name="wind_MW")
    if gap_at is not None:
        target.loc[pd.Timestamp(gap_at)] = np.nan
    return Dataset(target=target, weather=pd.DataFrame(index=index))


def test_issue_every_days_thins_issue_times_and_names_the_run():
    daily = backtest.BacktestSettings(**TEST_PERIOD)
    weekly = backtest.BacktestSettings(issue_every_days=2, **TEST_PERIOD)
    assert len(issue_times(daily)) == 4
    assert [t.day for t in issue_times(weekly)] == [2, 4]
    assert "_every" not in daily.run_name
    assert weekly.run_name.endswith("_every2d")
    with pytest.raises(ValueError):
        issue_times(backtest.BacktestSettings(issue_every_days=0, **TEST_PERIOD))


def test_sweep_runs_each_context_and_persistence_once(fake_models, tmp_path):
    paths = context_sweep.run_sweep([2, 1], ["chronos2"], dataset_loader=dataset, directory=tmp_path,
                                    overrides=TEST_PERIOD)
    assert sorted(paths) == [1, 2]
    assert all(p.exists() for p in paths.values())
    assert [m.name for m in fake_models] == ["Persistence", "Chronos-2", "Chronos-2"]
    # the second Chronos-2 run (2 days) gets twice the context of the first
    assert set(fake_models[1].context_lengths) == {96}
    assert set(fake_models[2].context_lengths) == {192}

    fake_models.clear()
    context_sweep.run_sweep([1, 2], ["chronos2"], dataset_loader=dataset, directory=tmp_path,
                            overrides=TEST_PERIOD, reuse=True)
    assert fake_models == []  # everything already saved


def test_table_uses_only_common_issue_days(fake_models, tmp_path):
    # a gap 45 h before the first issue time: inside the 2-day context only (persistence needs 39 h),
    # so only the 2-day run skips 2024-12-02
    gap = "2024-11-30T12:00:00Z"
    paths = context_sweep.run_sweep([1, 2], ["chronos2"], dataset_loader=lambda: dataset(gap),
                                    directory=tmp_path, overrides=TEST_PERIOD)
    predictions, timings = context_sweep.load_sweep(paths, ["chronos2"])
    assert set(predictions["context"]) == {"-", "1d", "2d"}
    assert len(context_sweep.common_issue_times(predictions)) == 3

    table = context_sweep.sweep_table(predictions, timings)
    assert set(table["days"]) == {3}
    assert list(table["context"]) == ["1d", "2d", "-"]  # sorted by model then context; persistence has none
    assert table.loc[table["model"] == "Persistence", "RMSE_vs_persistence"].item() == 0
    assert table["coverage80"].between(0, 1).all()
    assert table["s_per_forecast"].notna().all()
    assert list(context_sweep.pivot(table, "RMSE").columns) == ["1d", "2d"]

    out, csv_path = context_sweep.report(paths, ["chronos2"], out_dir=tmp_path / "sweep")
    assert csv_path.exists() and csv_path.with_suffix(".txt").exists()
    assert "1 issue days dropped" in csv_path.with_suffix(".txt").read_text(encoding="utf-8")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        context_sweep.model_name("xgboost")
