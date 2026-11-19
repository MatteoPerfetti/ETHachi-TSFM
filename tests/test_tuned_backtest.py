import pandas as pd
import json
import pytest

from src import backtest
from src.dataset import Dataset
from src.forecast import ForecastResult


def test_shared_backtest_pairs_adapter_and_base_and_preserves_history(monkeypatch, tmp_path):
    index = pd.date_range("2024-12-01", "2024-12-05", freq="15min", tz="UTC")
    dataset = Dataset(target=pd.Series(range(len(index)), index=index, name="wind_MW"),
                      weather=pd.DataFrame(index=index))
    record = {"base_revision": "revision-a"}
    monkeypatch.setattr("src.tsfm_runs.available_adapters", lambda: {"lora:chronos2:trial": record})
    calls = []

    class FakeModel:
        min_context = 1
        max_context = None
        uses_covariates = False
        def __init__(self, name):
            self.name = name
            self.configuration = {"model_key": "chronos2", "run_id": "trial"} if "lora" in name else {}
        def fit(self, *_args):
            pass
        def predict(self, requests):
            calls.append([(r.issue_time, tuple(r.context), tuple(r.forecast_index)) for r in requests])
            return [ForecastResult.from_model_output(self.name, r, {q: [1.0] * r.model_horizon for q in (0.1, 0.5, 0.9)}) for r in requests]

    def load(key, _settings, revisions):
        assert revisions == {"chronos2": "revision-a"}
        return FakeModel(key)
    monkeypatch.setattr(backtest, "load_forecaster", load)
    settings = backtest.BacktestSettings(test_start="2024-12-02T00:00:00Z", test_end="2024-12-04T00:00:00Z",
        context_steps=8, horizon_steps=4, covariates=[])
    result = backtest.run_backtest(dataset, ["chronos2", "lora:chronos2:trial"], settings)
    assert calls[0] == calls[1]
    assert result.predictions.groupby("model")["issue_time"].nunique().to_dict() == {"chronos2": 2, "lora:chronos2:trial": 2}
    first = backtest.save_backtest(result, tmp_path, preserve_history=True)
    second = backtest.save_backtest(result, tmp_path, preserve_history=True)
    assert first != second
    _, metadata = backtest.load_backtest(first.stem, tmp_path)
    assert metadata["scores_all"]["chronos2"]["days"] == 2
    assert metadata["model_configs"]["lora:chronos2:trial"]["run_id"] == "trial"


def test_failed_recorded_backtest_keeps_the_attempt(monkeypatch, tmp_path):
    def fail(*_args):
        raise ValueError("missing model")
    monkeypatch.setattr(backtest, "run_backtest", fail)
    with pytest.raises(ValueError, match="missing model"):
        backtest.run_recorded_backtest(None, ["chronos2"], backtest.BacktestSettings(), tmp_path)
    paths = list((tmp_path / "backtest_attempts").glob("*.json"))
    assert len(paths) == 1
    attempt = json.loads(paths[0].read_text(encoding="utf-8"))
    assert attempt["status"] == "failed"
    assert attempt["model_keys"] == ["chronos2"]
