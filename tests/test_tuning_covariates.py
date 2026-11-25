"""Chronos-2 training inputs with covariates: schema, windows drawn by Chronos, and the backtest covariate check."""
import numpy as np
import pandas as pd
import pytest
import torch

import config
from src import backtest
from src.tuning_data import build_covariate_inputs

COVARIATES = [f"c{k}" for k in range(1, 8)]          # sorted order = row order inside a Chronos group
OFFSET = 100_000                                     # covariate k = time index + k * OFFSET


def synthetic(start="2024-01-01", end="2025-01-01"):
    """Target = time index (0, 1, 2, ...), covariate k = time index + k * OFFSET: values identify their timestamp."""
    index = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
    position = np.arange(len(index), dtype=np.float64)
    target = pd.Series(position, index=index, name="wind_MW")
    weather = pd.DataFrame({name: position + k * OFFSET for k, name in enumerate(COVARIATES, start=1)}, index=index)
    return target, weather


def test_inputs_have_chronos_schema_and_december_windows():
    target, weather = synthetic()
    training, validation, origins = build_covariate_inputs(target, weather, context=96, horizon=24,
                                                           covariates=COVARIATES)
    assert len(training) == 1
    first = training[0]
    assert set(first) == {"target", "past_covariates", "future_covariates"}
    assert first["future_covariates"] == {name: None for name in COVARIATES}
    # training series: TRAIN_START (inclusive) to 1 December 2024 (exclusive)
    assert len(first["target"]) == (pd.Timestamp("2024-12-01", tz="UTC") - pd.Timestamp(config.TRAIN_START)) / pd.Timedelta("15min")
    assert origins[0] == pd.Timestamp("2024-12-01 09:00", tz="UTC")
    assert all(origin.hour == 9 and origin.minute == 0 for origin in origins)
    for window, origin in zip(validation, origins):
        assert len(window["target"]) == 96 + 24
        start = target.index.get_loc(origin) - 96            # first context step
        np.testing.assert_array_equal(window["target"], np.arange(start, start + 120, dtype=np.float32))
        np.testing.assert_array_equal(window["past_covariates"]["c3"], window["target"] + 3 * OFFSET)


def test_missing_values_in_training_period_raise():
    target, weather = synthetic()
    weather.iloc[500, 2] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        build_covariate_inputs(target, weather, context=96, horizon=24, covariates=COVARIATES)


def test_unknown_covariate_raises():
    target, weather = synthetic()
    with pytest.raises(ValueError, match="not in the weather"):
        build_covariate_inputs(target, weather, context=96, horizon=24, covariates=["nope"])


def test_random_training_windows_cut_target_and_covariates_at_the_same_timestamps():
    """30-day windows drawn by Chronos2Dataset (as in fit) keep target and covariates aligned."""
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    context, horizon, windows = 2880, 156, 8
    target, weather = synthetic()
    training, _, _ = build_covariate_inputs(target, weather, context, horizon, COVARIATES)
    rows = 1 + len(COVARIATES)
    np.random.seed(0)
    dataset = Chronos2Dataset(inputs=training, context_length=context, prediction_length=horizon,
                              batch_size=windows * rows, output_patch_size=16, min_past=context,
                              mode=DatasetMode.TRAIN)
    batch = next(iter(dataset))

    assert batch["context"].shape == (windows * rows, context)
    assert torch.equal(batch["group_ids"], torch.arange(windows).repeat_interleave(rows))
    for w in range(windows):
        group = slice(w * rows, (w + 1) * rows)
        past, future_target, future_cov = (batch[k][group] for k in ("context", "future_target", "future_covariates"))
        cut = int(past[0, -1]) + 1                       # first forecast step of this window
        assert cut >= context
        np.testing.assert_array_equal(past[0].numpy(), np.arange(cut - context, cut))
        np.testing.assert_array_equal(future_target[0].numpy(), np.arange(cut, cut + horizon))
        assert torch.isnan(future_cov[0]).all()          # the target is not known in the future
        assert torch.isnan(future_target[1:]).all()      # no loss on the covariate rows
        for k in range(1, rows):
            np.testing.assert_array_equal(past[k].numpy(), np.arange(cut - context, cut) + k * OFFSET)
            np.testing.assert_array_equal(future_cov[k].numpy(), np.arange(cut, cut + horizon) + k * OFFSET)


@pytest.mark.parametrize("settings_covariates, error", [
    ([], "trained with the covariates"),
    (COVARIATES[:3], "trained with the covariates"),
    (list(reversed(COVARIATES)), "Adapter files missing"),   # same set, any order: passes the covariate check
])
def test_adapter_trained_with_covariates_needs_the_same_covariates(monkeypatch, tmp_path, settings_covariates, error):
    record = {"target": "wind_MW", "covariates": COVARIATES, "backend": "peft",
              "base_checkpoint": config.MODELS_REGISTRY["chronos2"]["checkpoint"], "base_revision": "r"}
    monkeypatch.setattr("src.tsfm_runs.available_adapters", lambda: {"lora:chronos2:cov": record})
    monkeypatch.setattr(config, "BACKTEST_RESULTS_DIR", tmp_path)
    settings = backtest.BacktestSettings(covariates=settings_covariates, target_source="wind")
    with pytest.raises(ValueError, match=error):
        backtest.load_forecaster("lora:chronos2:cov", settings)
