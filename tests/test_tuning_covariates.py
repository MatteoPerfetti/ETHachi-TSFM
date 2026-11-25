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


def test_held_out_weeks_validate_with_30_days_and_hide_their_forecasts_from_training():
    target, weather = synthetic(start="2024-01-01", end="2025-01-01")
    context, horizon, validation_context = 96, 156, 2880
    training, validation, origins = build_covariate_inputs(target, weather, context, horizon, COVARIATES,
                                                           validation="weeks", validation_context=validation_context)
    # January has no 30-day history in this data; days 1-7 of February to December do
    assert len(origins) == 11 * 7
    assert {(o.month, o.day, o.hour) for o in origins} == {(m, d, 9) for m in range(2, 13) for d in range(1, 8)}
    for window, origin in zip(validation, origins):
        start = target.index.get_loc(origin) - validation_context
        np.testing.assert_array_equal(window["target"], np.arange(start, start + validation_context + horizon,
                                                                  dtype=np.float32))
    # training covers all of 2024; the target is NaN exactly over the validation forecasts, covariates are kept
    series = training[0]["target"]
    assert len(series) == len(target)
    hidden = np.zeros(len(series), dtype=bool)
    for origin in origins:
        first = target.index.get_loc(origin)
        hidden[first:first + horizon] = True
    assert np.isnan(series[hidden]).all() and np.isfinite(series[~hidden]).all()
    np.testing.assert_array_equal(training[0]["past_covariates"]["c3"], target.to_numpy() + 3 * OFFSET)
    first_hidden = target.index.get_loc(pd.Timestamp("2024-03-01 09:00", tz="UTC"))
    last_hidden = target.index.get_loc(pd.Timestamp("2024-03-08 23:45", tz="UTC"))
    assert np.isnan(series[first_hidden:last_hidden + 1]).all()
    assert np.isfinite(series[[first_hidden - 1, last_hidden + 1]]).all()


def test_missing_values_in_training_period_raise():
    target, weather = synthetic()
    weather.iloc[500, 2] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        build_covariate_inputs(target, weather, context=96, horizon=24, covariates=COVARIATES)


def test_unknown_covariate_raises():
    target, weather = synthetic()
    with pytest.raises(ValueError, match="not in the weather"):
        build_covariate_inputs(target, weather, context=96, horizon=24, covariates=["nope"])


def test_timestral_training_sampler_keeps_covariates_aligned_and_skips_empty_windows():
    from tuning.train_timestral import covariate_windows

    values = np.arange(10, dtype=np.float32)
    target = values.copy()
    target[1:6] = np.nan
    windows, weather, eligible = covariate_windows(
        {"target": target, "past_covariates": {"c1": values + OFFSET}}, ["c1"], context=3, horizon=3,
    )
    assert eligible.tolist() == [4]
    np.testing.assert_array_equal(windows[eligible].numpy(), [target[4:10]])
    np.testing.assert_array_equal(weather[eligible, 0].numpy(), [values[4:10] + OFFSET])


def test_timestral_covariates_condition_target_loss_and_backpropagate_with_missing_targets():
    from peft import LoraConfig, get_peft_model
    from timestral import TimestralConfig, TimestralForecaster, TimestralModel

    from tuning.train_timestral import TARGET_MODULES, training_loss

    torch.manual_seed(0)
    model = TimestralModel(TimestralConfig(
        patch_size=8, d_model=32, num_heads=4, mlp_hidden_dim=64,
        block_schedule=("time", "time_variate"), quantile_knots=(0.1, 0.5, 0.9),
        block_size=4, context_length=64,
    ))
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    adapter = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=list(TARGET_MODULES)))
    forecaster = TimestralForecaster(adapter.get_base_model())
    context, horizon = 17, 11  # Exercise padding at both ends of the input.
    batch = torch.randn(2, context + horizon)
    batch[0, 2:5] = torch.nan
    batch[:, context + 1:context + 5] = torch.nan
    weather = torch.randn(2, 2, context + horizon)
    changed = weather.clone()
    changed[:, :, context:] += 10

    loss = training_loss(forecaster, batch, context, weather)
    assert torch.isfinite(loss)
    assert abs((loss - training_loss(forecaster, batch, context, changed)).item()) > 1e-5
    separate = torch.stack([training_loss(forecaster, row[None], context, cov[None])
                            for row, cov in zip(batch, weather)]).mean()
    torch.testing.assert_close(loss, separate)
    loss.backward()
    grads = [p.grad for p in adapter.parameters() if p.requires_grad]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert any(torch.count_nonzero(grad) for grad in grads)


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
