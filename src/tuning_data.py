"""Training and validation windows for the standalone foundation-model trainers.

`load_windows` gives wind-only tensors (Chronos-2 custom/PEFT and Timestral). `load_covariate_inputs` gives the same
period as Chronos-2 input dicts with the weather covariates: `Chronos2Pipeline.fit` takes them directly, and the
Timestral trainer cuts its windows from them.

Three validation schemes for the covariate inputs:
- "december": the 09:00 forecasts of December 2024, with the training context; training stops on 1 December.
- "weeks": the 09:00 forecasts of days 1-7 of every month of 2024 whose history allows the validation context
  (January has none: the data starts on 28 December 2023). Training covers all of 2024, with the target set to
  NaN over every validation forecast, so no training window learns from (or sees) a held-out value.
- "weeks-loss": the same validation forecasts, but the training target keeps its real values and the training input
  gets a boolean "loss_mask" (True over every validation forecast). The trainer leaves those steps out of the loss
  only, so training contexts have no gaps (as in the 2025 test) while no held-out value is ever a label.
  Timestral trainer only; Chronos2Pipeline.fit does not accept the extra key.
"""

import numpy as np
import pandas as pd
import torch

import config
from src.forecast import STEP


VALIDATION_START = pd.Timestamp("2024-12-01T00:00:00Z")
VALIDATION_SCHEMES = ("december", "weeks", "weeks-loss")
HELD_OUT_DAYS = 7  # "weeks": days 1-7 of every month are validation forecasts


def load_windows(context: int, horizon: int) -> tuple[torch.Tensor, torch.Tensor, list[pd.Timestamp]]:
    start, end = pd.Timestamp(config.DATA_START), pd.Timestamp(config.DATA_END)
    path = config.DATA_DIR / "wind" / f"{config.RESOLUTIONCODE}_{start:%Y%m%d}_{end:%Y%m%dT}.parquet"
    regional = pd.read_parquet(path, columns=config.ELIA_SOURCES["wind"]["regions"])
    regional.index = pd.to_datetime(regional.index, utc=True)
    target = regional.sum(axis=1, min_count=1)
    train_start, train_end = pd.Timestamp(config.TRAIN_START), pd.Timestamp(config.TRAIN_END)
    train_index = pd.date_range(train_start, VALIDATION_START, freq=STEP, inclusive="left")
    values = torch.tensor(target.reindex(train_index).to_numpy(dtype=np.float32))
    if len(values) < context + horizon or not torch.isfinite(values).all():
        raise ValueError("Training history is too short or has missing wind values")
    training = values.unfold(0, context + horizon, 1)

    validation, origins = [], []
    issue_offset = pd.Timedelta(f"{config.BACKTEST_ISSUE_TIME}:00")
    for origin in pd.date_range(VALIDATION_START + issue_offset, train_end, freq="D", inclusive="left"):
        if origin + (horizon - 1) * STEP >= train_end:
            break
        index = pd.date_range(origin - context * STEP, periods=context + horizon, freq=STEP)
        window = target.reindex(index).to_numpy(dtype=np.float32)
        if np.isfinite(window).all():
            validation.append(window)
            origins.append(origin)
    if not validation:
        raise ValueError("No complete December validation windows")
    return training, torch.tensor(np.stack(validation)), origins


def _chronos_input(target: pd.Series, weather: pd.DataFrame, covariates: list[str]) -> dict:
    """One Chronos-2 input dict; future_covariates=None marks every covariate as known into the future."""
    return {"target": target.to_numpy(dtype=np.float32),
            "past_covariates": {name: weather[name].to_numpy(dtype=np.float32) for name in covariates},
            "future_covariates": {name: None for name in covariates}}


def build_covariate_inputs(target: pd.Series, weather: pd.DataFrame, context: int, horizon: int,
                           covariates: list[str], validation: str = "december",
                           validation_context: int | None = None) -> tuple[list[dict], list[dict], list[pd.Timestamp]]:
    """Training series and validation windows with covariates (see the module docstring for the schemes).

    The training input is one continuous series: Chronos2Dataset draws random windows of `context` steps plus
    `horizon` steps from it, cutting target and covariates at the same timestamps. Each validation input covers
    `validation_context + horizon` steps ending at the horizon of one 09:00 origin; Chronos uses its last
    `horizon` steps as labels and the known covariates over them as future inputs. `validation_context` defaults
    to `context`.
    """
    if not covariates:
        raise ValueError("No covariates given; use load_windows for wind-only training")
    if validation not in VALIDATION_SCHEMES:
        raise ValueError(f"validation must be one of {VALIDATION_SCHEMES}, got {validation!r}")
    missing = sorted(set(covariates) - set(weather.columns))
    if missing:
        raise ValueError(f"Covariates not in the weather data: {missing}")
    frame = weather[covariates].join(target.rename("__target__"), how="outer")
    if validation in ("weeks", "weeks-loss"):
        return _held_out_weeks(frame, context, horizon, covariates, validation_context or context,
                               hide_context=validation == "weeks")

    train_index = pd.date_range(pd.Timestamp(config.TRAIN_START), VALIDATION_START, freq=STEP, inclusive="left")
    train = frame.reindex(train_index)
    if len(train) < context + horizon:
        raise ValueError("Training history is too short for the requested context and horizon")
    if not np.isfinite(train.to_numpy(dtype=np.float64)).all():
        bad = train.columns[train.isna().any()].tolist()
        raise ValueError(f"Training period has missing values in {bad}")
    training = [_chronos_input(train["__target__"], train, covariates)]

    validation, origins = [], []
    issue_offset = pd.Timedelta(f"{config.BACKTEST_ISSUE_TIME}:00")
    train_end = pd.Timestamp(config.TRAIN_END)
    for origin in pd.date_range(VALIDATION_START + issue_offset, train_end, freq="D", inclusive="left"):
        if origin + (horizon - 1) * STEP > train_end:
            break
        window = frame.reindex(pd.date_range(origin - context * STEP, periods=context + horizon, freq=STEP))
        if np.isfinite(window.to_numpy(dtype=np.float64)).all():
            validation.append(_chronos_input(window["__target__"], window, covariates))
            origins.append(origin)
    if not validation:
        raise ValueError("No complete December validation windows")
    return training, validation, origins


def _held_out_weeks(frame: pd.DataFrame, context: int, horizon: int, covariates: list[str],
                    validation_context: int, hide_context: bool = True
                    ) -> tuple[list[dict], list[dict], list[pd.Timestamp]]:
    """Validation forecasts on days 1-7 of each month. hide_context=True ("weeks") sets the training target to NaN
    over them; False ("weeks-loss") keeps the target and marks them in a "loss_mask" of the training input."""
    train_start, train_end = pd.Timestamp(config.TRAIN_START), pd.Timestamp(config.TRAIN_END)
    issue_offset = pd.Timedelta(f"{config.BACKTEST_ISSUE_TIME}:00")
    validation, origins = [], []
    for month in pd.date_range(train_start.normalize(), train_end, freq="MS"):
        for day in range(HELD_OUT_DAYS):
            origin = month + pd.Timedelta(days=day) + issue_offset
            if origin + (horizon - 1) * STEP > train_end:
                continue
            window = frame.reindex(pd.date_range(origin - validation_context * STEP,
                                                 periods=validation_context + horizon, freq=STEP))
            if np.isfinite(window.to_numpy(dtype=np.float64)).all():   # skipped if the history is too short
                validation.append(_chronos_input(window["__target__"], window, covariates))
                origins.append(origin)
    if not validation:
        raise ValueError("No complete held-out-week validation windows")

    train = frame.reindex(pd.date_range(train_start, train_end, freq=STEP))
    if len(train) < context + horizon:
        raise ValueError("Training history is too short for the requested context and horizon")
    if not np.isfinite(train.to_numpy(dtype=np.float64)).all():
        bad = train.columns[train.isna().any()].tolist()
        raise ValueError(f"Training period has missing values in {bad}")
    held_out = np.zeros(len(train), dtype=bool)
    for origin in origins:
        first = train.index.get_loc(origin)
        held_out[first:first + horizon] = True
    if not hide_context:
        return [{**_chronos_input(train["__target__"], train, covariates), "loss_mask": held_out}], validation, origins
    train.loc[held_out, "__target__"] = np.nan   # Chronos masks NaN targets in the context and in the loss
    return [_chronos_input(train["__target__"], train, covariates)], validation, origins


def load_covariate_inputs(context: int, horizon: int, covariates: list[str], validation: str = "december",
                          validation_context: int | None = None) -> tuple[list[dict], list[dict], list[pd.Timestamp]]:
    """Wind target and weather exactly as the backtest sees them (src.dataset.load_dataset), as Chronos inputs."""
    from src.dataset import load_dataset

    dataset = load_dataset("wind")
    return build_covariate_inputs(dataset.target, dataset.weather, context, horizon, covariates,
                                  validation, validation_context)
