"""Shared wind-only windows for the standalone foundation-model trainers."""

import numpy as np
import pandas as pd
import torch

import config
from src.forecast import STEP


VALIDATION_START = pd.Timestamp("2024-12-01T00:00:00Z")


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
