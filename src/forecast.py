"""Common request/result types between the backtest engine and the models.

The backtest engine is the only place that cuts the data in time: it builds one ForecastRequest per issue time.
Every model takes requests and returns ForecastResults, so metrics and plots never depend on the model.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

QUANTILE_COLUMNS = {0.1: "q10", 0.5: "q50", 0.9: "q90"}  # quantile level -> result column
STEP = pd.Timedelta(config.RESOLUTIONCODE)


def _check_utc_index(index: pd.Index, what: str) -> None:
    """Raise ValueError unless `index` is a non-empty, UTC, regularly spaced DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        raise ValueError(f"{what}: expected a non-empty DatetimeIndex")
    if str(index.tz) != "UTC":
        raise ValueError(f"{what}: timestamps must be UTC, got tz={index.tz}")
    if len(index) > 1 and not (np.diff(index.asi8) == STEP.value).all():
        raise ValueError(f"{what}: timestamps must be regular with step {STEP}")


def _check_covariates(df: pd.DataFrame, index: pd.DatetimeIndex, what: str) -> None:
    """Raise ValueError unless `df` covers exactly `index` without missing values."""
    if not df.index.equals(index):
        raise ValueError(f"{what}: index must be exactly {index[0]}..{index[-1]} ({len(index)} steps)")
    if df.isna().any().any():
        raise ValueError(f"{what}: missing values in {df.columns[df.isna().any()].tolist()}")


@dataclass(frozen=True)
class ForecastRequest:
    """One forecast issued at `issue_time` for the timestamps in `forecast_index`.

    The model sees `context` (target observed up to the data latency before the issue time) and optionally
    weather covariates: `past_covariates` on the context timestamps, `future_covariates` on every step after
    the context, i.e. `model_index` (the latency gap plus the forecast timestamps).
    """
    issue_time: pd.Timestamp
    forecast_index: pd.DatetimeIndex
    context: pd.Series
    past_covariates: pd.DataFrame | None = None
    future_covariates: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.issue_time.tzinfo is None or str(self.issue_time.tz) != "UTC":
            raise ValueError(f"issue_time must be UTC, got {self.issue_time}")
        _check_utc_index(self.forecast_index, "forecast_index")
        _check_utc_index(self.context.index, "context")
        if self.context.isna().any():
            raise ValueError(f"context: {int(self.context.isna().sum())} missing target values")
        if self.context.index[-1] >= self.issue_time:
            raise ValueError(f"context ends at {self.context.index[-1]}, not before issue_time {self.issue_time}")
        if self.forecast_index[0] < self.issue_time:
            raise ValueError(f"forecast_index starts at {self.forecast_index[0]}, before issue_time {self.issue_time}")

        if self.past_covariates is not None:
            _check_covariates(self.past_covariates, self.context.index, "past_covariates")
        if self.future_covariates is not None:
            if self.past_covariates is None:
                raise ValueError("future_covariates need past_covariates with the same columns")
            if not self.future_covariates.columns.equals(self.past_covariates.columns):
                raise ValueError("past_covariates and future_covariates must have the same columns")
            _check_covariates(self.future_covariates, self.model_index, "future_covariates")

    @property
    def model_horizon(self) -> int:
        """Steps the model has to predict after the end of the context (latency gap + forecast timestamps)."""
        return int((self.forecast_index[-1] - self.context.index[-1]) / STEP)

    @property
    def model_index(self) -> pd.DatetimeIndex:
        """Timestamps of the `model_horizon` steps following the context."""
        return pd.date_range(self.context.index[-1] + STEP, periods=self.model_horizon, freq=STEP, name="timestamp")


@dataclass(frozen=True)
class ForecastResult:
    """Quantile forecast of one model for one request: index = forecast timestamps, columns = q10, q50, q90.

    q50 is the point forecast. Models without quantiles (e.g. persistence) leave q10 and q90 as NaN.
    """
    model: str
    issue_time: pd.Timestamp
    quantiles: pd.DataFrame

    def __post_init__(self) -> None:
        if list(self.quantiles.columns) != list(QUANTILE_COLUMNS.values()):
            raise ValueError(f"quantiles columns must be {list(QUANTILE_COLUMNS.values())}, "
                             f"got {list(self.quantiles.columns)}")
        _check_utc_index(self.quantiles.index, "quantiles")
        if self.quantiles["q50"].isna().any():
            raise ValueError(f"{self.model}: q50 (point forecast) has missing values")

    @classmethod
    def from_model_output(cls, model: str, request: ForecastRequest,
                          values: dict[float, np.ndarray]) -> "ForecastResult":
        """Build a result from raw model output over `request.model_index`, keeping only the forecast timestamps.

        `values` maps quantile level -> array of length `request.model_horizon`; missing levels become NaN.
        """
        frame = pd.DataFrame(index=request.model_index)
        for level, column in QUANTILE_COLUMNS.items():
            array = values.get(level)
            if array is None:
                frame[column] = np.nan
                continue
            array = np.asarray(array, dtype=float).reshape(-1)
            if len(array) != request.model_horizon:
                raise ValueError(f"{model}: quantile {level} has {len(array)} values, "
                                 f"expected model_horizon = {request.model_horizon}")
            frame[column] = array
        return cls(model=model, issue_time=request.issue_time, quantiles=frame.loc[request.forecast_index])

    def to_long(self) -> pd.DataFrame:
        """Rows (issue_time, timestamp, model, q10, q50, q90), the format of the backtest results table."""
        out = self.quantiles.rename_axis("timestamp").reset_index()
        out.insert(0, "issue_time", self.issue_time)
        out.insert(2, "model", self.model)
        return out


class Forecaster(ABC):
    """Interface of every model (traditional or TSFM) used by the backtest engine.

    min_context: steps of target history the model needs at least (e.g. the largest lag of a tabular model).
    max_context: longest context the architecture accepts, None = unlimited; longer contexts get cut by the model.
    uses_covariates: whether the request covariates enter the forecast.
    """
    name: str
    min_context: int = 1
    max_context: int | None = None
    uses_covariates: bool = False

    def fit(self, target: pd.Series, covariates: pd.DataFrame | None = None) -> None:
        """Train on history that the backtest engine already cut to the training period. Default: nothing to train."""

    @abstractmethod
    def predict(self, requests: list[ForecastRequest]) -> list[ForecastResult]:
        """Forecast every request, results in the same order."""
