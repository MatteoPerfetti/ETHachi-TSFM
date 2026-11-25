
"""Adapter between the data extraction pipeline and the forecasting code.

Only calls the extraction functions (datasource, merge helpers in features), it never changes them.
Everything downstream (models, backtest, metrics) uses `load_dataset()` and the `Dataset` it returns,
so a new extraction strategy only needs changes here.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from logger import logger
from src.datasource import hourly_variables, load_elia, load_weather_statistics, weather_points
from src.features import encode_directions, interpolate_to_15min, merge, merge_dwd_weather
from src.http_client import HttpClient


@dataclass
class Dataset:
    """Target and weather on the same regular 15 min UTC grid; missing values are explicit NaN, never dropped rows."""
    target: pd.Series        # generation in MW, name "<TARGET_SOURCE>_MW"
    weather: pd.DataFrame    # weather covariates, directions sin/cos encoded


@dataclass(frozen=True)
class RollingHorizon:
    """The forecast window of one daily issue time: `steps` timestamps from the issue time on.

    Owns which weather run each row may read, so the models only ask for plain covariate names. A row reads the
    previous day 1 run up to config.OPENMETEO_SOURCES["previous_runs"]["day1_max_lead_hours"] after the issue time
    and the previous day 2 run after that.
    """
    issue_time: pd.Timestamp
    steps: int = config.BACKTEST_HORIZON_STEPS
    step: pd.Timedelta = pd.Timedelta(config.RESOLUTIONCODE)

    @property
    def valid_times(self) -> pd.DatetimeIndex:
        return pd.date_range(self.issue_time, periods=self.steps, freq=self.step, name="timestamp")

    @property
    def lead_steps(self) -> np.ndarray:
        """1 for the issue time itself, up to `steps`."""
        return np.arange(1, self.steps + 1)

    def run_suffixes(self) -> np.ndarray:
        """Column suffix of the weather run each valid time reads: previous day 1 up to the limit, then day 2."""
        previous_runs = config.OPENMETEO_SOURCES["previous_runs"]
        _, day1, day2 = previous_runs["suffixes"]
        limit = pd.Timedelta(hours=previous_runs["day1_max_lead_hours"])
        return np.where(self.valid_times - self.issue_time <= limit, day1, day2)

    def covariates(self, weather: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Plain `columns` over the valid times, each row taken from the run its lead allows.

        `weather` must hold the run columns (<column>_previous_day1, <column>_previous_day2).
        """
        valid_times, suffixes = self.valid_times, self.run_suffixes()
        selected = pd.DataFrame(index=valid_times, columns=columns, dtype=float)
        for suffix in np.unique(suffixes):
            times = valid_times[suffixes == suffix]
            selected.loc[times] = weather.reindex(times)[[f"{column}{suffix}" for column in columns]].to_numpy()
        return selected


def load_sources(target_source: str = config.TARGET_SOURCE) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load the raw Elia data and the Open-Meteo weather statistics of target_source from the local parquet cache
    or via the API."""
    features_data = {}
    targets_data = {}
    for name, src in config.ELIA_SOURCES.items():
        with HttpClient(src["base_url"]) as client:
            targets_data[name] = load_elia(name, client, src["regions"], config.DATA_START, config.DATA_END, config.RESOLUTIONCODE)
    src = config.OPENMETEO_SOURCES
    with HttpClient(src["base_url"], timeout=120) as om:
        features_data["weather"] = load_weather_statistics(
            "weather", om, weather_points(target_source), hourly_variables(target_source), config.DATA_START,
            config.DATA_END, src["weather_sampling"][target_source]["cache_dir"])
    return features_data, targets_data


def load_targets(targets_data: dict[str, pd.DataFrame], target_source: str = config.TARGET_SOURCE) -> pd.DataFrame:
    """Target frame of target_source: one column per region if MULTILE_TARGETS, else the sum over regions."""
    target = targets_data[target_source]
    if config.MULTILE_TARGETS:
        return target.add_prefix(f"{target_source}_").add_suffix("_MW")
    # min_count=1 keeps NaN where every region is missing, instead of turning it into 0
    return target.sum(axis=1, min_count=1).to_frame(name=f"{target_source}_MW")


def load_weather(features_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align the weather to 15 min (Open-Meteo, plus DWD if configured) and encode directions as sin/cos."""
    dwd = config.OPENMETEO_SOURCES["dwd"]
    if dwd["use"]:
        weather = merge_dwd_weather(features_data["weather"], dwd["path"])
    else:
        weather = merge({name: interpolate_to_15min(df) for name, df in features_data.items()})
    return encode_directions(weather)


def regular_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Complete 15 min UTC grid from the first to the last timestamp of `index`."""
    index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return pd.date_range(index.min(), index.max(), freq=pd.Timedelta(config.RESOLUTIONCODE), name="timestamp")


def load_dataset(target_source: str = config.TARGET_SOURCE) -> Dataset:
    """Load target (default TARGET_SOURCE) and weather and put both on the regular grid spanned by the target."""
    if config.MULTILE_TARGETS:
        raise ValueError("load_dataset supports a single target only, set MULTILE_TARGETS = False")

    features_data, targets_data = load_sources(target_source)
    target = load_targets(targets_data, target_source).iloc[:, 0]
    weather = load_weather(features_data)

    grid = regular_grid(target.index)
    target = target.reindex(grid)
    weather = weather.reindex(grid)

    missing_target = int(target.isna().sum())
    missing_weather = int(weather.isna().any(axis=1).sum())
    if missing_target or missing_weather:
        logger.warning("Dataset %s..%s: %d of %d timestamps without target, %d without complete weather",
                       grid[0], grid[-1], missing_target, len(grid), missing_weather)
    logger.info("Dataset loaded: %d timestamps, target %s, %d weather columns",
                len(grid), target.name, weather.shape[1])
    return Dataset(target=target, weather=weather)


def _forecast_origins(
    index: pd.DatetimeIndex,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    prediction_length: int,
    forecast_time: str,
    timezone: str,
    frequency: str | pd.Timedelta,
) -> pd.DatetimeIndex:
    """Return daily local-time origins whose complete horizon is in the period."""
    step = pd.Timedelta(frequency)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts
    start_ts = (
        start_ts.tz_convert(index.tz)
        if index.tz is not None
        else start_ts.tz_convert(None)
    )
    end_ts = (
        end_ts.tz_convert(index.tz)
        if index.tz is not None
        else end_ts.tz_convert(None)
    )

    local_start = (
        start_ts.tz_localize("UTC").tz_convert(timezone)
        if index.tz is None
        else start_ts.tz_convert(timezone)
    )
    local_end = (
        end_ts.tz_localize("UTC").tz_convert(timezone)
        if index.tz is None
        else end_ts.tz_convert(timezone)
    )
    hour, minute = map(int, forecast_time.split(":"))
    local_days = pd.date_range(
        local_start.tz_localize(None).normalize(),
        (local_end - step).tz_localize(None).normalize(),
        freq="1D",
    )
    local_origins = (
        local_days + pd.Timedelta(hours=hour, minutes=minute)
    ).tz_localize(timezone)
    origins = (
        local_origins.tz_convert(index.tz)
        if index.tz is not None
        else local_origins.tz_convert("UTC").tz_localize(None)
    )
    last_target_offset = step * (prediction_length - 1)
    return origins[
        (origins >= start_ts)
        & (origins + last_target_offset < end_ts)
    ]


def build_origin_aligned_dataset(
    target: pd.Series | pd.DataFrame,
    covariates: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    context_length: int,
    prediction_length: int,
    forecast_time: str,
    timezone: str,
    frequency: str | pd.Timedelta = "15min",
    latency_steps: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build one pooled tabular row per forecast origin and lead step.

    Every horizon row receives the same target context ending
    ``latency_steps`` before its forecast origin. Target-time covariates and
    ``lead_step`` vary across the horizon. The resulting index is
    ``(forecast_origin, target_time)``.
    """
    target_series = (target.iloc[:, 0] if isinstance(target, pd.DataFrame) else target).astype(float).sort_index()
    covariates = covariates.sort_index()
    step = pd.Timedelta(frequency)
    origins = _forecast_origins(
        target_series.index,
        start,
        end,
        prediction_length,
        forecast_time,
        timezone,
        step,
    )
    context_columns = [
        f"{target_series.name}_lag_{lag}"
        for lag in range(1, context_length + 1)
    ]
    origin_index = origins.repeat(prediction_length)
    lead_steps = np.tile(
        np.arange(1, prediction_length + 1),
        len(origins),
    )
    target_offsets = pd.to_timedelta(
        (lead_steps - 1) * step.value,
        unit="ns",
    )
    target_index = origin_index + target_offsets
    pooled_index = pd.MultiIndex.from_arrays(
        [origin_index, target_index],
        names=["forecast_origin", "target_time"],
    )

    context_by_origin = np.column_stack([
        target_series.reindex(
            origins - (latency_steps + lag) * step
        ).to_numpy()
        for lag in range(1, context_length + 1)
    ])
    context = pd.DataFrame(
        np.repeat(context_by_origin, prediction_length, axis=0),
        index=pooled_index,
        columns=context_columns,
    )
    horizon = pd.DataFrame(
        {"lead_step": lead_steps},
        index=pooled_index,
    )
    target_covariates = covariates.reindex(target_index)
    target_covariates.index = pooled_index
    target_values = target_series.reindex(target_index)
    target_values.index = pooled_index

    pooled = pd.concat(
        [horizon, target_covariates, context, target_values.rename("__target__")],
        axis=1,
    ).dropna()
    pooled_target = pooled.pop("__target__").rename(target_series.name)
    return pooled, pooled_target
