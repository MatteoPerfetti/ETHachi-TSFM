"""Retrieval forecasts from production and archived, issue-time weather forecasts."""
from dataclasses import dataclass
from functools import cached_property
from hashlib import file_digest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.datasource import load_raft_weather_archive
from src.forecast import QUANTILE_COLUMNS, STEP, ForecastResult

CONTEXT_HOURS = 72
HORIZON_HOURS = 39
STRIDE_HOURS = 6
STEPS_PER_HOUR = int(pd.Timedelta(hours=1) / STEP)
RAFT_NAME = "RAFT"
WEATHER_WEIGHTS = {
    "solar": {
        "weather_direct_radiation": 0.52,
        "weather_diffuse_radiation": 0.26,
        "weather_direct_normal_irradiance": 0.22,
    },
    "wind": {
        "wind_speed_80m_mean": 0.32,
        "wind_speed_120m_mean": 0.26,
        "wind_speed_180m_mean": 0.19,
        "wind_speed_10m_mean": 0.13,
        "wind_gusts_10m_mean": 0.10,
    },
}


@dataclass(frozen=True)
class RaftSettings:
    weather_archive: str
    nominal_power_mw: float
    alpha: float = 0.5
    lambda_: float = 0.5
    gamma: float = 0.5
    neighbors: int = 15
    temperature: float = 0.1
    uniform_weather_weights: bool = False

    def __post_init__(self) -> None:
        for value, name in ((self.alpha, "alpha"), (self.lambda_, "lambda"), (self.gamma, "gamma")):
            if not 0 <= value <= 1:
                raise ValueError(f"RAFT {name} must be in [0, 1]")
        if self.neighbors not in (10, 15, 20):
            raise ValueError("RAFT neighbors must be 10, 15 or 20")
        for value, name in ((self.nominal_power_mw, "nominal power"), (self.temperature, "temperature")):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"RAFT {name} must be finite and positive")

    @cached_property
    def archive_digest(self) -> str:
        with Path(self.weather_archive).open("rb") as archive:
            return file_digest(archive, "sha256").hexdigest()


def production_distance(query: np.ndarray, keys: np.ndarray, nominal_power: float, alpha: float) -> np.ndarray:
    """Pearson shape distance and capacity-normalized level distance on 15-minute MW."""
    shape, level = _production_components(query, keys, nominal_power)
    return alpha * shape + (1 - alpha) * level


def _production_components(query: np.ndarray, keys: np.ndarray, nominal_power: float) -> tuple[np.ndarray, np.ndarray]:
    centered_query = query - query.mean()
    centered_keys = keys - keys.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered_query) * np.linalg.norm(centered_keys, axis=1)
    # Undefined Pearson is neutral, except that identical constant profiles have distance zero.
    correlation = (keys == query).all(axis=1).astype(float)
    defined = (np.ptp(query) > 0) & (np.ptp(keys, axis=1) > 0) & (norm > 0)
    np.divide(centered_keys @ centered_query, norm, out=correlation, where=defined)
    shape = (1 - np.clip(correlation, -1, 1)) / 2
    level = np.abs(keys - query).mean(axis=1) / nominal_power
    return shape, level


def _positive_median(values: np.ndarray) -> float:
    positive = values[values > 1e-12]
    return float(np.median(positive)) if positive.size else 1.0


def weighted_forecast(distances: np.ndarray, values: np.ndarray, neighbors: int, temperature: float) -> np.ndarray:
    nearest = np.argsort(distances, kind="stable")[:neighbors]
    selected = distances[nearest]
    weights = np.exp(-(selected - selected[0]) / temperature)
    return (weights / weights.sum()) @ values[nearest]


class RaftRetriever:
    """Train distance scales once, then retrieve using only completed historical episodes."""

    def __init__(self, settings: RaftSettings, target: pd.Series, target_source: str, issue_hour: int,
                 train_start: pd.Timestamp, train_end: pd.Timestamp):
        self.settings = settings
        weights = WEATHER_WEIGHTS[target_source]
        self.weather_columns = list(weights)
        self.beta = np.array(list(weights.values()))
        if settings.uniform_weather_weights:
            self.beta = np.full(len(self.beta), 1 / len(self.beta))

        self.archive = load_raft_weather_archive(settings.weather_archive, self.weather_columns)

        self.target = target.to_numpy(dtype=float, na_value=np.nan)
        self.grid = target.index
        self.context_steps = CONTEXT_HOURS * STEPS_PER_HOUR
        self.horizon_steps = HORIZON_HOURS * STEPS_PER_HOUR
        positions = np.flatnonzero(
            (self.grid == self.grid.floor("h"))
            & (self.grid.hour % STRIDE_HOURS == issue_hour % STRIDE_HOURS)
        )
        positions = positions[(positions >= self.context_steps) & (positions + self.horizon_steps <= len(target))]
        if not len(positions):
            raise ValueError("RAFT has no historical episode with enough target history")
        episodes = sliding_window_view(self.target, self.context_steps + self.horizon_steps)[
            positions - self.context_steps
        ]
        weather = np.stack([
            self.hourly_weather(self.grid[position], self.weather_columns).to_numpy(dtype=float, na_value=np.nan)
            for position in positions
        ])
        complete = np.isfinite(episodes).all(axis=1) & np.isfinite(weather).all(axis=(1, 2))
        self.origins = self.grid[positions[complete]]
        self.keys = episodes[complete, :self.context_steps]
        self.values = episodes[complete, self.context_steps:]
        self.weather = weather[complete]
        self.available = self.origins + HORIZON_HOURS * pd.Timedelta(hours=1)
        self.variable_scales, self.production_scale, self.weather_scale = self._fit_scales(train_start, train_end)

    def hourly_weather(self, issue: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
        horizon = pd.date_range(issue, periods=HORIZON_HOURS, freq="h", name="timestamp")
        try:
            rows = self.archive.xs(issue, level="issue_time")
        except KeyError:
            return pd.DataFrame(np.nan, index=horizon, columns=columns)
        return rows.reindex(index=horizon, columns=columns)

    def _fit_scales(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, float, float]:
        train = np.flatnonzero(
            (self.origins - pd.Timedelta(hours=CONTEXT_HOURS) >= start) & (self.available <= end)
        )
        rng = np.random.default_rng(0)
        shapes, levels, weather = [], [], []
        for query in rng.choice(train, size=min(len(train), 128), replace=False):
            earlier = train[self.available[train] <= self.origins[query]]
            if not len(earlier):
                continue
            candidates = rng.choice(earlier, size=min(len(earlier), 128), replace=False)
            shape, level = _production_components(
                self.keys[query], self.keys[candidates], self.settings.nominal_power_mw
            )
            shapes.append(shape)
            levels.append(level)
            weather.append(self._weather_errors(self.weather[query], candidates))
        if not shapes:
            raise ValueError("RAFT needs chronological pairs of complete episodes in the training period")
        errors = np.concatenate(weather)
        scales = np.array([_positive_median(column) for column in errors.T])
        weather_distances = (errors / scales) @ self.beta
        self.training_shape = np.concatenate(shapes)
        self.training_level = np.concatenate(levels)
        return scales, self.production_scale_for(self.settings.alpha), _positive_median(weather_distances)

    def production_scale_for(self, alpha: float) -> float:
        """Recombine the same training distance components for a different alpha."""
        return _positive_median(alpha * self.training_shape + (1 - alpha) * self.training_level)

    def _production_distance(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return production_distance(query, self.keys[candidates], self.settings.nominal_power_mw, self.settings.alpha)

    def _weather_errors(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return np.abs(self.weather[candidates] - query).mean(axis=1)

    def _query(self, issue: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        position = self.grid.get_loc(issue)
        context = self.target[max(0, position - self.context_steps):position]
        weather = self.hourly_weather(issue, self.weather_columns).to_numpy(dtype=float, na_value=np.nan)
        count = self.available.searchsorted(issue, side="right")
        if (count < self.settings.neighbors or len(context) != self.context_steps
                or not np.isfinite(context).all() or not np.isfinite(weather).all()):
            return None

        return context, weather, np.arange(count)

    def tuning_inputs(self, issue: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return reusable distances and values for a valid validation issue."""
        query = self._query(issue)
        if query is None:
            raise ValueError(f"RAFT tuning issue has incomplete data: {issue}")
        context, weather, candidates = query
        shape, level = _production_components(context, self.keys[candidates], self.settings.nominal_power_mw)
        meteo = (self._weather_errors(weather, candidates) / self.variable_scales) @ self.beta
        return shape, level, meteo / self.weather_scale, self.values[:len(candidates)]

    def predict(self, issue: pd.Timestamp) -> ForecastResult | None:
        query = self._query(issue)
        if query is None:
            return None
        context, weather, candidates = query
        production = self._production_distance(context, candidates) / self.production_scale
        weather_distances = (self._weather_errors(weather, candidates) / self.variable_scales) @ self.beta
        distances = self.settings.lambda_ * production + (1 - self.settings.lambda_) * (
            weather_distances / self.weather_scale
        )
        point = weighted_forecast(distances, self.values[:len(candidates)],
                                  self.settings.neighbors, self.settings.temperature)
        index = pd.date_range(issue, periods=self.horizon_steps, freq=STEP, name="timestamp")
        quantiles = pd.DataFrame(np.nan, index=index, columns=QUANTILE_COLUMNS.values())
        quantiles["q50"] = point
        return ForecastResult(RAFT_NAME, issue, quantiles)
