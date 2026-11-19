"""Live tests of the Elia and Open-Meteo fetch functions on a one day interval from config.DATA_START.

They call the real APIs with the sources, regions, variables and coordinates from config.py, so they need
network access. The cache is redirected to a temporary directory, nothing is written to data/.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src import datasource
from src.datasource import (fetch_elia, fetch_elia_production, fetch_openmeteo, fetch_openmeteo_region,
                            load_elia, load_openmeteo, reduce_openmeteo_region)
from src.http_client import HttpClient, HttpError


@pytest.fixture(scope="module")
def interval() -> tuple[str, str]:
    """[start, end) of one day, start = config.DATA_START."""
    end = pd.Timestamp(config.DATA_START) + pd.Timedelta("1D")
    return config.DATA_START, end.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture(scope="module", params=list(config.ELIA_SOURCES))
def elia_source(request) -> tuple[str, dict]:
    return request.param, config.ELIA_SOURCES[request.param]


@pytest.fixture(scope="module")
def openmeteo_source() -> dict:
    return config.OPENMETEO_SOURCES["weather"]


@pytest.fixture(scope="module")
def variables(openmeteo_source) -> list[str]:
    return openmeteo_source["hourly_pv"] + openmeteo_source["hourly_wind"]


@pytest.fixture
def cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(datasource, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def no_requests():
    """Client that fails on any request, to prove a result came from the cache."""
    class Offline(HttpClient):
        def _get(self, path, params):
            raise AssertionError("cache not used, the API was called")
    return Offline("https://offline.invalid")


def assert_utc_index(index: pd.Index, start: str, end: str) -> None:
    assert isinstance(index, pd.DatetimeIndex)
    assert str(index.tz) == "UTC"
    assert index.is_monotonic_increasing
    assert index.min() >= pd.Timestamp(start)
    assert index.max() < pd.Timestamp(end)


# ==========================================
# ------------- Elia -------------

def test_fetch_elia_returns_one_column_per_region(elia_source, interval):
    name, src = elia_source
    start, end = interval
    with HttpClient(src["base_url"]) as client:
        df = fetch_elia(client, src["regions"], start, end, config.RESOLUTIONCODE)

    steps_per_day = pd.Timedelta("1D") / pd.Timedelta(config.RESOLUTIONCODE)
    assert 0 < len(df) <= steps_per_day, name
    assert set(df.columns) <= set(src["regions"])
    assert df.notna().any().all(), "every region needs at least one measurement"
    assert (df.dropna(how="all") >= 0).all().all()
    assert_utc_index(df.index, start, end)


def test_fetch_elia_production_returns_energy_per_region(elia_source, interval):
    _, src = elia_source
    start, end = interval
    with HttpClient(src["base_url"]) as client:
        production = fetch_elia_production(client, src["regions"], start, end, config.RESOLUTIONCODE)

    assert set(production.index) <= set(src["regions"])
    assert len(production) > 0
    assert (production >= 0).all()


def test_fetch_elia_production_matches_summed_timeseries(elia_source, interval):
    """The server side sum (MW per step * hours per step) equals the sum over the fetched time series."""
    _, src = elia_source
    start, end = interval
    hours_per_step = pd.Timedelta(config.RESOLUTIONCODE) / pd.Timedelta("1h")
    with HttpClient(src["base_url"]) as client:
        series = fetch_elia(client, src["regions"], start, end, config.RESOLUTIONCODE)
        production = fetch_elia_production(client, src["regions"], start, end, config.RESOLUTIONCODE)

    expected = series.sum() * hours_per_step
    pd.testing.assert_series_equal(production.sort_index(), expected.sort_index(), check_names=False,
                                   check_dtype=False, rtol=1e-6)


def test_load_elia_caches_and_reads_back(elia_source, interval, cache_in_tmp, no_requests):
    name, src = elia_source
    start, end = interval
    with HttpClient(src["base_url"]) as client:
        fetched = load_elia(name, client, src["regions"], start, end, config.RESOLUTIONCODE)

    assert list(cache_in_tmp.rglob("*.parquet")), "load_elia must write the cache"
    cached = load_elia(name, no_requests, src["regions"], start, end, config.RESOLUTIONCODE)
    pd.testing.assert_frame_equal(cached, fetched)


def test_elia_rejects_unknown_region(interval):
    start, end = interval
    src = config.ELIA_SOURCES["wind"]
    with HttpClient(src["base_url"]) as client:
        try:
            df = fetch_elia(client, ["Atlantis"], start, end, config.RESOLUTIONCODE)
        except (HttpError, KeyError, ValueError):
            return
    assert df.empty


# ==========================================
# ------------- Open-Meteo -------------

def test_fetch_openmeteo_returns_hourly_columns_per_variable(openmeteo_source, variables, interval):
    start, end = interval
    with HttpClient(openmeteo_source["base_url"]) as client:
        df = fetch_openmeteo(client, openmeteo_source["latitude"], openmeteo_source["longitude"], variables,
                             start, end)

    assert list(df.columns) == variables
    # end_date is inclusive and Open-Meteo works on whole days: start day and end day, 24 h each
    assert len(df) == 48
    assert_utc_index(df.index, start, pd.Timestamp(end).normalize() + pd.Timedelta("1D"))
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert df.notna().any().all(), "every variable needs at least one value"


def test_fetch_openmeteo_values_are_physical(openmeteo_source, variables, interval):
    start, end = interval
    with HttpClient(openmeteo_source["base_url"]) as client:
        df = fetch_openmeteo(client, openmeteo_source["latitude"], openmeteo_source["longitude"], variables,
                             start, end)

    for column in [c for c in df.columns if c.startswith("cloud_cover")]:
        assert df[column].between(0, 100).all(), column
    for column in [c for c in df.columns if "radiation" in c or "irradiance" in c]:
        assert (df[column] >= 0).all(), column
    for column in [c for c in df.columns if c.startswith("wind_direction")]:
        assert df[column].between(0, 360).all(), column
    for column in [c for c in df.columns if c.startswith("wind_speed")]:
        assert (df[column] >= 0).all(), column


def test_load_openmeteo_caches_and_reads_back(openmeteo_source, variables, interval, cache_in_tmp, no_requests):
    start, end = interval
    args = (openmeteo_source["latitude"], openmeteo_source["longitude"], variables, start, end)
    with HttpClient(openmeteo_source["base_url"]) as client:
        fetched = load_openmeteo("weather", client, *args)

    assert list(cache_in_tmp.rglob("*.parquet")), "load_openmeteo must write the cache"
    cached = load_openmeteo("weather", no_requests, *args)
    pd.testing.assert_frame_equal(cached, fetched, check_freq=False)


@pytest.fixture(scope="module")
def region_points(openmeteo_source) -> pd.DataFrame:
    """Two points: the configured coordinate and one shifted by 0.5 degrees."""
    lat, lon = openmeteo_source["latitude"], openmeteo_source["longitude"]
    return pd.DataFrame({"latitude": [lat, lat + 0.5], "longitude": [lon, lon + 0.5]})


@pytest.fixture(scope="module")
def region_data(openmeteo_source, variables, region_points, interval) -> pd.DataFrame:
    start, end = interval
    with HttpClient(openmeteo_source["base_url"], timeout=120) as client:
        # chunk_size=1 sends one request per point, which also tests the chunking
        return fetch_openmeteo_region(client, region_points, variables, start, end, chunk_size=1,
                                      sleep_seconds=1.0)


def test_fetch_openmeteo_region_returns_long_format(region_data, region_points, variables, interval):
    start, end = interval
    assert list(region_data.columns) == ["timestamp", "point", *variables]
    assert sorted(region_data["point"].unique()) == list(range(len(region_points)))
    assert (region_data.groupby("point").size() == 48).all()
    assert (region_data[variables].dtypes == np.float32).all()
    assert str(region_data["timestamp"].dt.tz) == "UTC"


def test_fetch_openmeteo_region_single_chunk_equals_chunked(openmeteo_source, variables, region_points, region_data,
                                                            interval):
    """Both points in one request (a list response) give the same data as one request per point."""
    start, end = interval
    with HttpClient(openmeteo_source["base_url"], timeout=120) as client:
        one_request = fetch_openmeteo_region(client, region_points, variables, start, end, chunk_size=10)

    pd.testing.assert_frame_equal(one_request, region_data)


def test_fetch_openmeteo_region_resumes_from_chunk_dir(openmeteo_source, variables, region_points, region_data,
                                                       interval, tmp_path, no_requests):
    start, end = interval
    for first in range(len(region_points)):
        region_data[region_data["point"] == first].to_parquet(tmp_path / f"points_{first:05d}.parquet")

    resumed = fetch_openmeteo_region(no_requests, region_points, variables, start, end, chunk_size=1,
                                     chunk_dir=tmp_path)

    pd.testing.assert_frame_equal(resumed, region_data.reset_index(drop=True))


def test_reduce_openmeteo_region_gives_statistics_per_timestep(region_data, variables):
    stats = reduce_openmeteo_region(region_data, variables)

    assert len(stats) == 48
    assert list(stats.columns) == [f"{v}_{s}" for v in variables for s in config.STATISTICAL_PROPERTIES]
    assert stats.notna().all().all()
    for variable in variables:
        assert (stats[f"{variable}_min"] <= stats[f"{variable}_mean"] + 1e-4).all()
        assert (stats[f"{variable}_mean"] <= stats[f"{variable}_max"] + 1e-4).all()
