from pathlib import Path
from typing import Callable
import json
import shutil
import time

import pandas as pd
import numpy as np
from shapely import contains_xy
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

import config
from logger import logger
from src.http_client import HttpClient, HttpError

# Draws n (longitude, latitude) points inside a region: (region, n, rng) -> array of shape (n, 2)
RegionSampler = Callable[[str, int, np.random.Generator], np.ndarray]


def to_wide(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    """Pivot long to wide, summing rows that share the same index and column (e.g. several units per region)."""
    # min_count=1 keeps NaN where all summed values are missing
    return df.groupby([index, columns])[values].sum(min_count=1).unstack(columns)


def cache_path(name: str, start: str, end: str, resolutioncode: str) -> Path:
    return config.DATA_DIR / name / f"{resolutioncode}_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%dT}.parquet"


def cached(name: str, path: Path, compute: Callable, read: Callable = pd.read_parquet,
           write: Callable = pd.DataFrame.to_parquet, valid: Callable = lambda data: True):
    """Return the data cached at path if present and valid, else compute it, cache it at path and return it.

    read(path) loads the cache, write(data, path) stores it, valid(data) rejects an outdated cache.
    """
    if path.exists():
        data = read(path)
        if valid(data):
            logger.info("Loaded %s from cache %s", name, path)
            return data
    logger.info("Computing %s, not cached yet", name)
    data = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    write(data, path)
    logger.info("Cached %s to %s", name, path)
    return data

# ==========================================
# ------------- API calls-------------
# ==========================================

# ==========================================
# ------------- Elia API -------------

def elia_where(regions: list[str], start: str, end: str, resolutioncode: str) -> str:
    """Opendatasoft where clause selecting [start, end), one resolution and the given regions."""
    region = ", ".join(f'"{r}"' for r in regions)
    return (
        f"datetime >= date'{start}'"
        f" AND datetime < date'{end}'"
        f' AND resolutioncode = "{resolutioncode}"'
        f" AND region IN ({region})"
    )

def fetch_elia(client: HttpClient, regions: list[str], start: str, end: str, resolutioncode: str) -> pd.DataFrame:
    """Query an Elia Opendatasoft dataset and return it wide: tz-aware UTC DatetimeIndex, one column per region.

    start and end are UTC ISO 8601 strings, e.g. "2024-01-01T00:00:00Z".
    """
    df = client.get_parquet("exports/parquet", {
        "select": "datetime,region,measured",
        "where": elia_where(regions, start, end, resolutioncode),
        "order_by": "datetime",
    })
    df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
    return to_wide(df, index="timestamp", columns="region", values="measured")


def load_elia(name: str, client: HttpClient, regions: list[str], start: str, end: str,
              resolutioncode: str) -> pd.DataFrame:
    """Return the data from the local cache if present, else fetch it from the Elia API and cache it."""
    return cached(name, cache_path(name, start, end, resolutioncode),
                  lambda: fetch_elia(client, regions, start, end, resolutioncode))


def fetch_elia_production(client: HttpClient, regions: list[str], start: str, end: str,
                          resolutioncode: str) -> pd.Series:
    """Return the measured energy per region in MWh over [start, end), summed server side by the Elia API."""
    resp = client.get_json("records", {
        "select": "region, sum(measured) as measured",
        "where": elia_where(regions, start, end, resolutioncode),
        "group_by": "region",
        "limit": len(regions),
    })
    hours_per_step = pd.Timedelta(resolutioncode) / pd.Timedelta("1h")
    return pd.DataFrame(resp["results"]).set_index("region")["measured"] * hours_per_step


def load_region_probabilities(name: str, client: HttpClient, regions: list[str], year: int, resolutioncode: str,
                              directory: Path) -> pd.Series:
    """Return the sampling probability of each region: its share of the production of all `regions` in `year`.

    Uses the cached csv in `directory` if it holds all regions, else fetches the yearly production from the Elia
    dataset client is bound to and caches it. The probabilities are renormalized over `regions` and sum to 1.
    """
    def compute() -> pd.DataFrame:
        start, end = f"{year}-01-01T00:00:00Z", f"{year + 1}-01-01T00:00:00Z"
        production = fetch_elia_production(client, regions, start, end, resolutioncode)
        return pd.DataFrame({"production_mwh": production, "share": production / production.sum()}).rename_axis("region")

    df = cached(f"{name} region production shares {year}", directory / f"production_share_{name}_{year}.csv", compute,
                read=lambda path: pd.read_csv(path, index_col="region"), write=pd.DataFrame.to_csv,
                valid=lambda df: set(regions) <= set(df.index))
    share = df.loc[regions, "share"]
    return (share / share.sum()).rename("probability")

# ==========================================
# ------------- open-meteo API -------------

def openmeteo_params(latitude, longitude, variables: list[str], start: str, end: str) -> dict:
    """Open-Meteo hourly request parameters. start and end are UTC ISO 8601 strings. Open-Meteo only takes dates
    and its end_date is inclusive, so the whole day of `end` is requested."""
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start[:10],
        "end_date": end[:10],
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }


def get_json_waiting_for_limits(client: HttpClient, params: dict, minute_wait: float = 60.0,
                                hour_wait: float = 3600.0):
    """client.get_json that waits out Open-Meteo's minutely and hourly request limits (HTTP 429) and retries.

    Other errors, including the daily limit, are raised.
    """
    while True:
        try:
            return client.get_json(params=params)
        except HttpError as e:
            if e.status != 429 or "Daily" in str(e):
                raise
            wait = hour_wait if "Hourly" in str(e) else minute_wait
            logger.warning("Open-Meteo request limit hit (%s), waiting %d s", "hourly" if wait == hour_wait else "minutely", wait)
            time.sleep(wait)


def fetch_openmeteo_region(client: HttpClient, points: pd.DataFrame, variables: list[str], start: str, end: str,
                           chunk_size: int = 10, sleep_seconds: float = 20.0,
                           chunk_dir: Path | None = None) -> pd.DataFrame:
    """Query the Open-Meteo API for every point (latitude, longitude) and return hourly data in long format:
    columns timestamp (tz-aware UTC), point (row position in points) and one float32 column per variable.

    Several points are sent per request (chunk_size). Requests are spaced sleep_seconds apart (Open-Meteo
    accepted about 4 requests of 10 points over 2 years per minute). The hourly limit (5000 calls, one request
    counts locations x variables / 10 x days / 14, about 315 for a chunk of 10 points, 6 variables and 2 years)
    is waited out if hit. If chunk_dir is given, every chunk is saved there and chunks already present are not
    requested again. start and end as in openmeteo_params.
    """
    frames, requested = [], False
    for first in range(0, len(points), chunk_size):
        chunk_file = chunk_dir / f"points_{first:05d}.parquet" if chunk_dir else None
        if chunk_file and chunk_file.exists():
            frames.append(pd.read_parquet(chunk_file))
            continue
        if requested:
            time.sleep(sleep_seconds)
        chunk = points.iloc[first:first + chunk_size]
        logger.info("Fetching points %d-%d of %d", first, first + len(chunk) - 1, len(points))
        resp = get_json_waiting_for_limits(client, openmeteo_params(
            ",".join(f"{lat:.4f}" for lat in chunk["latitude"]),
            ",".join(f"{lon:.4f}" for lon in chunk["longitude"]),
            variables, start, end))
        requested = True
        # Open-Meteo returns a list for several locations, but a single object for one
        locations = []
        for point, location in enumerate(resp if isinstance(resp, list) else [resp], start=first):
            df = pd.DataFrame(location["hourly"]).astype({v: "float32" for v in variables})
            df["point"] = point
            locations.append(df)
        df = pd.concat(locations, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True)
        df = df[["timestamp", "point", *variables]]
        if chunk_file:
            chunk_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(chunk_file)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_openmeteo_region(name: str, client: HttpClient, points: pd.DataFrame, variables: list[str], start: str,
                          end: str, directory: Path, chunk_size: int = 10) -> pd.DataFrame:
    """Return the raw point data from the local cache in `directory` if present, else fetch it from the
    Open-Meteo API and cache it."""
    path = directory / f"{name}_points_{cache_path(name, start, end, 'PT1H').name}"
    chunk_dir = path.with_name(f"{path.stem}_chunks")
    df = cached(f"{name} for {len(points)} points", path,
                lambda: fetch_openmeteo_region(client, points, variables, start, end, chunk_size=chunk_size,
                                               chunk_dir=chunk_dir))
    shutil.rmtree(chunk_dir, ignore_errors=True)
    return df


def load_previous_runs_openmeteo_region(name: str, client: HttpClient, points: pd.DataFrame, variables: list[str],
                                        start: str, end: str, directory: Path) -> pd.DataFrame:
    """Like load_openmeteo_region, but from the Previous Runs API (client bound to the previous_runs base_url):
    every variable once per run (<variable>, <variable>_previous_day1, <variable>_previous_day2).

    The points are sent in chunks of at most config.OPENMETEO_SETTINGS["max_calls_per_request"] calls each (the call
    weight grows with the number of variables, three times as many as the plain request, and the days). The minutely
    and hourly limits are waited out (see get_json_waiting_for_limits). The daily limit is raised as an error; the
    chunks already fetched stay on disk and a later run continues where it stopped.
    """
    suffixes = config.OPENMETEO_SOURCES["previous_runs"]["suffixes"]
    names =[f"{variable}{suffix}" for variable in variables for suffix in suffixes]
    days = (pd.Timestamp(end[:10]) - pd.Timestamp(start[:10])).days + 1
    calls_per_point = len(names) / 10 * days / 14
    chunk_size = max(1, int(config.OPENMETEO_SETTINGS["max_calls_per_request"] / calls_per_point))
    total_calls = calls_per_point * len(points)
    daily_call_limit = config.OPENMETEO_SETTINGS["daily_call_limit"]
    logger.info("%s: %d points x %d variables x %d days = about %.0f calls in chunks of %d points",
                name, len(points), len(names), days, total_calls, chunk_size)
    if total_calls > daily_call_limit:
        logger.warning("%s needs about %.0f calls, above the daily limit of %d: expect it to stop and continue "
                       "on the next day from the cached chunks", name, total_calls, daily_call_limit)
    return load_openmeteo_region(name, client, points, names, start, end, directory, chunk_size)


def reduce_openmeteo_region(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Summarize all points per timestep: tz-aware UTC DatetimeIndex, one column per variable and statistic,
    e.g. direct_radiation_mean."""
    stats = df.groupby("timestamp")[variables].agg(config.STATISTICAL_PROPERTIES)
    stats.columns = [f"{variable}_{stat}" for variable, stat in stats.columns]
    # skew and kurt are undefined when all points agree (std = 0, e.g. no radiation at night): set to 0
    shape_columns = [c for c in stats.columns if c.endswith(("_skew", "_kurt"))]
    stats[shape_columns] = stats[shape_columns].fillna(0)
    return stats


def unique_locations(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Distinct (latitude, longitude) of `points`, and how often each (by row position in the first) was drawn."""
    coordinates = ["latitude", "longitude"]
    unique = points[coordinates].drop_duplicates().reset_index(drop=True)
    draws = points[coordinates].merge(unique.reset_index(names="point"), on=coordinates)["point"].value_counts()
    logger.info("%d samples cover %d distinct locations", len(points), len(unique))
    return unique, draws


def reduce_weighted(raw: pd.DataFrame, draws: pd.Series, variables: list[str]) -> pd.DataFrame:
    """reduce_openmeteo_region with every point's rows repeated once per draw."""
    return reduce_openmeteo_region(raw.loc[raw.index.repeat(raw["point"].map(draws))], variables)


def load_weather_statistics(name: str, client: HttpClient, points: pd.DataFrame, variables: list[str],
                            start: str, end: str, directory: Path) -> pd.DataFrame:
    """Return the per-timestep statistics over all sample points from `directory` if present, else compute
    them from the raw point data (see load_openmeteo_region) and cache them.

    A location drawn several times (e.g. a wind farm) is fetched once; its rows are repeated once per draw before
    the reduction.
    """
    def compute() -> pd.DataFrame:
        unique, draws = unique_locations(points)
        raw = load_openmeteo_region(name, client, unique, variables, start, end, directory)
        return reduce_weighted(raw, draws, variables)

    return cached(f"{name} statistics", directory / f"{name}_{cache_path(name, start, end, 'PT1H').name}", compute)


def load_previous_runs_weather_statistics(name: str, client: HttpClient, points: pd.DataFrame, variables: list[str],
                                          start: str, end: str, directory: Path) -> pd.DataFrame:
    """Like load_weather_statistics, from the Previous Runs API (see load_previous_runs_openmeteo_region).

    The run suffix goes at the end of the column name: <variable>_<stat> is the most recent run, then
    <variable>_<stat>_previous_day1 and <variable>_<stat>_previous_day2, e.g. wind_speed_80m_mean and
    wind_sin_120m_mean_previous_day1 (after encode_directions).
    """
    suffixes = config.OPENMETEO_SOURCES["previous_runs"]["suffixes"]

    def compute() -> pd.DataFrame:
        unique, draws = unique_locations(points)
        raw = load_previous_runs_openmeteo_region(name, client, unique, variables, start, end, directory)
        stats = reduce_weighted(raw, draws, [f"{variable}{suffix}" for variable in variables for suffix in suffixes])
        return stats.rename(columns={f"{variable}{suffix}_{stat}": f"{variable}_{stat}{suffix}"
                                     for variable in variables for suffix in suffixes
                                     for stat in config.STATISTICAL_PROPERTIES})

    return cached(f"{name} statistics", directory / f"{name}_{cache_path(name, start, end, 'PT1H').name}", compute)


def hourly_variables(target_source: str) -> list[str]:
    """Open-Meteo variables requested for target_source."""
    return config.OPENMETEO_SOURCES["hourly_pv"] if target_source == "solar" else config.OPENMETEO_SOURCES["hourly_wind"]


def load_previous_runs_weather(target_source: str = config.TARGET_SOURCE) -> pd.DataFrame:
    """Weather statistics of the most recent run and of the previous day 1 and day 2 runs for target_source over
    DATA_START..DATA_END, from the local parquet cache or via the Previous Runs API."""
    with HttpClient(config.OPENMETEO_SOURCES["previous_runs"]["base_url"], timeout=120) as om:
        return load_previous_runs_weather_statistics(
            "previous_runs", om, weather_points(target_source), hourly_variables(target_source), config.DATA_START,
            config.DATA_END, config.OPENMETEO_SOURCES["weather_sampling"][target_source]["cache_dir"])

# ==========================================
# ------------- Weather sample points -------------

def sample_points(probabilities: pd.Series, sample_in_region: RegionSampler, n: int, seed: int) -> pd.DataFrame:
    """Draw a region for each of the n samples by probability, then a location inside that region with
    sample_in_region. Returns latitude, longitude and region per sample, grouped by region."""
    rng = np.random.default_rng(seed)
    regions = rng.choice(probabilities.index.to_numpy(), size=n, p=probabilities.to_numpy())
    samples = []
    for region, count in zip(*np.unique(regions, return_counts=True)):
        longitude, latitude = sample_in_region(region, count, rng).T
        samples.append(pd.DataFrame({"latitude": latitude, "longitude": longitude, "region": region}))
    return pd.concat(samples, ignore_index=True)

# ------------- PV: uniform points inside the NUTS polygons -------------

def load_nuts_geometries(client: HttpClient, nuts_ids: dict[str, str]) -> dict[str, BaseGeometry]:
    """Return the Nuts2json polygon of each region, keyed by region name. nuts_ids maps region name -> NUTS id.

    The GeoJSON is cached under data/solar, named after its version, projection, scale and level.
    """
    cache_dir = config.OPENMETEO_SOURCES["weather_sampling"]["solar"]["cache_dir"]
    text = cached("NUTS polygons", cache_dir / "_".join(client.base_url.split("/")[-4:]), client.get_text,
                  read=lambda path: path.read_text(encoding="utf-8"),
                  write=lambda text, path: path.write_text(text, encoding="utf-8"))
    features = {f["properties"]["id"]: f["geometry"] for f in json.loads(text)["features"]}
    return {region: shape(features[nuts_id]) for region, nuts_id in nuts_ids.items()}


def sample_points_in(geometry: BaseGeometry, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n uniform (longitude, latitude) points inside geometry by rejection sampling from its bounding box."""
    min_x, min_y, max_x, max_y = geometry.bounds
    points = np.empty((0, 2))
    while len(points) < n:
        candidates = rng.uniform((min_x, min_y), (max_x, max_y), size=(2 * n, 2))
        points = np.vstack([points, candidates[contains_xy(geometry, candidates[:, 0], candidates[:, 1])]])
    return points[:n]


def generate_pv_weather_samples(elia_client: HttpClient, nuts_client: HttpClient,
                                regions: list[str] = config.ELIA_SOURCES["solar"]["sample_regions"],
                                n_samples: int = config.OPENMETEO_SOURCES["weather_sampling"]["solar"]["n_samples"],
                                seed: int = config.OPENMETEO_SOURCES["weather_sampling"]["solar"]["seed"]
                                ) -> pd.DataFrame:
    """Return n_samples random weather sample points (latitude, longitude, region), spread over the regions
    in proportion to their solar production.

    elia_client is bound to the Elia solar dataset, nuts_client to the solar weather_sampling nuts_url.
    """
    nuts = config.OPENMETEO_SOURCES["weather_sampling"]["solar"]["nuts"]
    probabilities = load_region_probabilities("solar", elia_client, regions,
                                              config.ELIA_SOURCES["solar"]["production_year"],
                                              config.RESOLUTIONCODE, config.INPUTS_DIR)
    geometries = load_nuts_geometries(nuts_client, {region: nuts[region] for region in regions})
    return sample_points(probabilities, lambda region, n, rng: sample_points_in(geometries[region], n, rng),
                         n_samples, seed)

# ------------- Wind: wind farm locations weighted by capacity -------------

def wind_farm_sampler(farms: pd.DataFrame) -> RegionSampler:
    """Sampler drawing wind farms of a region weighted by their capacity_mw. Farms can be drawn more than once."""
    def sample(region: str, n: int, rng: np.random.Generator) -> np.ndarray:
        in_region = farms[farms["regions"] == region]
        weights = in_region["capacity_mw"] / in_region["capacity_mw"].sum()
        picked = in_region.iloc[rng.choice(len(in_region), size=n, p=weights.to_numpy())]
        return picked[["lon", "lat"]].to_numpy()
    return sample


def generate_wind_weather_samples(elia_client: HttpClient, regions: list[str] = config.ELIA_SOURCES["wind"]["regions"],
                                  n_samples: int = config.OPENMETEO_SOURCES["weather_sampling"]["wind"]["n_samples"],
                                  seed: int = config.OPENMETEO_SOURCES["weather_sampling"]["wind"]["seed"]
                                  ) -> pd.DataFrame:
    """Return n_samples random weather sample points (latitude, longitude, region) at wind farm locations,
    spread over the regions in proportion to their wind production. elia_client is bound to the Elia wind dataset."""
    probabilities = load_region_probabilities("wind", elia_client, regions,
                                              config.ELIA_SOURCES["wind"]["production_year"],
                                              config.RESOLUTIONCODE, config.INPUTS_DIR)
    farms = pd.read_csv(config.OPENMETEO_SOURCES["weather_sampling"]["wind"]["farms_path"], sep=";")
    return sample_points(probabilities, wind_farm_sampler(farms), n_samples, seed)


def weather_points(target_source: str) -> pd.DataFrame:
    """Weather sample points (latitude, longitude, region) of target_source: provinces for solar, farms for wind."""
    with HttpClient(config.ELIA_SOURCES[target_source]["base_url"]) as elia:
        if target_source == "solar":
            with HttpClient(config.OPENMETEO_SOURCES["weather_sampling"]["solar"]["nuts_url"]) as nuts:
                return generate_pv_weather_samples(elia, nuts)
        return generate_wind_weather_samples(elia)


if __name__ == "__main__":
    # Fetch and cache the previous runs weather of TARGET_SOURCE: python -m src.datasource
    # Chunks already fetched stay on disk; after a rate limit or the daily limit the same command continues.
    weather = load_previous_runs_weather()
    logger.info("Previous runs weather of %s: %d timestamps, %d columns, %s..%s", config.TARGET_SOURCE,
                len(weather), weather.shape[1], weather.index[0], weather.index[-1])
    run_suffixes = config.OPENMETEO_SOURCES["previous_runs"]["suffixes"]
    for suffix in run_suffixes:
        columns = [c for c in weather.columns if c.endswith(suffix)] if suffix else [
            c for c in weather.columns if not c.endswith(run_suffixes[1:])]
        logger.info("Run %r: %d columns, %.1f%% missing", suffix, len(columns), 100 * weather[columns].isna().to_numpy().mean())
