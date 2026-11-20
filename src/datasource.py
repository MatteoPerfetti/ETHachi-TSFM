from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, local
import json
import shutil
import time

import pandas as pd
import numpy as np
import requests
from shapely import contains_xy
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from config import (DATA_DIR, SOLAR_REGION_DIR, DATA_PV_NUTS, DATA_PV_PRODUCTION_YEAR, DATA_PV_RANDOM_SEED, DATA_PV_REGIONS,
                    DATA_PV_WEATHER_SAMPLES, RESOLUTIONCODE, STATISTICAL_PROPERTIES)
from logger import logger
from src.http_client import HttpClient, HttpError

def to_wide(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    """Pivot long to wide, summing rows that share the same index and column (e.g. several units per region)."""
    # min_count=1 keeps NaN where all summed values are missing, instead of turning it into 0
    return df.groupby([index, columns])[values].sum(min_count=1).unstack(columns)


def cache_path(name: str, start: str, end: str, resolutioncode: str) -> Path:
    return DATA_DIR / name / f"{resolutioncode}_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%dT}.parquet"

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
    path = cache_path(name, start, end, resolutioncode)
    if path.exists():
        logger.info("Loaded %s from cache %s", name, path)
        return pd.read_parquet(path)
    logger.info("Fetching %s from the Elia API", name)
    df = fetch_elia(client, regions, start, end, resolutioncode)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Cached %s (%d rows) to %s", name, len(df), path)
    return df

# ==========================================
# ------------- open-meteo API -------------

def fetch_openmeteo(client: HttpClient, latitude: float, longitude: float, variables: list[str], start: str,
                    end: str) -> pd.DataFrame:
    """Query the Open-Meteo API and return hourly data: tz-aware UTC DatetimeIndex, one column per variable.

    start and end are UTC ISO 8601 strings. Open-Meteo only takes dates and its end_date is inclusive,
    so the whole day of `end` is downloaded.
    """
    resp = client.get_json(params={
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start[:10],
        "end_date": end[:10],
        "hourly": ",".join(variables),
        "timezone": "UTC",
    })
    df = pd.DataFrame(resp["hourly"]).rename(columns={"time": "datetime"})
    df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("timestamp").drop(columns="datetime")


def load_openmeteo(name: str, client: HttpClient, latitude: float, longitude: float, variables: list[str],
                   start: str, end: str) -> pd.DataFrame:
    """Return the data from the local cache if present, else fetch it from the Open-Meteo API and cache it."""
    path = cache_path(name, start, end, "PT1H")
    if path.exists():
        logger.info("Loaded %s from cache %s", name, path)
        df = pd.read_parquet(path).drop(columns=["datetime", "timestamp"], errors="ignore")
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    logger.info("Fetching %s from the Open-Meteo API", name)
    df = fetch_openmeteo(client, latitude, longitude, variables, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Cached %s (%d rows) to %s", name, len(df), path)
    return df

# ==========================================
# ------------- PV weather sample points -------------

def fetch_elia_production(client: HttpClient, regions: list[str], start: str, end: str,
                          resolutioncode: str) -> pd.Series:
    """Return the measured energy per region in MWh over [start, end), summed server side by the Elia API."""
    resp = client.get_json("records", {
        "select": "regions, sum(measured) as measured",
        "where": elia_where(regions, start, end, resolutioncode),
        "group_by": "regions",
        "limit": len(regions),
    })
    hours_per_step = pd.Timedelta(resolutioncode) / pd.Timedelta("1h")
    return pd.DataFrame(resp["results"]).set_index("region")["measured"] * hours_per_step

def load_pv_region_probabilities(client: HttpClient, regions: list[str] = DATA_PV_REGIONS,
                                 year: int = DATA_PV_PRODUCTION_YEAR,
                                 resolutioncode: str = RESOLUTIONCODE) -> pd.Series:
    """Return the sampling probability of each region: its share of the Belgian solar production in `year`.

    Uses the cached csv if it holds all regions, else fetches the yearly production from the Elia solar
    dataset and caches it. The probabilities are renormalized over `regions`, so they always sum to 1.
    """
    path = SOLAR_REGION_DIR / f"production_share_{year}.csv"
    df = pd.read_csv(path, index_col="region") if path.exists() else None
    if df is not None and set(regions) <= set(df.index):
        logger.info("Loaded PV region production shares from cache %s", path)
    else:
        logger.info("Fetching %d PV region production from the Elia API", year)
        start, end = f"{year}-01-01T00:00:00Z", f"{year + 1}-01-01T00:00:00Z"
        production = fetch_elia_production(client, [*regions, "Belgium"], start, end, resolutioncode)
        belgium = production.pop("Belgium")
        df = pd.DataFrame({"production_mwh": production, "share": production / belgium}).rename_axis("region")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        logger.info("Cached PV region production shares to %s", path)
    share = df.loc[regions, "share"]
    return (share / share.sum()).rename("probability")


def load_nuts_geometries(client: HttpClient, nuts_ids: dict[str, str]) -> dict[str, BaseGeometry]:
    """Return the Nuts2json polygon of each region, keyed by region name. nuts_ids maps region name -> NUTS id.

    The GeoJSON is cached under data/solar-region, named after its version, projection, scale and level.
    """
    path = SOLAR_REGION_DIR / "_".join(client.base_url.split("/")[-4:])
    if path.exists():
        logger.info("Loaded NUTS polygons from cache %s", path)
        text = path.read_text(encoding="utf-8")
    else:
        logger.info("Fetching NUTS polygons from %s", client.base_url)
        text = client.get_text()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
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


def sample_region_points(probabilities: pd.Series, geometries: dict[str, BaseGeometry], n: int,
                         seed: int) -> pd.DataFrame:
    """Draw a region for each of the n samples by probability, then a uniform point inside that region."""
    rng = np.random.default_rng(seed)
    regions = rng.choice(probabilities.index.to_numpy(), size=n, p=probabilities.to_numpy())
    df = pd.DataFrame({"latitude": np.nan, "longitude": np.nan, "region": regions})
    for region, idx in df.groupby("region").indices.items():
        df.loc[idx, ["longitude", "latitude"]] = sample_points_in(geometries[region], len(idx), rng)
    return df


def generate_pv_weather_samples(elia_client: HttpClient, nuts_client: HttpClient,
                                regions: list[str] = DATA_PV_REGIONS, n_samples: int = DATA_PV_WEATHER_SAMPLES,
                                seed: int = DATA_PV_RANDOM_SEED) -> pd.DataFrame:
    """Return n_samples random weather sample points (latitude, longitude, region), spread over the regions
    in proportion to their solar production.

    elia_client is bound to the Elia solar dataset, nuts_client to DATA_PV_NUTS_URL.
    """
    probabilities = load_pv_region_probabilities(elia_client, regions)
    geometries = load_nuts_geometries(nuts_client, {region: DATA_PV_NUTS[region] for region in regions})
    return sample_region_points(probabilities, geometries, n_samples, seed)


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

    Several points are sent per request (chunk_size) to keep the responses small. Open-Meteo rejects requests
    above a per-minute limit (about 4 requests of 10 points over 2 years were accepted per minute), so we
    sleep sleep_seconds between requests. The hourly limit (5000 calls, one request counts locations x
    variables / 10 x days / 14, about 315 for a chunk of 10 points, 6 variables and 2 years) is waited out
    if hit. If chunk_dir is given, every chunk is saved there and chunks already present are not requested
    again, so an interrupted run continues where it stopped. start and end as in fetch_openmeteo.
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
        resp = get_json_waiting_for_limits(client, {
            "latitude": ",".join(f"{lat:.4f}" for lat in chunk["latitude"]),
            "longitude": ",".join(f"{lon:.4f}" for lon in chunk["longitude"]),
            "start_date": start[:10],
            "end_date": end[:10],
            "hourly": ",".join(variables),
            "timezone": "UTC",
        })
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
                          end: str) -> pd.DataFrame:
    """Return the raw point data from the local cache if present, else fetch it from the Open-Meteo API and
    cache it."""
    path = SOLAR_REGION_DIR / f"{name}_points_{cache_path(name, start, end, 'PT1H').name}"
    if path.exists():
        logger.info("Loaded %s from cache %s", name, path)
        return pd.read_parquet(path)
    logger.info("Fetching %s for %d points from the Open-Meteo API", name, len(points))
    chunk_dir = path.with_name(f"{path.stem}_chunks")
    df = fetch_openmeteo_region(client, points, variables, start, end, chunk_dir=chunk_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    shutil.rmtree(chunk_dir, ignore_errors=True)
    logger.info("Cached %s (%d rows) to %s", name, len(df), path)
    return df


def reduce_openmeteo_region(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Summarize all points per timestep: tz-aware UTC DatetimeIndex, one column per variable and statistic,
    e.g. direct_radiation_mean."""
    stats = df.groupby("timestamp")[variables].agg(STATISTICAL_PROPERTIES)
    stats.columns = [f"{variable}_{stat}" for variable, stat in stats.columns]
    # skew and kurt are undefined when all points agree (std = 0, e.g. no radiation at night): no asymmetry, no tails
    shape_columns = [c for c in stats.columns if c.endswith(("_skew", "_kurt"))]
    stats[shape_columns] = stats[shape_columns].fillna(0)
    return stats


def load_pv_weather_statistics(name: str, client: HttpClient, points: pd.DataFrame, variables: list[str],
                               start: str, end: str) -> pd.DataFrame:
    """Return the per-timestep statistics over all points from data/solar-region if present, else compute them
    from the raw point data (see load_openmeteo_region) and cache them."""
    path = SOLAR_REGION_DIR / f"{name}_{cache_path(name, start, end, 'PT1H').name}"
    if path.exists():
        logger.info("Loaded %s statistics from cache %s", name, path)
        return pd.read_parquet(path)
    stats = reduce_openmeteo_region(load_openmeteo_region(name, client, points, variables, start, end), variables)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(path)
    logger.info("Cached %s statistics (%d rows) to %s", name, len(stats), path)
    return stats

# Issue-time weather archive for RAFT

def load_raft_weather_archive(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Load RAFT's hourly forecast vintages, indexed by issue time and valid time."""
    archive = pd.read_parquet(path)
    dates = ["issue_time", "timestamp"]
    missing = set(dates + columns) - set(archive.columns)
    if missing:
        raise ValueError(f"RAFT weather archive is missing columns: {sorted(missing)}")
    archive[dates] = archive[dates].apply(pd.to_datetime, utc=True)
    if archive[dates].isna().any().any() or archive.duplicated(dates).any():
        raise ValueError("RAFT weather archive requires unique, non-missing issue_time and timestamp pairs")
    return archive.set_index(dates).sort_index()


_RAFT_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
_RAFT_POINTS = ((51.05, 3.72), (50.47, 4.87), (51.65, 2.85))
_RAFT_CAPACITY = (1819.64, 1357.44, 2262.10)  # Elia 2024 regional capacity weights
_RAFT_SPEEDS = ("wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_gusts_10m")
_RAFT_DIRECTION = "wind_direction_120m"
_RAFT_MODEL = "ecmwf_ifs"


class _RaftWeatherDownloader:
    def __init__(self, cache):
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.next_request = 0.0
        self.thread = local()

    def fetch(self, run):
        path = self.cache / f"{run:%Y%m%dT%H}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if not hasattr(self.thread, "session"):
            self.thread.session = requests.Session()
        for attempt in range(4):
            while True:
                with self.lock:
                    now = time.monotonic()
                    wait = self.next_request - now
                    if wait <= 0:
                        self.next_request = now + 0.90
                        break
                time.sleep(wait)
            try:
                response = self.thread.session.get(_RAFT_URL, timeout=30, params={
                    "run": run.strftime("%Y-%m-%dT%H:%M"), "models": _RAFT_MODEL,
                    "latitude": ",".join(str(point[0]) for point in _RAFT_POINTS),
                    "longitude": ",".join(str(point[1]) for point in _RAFT_POINTS),
                    "forecast_hours": 54, "wind_speed_unit": "ms", "cell_selection": "nearest",
                    "hourly": ",".join((*_RAFT_SPEEDS, _RAFT_DIRECTION)),
                })
                if response.status_code == 429:
                    delay = response.headers.get("Retry-After", "")
                    with self.lock:
                        self.next_request = max(self.next_request, time.monotonic() + (
                            int(delay) if delay.isdigit() else 60
                        ))
                if response.status_code >= 400 and response.status_code != 429 and response.status_code < 500:
                    raise ValueError(f"{run}: HTTP {response.status_code}: {response.text[:250]}")
                response.raise_for_status()
                payload = response.json()
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(payload), encoding="utf-8")
                temp.replace(path)
                return payload
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                if attempt == 3:
                    raise
                time.sleep(3 * 2 ** attempt)

    def build(self, run, issues):
        payload = self.fetch(run)
        if len(payload) != len(_RAFT_POINTS):
            raise ValueError(f"{run}: expected one response per location")
        locations = []
        for location in payload:
            if any(location["hourly_units"][name] != "m/s" for name in _RAFT_SPEEDS):
                raise ValueError(f"{run}: expected units of m/s")
            hourly = pd.DataFrame(location["hourly"])
            hourly.index = pd.to_datetime(hourly.pop("time"), utc=True)
            locations.append(hourly)
        frames = []
        for issue in issues:
            index = pd.date_range(issue, periods=39, freq="h", name="timestamp")
            values = np.stack([hourly.reindex(index)[[*_RAFT_SPEEDS, _RAFT_DIRECTION]].to_numpy(dtype=float) for hourly in locations])
            if not np.isfinite(values).all():
                raise ValueError(f"{run}: missing forecast values for {issue}")
            frame = pd.DataFrame(np.average(values[:, :, :-1], weights=_RAFT_CAPACITY, axis=0),
                                 index=index, columns=[f"{name}_mean" for name in _RAFT_SPEEDS])
            angle = np.deg2rad(values[:, :, -1])
            frame["wind_sin_120m_mean"] = np.average(np.sin(angle), weights=_RAFT_CAPACITY, axis=0)
            frame["wind_cos_120m_mean"] = np.average(np.cos(angle), weights=_RAFT_CAPACITY, axis=0)
            frames.append(frame.reset_index().assign(issue_time=issue, run_time=run, weather_model=_RAFT_MODEL))
        return frames


def build_raft_weather_archive(start: str, end: str, output: Path) -> None:
    """Write the Belgian wind RAFT archive from historical ECMWF runs."""
    start, end = (pd.Timestamp(value).tz_convert("UTC") for value in (start, end))
    if start != start.floor("h") or start.hour % 6 != 3 or end < start:
        raise ValueError("Use an issue interval on the 03/09/15/21 UTC grid.")
    # The 00/12 UTC forecast is at least 9 hours old at the chosen issuance.
    by_run = {}
    for issue in pd.date_range(start, end, freq="6h"):
        run = (issue - pd.Timedelta(hours=9)).floor("12h")
        by_run.setdefault(run, []).append(issue)
    downloader = _RaftWeatherDownloader(output.parent / "ecmwf_ifs_run_cache")
    frames, missing = [], 0
    with ThreadPoolExecutor(max_workers=7) as pool:
        jobs = {run: pool.submit(downloader.build, run, issues) for run, issues in by_run.items()}
        for i, (run, job) in enumerate(jobs.items(), start=1):
            try:
                frames.extend(job.result())
            except (requests.RequestException, ValueError, KeyError) as error:
                missing += len(by_run[run])
                print(f"Skipped {run}: {error}", flush=True)
            if i % 30 == 0 or i == len(jobs):
                print(f"Checked {i}/{len(jobs)} archived runs, {len(frames)} usable issues", flush=True)
    if not frames:
        raise ValueError("No complete forecasts to write")
    archive = pd.concat(frames, ignore_index=True)
    temp = output.with_suffix(".tmp")
    archive.to_parquet(temp, index=False)
    temp.replace(output)
    print(f"Saved {len(frames)} issues to {output}; {missing} unavailable", flush=True)
