from pathlib import Path

import pandas as pd

from config import DATA_DIR
from logger import logger
from src.http_client import HttpClient

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

def fetch_elia(client: HttpClient, regions: list[str], start: str, end: str, resolutioncode: str) -> pd.DataFrame:
    """Query an Elia Opendatasoft dataset and return it wide: tz-aware UTC DatetimeIndex, one column per region.

    start and end are UTC ISO 8601 strings, e.g. "2024-01-01T00:00:00Z".
    """
    region = ", ".join(f'"{r}"' for r in regions)
    where = (
        f"datetime >= date'{start}'"
        f" AND datetime < date'{end}'"
        f' AND resolutioncode = "{resolutioncode}"'
        f" AND region IN ({region})"
    )
    df = client.get_parquet("exports/parquet", {
        "select": "datetime,region,measured",
        "where": where,
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
    return df.set_index("datetime")


def load_openmeteo(name: str, client: HttpClient, latitude: float, longitude: float, variables: list[str],
                   start: str, end: str) -> pd.DataFrame:
    """Return the data from the local cache if present, else fetch it from the Open-Meteo API and cache it."""
    path = cache_path(name, start, end, "PT1H")
    if path.exists():
        logger.info("Loaded %s from cache %s", name, path)
        return pd.read_parquet(path)
    logger.info("Fetching %s from the Open-Meteo API", name)
    df = fetch_openmeteo(client, latitude, longitude, variables, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Cached %s (%d rows) to %s", name, len(df), path)
    return df
