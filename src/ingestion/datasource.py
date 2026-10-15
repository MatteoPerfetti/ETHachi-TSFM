from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..implement_config import SourceConfig, Config
from .http_client import HttpClient

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SOURCES: dict[str, type["DataSource"]] = {}

def register(name: str):
    def deco(cls):
        if name in SOURCES:
            raise ValueError(f"Source '{name}' already registered")
        SOURCES[name] = cls
        return cls
    return deco

def create_source(src_cfg: SourceConfig, client: HttpClient) -> "DataSource":
    try:
        cls = SOURCES[src_cfg.name]
    except KeyError:
        raise ValueError(f"Unknown source '{src_cfg.name}'. Registered: {sorted(SOURCES)}") from None
    return cls(src_cfg, client)

class DataSource(ABC):
    def __init__(self, src_cfg: SourceConfig, client: HttpClient):
        self.src_cfg, self.client = src_cfg, client,

    def cache_path(self, start: pd.Timestamp, end: pd.Timestamp, resolution: str) -> Path:
        return DATA_DIR / self.src_cfg.name / f"{resolution}_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}.parquet"

    def load(self, start: pd.Timestamp, end: pd.Timestamp, resolution: str) -> pd.DataFrame:
        """Return the data from the local cache if present, else fetch from the API and cache it."""
        path = self.cache_path(start, end, resolution)
        if path.exists():
            return pd.read_parquet(path)
        df = self._fetch(start, end, resolution)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df

    @abstractmethod
    def _fetch(self, start: pd.Timestamp, end: pd.Timestamp, resolution: str) -> pd.DataFrame:
        """Call the API and return a DataFrame with a tz-aware UTC DatetimeIndex."""
