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

def to_wide(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    """Pivot long to wide, summing rows that share the same index and column (e.g. several units per region)."""
    # min_count=1 keeps NaN where all summed values are missing, instead of turning it into 0
    return df.groupby([index, columns])[values].sum(min_count=1).unstack(columns)

class DataSource(ABC):
    def __init__(self, src_cfg: SourceConfig, client: HttpClient):
        self.src_cfg, self.client = src_cfg, client

    def cache_path(self, start: pd.Timestamp, end: pd.Timestamp, resolutioncode: str) -> Path:
        return DATA_DIR / self.src_cfg.name / f"{resolutioncode}_{start:%Y%m%d}_{end:%Y%m%dT}.parquet"

    def load(self, start: pd.Timestamp, end: pd.Timestamp, resolutioncode: str) -> pd.DataFrame:
        """Return the data from the local cache if present, else fetch from the API and cache it."""
        path = self.cache_path(start, end, resolutioncode)
        if path.exists():
            return pd.read_parquet(path)
        df = self._fetch(start, end, resolutioncode)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df

    @abstractmethod
    def _fetch(self, start: pd.Timestamp, end: pd.Timestamp, resolutioncode: str) -> pd.DataFrame:
        """Call the API and return a DataFrame with a tz-aware UTC DatetimeIndex."""
