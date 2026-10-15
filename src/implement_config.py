"""Builds and validates the typed Config object from the plain parameters in config.py."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import pandas as pd

from . import config

@dataclass(frozen=True)
class SourceConfig:
    name: str      # registry key, also the cache subfolder under data/
    base_url: str
    # source-specific entries from config.py, parsed and validated by the source class itself
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    data_dir: Path 
    sources: list[SourceConfig]
    start: pd.Timestamp
    end: pd.Timestamp
    resolution: str

    def __post_init__(self):
        if not self.sources:
            raise ValueError("No sources configured")
        if self.start >= self.end:
            raise ValueError(f"START ({self.start}) must be before END ({self.end})")
        try:
            pd.Timedelta(self.resolution)
        except ValueError:
            raise ValueError(f"RESOLUTION '{self.resolution}' is not an ISO 8601 duration like 'PT15M'") from None


def _build_source(name: str, raw: dict) -> SourceConfig:
    raw = dict(raw)
    try:
        base_url = raw.pop("base_url")
    except KeyError:
        raise ValueError(f"{name}: missing 'base_url'") from None
    return SourceConfig(name, base_url, raw)


def build_config() -> Config:
    return Config(
        data_dir=config.DATA_DIR,
        sources=[_build_source(name, raw) for name, raw in config.SOURCES.items()],
        start=pd.Timestamp(config.START, tz="UTC"),
        end=pd.Timestamp(config.END, tz="UTC"),
        resolution=config.RESOLUTION,
    )
