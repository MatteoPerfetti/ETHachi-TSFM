import pandas as pd
from pathlib import Path

from .implement_config import Config, build_config
from .ingestion import sources  # noqa: F401  (import fills the SOURCES registry)
from .ingestion.datasource import create_source
from .ingestion.http_client import HttpClient


def load_sources(cfg: Config) -> dict[str, pd.DataFrame]:
    """Load every configured source: from data/ if cached, otherwise from its API."""
    frames = {}
    for src_cfg in cfg.sources:
        with HttpClient(src_cfg.base_url) as client:
            source = create_source(src_cfg, client)
            frames[src_cfg.name] = source.load(cfg.start, cfg.end, cfg.resolutioncode)
    return frames


def run() -> None:
    cfg = build_config()  # config.py is read once here; source-specific params are validated by each source
    frames = load_sources(cfg)
