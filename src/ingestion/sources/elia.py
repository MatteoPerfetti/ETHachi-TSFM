from dataclasses import dataclass

import pandas as pd

from ...implement_config import SourceConfig
from ..datasource import DataSource, register
from ..http_client import HttpClient


@dataclass(frozen=True)
class EliaParams:
    fueltypeentsoe: tuple[str, ...]

    def __post_init__(self):
        if isinstance(self.fueltypeentsoe, str) or not self.fueltypeentsoe:
            raise ValueError("elia: 'fueltypeentsoe' must be a non-empty list of fuel type names")
        if not all(isinstance(f, str) and f for f in self.fueltypeentsoe):
            raise ValueError(f"elia: invalid 'fueltypeentsoe' entries: {self.fueltypeentsoe!r}")
        # sorted, so the same selection in a different order hits the same cache file
        object.__setattr__(self, "fueltypeentsoe", tuple(sorted(self.fueltypeentsoe)))


@register("elia")
class EliaSource(DataSource):
    """Elia power generation per ENTSO-E fuel type (dataset ods201), wide format: one column per fuel type."""

    def __init__(self, src_cfg: SourceConfig, client: HttpClient):
        super().__init__(src_cfg, client)
        try:
            self.params = EliaParams(fueltypeentsoe=src_cfg.params["fueltypeentsoe"])
        except KeyError:
            raise ValueError("elia: missing 'fueltypeentsoe'") from None

    def _fetch(self, start: pd.Timestamp, end: pd.Timestamp, resolution: str) -> pd.DataFrame:
        fuels = ", ".join(f'"{f}"' for f in self.params.fueltypeentsoe)
        where = (
            f"datetime >= date'{start:%Y-%m-%dT%H:%M:%SZ}'"
            f" AND datetime < date'{end:%Y-%m-%dT%H:%M:%SZ}'"
            f' AND resolutioncode = "{resolution}"'
            f" AND fueltypeentsoe IN ({fuels})"
        )
        df = self.client.get_parquet(f"exports/parquet", {
            "select": "datetime,fueltypeentsoe,generatedpower",
            "where": where,
            "order_by": "datetime",
        })
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.pivot(index="datetime", columns="fueltypeentsoe", values="generatedpower")
