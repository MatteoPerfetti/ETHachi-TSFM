from dataclasses import dataclass

import pandas as pd

from ...implement_config import SourceConfig
from ..datasource import DataSource, register
from ..http_client import HttpClient


@dataclass(frozen=True)
class EliaParams:
    region: tuple[str, ...]

    def __post_init__(self):
        if isinstance(self.region, str) or not self.region:
            raise ValueError("elia: 'region' must be a non-empty list of fuel type names")
        if not all(isinstance(f, str) and f for f in self.region):
            raise ValueError(f"elia: invalid 'region' entries: {self.region!r}")
        # sorted, so the same selection in a different order hits the same cache file
        object.__setattr__(self, "region", tuple(sorted(self.region)))


@register("elia")
class EliaSource(DataSource):
    """Elia power generation per ENTSO-E fuel type (dataset ods201), wide format: one column per fuel type."""

    def __init__(self, src_cfg: SourceConfig, client: HttpClient):
        super().__init__(src_cfg, client)
        try:
            self.params = EliaParams(region=src_cfg.params["region"])
        except KeyError:
            raise ValueError("elia: missing 'region'") from None

    def _fetch(self, start: pd.Timestamp, end: pd.Timestamp, resolutioncode: str) -> pd.DataFrame:
        region = ", ".join(f'"{f}"' for f in self.params.region)
        where = (
            f"datetime >= date'{start:%Y-%m-%dT%H:%M:%SZ}'"
            f" AND datetime < date'{end:%Y-%m-%dT%H:%M:%SZ}'"
            f' AND resolutioncode = "{resolutioncode}"'
            f" AND region IN ({region})"
        )
        df = self.client.get_parquet(f"exports/parquet", {
            "select": "datetime,region,measured",
            "where": where,
            "order_by": "datetime",
        })
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.pivot(index="datetime", columns="region", values="measured")
