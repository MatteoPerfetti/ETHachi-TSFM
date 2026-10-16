from dataclasses import dataclass

import pandas as pd

from ...implement_config import SourceConfig
from ..datasource import DataSource, register, to_wide
from ..http_client import HttpClient


@dataclass(frozen=True)
class SolarParams:
    region: tuple[str, ...]

    def __post_init__(self):
        if isinstance(self.region, str) or not self.region:
            raise ValueError("solar: 'region' must be a non-empty list of fuel type names")
        if not all(isinstance(f, str) and f for f in self.region):
            raise ValueError(f"solar: invalid 'region' entries: {self.region!r}")
        # sorted, so the same selection in a different order hits the same cache file
        object.__setattr__(self, "region", tuple(sorted(self.region)))


@register("solar")
class SolarSource(DataSource):
    """Solar power generation  (dataset ods032), wide format: one column per region."""

    def __init__(self, src_cfg: SourceConfig, client: HttpClient):
        super().__init__(src_cfg, client)
        try:
            self.params = SolarParams(region=src_cfg.params["region"])
        except KeyError:
            raise ValueError("solar: missing 'region'") from None

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
        return to_wide(df, index="datetime", columns="region", values="measured")