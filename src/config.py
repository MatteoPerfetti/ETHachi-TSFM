#### Data Source Configuration ####
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCES = {
    # Elia data set ods201: power generation per ENTSO-E fuel type
    "elia": {
        "base_url": "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods201/",
        "fueltypeentsoe": ["Wind Onshore", "Wind Offshore", "Solar"],
    },
}
START = "2025-01-01"           # UTC, inclusive
END = "2025-02-01"             # UTC, exclusive
RESOLUTION = "PT15M"           # ISO 8601 duration, as used by Elia's resolutioncode
