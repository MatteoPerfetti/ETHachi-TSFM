#### Data Source Configuration ####

# from pathlib import Path
# DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCES = {
    # Elia data set ods031: onshore & offshore wind generation per 15min interval, in MW
    "wind": {
        "base_url": "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods031/",
        "region": ['Flanders', 'Wallonia', 'Federal'],
    },
    # Elia data set ods032: solar generation per 15min interval, in MW
    "solar": {
            "base_url": "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/",
            "region": ['Belgium'],
        }
}
START = "2024-01-01"           # UTC, inclusive
END = "2024-02-01"             # UTC, exclusive
RESOLUTIONCODE = "PT15M"           # ISO 8601 duration, as used by Elia's resolutioncode
