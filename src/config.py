# ==========================================
# ------------- General config -------------
# ==========================================

START = "2024-01-01T00:00:00Z"     # YYYY-MM-DDTHH:MM:SSZ, UTC, inclusive
END = "2024-01-31T23:45:00Z"       # YYYY-MM-DDTHH:MM:SSZ, UTC, exclusive (open-meteo: end date is inclusive, the whole day is downloaded)
RESOLUTIONCODE = "PT15M"           # ISO 8601 duration, as used by Elia's resolutioncode


# ==========================================
# ------------- Training split config ------
# ==========================================

# train_test_split
TEST_SIZE = 0.2                    # fraction of rows used for testing
SHUFFLE = False                    # keep False for time series (no leakage from the future)
RANDOM_STATE = 42                  # only relevant if SHUFFLE is True

# TimeSeriesSplit
CV_N_SPLITS = 5
CV_TEST_SIZE = None                # rows per validation fold, None = n_samples // (n_splits + 1)
CV_GAP = 0                         # rows left out between train and validation fold
CV_MAX_TRAIN_SIZE = None           # None = expanding window

# ==========================================
# ------------- Data Source config ---------
# ==========================================

from pathlib import Path
# data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ------------- API sources -------------
ELIA_SOURCES = {
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

OPENMETEO_SOURCES = {
    # open-meteo: weather forecasts for Belgium, hourly, in UTC
    "weather": {
        "base_url": "https://historical-forecast-api.open-meteo.com/v1/forecast",
        "latitude": 50.2581,
        "longitude": 4.9117,
        "hourly": ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance"]
    }
}
# Temporary: use local dwd weather data for wind forecast including stochastic features
DATA_USE_DWD = True
DATA_DWD_PATH = DATA_DIR / "weather" / "gen_weather_dwd.parquet"
# ==========================================

LOG_LEVEL = "INFO"                 # DEBUG, INFO, WARNING, ERROR
LOG_FILE = None                    # e.g. DATA_DIR / "logs" / "run.log"; None = console only


