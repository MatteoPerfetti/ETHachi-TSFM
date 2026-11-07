from pathlib import Path

# ==========================================
# ------------- General config ----
# ==========================================
COUNTRY = "BE"                         # country code for holidays (ISO 3166-1 alpha-2)
TIMEZONE = "Europe/Brussels"          # timezone for localizing timestamps (e.g, "Europe/Brussels")

# ==========================================
# ------------- Model selection ----
# ==========================================
MODELS_REGISTRY = {
    "xgboost": {
        "type": "traditional",
        "name": "XGBoost Regressor",
    },
    "chronos2": {
        "type": "tsfm",
        "checkpoint": "amazon/chronos-2",
        "provider": "amazon",
    },
    "timestral_209m": {
        "type": "tsfm",
        "checkpoint": "mistralai/Timestral-209M",
        "provider": "mistral",
    },
    "timesfm3": {
        "type": "tsfm",
        "checkpoint": "google/timesfm-3.0-pytorch",
        "provider": "google",
    },
    "moirai11rlarge": {
        "type": "tsfm",
        "checkpoint": "Salesforce/moirai-1.1-R-large",
        "provider": "salesforce",
    },
}
TRADITIONAL_MODELS = [k for k, v in MODELS_REGISTRY.items() if v["type"] == "traditional"]
TSFM_MODELS = [k for k, v in MODELS_REGISTRY.items() if v["type"] == "tsfm"]
MODEL_TYPE = "xgboost"                 # "xgboost"

# ==========================================
# ------------- Forecast config ----
# ==========================================
TARGET_SOURCE = "wind"                 # name of the target source
MULTILE_TARGETS = False                # True: multi-target model, False: single-target model
COVARIATES = True                      # True: use covariates (weather, time features), False: only use lagged target values

# TSFM prediction horizons & context windows
PREDICTION_HORIZON_HOURS = 39
HORIZON_STEPS = 4 * PREDICTION_HORIZON_HOURS  # 156 timesteps (15 minuti)
PREDICTION_LENGTH = HORIZON_STEPS             # backward compatibility for lag features pipeline

CONTEXT_WINDOWS = {
    "2D": 192,                         # 2 giorni
    "7D": 672,                         # 7 giorni
    "14D": 1344,                       # 14 giorni
    "30D": 2880,                       # 30 giorni
}
DEFAULT_CONTEXT = 1344                 # 14 giorni come default bilanciato
CONTEXT_LENGTH = 96                    # number of past steps used as input for traditional lag models
# FORECAST_DAYTIME = "09:00"            # time of day for the forecast (UTC), 
PLOT_DIRECTORY = Path(__file__).resolve().parent / "plots"  # directory to save forecast plots

# ==========================================
# ------------- Train/Test split config ----
# ==========================================
# Boolean flags to control whether to use train/test split or cross-validation
USE_CROSS_VALIDATION = False           # True: cross-validation, False: train/test split

TRAIN_START = "2024-01-01T00:00:00Z"   # YYYY-MM-DDTHH:MM:SSZ, UTC, inclusive
TRAIN_END = "2024-12-31T23:45:00Z"     # YYYY-MM-DDTHH:MM:SSZ, UTC, exclusive (open-meteo: end date is inclusive, the whole day is downloaded)
TEST_START = "2025-01-01T00:00:00Z"    # YYYY-MM-DDTHH:MM:SSZ, UTC, inclusive
TEST_END = "2025-12-31T23:45:00Z"      # YYYY-MM-DDTHH:MM:SSZ, UTC, exclusive (open-meteo: end date is inclusive, the whole day is downloaded)

# Cross validation (time-based, durations are pandas timedelta strings, e.g. "6h", "2D")
CV_N_SPLITS = 5
CV_TEST_SIZE = None                    # length of each validation fold, None = train span // (n_splits + 1)
CV_GAP = "0min"                        # time left out between train and validation fold
CV_MAX_TRAIN_SIZE = None               # max length of the train window, None = expanding window
# ==========================================
# ------------- Data Source config ---------
# ==========================================
DATA_START = "2023-12-28T00:00:00Z"    # YYYY-MM-DDTHH:MM:SSZ, UTC, inclusive
DATA_END = "2025-12-31T23:45:00Z"      # YYYY-MM-DDTHH:MM:SSZ, UTC, exclusive (open-meteo: end date is inclusive, the whole day is downloaded)


# data folder
DATA_DIR = Path(__file__).resolve().parent / "data"

# ------------- Elia API -------------
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
RESOLUTIONCODE = "PT15M"               # ISO 8601 duration, as used by Elia's resolutioncode

# ------------- Open-Meteo API -------------
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
DATA_USE_DWD = False
DATA_DWD_PATH = DATA_DIR / "dwd" / "gen_weather_dwd.parquet"
# ==========================================

LOG_LEVEL = "INFO"                     # DEBUG, INFO, WARNING, ERROR
LOG_FILE = None                        # e.g. DATA_DIR / "logs" / "run.log"; None = console only
