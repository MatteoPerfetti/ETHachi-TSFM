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
    "persistence": {
        "type": "baseline",
        "name": "Persistence",             # repeats the last HORIZON_STEPS observed values
    },
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
}
TRADITIONAL_MODELS = [k for k, v in MODELS_REGISTRY.items() if v["type"] == "traditional"]
RUN_TRADITIONAL_MODELS = False         # False: main.py skips the traditional models (XGBoost silenced while being reworked)
TSFM_MODELS = [k for k, v in MODELS_REGISTRY.items() if v["type"] == "tsfm"]
MODEL_TYPE = "xgboost"                 # "xgboost"

# ==========================================
# ------------- Forecast config ----
# ==========================================
TARGET_SOURCE = "wind"                 # name of the target source
MULTILE_TARGETS = False                # True: multi-target model, False: single-target model
COVARIATES = True                      # True: use covariates (weather, time features), False: only use lagged target values

# TSFM weather covariates, passed as known future covariates (Chronos-2: future_df)
TSFM_USE_COVARIATES = True             # True: also run Chronos-2 with weather covariates next to the univariate run
TSFM_COVARIATES = {
    # DWD columns (requires DATA_USE_DWD = True); directions are sin/cos encoded by encode_directions
    "wind": ["wind_speed_10m_mean", "wind_speed_80m_mean", "wind_speed_120m_mean", "wind_speed_180m_mean",
             "wind_gusts_10m_mean", "wind_sin_120m_mean", "wind_cos_120m_mean"],
    # Open-Meteo columns, prefixed with the OPENMETEO_SOURCES key by merge()
    "solar": ["weather_direct_radiation", "weather_diffuse_radiation", "weather_direct_normal_irradiance",
              "weather_cloud_cover_low", "weather_cloud_cover_mid", "weather_cloud_cover_high"],
}

# TSFM prediction horizons & context windows
PREDICTION_HORIZON_HOURS = 39
HORIZON_STEPS = 4 * PREDICTION_HORIZON_HOURS  # 156 timesteps (15 minutes)
PREDICTION_LENGTH = HORIZON_STEPS             # backward compatibility for lag features pipeline

CONTEXT_WINDOWS = {
    "2D": 192,                         # 2 days
    "7D": 672,                         # 7 days
    "14D": 1344,                       # 14 days
    "30D": 2880,                       # 30 days
}
DEFAULT_CONTEXT = 1344                 # 14 days as a balanced default
CONTEXT_LENGTH = 96                    # number of past steps used as input for traditional lag models
# FORECAST_DAYTIME = "09:00"            # time of day for the forecast (UTC), 
PLOT_DIRECTORY = Path(__file__).resolve().parent / "plots"  # directory to save forecast plots

# ==========================================
# ------------- Backtest (rolling window) config ----
# ==========================================
# One forecast per day of the test period (TEST_START..TEST_END), issued at BACKTEST_ISSUE_TIME UTC.
# TSFMs always get the TSFM_COVARIATES of the target as known future covariates.
BACKTEST_MODELS = ["persistence", "chronos2", "timestral_209m", "timesfm3"]  # keys of MODELS_REGISTRY
BACKTEST_ISSUE_TIME = "09:00"          # HH:MM, UTC, same every day
BACKTEST_HORIZON_STEPS = 156           # steps forecast from the issue time (09:00 -> 23:45 of the next day = 60 + 96)
BACKTEST_CONTEXT_STEPS = 3 * 96        # target history given to the models (raised to a model's min_context if needed)
BACKTEST_LATENCY_STEPS = 0             # steps between the last available measurement and the issue time
BACKTEST_RESULTS_DIR = Path(__file__).resolve().parent / "results"
BACKTEST_EVAL_WINDOW = "all"           # "all": every forecast step, "d+1": only the next day 00:00-23:45 (as Predico)
# Forecast plot: D+1 part of consecutive forecasts stitched over [START, END), one panel per model
BACKTEST_PLOT_START = "2025-03-01"     # YYYY-MM-DD, UTC, inclusive
BACKTEST_PLOT_END = "2025-03-15"       # YYYY-MM-DD, UTC, exclusive
BACKTEST_PLOT_MODELS = None            # model names as in the results (e.g. ["Chronos-2", "Persistence"]), None = all

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
DATA_USE_DWD = True                    # required for the wind TSFM_COVARIATES
DATA_DWD_PATH = DATA_DIR / "dwd" / "gen_weather_dwd.parquet"
# ==========================================

LOG_LEVEL = "INFO"                     # DEBUG, INFO, WARNING, ERROR
LOG_FILE = None                        # e.g. DATA_DIR / "logs" / "run.log"; None = console only
