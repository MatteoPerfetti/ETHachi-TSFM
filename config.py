from pathlib import Path
# ==========================================
# ------------- General config ----
# ==========================================
COUNTRY = "BE"                         # country code for holidays (ISO 3166-1 alpha-2)
TIMEZONE = "Europe/Brussels"          # timezone for localizing timestamps (e.g, "Europe/Brussels")

# ==========================================
# ------------- Model selection ----
# ==========================================
TRADITIONAL_MODELS = ["xgboost"]
TSFM_MODELS = ["citras_fm"]
MODEL_TYPE = "xgboost"              # "xgboost"
# ==========================================
# ------------- XGBoost config ----
# ==========================================

UNTUNED = False  # True: use default hyperparameters, False: tune hyperparameters
MAX_DEPTH = [1, 3, 5, 7, 9]                # maximum depth of the trees
N_ESTIMATORS = [50, 100, 200, 300, 400]       # number of trees
LEARNING_RATE = [0.01, 0.1, 0.2, 0.3]      # learning rate for boosting
OPTUNA = True  # True: use Optuna for hyperparameter tuning, False: use grid search
OPTUNA_N_TRIALS = 100  # number of trials for Optuna hyperparameter tuning

# ==========================================
# ------------- Forecast config ----
# ==========================================
TARGET_SOURCE = "wind"              # name of the target source
MULTILE_TARGETS = False             # True: multi-target model, False: single-target model
COVARIATES = True                   # True: use covariates (weather, time features), False: only use lagged target values
CONTEXT_LENGTH = 96                 # number of past steps used as input to the model
PREDICTION_LENGTH = 4*39              # number of future steps predicted by the model 
# FORECAST_DAYTIME = "09:00"           # time of day for the forecast (UTC), 
PLOT_DIRECTORY = Path(__file__).resolve().parent / "plots"                  # directory to save forecast plots

# ==========================================
# ------------- Train/Test split config ----
# ==========================================
# Boolean flags to control whether to use train/test split or cross-validation
USE_CROSS_VALIDATION = False            # True: cross-validation, False: train/test split

TRAIN_START = "2024-01-04T00:00:00Z"  # YYYY-MM-DDTHH:MM:SSZ, UTC, inclusive
TRAIN_END = "2024-01-24T23:45:00Z"    # YYYY-MM-DDTHH:MM:SSZ, UTC, exclusive (open-meteo: end date is inclusive, the whole day is downloaded)
TEST_START = "2025-01-25T00:00:00Z"  # YYYY-MM-DDTHH:MM:SSZ, UTC, inclusive
TEST_END = "2025-01-31T23:45:00Z"    # YYYY-MM-DDTHH:MM:SSZ, UTC, exclusive (open-meteo: end date is inclusive, the whole day is downloaded)

# Cross validation (time-based, durations are pandas timedelta strings, e.g. "6h", "2D")
CV_N_SPLITS = 5
CV_TEST_SIZE = None                # length of each validation fold, None = train span // (n_splits + 1)
CV_GAP = "0min"                    # time left out between train and validation fold
CV_MAX_TRAIN_SIZE = None           # max length of the train window, None = expanding window
# ==========================================
# ------------- Data Source config ---------
# ==========================================
DATA_START = "2024-01-01T00:00:00Z"     # YYYY-MM-DDTHH:MM:SSZ, UTC, inclusive
DATA_END = "2025-01-31T23:45:00Z"       # YYYY-MM-DDTHH:MM:SSZ, UTC, exclusive (open-meteo: end date is inclusive, the whole day is downloaded)


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
RESOLUTIONCODE = "PT15M"           # ISO 8601 duration, as used by Elia's resolutioncode

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
DATA_USE_DWD = True
DATA_DWD_PATH = DATA_DIR / "dwd" / "gen_weather_dwd.parquet"
# ==========================================

LOG_LEVEL = "INFO"                 # DEBUG, INFO, WARNING, ERROR
LOG_FILE = None                    # e.g. DATA_DIR / "logs" / "run.log"; None = console only


