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
# ------------- Local Plots & tables -------
# ==========================================
PLOT_DIRECTORY = Path(__file__).resolve().parent / "plots"
FEATURES_DIR = Path(__file__).resolve().parent / "data" / "features"  
TARGET_DIR = Path(__file__).resolve().parent / "data" / "target"      
# ==========================================
# ------------- Tree-based feature selection config ----
# ==========================================
TREE_SELECTION_BOOL = False           # True: rank features by RandomForest permutation importance on X_train / y_train
TREE_SELECTION_RF_PARAMS = {           # RandomForestRegressor params; max_features="sqrt" keeps the fits fast
    "n_estimators": 100,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": 42,
}
TREE_SELECTION_N_REPEATS = 3           # shuffles per feature and fold for permutation importance
TREE_SELECTION_SUBSET_SIZES = [5, 10, 20, 30, 40, 60, 80, 120]   # top-k sizes evaluated by CV RMSE (all features always added)
TREE_SELECTION_TOLERANCE = 0.01        # select the smallest top-k with CV RMSE within 1% of all features
TREE_SELECTION_TOP_N_PLOT = 30         # number of features shown in the importance plot
TREE_GROUP_SELECTION_BOOL = True       # True: same selection on feature groups (weather variable, lag block, calendar dummy)
TREE_GROUP_LAG_BLOCK = 12              # consecutive lags per group (24 steps = 6 h at 15 min)
TREE_SELECTION_DIR = BACKTEST_RESULTS_DIR / "feature_selection"  # saved rankings shown in the dashboard
TREE_SELECTION_MODES = {               # mode -> (curve size column, unit, dashboard label)
    "groups": ("n_groups", "groups", "Feature groups"),
    "features": ("n_features", "features", "Single features"),
}
TREE_SELECTION_LAGS = False            # default: target lags as candidates (ranked only, never TSFM covariates)
TREE_SELECTION_CALENDAR = False        # default: calendar dummies as candidates (ranked only, never TSFM covariates)
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
        "regions": ['Flanders', 'Wallonia', 'Federal'],
    },
    # Elia data set ods032: solar generation per 15min interval, in MW
    "solar": {
        "base_url": "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/",
        "regions": ['Belgium'],
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
        "hourly": ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "direct_radiation", "diffuse_radiation"]
    }
}
# Temporary: use local dwd weather data for wind forecast including stochastic features
DATA_USE_DWD = True                    # required for the wind TSFM_COVARIATES
DATA_DWD_PATH = DATA_DIR / "dwd" / "gen_weather_dwd.parquet"
# PV data sample points
DATA_PV_REGIONS_BOOL = False            # True: use the PV sample points for the solar TSFM_COVARIATES, False: use the open-meteo weather data
DATA_PV_REGIONS = ["Flemish-Brabant", "West-Flanders", "East-Flanders", "Limburg", "Liège", "Namur", "Brussels", "Hainaut", "Walloon-Brabant", "Antwerp", "Luxembourg"]
# -- redundant with ELIA_SOURCES["solar"]["regions"]

SOLAR_REGION_DIR = DATA_DIR / "solar-region"
DATA_PV_PRODUCTION_YEAR = 2025
DATA_PV_WEATHER_SAMPLES = 200
DATA_PV_RANDOM_SEED = 42
# statistics that summarize all PV weather sample points per timestep and variable (pandas aggregation names)
STATISTICAL_PROPERTIES = ["mean", "std", "min", "max"]
# STATISTICAL_PROPERTIES = ["mean", "std", "min", "max", "skew", "kurt"]
# Elia region -> NUTS 2 code, polygons from Nuts2json (2024 NUTS, WGS84 lon/lat, 1:10M)
DATA_PV_NUTS = {
    "Brussels": "BE10",
    "Antwerp": "BE21",
    "Limburg": "BE22",
    "East-Flanders": "BE23",
    "Flemish-Brabant": "BE24",
    "West-Flanders": "BE25",
    "Walloon-Brabant": "BE31",
    "Hainaut": "BE32",
    "Liège": "BE33",
    "Luxembourg": "BE34",
    "Namur": "BE35",
}
DATA_PV_NUTS_URL = "https://raw.githubusercontent.com/eurostat/Nuts2json/master/pub/v2/2024/4326/10M/nutsrg_2.json"
# ==========================================

LOG_LEVEL = "INFO"                     # DEBUG, INFO, WARNING, ERROR
LOG_FILE = None                        # e.g. DATA_DIR / "logs" / "run.log"; None = console only
