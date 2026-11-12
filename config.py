from pathlib import Path

# ==========================================
# ------------- General config ------------
# ==========================================
COUNTRY = "BE"
TIMEZONE = "Europe/Brussels"

# ==========================================
# ------------- Model selection -----------
# ==========================================
MODELS_REGISTRY = {
    "persistence": {"type": "baseline", "name": "Persistence"},
    "xgboost": {"type": "traditional", "name": "XGBoost"},
    "lightgbm": {"type": "traditional", "name": "LightGBM"},
    "catboost": {"type": "traditional", "name": "CatBoost"},
    "ngboost": {"type": "traditional", "name": "NGBoost"},
    "tabnet": {"type": "traditional", "name": "TabNet"},
    "equal_mean": {"type": "ensemble", "name": "Equal mean ensemble"},
    "top_k_exponential": {"type": "ensemble", "name": "Top-K exponential ensemble"},
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
TRADITIONAL_MODELS = [key for key, value in MODELS_REGISTRY.items() if value["type"] == "traditional"]
TSFM_MODELS = [key for key, value in MODELS_REGISTRY.items() if value["type"] == "tsfm"]
ENSEMBLE_MODELS = [key for key, value in MODELS_REGISTRY.items() if value["type"] == "ensemble"]

# Dashboard runs use configured/default parameters. Optuna remains available
# as an offline workflow in src/MLtrad.py and is never started by app.py.
TRADITIONAL_QUANTILES = (0.1, 0.5, 0.9)
TRADITIONAL_STRATEGIES = ("origin_aligned", "fixed_gap")
TRADITIONAL_STRATEGY = "fixed_gap"
TRADITIONAL_CONTEXT_STEPS = 96
TRADITIONAL_MODEL_PARAMETERS = {
    "xgboost": {},
    "lightgbm": {},
    "catboost": {},
    "ngboost": {},
    "tabnet": {},
}
OPTUNA_N_TRIALS = 10

# Ensemble dependencies are run automatically with the dashboard's selected
# traditional feature mode. Ensemble parameters live in src/ensembles.py.
ENSEMBLE_BASE_MODELS = list(TRADITIONAL_MODELS)

# ==========================================
# ------------- Forecast config -----------
# ==========================================
TARGET_SOURCE = "wind"
MULTILE_TARGETS = False
COVARIATES = True

# Weather covariates passed as known future covariates. Traditional models use
# the same selected columns so every model is evaluated with the same inputs.
TSFM_USE_COVARIATES = True
TSFM_COVARIATES = {
    "wind": [
        "wind_speed_10m_mean",
        "wind_speed_80m_mean",
        "wind_speed_120m_mean",
        "wind_speed_180m_mean",
        "wind_gusts_10m_mean",
        "wind_sin_120m_mean",
        "wind_cos_120m_mean",
    ],
    "solar": [
        "weather_direct_radiation",
        "weather_diffuse_radiation",
        "weather_direct_normal_irradiance",
        "weather_cloud_cover_low",
        "weather_cloud_cover_mid",
        "weather_cloud_cover_high",
    ],
}

PREDICTION_HORIZON_HOURS = 39
HORIZON_STEPS = 4 * PREDICTION_HORIZON_HOURS
PREDICTION_LENGTH = HORIZON_STEPS
CONTEXT_LENGTH = TRADITIONAL_CONTEXT_STEPS

CONTEXT_WINDOWS = {"2D": 192, "7D": 672, "14D": 1344, "30D": 2880}
DEFAULT_CONTEXT = 1344

# ==========================================
# ------------- Rolling backtest ----------
# ==========================================
BACKTEST_MODELS = ["persistence", "xgboost", "chronos2", "timestral_209m", "timesfm3"]
BACKTEST_ISSUE_TIME = "09:00"
BACKTEST_HORIZON_STEPS = 156
BACKTEST_CONTEXT_STEPS = 3 * 96
BACKTEST_LATENCY_STEPS = 0
BACKTEST_RESULTS_DIR = Path(__file__).resolve().parent / "results"
BACKTEST_EVAL_WINDOW = "all"
BACKTEST_PLOT_START = "2025-03-01"
BACKTEST_PLOT_END = "2025-03-15"
BACKTEST_PLOT_MODELS = None

# ==========================================
# ------------- Train/test & tuning -------
# ==========================================
USE_CROSS_VALIDATION = False
TRAIN_START = "2024-01-01T00:00:00Z"
TRAIN_END = "2024-12-31T23:45:00Z"
TEST_START = "2025-01-01T00:00:00Z"
TEST_END = "2025-12-31T23:45:00Z"

CV_N_SPLITS = 5
CV_TEST_SIZE = None
CV_GAP = "0min"
CV_MAX_TRAIN_SIZE = None

# ==========================================
# ------------- Local plots & tables ------
# ==========================================
PLOT_DIRECTORY = Path(__file__).resolve().parent / "plots"
FEATURES_DIR = Path(__file__).resolve().parent / "data" / "features"
TARGET_DIR = Path(__file__).resolve().parent / "data" / "target"

# ==========================================
# -------- Tree-based feature selection ----
# ==========================================
TREE_SELECTION_BOOL = False
TREE_SELECTION_RF_PARAMS = {
    "n_estimators": 100,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": 42,
}
TREE_SELECTION_N_REPEATS = 3
TREE_SELECTION_SUBSET_SIZES = [5, 10, 20, 30, 40, 60, 80, 120]
TREE_SELECTION_TOLERANCE = 0.01
TREE_SELECTION_TOP_N_PLOT = 30
TREE_GROUP_SELECTION_BOOL = True
TREE_GROUP_LAG_BLOCK = 12
TREE_SELECTION_DIR = BACKTEST_RESULTS_DIR / "feature_selection"
TREE_SELECTION_MODES = {
    "groups": ("n_groups", "groups", "Feature groups"),
    "features": ("n_features", "features", "Single features"),
}
TREE_SELECTION_LAGS = False
TREE_SELECTION_CALENDAR = False

# ==========================================
# ------------- Data sources --------------
# ==========================================
DATA_START = "2023-12-28T00:00:00Z"
DATA_END = "2025-12-31T23:45:00Z"
DATA_DIR = Path(__file__).resolve().parent / "data"

ELIA_SOURCES = {
    "wind": {
        "base_url": "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods031/",
        "regions": ["Flanders", "Wallonia", "Federal"],
    },
    "solar": {
        "base_url": "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/",
        "regions": ["Belgium"],
    },
}
RESOLUTIONCODE = "PT15M"

OPENMETEO_SOURCES = {
    "weather": {
        "base_url": "https://historical-forecast-api.open-meteo.com/v1/forecast",
        "latitude": 50.2581,
        "longitude": 4.9117,
        "hourly": [
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "direct_radiation",
            "diffuse_radiation",
            "direct_normal_irradiance",
        ],
    }
}
DATA_USE_DWD = True
DATA_DWD_PATH = DATA_DIR / "dwd" / "gen_weather_dwd.parquet"

# PV data sample points and spatial weather aggregation.
DATA_PV_REGIONS_BOOL = False
DATA_PV_REGIONS = [
    "Flemish-Brabant",
    "West-Flanders",
    "East-Flanders",
    "Limburg",
    "Liège",
    "Namur",
    "Brussels",
    "Hainaut",
    "Walloon-Brabant",
    "Antwerp",
    "Luxembourg",
]
SOLAR_REGION_DIR = DATA_DIR / "solar-region"
DATA_PV_PRODUCTION_YEAR = 2025
DATA_PV_WEATHER_SAMPLES = 200
DATA_PV_RANDOM_SEED = 42
STATISTICAL_PROPERTIES = ["mean", "std", "min", "max"]
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
DATA_PV_NUTS_URL = (
    "https://raw.githubusercontent.com/eurostat/Nuts2json/master/pub/v2/2024/4326/10M/nutsrg_2.json"
)

LOG_LEVEL = "INFO"
LOG_FILE = None
