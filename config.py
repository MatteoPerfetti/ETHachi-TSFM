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

# PCA on the weather covariates, fitted on the training period only. It runs after each model's covariate frame is
# assembled (after build_origin_aligned_dataset for the traditional models, on the request covariates for the TSFMs).
# int = number of components, float in (0, 1) = share of explained variance, None = no PCA.
PCA_N_COMPONENTS = 0.95
PCA_PREFIX = "pc"

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
BACKTEST_ISSUE_EVERY_DAYS = 1          # 1 = one forecast every day of the test period; N = every N-th day (quick runs)
BACKTEST_HORIZON_STEPS = 156
BACKTEST_CONTEXT_DAYS = 3              # target history given to the models, in days
BACKTEST_CONTEXT_STEPS = BACKTEST_CONTEXT_DAYS * STEPS_PER_DAY
BACKTEST_LATENCY_STEPS = 0

# Context-length sweep (context_sweep.py): every model below runs once per context length.
# Persistence is added once as reference (it does not depend on the context).
CONTEXT_SWEEP_DAYS = [3, 7]            # next: 14, 30, 60
CONTEXT_SWEEP_MODELS = ["chronos2", "timestral_209m", "timesfm3"]
CONTEXT_SWEEP_DIR = Path(__file__).resolve().parent / "results" / "context_sweep"
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
INPUTS_DIR = DATA_DIR / "inputs"
RESOLUTIONCODE = "PT15M"
STATISTICAL_PROPERTIES = ["mean", "std", "min", "max"]

# Elia targets. "regions" are queried from Elia; "production_year" sets the production shares that weight the
# weather sample points. Solar samples over the provinces ("sample_regions"), wind over its "regions".
ELIA_SOURCES = {
    "wind": {
        "base_url": "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods031/",
        "regions": ["Flanders", "Wallonia", "Federal"],
        "production_year": 2025,
    },
    "solar": {
        "base_url": "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/",
        "regions": ["Belgium"],
        "production_year": 2025,
        "sample_regions": [
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
        ],
    },
}

# Open-Meteo settings shared by all sources.
# Open-Meteo counts a request as locations x variables / 10 x days / 14 calls (limits: 600/min, 5000/h, 10000/day).
# The previous runs request is split into chunks of points that stay below max_calls_per_request calls each.
OPENMETEO_SETTINGS = {
    "max_calls_per_request": 250,
    "daily_call_limit": 10000,
}

OPENMETEO_SOURCES = {
    "base_url": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "hourly_pv": [
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
    ],
    "hourly_wind": [
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "surface_pressure",
        "wind_speed_10m",
        "wind_speed_80m",
        "wind_speed_120m",
        "wind_speed_180m",
        "wind_direction_10m",
        "wind_direction_80m",
        "wind_direction_120m",
        "wind_direction_180m",
        "wind_gusts_10m",
    ],
    "previous_runs": {
        "base_url": "https://previous-runs-api.open-meteo.com/v1/forecast",
        # Suffix of every requested variable: "" = most recent run, then the runs of the previous day 1 and day 2
        "suffixes": ("", "_previous_day1", "_previous_day2"),
        # Run a forecast row reads: previous day 1 up to this many hours after the issue time (D+1 00:00 for an
        # issue at 09:00), previous day 2 after that. Conservative: a run is never used before it was published.
        "day1_max_lead_hours": 15,
    },
    "dwd": {
        "use": True,
        "path": DATA_DIR / "dwd" / "gen_weather_dwd.parquet",
    },
    # Sample points and spatial weather aggregation, per Elia source (regions: see ELIA_SOURCES).
    "weather_sampling": {
        "solar": {
            "n_samples": 200,
            "seed": 42,
            "cache_dir": DATA_DIR / "solar-region",
            "nuts": {
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
            },
            "nuts_url": "https://raw.githubusercontent.com/eurostat/Nuts2json/master/pub/v2/2024/4326/10M/nutsrg_2.json",
        },
        "wind": {
            "n_samples": 20,
            "seed": 42,
            "cache_dir": DATA_DIR / "wind-region",
            "farms_path": INPUTS_DIR / "belgium_wind_farms.csv",
        },
    },
}

LOG_LEVEL = "INFO"
LOG_FILE = None
