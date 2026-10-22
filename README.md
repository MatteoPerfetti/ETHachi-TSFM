# ETHachi-TSFM
Repository TSFM for Hitachi Energy (Energy Now! 5.0 Challenge): Benchmarking Time Series Foundation Models for Renewable Energy Forecasting

## Forecast
The goal is to forecast the Wind and PV generation in MW for the Belgian Power Grid in 15min resolution. The initial timehorizon is to predict generation from 09:00 to midnight and throughout the following day. This is sums up to 39 hours or 166 15min-timesteps.

## How to run

## Datasources
### PV and Wind generation
Historical generation of PV and Wind in MW for the years 2024, 2025 and Januar 2026 using the API by Elia (Link: https://opendata.elia.be/api-console/explore/v2.1/)
- wind: ods031
Relevant parameters
    "datetime":"2021-01-15T18:15:00+00:00",
    "resolutioncode":"PT15M",
    "region":"Flanders","Wallonia","Federal"
    "measured": [generated Power in MW],
- solar: ods032
Relevant parameters
    "datetime":"2025-07-10T14:00:00+00:00",
    "resolutioncode":"PT15M",
    "region":"Belgium",
    "measured": [generated Power in MW],
### Temporal dummy variables
- Temporal dummy variables on month (i.e. season), weekday, national holiday (Link: https://pypi.org/project/holidays/)
### Weather covariates (optional)
- DWD parquet file for wind covariates (already in PT15M). 
    Variables: temperature_2m (°C), relative_humidity_2m (%), pressure_msl / surface_pressure (hPa), wind_speed_<h>m (km/h), wind_direction_<h>m (°), wind_gusts_10m (km/h) 

    Available wind heights 10/80/120/180 m. THe parquet file includes both mean as well as standard deviation values for all 
- Open meteo data for solar covariates (hourly, needs to be interpolated)
    Variables: ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance"]

## Data splits
### Train/feed 2024, Test 2025 (implemented)
### Train feed 2024 & 2025, Test Jan 2026 (not yet implemented)

## ML pipeline
1. Config datasources and models
2. Data aquisation from API sources
3. Data cleaning, testing and validation
4. Features engineering for covariates using (Kernel) PCA
5. Feed data (training period) to model
6. Generate forecast (potential hyperparameter tuning)
7. Test forecast on ground truth test period
8. Compare and reason results

## How-to-run
### Requirements
- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- DWD weather file at `data/dwd/gen_weather_dwd.parquet` currently not in gitignore to pass it over (temporary)

### Setup
```bash
git clone <repo-url>
cd ETHachi-TSFM
uv sync                        # creates .venv and installs locked dependencies
```
Activate the environment:
```bash
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/macOS
```
Without uv: `python -m venv .venv`, activate it, then `pip install holidays ipykernel matplotlib pandas pyarrow pytest requests scikit-learn xgboost`

### Run
1. Adjust `config.py` (target, model, date ranges).
2. Run the pipeline:
```bash
python main.py
```
Elia and Open-Meteo data are downloaded on first run and cached in `data/`. Forecast plots are saved to `plots/`.

## Later steps (not part of current scope)
### Switch to direct forecasting
- Instead of recursive iterated models (fast and leightweight), choose direct forecasting (one model per predicted timestep)
### Other covariates
- Planned and forced unavailabilities of technical units
#### Adopt the Previous Model Runs API or the Single Runs API (model specific) by openmeteo
- For Previous Model Runs use the _previous_day2 values 
- For Single Runs API choose the proper models
#### Stochastic features
- include stochastic featues (now only deterministic - mean)
#### Increase spatial resolution PV
- Production of PV by Belgian region
### Other forecasting TSO regions
- Expand forecast to North-East Germany, specifically the 50Hertz (daugther company of elia group) TSO region. 

### Chronos-2 implementation

what was done so far:
- install on the terminal two libraries using: pip install chronos-forecasting torch
- downloaded and used the library to have the model weights
- succesfully implemented zero shot for wind_MW over the first 39 time steps of 2026 (39*15 minutes) with context = target variable in 2024

next steps:

sticking to the official repo tutorial: https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb

- insert covariates (future values of weather) in the context input
- implementing the rolling window algorithm (for now, we are only predicting a few hours; we want a loop that increases the context and predicts the whole 2025 one iteration at a time)
- implementing a metrics_evaluation script/function that computes our two metrics confronting the prediction with the true value

notes:
- context_df starts in DATA_START and finishes in DATA_END (TRAIN/TEST_START/END are not relevant for this zero shot approach as we don't do train --> test but only give context and get the prediction)
- pred_df starts in DATA_END and finishes in DATA_END + PREDICTION_LENGHT

### Timestral implementation (Mistral AI)

what was done so far:
- installed Mistral official timestral library, on hugging face there is the simple command to install it
- integrated mistralai/Timestral-209M for zero-shot probabilistic forecasting in main.py
- prediction horizon (39 hours = 156 timesteps, Jan 1-2 2025)
- computed 39h RMSE against Elia ground truth and generated unified comparative plots

next steps:
sticking to the official Mistral repository: https://github.com/mistralai/timestral
- inject weather covariates using known_covariates and past_covariates
- integrate into day-ahead rolling window across 2025 (same as Chronos)


notes:
- input tensor: (series=1, variates=1, timesteps)
- output: 21 quantiles from 0.01 to 0.99
- zero-shot: no dataset-specific fine-tuning required

### Times-FM 3.0 Implementation (Google Research)

what was done so far:
- installed the official library using: `uv add timesfm`
- integrated `google/timesfm-3.0-pytorch` (330M parameters) into `main.py` for zero-shot forecasting
- harmonized shared configuration with Chronos-2 and Timestral-209M:
  - context: historical wind generation (`wind_MW`) up to `2024-12-31 23:45:00 UTC`
  - prediction horizon: 39 hours (156 timesteps of 15 min, Jan 1-2, 2025)
- extracted the outputs
- computed 39h RMSE against actual Elia measurements
- integrated TimesFM-3 into the consolidated terminal comparison table and comparative 4-way plot

next steps:
Google Research repository: https://github.com/google-research/timesfm
- inject numerical weather covariates using TimesFM 3.0 native `past_only_covariates` and `past_future_covariates` arguments
- integrate TimesFM into the day-ahead rolling window evaluation loop across the entire 2025 test set
- evaluate GPU acceleration vs CPU inference latency

notes:
- input format: expects a 1D NumPy array `(timesteps,)` with `dtype=np.float32` for single-series forecasting
- output format: `ForecastOutput` object with `.forecast` (median) and `.quantiles` matrix of shape `(horizon, 9)` corresponding to levels `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`

## Test Infrastructure & Architecture

The testing and benchmarking framework is structured as a modular pipeline designed to evaluate traditional baselines (e.g. XGBoost) alongside Time Series Foundation Models (Chronos-2, Timestral, TimesFM 3.0) on the Elia renewable forecasting benchmark.

### Directory Structure

```text
ETHachi-TSFM/
├── config.py             <- Centralized registry: models, forecast horizons, context windows, API configs
├── main.py               <- CLI benchmark runner and headless execution engine
├── app.py                <- Interactive Streamlit dashboard for real-time benchmarking and visual inspection
├── data/                 <- Local parquet storage (Elia grid actuals, DWD weather, Open-Meteo)
│   └── predictions_cache.parquet <- Local cache for fast dashboard reloads without re-computing inference
├── plots/                <- Generated comparison charts (PNG) from headless runs
└── src/
    ├── datasource.py     <- API connectors for Elia Open Data and Open-Meteo
    ├── features.py       <- Feature engineering (calendar, cyclic encoding, lag features, DWD integration)
    ├── http_client.py    <- HTTP session handler with retries
    ├── metrics.py        <- Evaluation metrics (RMSE, MAE, WAPE, and 80% CI PICP coverage)
    ├── models.py         <- Traditional ML pipeline (XGBoost training & iterative inference)
    ├── models_tsfm.py    <- Unified Adapter pattern (TSFMRunner) for Chronos-2, Timestral-209M, and TimesFM 3.0
    ├── plotter.py        <- Matplotlib static charts & interactive Plotly visualizer with confidence bands
    ├── tuning.py         <- Hyperparameter optimization routines
    └── validate_config.py<- Configuration sanity checks
```

### Core Infrastructure Components

1. **Centralized Configuration Registry (`config.py`)**:
   - **Models Registry**: Declarative catalogue mapping model IDs (`xgboost`, `chronos2`, `timestral_209m`, `timesfm3`) to provider packages and Hugging Face checkpoints.
   - **Harmonized Horizons**: Standard 39-hour forecast horizon (156 steps of 15 min, from 09:00 UTC through midnight of next day).
   - **Context Window Presets**: Parametric context sizes (`2D`: 192 steps, `7D`: 672 steps, `14D`: 1344 steps, `30D`: 2880 steps) with 14D as default.

2. **Unified Foundation Model Adapter (`src/models_tsfm.py`)**:
   - `TSFMRunner`: Implements an Adapter pattern providing a consistent execution interface across heterogeneous libraries:
     - **Amazon Chronos-2** (`amazon/chronos-2`) via `chronos-forecasting`
     - **Mistral Timestral** (`mistralai/Timestral-209M`) via `timestral`
     - **Google TimesFM** (`google/timesfm-3.0-pytorch`) via `timesfm`
   - Handles device allocation (automatic fallback: CUDA -> MPS -> CPU), input tensor formatting, and standardization into `TSFMResult` (median forecast, 10th percentile, 90th percentile).

3. **Standardized Evaluation Metrics (`src/metrics.py`)**:
   - **Deterministic Metrics**:
     - RMSE (Root Mean Squared Error in MW)
     - MAE (Mean Absolute Error in MW)
     - WAPE (Weighted Absolute Percentage Error in %)
   - **Probabilistic Calibration**:
     - Coverage 80% (Prediction Interval Coverage Probability / PICP between q10 and q90).

4. **Interactive Dashboard (`app.py` via Streamlit)**:
   - Run with:
     ```bash
     streamlit run app.py
     ```
   - Features:
     - Interactive date selection for rolling evaluation.
     - Context window toggle (2D, 7D, 14D, 30D).
     - Multi-model selection with on-the-fly execution.
     - Local parquet prediction caching (`data/predictions_cache.parquet`) for instantaneous recall.
     - Rich Plotly charts with interactive tooltips and shaded uncertainty bands (q10 - q90).
     - Comparative KPI cards and sortable evaluation summary table.
