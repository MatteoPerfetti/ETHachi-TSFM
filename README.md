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
