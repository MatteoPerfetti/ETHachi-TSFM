# ETHachi-TSFM
Repository TSFM for Hitachi Energy (Energy Now! 5.0 Challenge): Benchmarking Time Series Foundation Models for Renewable Energy Forecasting

## Forecast
The goal is to forecast the Wind and PV generation in MW for the Belgian Power Grid in 15min resolution. The initial timehorizon is to predict generation from 09:00 to midnight and throughout the following day. This is sums up to 39 hours or 166 15min-timesteps.

## Datasources
### PV and Wind generation
Historical generation of PV and Wind in MW for the years 2024, 2025 and Januar 2026 using the API by Elia (Link: https://opendata.elia.be/api-console/explore/v2.1/)
- wind: ods031
Relevant parameters
    "datetime":"2021-01-15T18:15:00+00:00",
    "resolutioncode":"PT15M",
    "region":"Flanders","Wallonia","Federal"
    "measured":2.82,
- solar: ods032
Relevant parameters
    "datetime":"2025-07-10T14:00:00+00:00",
    "resolutioncode":"PT15M",
    "region":"Belgium",
    "measured":5092.607,
### Temporal dummy variables
- Temporal dummy variables on month (i.e. season), weekday, national holiday (Link: https://pypi.org/project/holidays/)
### Weather covariates (optional)
- Column names follow <variable>_<height>_<stat>[_<provider>]. 

    Variables: temperature_2m (°C), relative_humidity_2m (%), pressure_msl / surface_pressure (hPa), wind_speed_<h>m (km/h), wind_direction_<h>m (°), wind_gusts_10m (km/h) 

    Available wind heights depend on the provider (10, 50, 80, 100, 120, 150, 180, 200 m). DWD, for example, has 10/80/120/180 m. 
## Data splits
### Train/feed 2024, Test 2025
### Train feed 2024 & 2025, Test Jan 2026

## ML pipeline
1. Config datasources and models
2. Data aquisation from API sources
3. Data cleaning, testing and validation
4. Features engineering for covariates using (Kernel) PCA
5. Feed data (training period) to model
6. Generate forecast (potential hyperparameter tuning)
7. Test forecast on ground truth test period
8. Compare and reason results
## Code base architecture
[to be added]
### Requirements

## Later steps (not part of current scope)
### Other covariates
- Planned and forced unavailabilities of technical units
### Adopt the Previous Model Runs API or the Single Runs API (model specific) by openmeteo
- For Previous Model Runs use the _previous_day2 values 
- For Single Runs API choose the proper models
### Increase spatial resolution PV
- Production of PV by Belgian region
### Other forecasting TSO regions
- Expand forecast to North-East Germany, specifically the 50Hertz (daugther company of elia group) TSO region. 