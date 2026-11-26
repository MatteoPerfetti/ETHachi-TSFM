import holidays
import numpy as np
import pandas as pd
from logger import logger


def interpolate_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    # if 'timestamp' shows up as a column, we convert it in DatatimeIndex
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df.resample("15min").interpolate(method="time")


def merge(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge frames on their shared DatetimeIndex, prefixing columns with the dict key.

    Rows where any frame has no value are dropped, with a WARNING.
    """
    merged = pd.concat([df.add_prefix(f"{name}_") for name, df in dfs.items()], axis=1, join="outer").sort_index()
    complete = merged.dropna()
    dropped = len(merged) - len(complete)
    if dropped:
        missing = merged.isna().sum()
        logger.warning("Dropped %d of %d rows with missing values. Missing per column: %s",
                       dropped, len(merged), missing[missing > 0].to_dict())
    return complete

# ------
# DWD/ICON parquet helpers

def load_dwd_weather(dwd_path: str) -> pd.DataFrame:
    """Load the 15-minute DWD/ICON weather features from parquet."""
    weather = (
        pd.read_parquet(dwd_path)
        .set_index("timestamp")
        .drop(columns="measured", errors="ignore")
        .sort_index()
    )
    weather.index = pd.to_datetime(weather.index, utc=True)
    complete = weather.dropna()
    dropped = len(weather) - len(complete)
    if dropped:
        missing = weather.isna().sum()
        logger.warning(
            "Dropped %d of %d DWD rows with missing values. Missing per column: %s",
            dropped,
            len(weather),
            missing[missing > 0].to_dict(),
        )
    logger.info("Loaded DWD/ICON weather data from %s (%d rows)", dwd_path, len(complete))
    return complete


def merge_dwd_weather(df: pd.DataFrame, dwd_path: str) -> pd.DataFrame:
    """Merge the DWD weather data with the Open-Meteo weather data."""
    # Interpolate the Open-Meteo data to 15min steps, then merge with the DWD data on timestamp
    # Same "weather_" prefix as merge() gives the Open-Meteo columns, so column names don't depend on the dwd "use" setting
    df = interpolate_to_15min(df).add_prefix("weather_")
    # Drop 'measured' column from dwd_df
    dwd_df = pd.read_parquet(dwd_path)
    dwd_df["timestamp"] = pd.to_datetime(dwd_df["timestamp"], utc=True)
    dwd_df = dwd_df.set_index("timestamp").drop(columns="measured", errors="ignore")
    logger.info("Loaded DWD weather data from %s (%d rows)", dwd_path, len(dwd_df))
    # Merge DWD data (PM15M) with Open-Meteo data (PT1H) on timestamp, keeping only rows where both have values
    merged = df.join(dwd_df, how="outer").sort_index()
    complete = merged.dropna()
    dropped = len(merged) - len(complete)
    if dropped:
        missing = merged.isna().sum()
        logger.warning(
            "Dropped %d of %d rows with missing values after weather alignment. Missing per column: %s",
            dropped,
            len(merged),
            missing[missing > 0].to_dict(),
        )
    return complete

# ------

# ====================================================
# ------------- Add lag & calendar features -----------
def add_lag_features(df: pd.DataFrame, context_length: int, prediction_length: int = 1,
                     columns: list[str] | None = None) -> pd.DataFrame:
    """Return only the columns <col>_lag_<k> for k = prediction_length..prediction_length + context_length - 1
    (shifted by k steps) for the given columns (None = all).

    Starting at lag prediction_length means every row only uses values known prediction_length steps earlier,
    so each row is an honest forecast for any horizon up to prediction_length.
    The first prediction_length + context_length - 1 rows have no complete lag history and are dropped.
    """
    columns = list(df.columns) if columns is None else columns
    max_lag = prediction_length + context_length - 1
    lags = [df[columns].shift(k).add_suffix(f"_lag_{k}") for k in range(prediction_length, max_lag + 1)]
    return pd.concat(lags, axis=1).iloc[max_lag:]



def add_calendar_features(df: pd.DataFrame, timezone: str, country: str) -> pd.DataFrame:
    """Add dummy columns month_<1..12>, weekday_<0..6> (0 = Monday) and is_holiday (national holidays of `country`).

    Calendar values are taken in local `timezone`; a naive index is assumed to be UTC. All months and weekdays get
    a column even if absent from the data, so train and test frames have the same columns.
    """
    index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
    local = index.tz_convert(timezone)
    month = pd.Categorical(local.month, categories=range(1, 13))
    weekday = pd.Categorical(local.weekday, categories=range(7))
    calendar = pd.concat([pd.get_dummies(month, prefix="month", dtype=int),
                          pd.get_dummies(weekday, prefix="weekday", dtype=int)], axis=1)
    national_holidays = holidays.country_holidays(country, years=local.year.unique().tolist())
    calendar["is_holiday"] = pd.Index(local.date).isin(list(national_holidays)).astype(int)
    calendar.index = df.index
    return pd.concat([df, calendar], axis=1)


def encode_directions(df: pd.DataFrame) -> pd.DataFrame:
    """Replace every column containing "direction" (in degrees) with <col>_sin and <col>_cos, at the same position."""
    encoded = {}
    for col in df.columns:
        if "direction" in col:
            radians = np.deg2rad(df[col])
            encoded[col.replace("direction", "sin")] = np.sin(radians)
            encoded[col.replace("direction", "cos")] = np.cos(radians)
        else:
            encoded[col] = df[col]
    return pd.DataFrame(encoded, index=df.index)


