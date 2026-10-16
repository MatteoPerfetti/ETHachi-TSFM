from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from ..logger import logger


def interpolate_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Upsample a frame with a DatetimeIndex (e.g. hourly) to 15min steps by time-based linear interpolation."""
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
# Temporary workaround to merge dwd parquet file with weather

def merge_dwd_weather(df: pd.DataFrame, dwd_path: str) -> pd.DataFrame:
    """Merge the DWD weather data with the Open-Meteo weather data."""
    # Interpolate the Open-Meteo data to 15min steps, then merge with the DWD data on timestamp
    df = interpolate_to_15min(df)
    # Drop 'measured' column from dwd_df
    dwd_df = pd.read_parquet(dwd_path).set_index("timestamp").drop(columns="measured", errors="ignore")
    # Merge DWD data (PM15M) with Open-Meteo data (PT1H) on timestamp, keeping only rows where both have values
    merged = df.join(dwd_df, how="outer").sort_index()
    complete = merged.dropna()
    dropped = len(merged) - len(complete)
    if dropped:
        missing = merged.isna().sum()
        logger.warning("Dropped %d of %d rows with missing values. Missing per column: %s",
                        dropped, len(merged), missing[missing > 0].to_dict())
    return complete

# ------


def split_train_test(df: pd.DataFrame, test_size: float, shuffle: bool,
                     random_state: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test frames with sklearn's train_test_split."""
    train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state)
    return train, test


def cross_validation_splits(df: pd.DataFrame, n_splits: int, test_size: int | None, gap: int,
                            max_train_size: int | None) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train, validation) frames for expanding-window cross validation with TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap, max_train_size=max_train_size)
    for train_idx, val_idx in tscv.split(df):
        yield df.iloc[train_idx], df.iloc[val_idx]
