from collections.abc import Iterator

import pandas as pd


# ==========================================
# ------------- Train/Test split -----------

def _to_index_tz(value: str | pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Parse a timestamp (UTC if no zone is given) and match the timezone awareness of the index."""
    index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return ts if index.tz is not None else ts.tz_convert(None)


def split_train_test(df: pd.DataFrame, train_start: str, train_end: str, test_start: str,
                     test_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by timestamp into train and test frames. Starts are inclusive, ends exclusive."""
    bounds = [_to_index_tz(t, df.index) for t in (train_start, train_end, test_start, test_end)]
    train_start, train_end, test_start, test_end = bounds
    train = df[(df.index >= train_start) & (df.index < train_end)]
    test = df[(df.index >= test_start) & (df.index < test_end)]
    return train, test


def cross_validation_splits(
    df: pd.DataFrame,
    n_splits: int,
    test_size: str | pd.Timedelta | None,
    gap: str | pd.Timedelta,
    max_train_size: str | pd.Timedelta | None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train, validation) frames for expanding-window cross validation, split by timestamp.

    The validation folds are consecutive time windows of length test_size ending at the last timestamp
    (None = span // (n_splits + 1)). Train ends `gap` before its validation fold starts and covers at most
    `max_train_size` (None = expanding window). All durations are pandas timedelta strings, e.g. "6h".
    """
    step = df.index.to_series().diff().median()
    start, end = df.index[0], df.index[-1] + step
    val_len = pd.Timedelta(test_size) if test_size is not None else (end - start) / (n_splits + 1)
    gap = pd.Timedelta(gap)
    for i in range(n_splits):
        val_end = end - (n_splits - 1 - i) * val_len
        val_start = val_end - val_len
        train_end = val_start - gap
        train_start = start if max_train_size is None else max(start, train_end - pd.Timedelta(max_train_size))
        train = df[(df.index >= train_start) & (df.index < train_end)]
        val = df[(df.index >= val_start) & (df.index < val_end)]
        yield train, val
