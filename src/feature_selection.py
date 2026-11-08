import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

import config
from logger import logger
from src.features import add_calendar_features, add_lag_features

# ==========================================
# ------------- Tree-based selection -------


def _folds(X: pd.DataFrame, y: pd.DataFrame, folds: list[tuple[pd.DataFrame, pd.DataFrame]]):
    """Yield (X_fit, y_fit, X_val, y_val) per fold; y is squeezed to a Series for a single target."""
    y = y.squeeze(axis=1)
    for X_fit, X_val in folds:
        yield X.loc[X_fit.index], y.loc[X_fit.index], X.loc[X_val.index], y.loc[X_val.index]


def tree_feature_importance(X_train: pd.DataFrame, y_train: pd.DataFrame,
                            folds: list[tuple[pd.DataFrame, pd.DataFrame]],
                            rf_params: dict, n_repeats: int) -> pd.DataFrame:
    """Rank every column of X_train by RandomForest permutation importance on time-ordered validation folds.

    Per fold, a RandomForest is fit on the train part; each feature is then shuffled n_repeats times on the
    validation part and the RMSE increase is recorded. Importance is averaged over folds (std = fold spread).
    impurity_mean is the built-in (MDI) importance, for comparison only: it is biased towards continuous features.
    Correlated features share credit, so near-duplicates (e.g. adjacent lags) can each look unimportant.
    """
    permutation, impurity = {}, {}
    for i, (X_fit, y_fit, X_val, y_val) in enumerate(_folds(X_train, y_train, folds)):
        logger.info("Tree selection fold %d/%d: fit %d rows, validate %d rows", i + 1, len(folds), len(X_fit), len(X_val))
        model = RandomForestRegressor(**rf_params).fit(X_fit, y_fit)
        result = permutation_importance(model, X_val, y_val, scoring="neg_root_mean_squared_error",
                                        n_repeats=n_repeats, random_state=rf_params.get("random_state"),
                                        n_jobs=rf_params.get("n_jobs"))
        permutation[i] = result.importances_mean
        impurity[i] = model.feature_importances_

    permutation = pd.DataFrame(permutation, index=X_train.columns)
    importance = pd.DataFrame({
        "permutation_mean": permutation.mean(axis=1),
        "permutation_std": permutation.std(axis=1),
        "impurity_mean": pd.DataFrame(impurity, index=X_train.columns).mean(axis=1),
    }).sort_values("permutation_mean", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance


def _cv_rmse(X_train: pd.DataFrame, y_train: pd.DataFrame, columns: pd.Index,
             folds: list[tuple[pd.DataFrame, pd.DataFrame]], rf_params: dict) -> dict:
    """Mean and std over folds of the validation RMSE of a RandomForest fit on the given columns only."""
    rmses = [root_mean_squared_error(y_val, RandomForestRegressor(**rf_params).fit(X_fit[columns], y_fit)
                                     .predict(X_val[columns]))
             for X_fit, y_fit, X_val, y_val in _folds(X_train, y_train, folds)]
    return {"cv_rmse": np.mean(rmses), "cv_rmse_std": np.std(rmses)}


def cv_rmse_by_subset_size(X_train: pd.DataFrame, y_train: pd.DataFrame, ranking: pd.Index,
                           folds: list[tuple[pd.DataFrame, pd.DataFrame]], rf_params: dict,
                           subset_sizes: list[int]) -> pd.DataFrame:
    """Mean CV RMSE of a RandomForest on the top-k ranked features, for each k in subset_sizes (plus all features)."""
    sizes = sorted({k for k in subset_sizes if k < len(ranking)} | {len(ranking)})
    rows = []
    for k in sizes:
        rows.append({"n_features": k, **_cv_rmse(X_train, y_train, ranking[:k], folds, rf_params)})
        logger.info("Top %d features: CV RMSE %.2f", k, rows[-1]["cv_rmse"])
    return pd.DataFrame(rows)


def select_smallest_subset(curve: pd.DataFrame, ranking: pd.Index, tolerance: float,
                           size_col: str = "n_features") -> list[str]:
    """Return the top-k entries of ranking for the smallest k (curve[size_col]) whose CV RMSE is within
    (1 + tolerance) of the full model. Works for ranked features as well as ranked feature groups."""
    full_rmse = curve.loc[curve[size_col].idxmax(), "cv_rmse"]
    k = curve.loc[curve["cv_rmse"] <= (1 + tolerance) * full_rmse, size_col].min()
    logger.info("Selected %d of %d (%s, CV RMSE within %.1f%% of all: %.2f)",
                k, len(ranking), size_col, tolerance * 100, full_rmse)
    return list(ranking[:k])


# ==========================================
# ------------- Grouped tree-based selection
def feature_groups(columns: pd.Index, statistics: list[str], lag_block: int) -> pd.Series:
    """Map every column to the group it is fetched or computed with (Series: column -> group name).

    - <base>_lag_<k>        -> <base>_lag_<first>-<last>: blocks of lag_block consecutive lags, from the smallest lag
    - <var>_<stat>          -> <var> for stat in statistics: all statistics come from one fetch of <var>
    - <prefix>_<int>        -> <prefix>: dummy columns, e.g. month_1, weekday_0
    - anything else         -> the column itself, e.g. is_holiday
    """
    lag_pattern = re.compile(r"^(.+)_lag_(\d+)$")
    lag_of = {col: (m[1], int(m[2])) for col in columns if (m := lag_pattern.match(col))}
    bounds = pd.DataFrame(list(lag_of.values()), columns=["base", "lag"]).groupby("base")["lag"].agg(["min", "max"])

    groups = {}
    for col in columns:
        prefix, _, suffix = col.rpartition("_")
        if col in lag_of:
            base, lag = lag_of[col]
            first, last = bounds.loc[base]
            start = first + (lag - first) // lag_block * lag_block
            groups[col] = f"{base}_lag_{start}-{min(start + lag_block - 1, last)}"
        elif prefix and (suffix in statistics or suffix.isdigit()):
            groups[col] = prefix
        else:
            groups[col] = col
    return pd.Series(groups, name="group")


def _grouped_permutation_importance(model: RandomForestRegressor, X_val: pd.DataFrame, y_val: pd.Series,
                                    groups: pd.Series, n_repeats: int, rng: np.random.Generator) -> pd.Series:
    """Mean RMSE increase on X_val when all columns of a group are shuffled together (same row permutation)."""
    baseline = root_mean_squared_error(y_val, model.predict(X_val))
    values = X_val.to_numpy()
    importance = {}
    for group, columns in groups.groupby(groups).groups.items():
        positions = X_val.columns.get_indexer(columns)
        increases = []
        for _ in range(n_repeats):
            shuffled = values.copy()
            shuffled[:, positions] = values[rng.permutation(len(values))][:, positions]
            X_shuffled = pd.DataFrame(shuffled, index=X_val.index, columns=X_val.columns)
            increases.append(root_mean_squared_error(y_val, model.predict(X_shuffled)) - baseline)
        importance[group] = np.mean(increases)
    return pd.Series(importance)


def tree_group_importance(X_train: pd.DataFrame, y_train: pd.DataFrame, groups: pd.Series,
                          folds: list[tuple[pd.DataFrame, pd.DataFrame]],
                          rf_params: dict, n_repeats: int) -> pd.DataFrame:
    """Rank feature groups by grouped RandomForest permutation importance on time-ordered validation folds.

    Same procedure as tree_feature_importance, but all columns of a group are shuffled together, so correlated
    columns inside a group (lags, statistics of one variable) no longer hide each other.
    impurity_sum is the summed built-in (MDI) importance of the group's columns, for comparison only.
    """
    rng = np.random.default_rng(rf_params.get("random_state"))
    permutation, impurity = {}, {}
    for i, (X_fit, y_fit, X_val, y_val) in enumerate(_folds(X_train, y_train, folds)):
        logger.info("Grouped tree selection fold %d/%d: fit %d rows, validate %d rows",
                    i + 1, len(folds), len(X_fit), len(X_val))
        model = RandomForestRegressor(**rf_params).fit(X_fit, y_fit)
        permutation[i] = _grouped_permutation_importance(model, X_val, y_val, groups, n_repeats, rng)
        impurity[i] = pd.Series(model.feature_importances_, index=X_train.columns).groupby(groups).sum()

    suffixes = pd.Series(groups.index.str.rpartition("_").get_level_values(2), index=groups.index)
    is_statistic = suffixes.isin(config.STATISTICAL_PROPERTIES) & (groups != groups.index)

    permutation = pd.DataFrame(permutation)
    importance = pd.DataFrame({
        "permutation_mean": permutation.mean(axis=1),
        "permutation_std": permutation.std(axis=1),
        "impurity_sum": pd.DataFrame(impurity).mean(axis=1),
        "n_features": groups.value_counts(),
        "statistics": suffixes[is_statistic].groupby(groups[is_statistic]).agg(", ".join),
    }).sort_values("permutation_mean", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance


def cv_rmse_by_group_count(X_train: pd.DataFrame, y_train: pd.DataFrame, groups: pd.Series, ranking: pd.Index,
                           folds: list[tuple[pd.DataFrame, pd.DataFrame]], rf_params: dict) -> pd.DataFrame:
    """Mean CV RMSE of a RandomForest on all columns of the top-k ranked groups, for every k = 1..n_groups."""
    rows = []
    for k in range(1, len(ranking) + 1):
        columns = groups.index[groups.isin(ranking[:k])]
        rows.append({"n_groups": k, "n_features": len(columns), **_cv_rmse(X_train, y_train, columns, folds, rf_params)})
        logger.info("Top %d groups (%d features): CV RMSE %.2f", k, len(columns), rows[-1]["cv_rmse"])
    return pd.DataFrame(rows)


# ==========================================
# ------------- Selection run for the TSFM covariates (dashboard)
@dataclass
class SelectionResult:
    """Ranking of the candidate features of one target, with the CV RMSE curve of the top-k."""
    target_source: str
    mode: str                    # "features" or "groups"
    lags: bool                   # target lags were candidates
    calendar: bool               # calendar dummies were candidates
    importance: pd.DataFrame
    curve: pd.DataFrame
    columns: list[str]           # every candidate column
    weather_columns: list[str]   # the candidates that can be TSFM covariates


def selection_features(target: pd.Series, weather: pd.DataFrame, lags: bool, calendar: bool) -> pd.DataFrame:
    """Candidate features: the weather columns, plus optionally lags of the target and calendar dummies.

    Lags run from PREDICTION_LENGTH to PREDICTION_LENGTH + CONTEXT_LENGTH - 1 steps back (add_lag_features), so they
    are known at forecast time. Lags and calendar dummies only compete in the ranking, they are never TSFM covariates:
    the TSFMs get the target history as context and build such features internally.
    """
    X = weather
    if lags:
        X = X.join(add_lag_features(target.to_frame(), config.CONTEXT_LENGTH, config.PREDICTION_LENGTH))
    if calendar:
        X = add_calendar_features(X, config.TIMEZONE, config.COUNTRY)
    return X


def time_folds(X: pd.DataFrame, n_splits: int = config.CV_N_SPLITS, test_size: str | None = config.CV_TEST_SIZE,
               gap: str = config.CV_GAP, max_train_size: str | None = config.CV_MAX_TRAIN_SIZE
               ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Time-ordered (X_fit, X_val) folds from the CV_* settings; durations are converted to rows of X's median step."""
    step = X.index.to_series().diff().median()

    def rows(duration: str | None) -> int | None:
        return None if duration is None else int(pd.Timedelta(duration) / step)

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=rows(test_size), gap=rows(gap),
                               max_train_size=rows(max_train_size))
    return [(X.iloc[fit], X.iloc[val]) for fit, val in splitter.split(X)]


def run_tree_selection(target_source: str, target: pd.Series, weather: pd.DataFrame, mode: str,
                       lags: bool, calendar: bool) -> SelectionResult:
    """Rank the candidate features of `target` on the training period (TRAIN_START..TRAIN_END, rows with gaps dropped).

    mode "features" ranks single columns (curve over TREE_SELECTION_SUBSET_SIZES), "groups" ranks feature_groups
    (curve over every group count).
    """
    X = selection_features(target, weather, lags, calendar)
    train = slice(pd.Timestamp(config.TRAIN_START), pd.Timestamp(config.TRAIN_END) - pd.Timedelta(config.RESOLUTIONCODE))
    data = X.join(target).loc[train].dropna()
    X, y = data[X.columns], data[[target.name]]
    folds = time_folds(X)
    params, repeats = config.TREE_SELECTION_RF_PARAMS, config.TREE_SELECTION_N_REPEATS
    logger.info("Feature selection %s (%s): %d rows, %d candidates", target_source, mode, len(X), X.shape[1])
    if mode == "groups":
        groups = feature_groups(X.columns, config.STATISTICAL_PROPERTIES, config.TREE_GROUP_LAG_BLOCK)
        importance = tree_group_importance(X, y, groups, folds, params, repeats)
        curve = cv_rmse_by_group_count(X, y, groups, importance.index, folds, params)
    else:
        importance = tree_feature_importance(X, y, folds, params, repeats)
        curve = cv_rmse_by_subset_size(X, y, importance.index, folds, params, config.TREE_SELECTION_SUBSET_SIZES)
    return SelectionResult(target_source, mode, lags, calendar, importance, curve,
                           list(X.columns), list(weather.columns))


def tsfm_covariates(result: SelectionResult, selected: list[str]) -> list[str]:
    """Weather columns of the selected features or groups, in rank order; lags and calendar dummies are left out."""
    if result.mode == "groups":
        groups = feature_groups(pd.Index(result.columns), config.STATISTICAL_PROPERTIES, config.TREE_GROUP_LAG_BLOCK)
        selected = [col for group in selected for col in groups.index[groups == group]]
    return [col for col in selected if col in result.weather_columns]


def _selection_stem(target_source: str, mode: str, lags: bool, calendar: bool) -> Path:
    """e.g. results/feature_selection/wind_groups_lags_calendar"""
    extras = "".join(name for name, on in (("_lags", lags), ("_calendar", calendar)) if on)
    return config.TREE_SELECTION_DIR / f"{target_source}_{mode}{extras}"


def save_selection(result: SelectionResult) -> Path:
    """Write <stem>_importance.parquet, <stem>_curve.parquet and <stem>.json (candidate columns, training period)."""
    stem = _selection_stem(result.target_source, result.mode, result.lags, result.calendar)
    stem.parent.mkdir(parents=True, exist_ok=True)
    result.importance.to_parquet(stem.with_name(f"{stem.name}_importance.parquet"))
    result.curve.to_parquet(stem.with_name(f"{stem.name}_curve.parquet"))
    meta = {"columns": result.columns, "weather_columns": result.weather_columns,
            "train": [config.TRAIN_START, config.TRAIN_END]}
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Feature selection saved to %s", stem)
    return stem


def load_selection(target_source: str, mode: str, lags: bool, calendar: bool) -> SelectionResult | None:
    """Read back a saved selection, None if it was not run with these settings yet."""
    stem = _selection_stem(target_source, mode, lags, calendar)
    if not stem.with_suffix(".json").exists():
        return None
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    return SelectionResult(target_source, mode, lags, calendar,
                           pd.read_parquet(stem.with_name(f"{stem.name}_importance.parquet")),
                           pd.read_parquet(stem.with_name(f"{stem.name}_curve.parquet")),
                           meta["columns"], meta["weather_columns"])
