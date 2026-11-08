import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import RobustScaler

from logger import logger

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

    permutation = pd.DataFrame(permutation)
    importance = pd.DataFrame({
        "permutation_mean": permutation.mean(axis=1),
        "permutation_std": permutation.std(axis=1),
        "impurity_sum": pd.DataFrame(impurity).mean(axis=1),
        "n_features": groups.value_counts(),
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
