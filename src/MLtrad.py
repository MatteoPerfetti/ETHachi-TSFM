"""Traditional machine-learning benchmarks for the Hitachi comparison.

Assumptions and differences from the Hitachi technical report:
- The implemented model families are XGBoost, LightGBM, CatBoost, NGBoost,
  and TabNet, with Q10, Q50, and Q90 outputs.
- XGBoost, LightGBM, CatBoost, and TabNet fit one model per quantile. NGBoost
  fits one Normal predictive distribution and extracts all three quantiles.
- The report does not disclose its hyperparameters, so this module uses
  explicit defaults and time-based Optuna tuning with mean pinball loss.
- The current project uses only the available ICON/DWD weather dataset,
  rather than separate datasets from eight numerical weather models.
- The current target is total Belgian wind generation in MW and is not
  normalized by available capacity; the report forecasts offshore wind and
  trains on normalized power.
- The report's adaptive multi-provider ensemble is not reproduced here. Each
  model is evaluated independently through the shared project pipeline.
- Historical DWD covariates are not verified forecast vintages. Results are
  therefore development benchmarks, not an exact operational reproduction.
- The optional origin-aligned pooled dataset is a project extension. The
  report does not specify target context windows or pooled forecast horizons.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss, root_mean_squared_error

import config
from src.models import cross_validation_splits


DEFAULT_PARAMETERS = {
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "lightgbm": {
        "n_estimators": 300,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "catboost": {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
    },
    "ngboost": {
        "n_estimators": 150,              # default 500
        "learning_rate": 0.05,            # default 0.03
        "minibatch_frac": 0.5,            # default 1.0
        "col_sample": 0.7,                # default 1.0
    },
    "tabnet": {
        "n_d": 8,                        # default 16
        "n_a": 8,                        # default 16
        "n_steps": 3,                    # default 5
        "gamma": 1.3,                    # default 1.5
        "lambda_sparse": 1e-4,           # default 1e-4
        "optimizer_lr": 0.02,            # default 2e-2
        "max_epochs": 25,                # default 100    
    },
}


@dataclass
class FittedTraditionalModel:
    name: str
    quantiles: tuple[float, ...]
    estimators: dict[float, Any]


def quantile_column(quantile: float) -> str:
    """Return the report-style column name for a quantile."""
    return f"q{int(round(100 * quantile)):02d}"


def _target_series(target: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(target, pd.DataFrame):
        return target.iloc[:, 0].astype(float)
    return target.astype(float)


def _forecast_origins(
    index: pd.DatetimeIndex,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    prediction_length: int,
    forecast_time: str,
    timezone: str,
    frequency: str | pd.Timedelta,
) -> pd.DatetimeIndex:
    """Return daily local-time origins whose complete horizon is in the period."""
    step = pd.Timedelta(frequency)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts
    start_ts = (
        start_ts.tz_convert(index.tz)
        if index.tz is not None
        else start_ts.tz_convert(None)
    )
    end_ts = (
        end_ts.tz_convert(index.tz)
        if index.tz is not None
        else end_ts.tz_convert(None)
    )

    local_start = (
        start_ts.tz_localize("UTC").tz_convert(timezone)
        if index.tz is None
        else start_ts.tz_convert(timezone)
    )
    local_end = (
        end_ts.tz_localize("UTC").tz_convert(timezone)
        if index.tz is None
        else end_ts.tz_convert(timezone)
    )
    hour, minute = map(int, forecast_time.split(":"))
    local_days = pd.date_range(
        local_start.tz_localize(None).normalize(),
        (local_end - step).tz_localize(None).normalize(),
        freq="1D",
    )
    local_origins = (
        local_days + pd.Timedelta(hours=hour, minutes=minute)
    ).tz_localize(timezone)
    origins = (
        local_origins.tz_convert(index.tz)
        if index.tz is not None
        else local_origins.tz_convert("UTC").tz_localize(None)
    )
    last_target_offset = step * (prediction_length - 1)
    return origins[
        (origins >= start_ts)
        & (origins + last_target_offset < end_ts)
    ]


def build_origin_aligned_dataset(
    target: pd.Series | pd.DataFrame,
    covariates: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    context_length: int,
    prediction_length: int,
    forecast_time: str,
    timezone: str,
    frequency: str | pd.Timedelta = "15min",
    latency_steps: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build one pooled tabular row per forecast origin and lead step.

    Every horizon row receives the same target context ending
    ``latency_steps`` before its forecast origin. Target-time covariates and
    ``lead_step`` vary across the horizon. The resulting index is
    ``(forecast_origin, target_time)``.
    """
    target_series = _target_series(target).sort_index()
    covariates = covariates.sort_index()
    step = pd.Timedelta(frequency)
    origins = _forecast_origins(
        target_series.index,
        start,
        end,
        prediction_length,
        forecast_time,
        timezone,
        step,
    )
    context_columns = [
        f"{target_series.name}_lag_{lag}"
        for lag in range(1, context_length + 1)
    ]
    origin_index = origins.repeat(prediction_length)
    lead_steps = np.tile(
        np.arange(1, prediction_length + 1),
        len(origins),
    )
    target_offsets = pd.to_timedelta(
        (lead_steps - 1) * step.value,
        unit="ns",
    )
    target_index = origin_index + target_offsets
    pooled_index = pd.MultiIndex.from_arrays(
        [origin_index, target_index],
        names=["forecast_origin", "target_time"],
    )

    context_by_origin = np.column_stack([
        target_series.reindex(
            origins - (latency_steps + lag) * step
        ).to_numpy()
        for lag in range(1, context_length + 1)
    ])
    context = pd.DataFrame(
        np.repeat(context_by_origin, prediction_length, axis=0),
        index=pooled_index,
        columns=context_columns,
    )
    horizon = pd.DataFrame(
        {"lead_step": lead_steps},
        index=pooled_index,
    )
    target_covariates = covariates.reindex(target_index)
    target_covariates.index = pooled_index
    target_values = target_series.reindex(target_index)
    target_values.index = pooled_index

    pooled = pd.concat(
        [horizon, target_covariates, context, target_values.rename("__target__")],
        axis=1,
    ).dropna()
    pooled_target = pooled.pop("__target__").rename(target_series.name)
    return pooled, pooled_target


def _origin_aligned_validation_splits(
    features: pd.DataFrame,
    n_splits: int,
    test_size: str | pd.Timedelta | None,
    gap: str | pd.Timedelta,
    max_train_size: str | pd.Timedelta | None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split pooled forecasts by origin and purge overlapping target windows."""
    origin_index = features.index.get_level_values("forecast_origin")
    target_index = features.index.get_level_values("target_time")
    origins = pd.DatetimeIndex(origin_index.unique()).sort_values()
    origin_step = origins.to_series().diff().median()
    start = origins[0]
    end = origins[-1] + origin_step
    validation_span = (
        pd.Timedelta(test_size)
        if test_size is not None
        else (end - start) / (n_splits + 1)
    )
    gap = pd.Timedelta(gap)
    last_target_by_origin = (
        pd.Series(target_index, index=origin_index)
        .groupby(level=0)
        .max()
    )

    for fold in range(n_splits):
        validation_end = end - (n_splits - 1 - fold) * validation_span
        validation_start = validation_end - validation_span
        validation_origins = origins[
            (origins >= validation_start)
            & (origins < validation_end)
        ]
        first_validation_origin = validation_origins[0]
        training_origins = last_target_by_origin[
            last_target_by_origin < first_validation_origin - gap
        ].index
        if max_train_size is not None:
            train_start = first_validation_origin - gap - pd.Timedelta(max_train_size)
            training_origins = training_origins[training_origins >= train_start]

        yield (
            features[origin_index.isin(training_origins)],
            features[origin_index.isin(validation_origins)],
        )


def _build_quantile_estimator(
    model_name: str,
    quantile: float,
    parameters: dict,
):
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            tree_method="hist",
            random_state=42,
            **parameters,
        )

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            random_state=42,
            verbosity=-1,
            **parameters,
        )

    if model_name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function=f"Quantile:alpha={quantile}",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            **parameters,
        )

    from pytorch_tabnet.tab_model import TabNetRegressor

    tabnet_parameters = parameters.copy()
    optimizer_lr = tabnet_parameters.pop("optimizer_lr")
    tabnet_parameters.pop("max_epochs")
    return TabNetRegressor(
        optimizer_params={"lr": optimizer_lr},
        seed=42,
        verbose=0,
        **tabnet_parameters,
    )


def _tabnet_quantile_loss(quantile: float):
    import torch

    class PinballLoss(torch.nn.Module):
        def forward(self, prediction, target):
            error = target - prediction
            return torch.maximum(
                quantile * error,
                (quantile - 1) * error,
            ).mean()

    return PinballLoss()


def fit_traditional_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    quantiles: tuple[float, ...],
    model_parameters: dict | None = None,
) -> FittedTraditionalModel:
    """Fit one probabilistic model or one independent model per quantile."""
    parameters = {
        **DEFAULT_PARAMETERS[model_name],
        **(model_parameters or {}),
    }
    target = _target_series(y_train)

    if model_name == "ngboost":
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from ngboost.scores import LogScore

        estimator = NGBRegressor(
            Dist=Normal,
            Score=LogScore,
            random_state=42,
            verbose=False,
            **parameters,
        )
        estimator.fit(X_train, target.to_numpy())
        return FittedTraditionalModel(model_name, quantiles, {0.5: estimator})

    estimators = {}
    for quantile in quantiles:
        estimator = _build_quantile_estimator(
            model_name,
            quantile,
            parameters,
        )
        if model_name == "tabnet":
            estimator.fit(
                X_train.to_numpy(dtype=np.float32, copy=True),
                target.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1),
                loss_fn=_tabnet_quantile_loss(quantile),
                max_epochs=parameters["max_epochs"],
                patience=0,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
            )
        else:
            estimator.fit(X_train, target)
        estimators[quantile] = estimator

    return FittedTraditionalModel(model_name, quantiles, estimators)


def predict_quantiles(
    fitted_model: FittedTraditionalModel,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Return timestamp-aligned Q10, Q50, and Q90 predictions."""
    if fitted_model.name == "ngboost":
        estimator = fitted_model.estimators[0.5]
        distribution = estimator.pred_dist(X)
        predictions = {
            quantile_column(quantile): np.asarray(
                distribution.ppf(quantile)
            ).reshape(-1)
            for quantile in fitted_model.quantiles
        }
    else:
        model_input = (
            X.to_numpy(dtype=np.float32, copy=True)
            if fitted_model.name == "tabnet"
            else X
        )
        predictions = {
            quantile_column(quantile): np.asarray(
                fitted_model.estimators[quantile].predict(model_input)
            ).reshape(-1)
            for quantile in fitted_model.quantiles
        }

    return pd.DataFrame(predictions, index=X.index)


def mean_quantile_loss(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    quantiles: tuple[float, ...],
) -> float:
    losses = [
        mean_pinball_loss(
            y_true,
            predictions[quantile_column(quantile)],
            alpha=quantile,
        )
        for quantile in quantiles
    ]
    return float(np.mean(losses))


def mean_winkler_interval(
    y_true: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    interval_alpha: float,
) -> float:
    """Calculate the mean Winkler score for the central prediction interval."""
    y = y_true.to_numpy(dtype=float)
    lower_values = lower.to_numpy(dtype=float)
    upper_values = upper.to_numpy(dtype=float)
    score = upper_values - lower_values
    score += (2 / interval_alpha) * (lower_values - y) * (y < lower_values)
    score += (2 / interval_alpha) * (y - upper_values) * (y > upper_values)
    return float(np.mean(score))


def score_quantile_forecast(
    y_true: pd.Series | pd.DataFrame,
    predictions: pd.DataFrame,
    quantiles: tuple[float, ...],
) -> dict[str, float]:
    """Score Q50 with RMSE and Q10-Q90 with probabilistic metrics."""
    target = _target_series(y_true)
    lower_quantile = quantiles[0]
    median_quantile = quantiles[len(quantiles) // 2]
    upper_quantile = quantiles[-1]
    lower = predictions[quantile_column(lower_quantile)]
    median = predictions[quantile_column(median_quantile)]
    upper = predictions[quantile_column(upper_quantile)]

    return {
        "rmse": float(root_mean_squared_error(target, median)),
        "mean_pinball_loss": mean_quantile_loss(
            target,
            predictions,
            quantiles,
        ),
        "mean_winkler_interval": mean_winkler_interval(
            target,
            lower,
            upper,
            interval_alpha=1 - upper_quantile + lower_quantile,
        ),
        "interval_coverage": float(
            ((target >= lower) & (target <= upper)).mean()
        ),
    }


def _suggest_parameters(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, step=10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 800, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }

    if model_name == "ngboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "minibatch_frac": trial.suggest_float("minibatch_frac", 0.5, 1.0),
            "col_sample": trial.suggest_float("col_sample", 0.5, 1.0),
        }

    width = trial.suggest_categorical("width", [8, 16, 32])
    return {
        "n_d": width,
        "n_a": width,
        "n_steps": trial.suggest_int("n_steps", 3, 7),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-3, 3e-2, log=True),
        "max_epochs": trial.suggest_int("max_epochs", 50, 200, step=50),
    }


def cross_validate_traditional_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    quantiles: tuple[float, ...],
    model_parameters: dict | None = None,
) -> tuple[float, list[float]]:
    """Evaluate one parameter set with expanding-window validation."""
    target = _target_series(y_train)
    fold_losses = []

    if "forecast_origin" in X_train.index.names:
        folds = _origin_aligned_validation_splits(
            X_train,
            config.CV_N_SPLITS,
            config.CV_TEST_SIZE,
            config.CV_GAP,
            config.CV_MAX_TRAIN_SIZE,
        )
    else:
        folds = cross_validation_splits(
            X_train,
            config.CV_N_SPLITS,
            config.CV_TEST_SIZE,
            config.CV_GAP,
            config.CV_MAX_TRAIN_SIZE,
        )

    for X_fold_train, X_fold_validation in folds:
        fitted_model = fit_traditional_model(
            model_name,
            X_fold_train,
            target.loc[X_fold_train.index],
            quantiles,
            model_parameters,
        )
        predictions = predict_quantiles(
            fitted_model,
            X_fold_validation,
        )
        fold_losses.append(
            mean_quantile_loss(
                target.loc[X_fold_validation.index],
                predictions,
                quantiles,
            )
        )

    return float(np.mean(fold_losses)), fold_losses


def tune_traditional_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    quantiles: tuple[float, ...],
    n_trials: int,
) -> tuple[dict, float]:
    """Tune a traditional model on mean pinball loss across time folds."""
    import optuna

    def objective(trial: optuna.Trial) -> float:
        parameters = _suggest_parameters(trial, model_name)
        trial.set_user_attr("model_parameters", parameters)
        mean_loss, _ = cross_validate_traditional_model(
            model_name,
            X_train,
            y_train,
            quantiles,
            parameters,
        )
        return mean_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return (
        study.best_trial.user_attrs["model_parameters"],
        float(study.best_value),
    )
