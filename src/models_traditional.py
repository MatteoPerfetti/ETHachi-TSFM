"""Adapters that expose the traditional ML benchmarks to the shared backtest.

The estimator implementations, defaults, quantile losses, cross-validation,
and Optuna routines stay in :mod:`src.MLtrad`. This module only translates the
shared ``ForecastRequest`` objects into the two supported tabular feature
layouts. Dashboard runs never invoke Optuna; they use the defaults from
``MLtrad.DEFAULT_PARAMETERS`` plus optional overrides from config.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from logger import logger
from src.features import add_calendar_features, add_lag_features
from src.forecast import STEP, Forecaster, ForecastRequest, ForecastResult
from src.MLtrad import (
    build_origin_aligned_dataset,
    fit_traditional_model,
    predict_quantiles,
)


class TraditionalMLAdapter(Forecaster):
    """One traditional model evaluated with one multi-horizon feature mode."""

    uses_covariates = True

    def __init__(
        self,
        model_key: str,
        strategy: str,
        horizon_steps: int,
        latency_steps: int,
        context_steps: int = config.TRADITIONAL_CONTEXT_STEPS,
        parameters: dict | None = None,
        issue_time: str = config.BACKTEST_ISSUE_TIME,
    ) -> None:
        if model_key not in config.TRADITIONAL_MODELS:
            raise ValueError(f"Unknown traditional model: {model_key}")
        if strategy not in config.TRADITIONAL_STRATEGIES:
            raise ValueError(f"Unknown traditional strategy: {strategy}")

        self.model_key = model_key
        self.strategy = strategy
        self.horizon_steps = horizon_steps
        self.latency_steps = latency_steps
        self.context_steps = context_steps
        self.parameters = dict(parameters or {})
        self.issue_time = issue_time
        self.name = f"{config.MODELS_REGISTRY[model_key]['name']} [{strategy}]"
        self.min_context = (
            context_steps
            if strategy == "origin_aligned"
            else horizon_steps + latency_steps + context_steps - 1
        )
        self.feature_columns: list[str] = []
        self.target_name = "target"
        self.fitted_model = None

    @property
    def configuration(self) -> dict:
        """Serializable settings stored next to the saved dashboard run."""
        return {
            "model_key": self.model_key,
            "strategy": self.strategy,
            "context_steps": self.context_steps,
            "issue_time": self.issue_time,
            "parameters": self.parameters or "defaults",
        }

    @staticmethod
    def _with_calendar(
        index: pd.DatetimeIndex,
        covariates: pd.DataFrame | None,
    ) -> pd.DataFrame:
        base = pd.DataFrame(index=index) if covariates is None else covariates.copy()
        return add_calendar_features(base, config.TIMEZONE, config.COUNTRY)

    def fit(self, target: pd.Series, covariates: pd.DataFrame | None = None) -> None:
        target = target.astype(float).sort_index()
        self.target_name = target.name or "target"
        features = self._with_calendar(target.index, covariates)

        if self.strategy == "origin_aligned":
            X_train, y_train = build_origin_aligned_dataset(
                target=target,
                covariates=features,
                start=target.index[0],
                end=target.index[-1] + STEP,
                context_length=self.context_steps,
                prediction_length=self.horizon_steps,
                forecast_time=self.issue_time,
                timezone="UTC",
                frequency=STEP,
                latency_steps=self.latency_steps,
            )
        else:
            safe_gap = self.horizon_steps + self.latency_steps
            lagged = add_lag_features(
                target.to_frame(self.target_name),
                context_length=self.context_steps,
                prediction_length=safe_gap,
            )
            X_train = features.join(lagged, how="inner").dropna()
            y_train = target.reindex(X_train.index)

        self.feature_columns = X_train.columns.tolist()
        logger.info(
            "%s training matrix: %d rows x %d features",
            self.name,
            X_train.shape[0],
            X_train.shape[1],
        )
        self.fitted_model = fit_traditional_model(
            self.model_key,
            X_train,
            y_train,
            config.TRADITIONAL_QUANTILES,
            self.parameters,
        )

    def _origin_aligned_features(self, request: ForecastRequest) -> pd.DataFrame:
        index = request.forecast_index
        target_name = request.context.name or self.target_name
        covariates = None
        if request.future_covariates is not None:
            covariates = request.future_covariates.loc[index]
        future = self._with_calendar(index, covariates)
        horizon = pd.DataFrame(
            {"lead_step": np.arange(1, len(index) + 1)},
            index=index,
        )
        context = {
            f"{target_name}_lag_{lag}": np.repeat(
                request.context.iloc[-lag],
                len(index),
            )
            for lag in range(1, self.context_steps + 1)
        }
        return pd.concat([horizon, future, pd.DataFrame(context, index=index)], axis=1)

    def _fixed_gap_features(self, request: ForecastRequest) -> pd.DataFrame:
        index = request.forecast_index
        target_name = request.context.name or self.target_name
        covariates = None
        if request.future_covariates is not None:
            covariates = request.future_covariates.loc[index]
        future = self._with_calendar(index, covariates)
        safe_gap = self.horizon_steps + self.latency_steps
        lagged = {
            f"{target_name}_lag_{lag}": request.context.reindex(
                index - lag * STEP
            ).to_numpy()
            for lag in range(safe_gap, safe_gap + self.context_steps)
        }
        return pd.concat([future, pd.DataFrame(lagged, index=index)], axis=1)

    def predict(self, requests: list[ForecastRequest]) -> list[ForecastResult]:
        if self.fitted_model is None:
            raise RuntimeError(f"{self.name} must be fitted before prediction")

        results = []
        for request in requests:
            if self.strategy == "origin_aligned":
                features = self._origin_aligned_features(request)
            else:
                features = self._fixed_gap_features(request)
            features = features.reindex(columns=self.feature_columns)
            predictions = predict_quantiles(self.fitted_model, features)
            results.append(
                ForecastResult(
                    model=self.name,
                    issue_time=request.issue_time,
                    quantiles=predictions,
                )
            )
        return results


def load_traditional(
    model_key: str,
    strategy: str,
    horizon_steps: int,
    latency_steps: int,
    issue_time: str,
) -> TraditionalMLAdapter:
    """Create an untuned/configured traditional benchmark adapter."""
    return TraditionalMLAdapter(
        model_key=model_key,
        strategy=strategy,
        horizon_steps=horizon_steps,
        latency_steps=latency_steps,
        parameters=config.TRADITIONAL_MODEL_PARAMETERS[model_key],
        issue_time=issue_time,
    )
