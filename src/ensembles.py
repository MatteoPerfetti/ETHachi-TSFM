"""Forecast ensembles for the shared backtest prediction table.

The functions in this module combine forecasts *after* the base models have
run.  They therefore work with traditional models from ``MLtrad.py`` and with
foundation models without importing or fitting either family again.

Implemented ensembles
---------------------
``equal_mean_ensemble``
    Gives every available base model the same weight at each issue time.

``top_k_exponential_ensemble``
    Reproduces the method described in the technical report: rank the
    base models on each recent day, retain that day's top K models, give more
    recent days exponentially larger weights, and share each day's weight
    equally among its selected models.  A model selected on several days
    receives the sum of those contributions.

Assumptions
---------------------------------------------------------
* A model's daily performance is the RMSE of its q50 forecast over the full
  forecast horizon.
* An origin can influence a new forecast only after its complete horizon is
  observable.  This is essential for overlapping 39-hour forecasts and avoids
  using future targets.
* The day weight is ``exp(-decay * age)``.  With ``decay=1`` and three history
  days this gives 0.67, 0.24, and 0.09 after rounding, matching the report.
* q10 and q90 are combined only when every positively weighted model supplies
  both bounds for the complete issue.  Otherwise the ensemble remains a point
  forecast and both interval columns are NaN.

The report says that lookback days and top K are retuned weekly, but does not
state the tuning loss or validation protocol.  They remain explicit arguments
here; a leakage-free weekly tuner should select them on historical walk-forward
RMSE and then hold them fixed for the following week.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecast import STEP


KEY_COLUMNS = ["issue_time", "timestamp", "lead_step"]
QUANTILE_COLUMNS = ["q10", "q50", "q90"]
PREDICTION_COLUMNS = [*KEY_COLUMNS, "model", *QUANTILE_COLUMNS, "actual"]

# Operational defaults used by main.py and the dashboard.
# The equal mean has no fitted parameters; the top_k_exponential
# ensemble uses the report's illustrative three-day/top-three
# setup and decay. Offline walk-forward tuning can update these values before a
# new evaluation run.

DEFAULT_PARAMETERS = {
    "equal_mean": {},
    "top_k_exponential": {
        "lookback_days": 3,
        "top_k": 3,
        "decay": 1.0,
    },
}


@dataclass(frozen=True)
class EnsembleResult:
    """Combined forecasts and the base-model weights used at each issue time."""

    predictions: pd.DataFrame
    weights: pd.DataFrame


def _select_models(predictions: pd.DataFrame, models: list[str] | None) -> pd.DataFrame:
    """Return the requested base-model rows in canonical column order."""
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")

    selected = predictions if models is None else predictions[predictions["model"].isin(models)]
    if selected.empty:
        raise ValueError("no base-model predictions were selected")
    if selected.duplicated([*KEY_COLUMNS, "model"]).any():
        raise ValueError("expected one prediction per issue, timestamp, lead step, and model")
    return selected[PREDICTION_COLUMNS].copy()


def _combine_issue(rows: pd.DataFrame, weights: pd.Series, name: str) -> pd.DataFrame:
    """Apply one model-weight vector to all target timestamps of one issue."""
    point = rows.pivot(index=KEY_COLUMNS, columns="model", values="q50")
    usable = [model for model in weights.index if model in point and point[model].notna().all()]
    weights = weights.loc[usable]
    weights = weights / weights.sum()

    combined = pd.DataFrame(index=point.index)
    combined["q50"] = point[usable].mul(weights, axis="columns").sum(axis="columns")

    lower = rows.pivot(index=KEY_COLUMNS, columns="model", values="q10").reindex(columns=usable)
    upper = rows.pivot(index=KEY_COLUMNS, columns="model", values="q90").reindex(columns=usable)
    complete_interval = lower.notna().all().all() and upper.notna().all().all()
    combined["q10"] = lower.mul(weights, axis="columns").sum(axis="columns") if complete_interval else np.nan
    combined["q90"] = upper.mul(weights, axis="columns").sum(axis="columns") if complete_interval else np.nan

    actual = rows.groupby(KEY_COLUMNS, sort=False)["actual"].first()
    combined["actual"] = actual.reindex(combined.index)
    combined = combined.reset_index()
    combined.insert(3, "model", name)
    return combined[PREDICTION_COLUMNS]


def equal_mean_ensemble(
    predictions: pd.DataFrame,
    models: list[str] | None = None,
    name: str = "Equal mean ensemble",
) -> EnsembleResult:
    """Average the selected base forecasts with equal weights.

    Weights are recalculated at each issue time so an issue is still usable if
    one of the requested models did not produce a forecast on that day.
    """
    selected = _select_models(predictions, models)
    forecasts, weight_rows = [], []

    for issue, rows in selected.groupby("issue_time", sort=True):
        available = rows.groupby("model")["q50"].apply(lambda values: values.notna().all())
        model_names = available[available].index
        weights = pd.Series(1 / len(model_names), index=model_names, dtype=float)
        forecasts.append(_combine_issue(rows, weights, name))
        weight_rows.extend(
            {"issue_time": issue, "model": model, "weight": weight, "source_days": 0}
            for model, weight in weights.items()
        )

    return EnsembleResult(
        predictions=pd.concat(forecasts, ignore_index=True),
        weights=pd.DataFrame(weight_rows),
    )


def top_k_exponential_ensemble(
    predictions: pd.DataFrame,
    lookback_days: int,
    top_k: int,
    models: list[str] | None = None,
    decay: float = 1.0,
    latency_steps: int = 0,
    name: str = "Top-K exponential ensemble",
) -> EnsembleResult:
    """Combine recent daily winners using exponentially decaying day weights.

    For each current issue, only earlier forecast origins whose complete target
    horizon was observable before the current context cutoff are eligible for
    ranking.  The most recent ``lookback_days`` eligible origins are retained.
    Within every retained origin, the ``top_k`` models with the lowest full-
    horizon q50 RMSE share that origin's weight equally.  Until a complete
    historical origin exists, the function falls back to the equal mean.

    ``decay=1`` reproduces the report's illustrative three-day weights.  The
    returned weight table records the final aggregated model weights used for
    each new issue and how many historical days selected each model.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if decay < 0:
        raise ValueError("decay must be non-negative")

    selected = _select_models(predictions, models)
    selected["squared_error"] = (selected["actual"] - selected["q50"]) ** 2

    score_rows = (
        selected.groupby(["issue_time", "model"], sort=True)
        .agg(
            rmse=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            scored_steps=("squared_error", "count"),
            forecast_steps=("q50", "size"),
        )
        .reset_index()
    )
    score_rows = score_rows[score_rows["scored_steps"] == score_rows["forecast_steps"]]
    last_targets = selected.groupby("issue_time")["timestamp"].max().sort_index()

    forecasts, weight_rows = [], []
    for issue, rows in selected.groupby("issue_time", sort=True):
        current_models = rows.groupby("model")["q50"].apply(lambda values: values.notna().all())
        current_models = current_models[current_models].index
        context_end = issue - (latency_steps + 1) * STEP
        eligible_origins = last_targets[last_targets <= context_end].index[-lookback_days:]

        history = score_rows[
            score_rows["issue_time"].isin(eligible_origins)
            & score_rows["model"].isin(current_models)
        ]
        history_origins = sorted(history["issue_time"].unique())

        contributions: dict[str, float] = {}
        selection_counts: dict[str, int] = {}
        raw_day_weights = np.exp(-decay * np.arange(len(history_origins) - 1, -1, -1))
        raw_day_weights = raw_day_weights / raw_day_weights.sum() if len(raw_day_weights) else raw_day_weights

        for origin, day_weight in zip(history_origins, raw_day_weights):
            day_scores = history[history["issue_time"] == origin].nsmallest(top_k, "rmse")
            selected_models = day_scores["model"].tolist()
            model_share = day_weight / len(selected_models)
            for model in selected_models:
                contributions[model] = contributions.get(model, 0.0) + model_share
                selection_counts[model] = selection_counts.get(model, 0) + 1

        if contributions:
            weights = pd.Series(contributions, dtype=float).sort_index()
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1 / len(current_models), index=current_models, dtype=float)

        forecasts.append(_combine_issue(rows, weights, name))
        weight_rows.extend(
            {
                "issue_time": issue,
                "model": model,
                "weight": weight,
                "source_days": selection_counts.get(model, 0),
            }
            for model, weight in weights.items()
        )

    return EnsembleResult(
        predictions=pd.concat(forecasts, ignore_index=True),
        weights=pd.DataFrame(weight_rows),
    )


def build_ensemble(
    ensemble_key: str,
    predictions: pd.DataFrame,
    models: list[str],
    strategy: str,
    latency_steps: int,
) -> EnsembleResult:
    """Build one configured ensemble for the shared backtest engine."""
    name = {
        "equal_mean": "Equal mean ensemble",
        "top_k_exponential": "Top-K exponential ensemble",
    }[ensemble_key]
    name = f"{name} [{strategy}]"

    if ensemble_key == "equal_mean":
        return equal_mean_ensemble(
            predictions=predictions,
            models=models,
            name=name,
        )
    return top_k_exponential_ensemble(
        predictions=predictions,
        models=models,
        latency_steps=latency_steps,
        name=name,
        **DEFAULT_PARAMETERS[ensemble_key],
    )
