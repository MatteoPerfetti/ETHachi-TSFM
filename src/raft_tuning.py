"""Chronological Optuna search for the RAFT/TSFM point forecast."""
from dataclasses import dataclass, replace

import numpy as np
import optuna
import pandas as pd

from src.backtest import BacktestSettings, run_backtest
from src.dataset import Dataset
from src.forecast import STEP
from src.raft import RAFT_NAME, RaftRetriever, RaftSettings, weighted_forecast

NEIGHBORS = [10, 15, 20]
TEMPERATURES = [0.05, 0.1, 0.2]


@dataclass(frozen=True)
class RaftTuningResult:
    settings: RaftSettings
    metadata: dict


def tune_raft(dataset: Dataset, tsfm_keys: list[str], settings: BacktestSettings, validation_start: str,
              trials: int = 50, seed: int = 42) -> RaftTuningResult:
    """Select one shared blend using the mean daily RMSE over the selected TSFMs."""
    raft_settings = settings.raft
    if raft_settings is None or trials < 1 or not tsfm_keys:
        raise ValueError("RAFT tuning requires RAFT settings, at least one TSFM and a positive trial count")
    start = pd.to_datetime(validation_start, utc=True)
    train_end = min(pd.Timestamp(settings.train_end), start)
    # issue_times uses an exclusive end. The last validation target must precede the test.
    end = pd.Timestamp(settings.test_start) - (settings.horizon_steps - 1) * STEP
    if not pd.Timestamp(settings.train_start) < train_end or start >= end:
        raise ValueError("RAFT validation needs training history and complete horizons before the test")
    validation = replace(settings, train_end=train_end.isoformat(), test_start=start.isoformat(),
                         test_end=end.isoformat(), raft=replace(raft_settings, neighbors=max(NEIGHBORS)))
    # Run the TSFMs once. The largest candidate count fixes identical issues for every trial.
    base = run_backtest(dataset, tsfm_keys, validation).predictions
    index = ["issue_time", "lead_step"]
    reference = base.loc[base["model"].eq(RAFT_NAME)].set_index(index).sort_index()["actual"]
    tsfm = base.loc[base["model"].ne(RAFT_NAME) & ~base["model"].str.endswith(" + RAFT")]
    tsfm_points = tsfm.pivot(index=index, columns="model", values="q50").reindex(reference.index)
    issues = reference.index.get_level_values("issue_time").unique()
    actual = reference.to_numpy(dtype=float).reshape(len(issues), settings.horizon_steps)
    forecasts = tsfm_points.to_numpy(dtype=float).T.reshape(-1, *actual.shape)
    if not np.isfinite(forecasts).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite TSFM forecasts or targets in RAFT validation")

    retriever = RaftRetriever(validation.raft, dataset.target, settings.target_source,
                             int(settings.issue_time.split(":")[0]), pd.Timestamp(settings.train_start), train_end)
    inputs = [retriever.tuning_inputs(issue) for issue in issues]

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        lambda_ = trial.suggest_float("lambda_", 0.0, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 1.0)
        neighbors = trial.suggest_categorical("neighbors", NEIGHBORS)
        temperature = trial.suggest_categorical("temperature", TEMPERATURES)
        scale = retriever.production_scale_for(alpha)
        retrieval = np.stack([
            weighted_forecast(lambda_ * (alpha * shape + (1 - alpha) * level) / scale
                              + (1 - lambda_) * weather, values, neighbors, temperature)
            for shape, level, weather, values in inputs
        ])
        blended = gamma * forecasts + (1 - gamma) * retrieval
        return float(np.sqrt(np.mean((blended - actual) ** 2, axis=-1)).mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    if raft_settings.temperature in TEMPERATURES:
        initial = {name: getattr(raft_settings, name) for name in
                   ("alpha", "lambda_", "gamma", "neighbors", "temperature")}
        study.enqueue_trial(initial)
        if trials > 1 and raft_settings.gamma != 1:
            study.enqueue_trial({**initial, "gamma": 1.0})
    study.optimize(objective, n_trials=trials)
    return RaftTuningResult(replace(raft_settings, **study.best_params), {
        "objective": "mean_daily_rmse_mw_across_tsfms",
        "validation_rmse_mw": study.best_value,
        "best_params": study.best_params,
        "sampler_seed": seed,
        "trials": len(study.trials),
        "validation_issues": len(issues),
        "validation_first_issue": issues[0].isoformat(),
        "validation_last_issue": issues[-1].isoformat(),
        "validation_train_end": train_end.isoformat(),
        "test_start": settings.test_start,
        "tsfm_models": list(tsfm_points.columns),
    })
