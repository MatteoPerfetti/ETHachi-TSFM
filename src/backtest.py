"""Rolling-window backtest: one forecast per day of the test period, for every model, saved as one long table.

This is the only place that cuts the data in time. For each issue time it builds the ForecastRequests
(context up to the data latency before the issue time, covariates, forecast timestamps); the models only see those.
An issue time is skipped for every model if any of them would get incomplete data, so all models are scored
on the same days. Metrics are computed afterwards from the saved table (src/metrics.py).
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

import config
from logger import logger
from src.baselines import Persistence
from src.dataset import Dataset
from src.ensembles import DEFAULT_PARAMETERS as ENSEMBLE_PARAMETERS
from src.ensembles import build_ensemble
from src.forecast import STEP, Forecaster, ForecastRequest
from src.raft import CONTEXT_HOURS, HORIZON_HOURS, STEPS_PER_HOUR, RaftRetriever, RaftSettings


@dataclass
class BacktestSettings:
    """Everything that defines a backtest run; defaults come from config.py."""
    test_start: str = config.TEST_START
    test_end: str = config.TEST_END
    train_start: str = config.TRAIN_START
    train_end: str = config.TRAIN_END
    issue_time: str = config.BACKTEST_ISSUE_TIME
    horizon_steps: int = config.BACKTEST_HORIZON_STEPS
    context_steps: int = config.BACKTEST_CONTEXT_STEPS
    latency_steps: int = config.BACKTEST_LATENCY_STEPS
    covariates: list[str] = field(default_factory=lambda: list(config.TSFM_COVARIATES[config.TARGET_SOURCE]))
    traditional_strategy: str = config.TRADITIONAL_STRATEGY
    target_source: str = config.TARGET_SOURCE
    raft: RaftSettings | None = None

    @property
    def run_name(self) -> str:
        """e.g. wind_ctx288_h156_lat0_cov7-1a2b3c: covariate count + short hash of the (sorted) covariate names."""
        if self.covariates:
            digest = hashlib.md5(",".join(sorted(self.covariates)).encode()).hexdigest()[:6]
            covariates = f"cov{len(self.covariates)}-{digest}"
        else:
            covariates = "nocov"
        name = (f"{self.target_source}_ctx{self.context_steps}_h{self.horizon_steps}"
                f"_lat{self.latency_steps}_{covariates}")
        if self.raft is not None:
            identity = json.dumps({
                "raft": asdict(self.raft),
                "archive": self.raft.archive_digest,
                "model_weather": "dataset",
                "train": [self.train_start, self.train_end],
                "test": [self.test_start, self.test_end],
                "issue_time": self.issue_time,
                "covariates": self.covariates,
            }, sort_keys=True)
            name += f"_raft-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
        return name


@dataclass
class BacktestResult:
    """predictions: one row per (issue_time, timestamp, model) with lead_step, q10, q50, q90 and actual."""
    settings: BacktestSettings
    predictions: pd.DataFrame
    skipped: list[pd.Timestamp]
    timings: dict[str, dict[str, float]]
    model_configs: dict[str, dict] = field(default_factory=dict)


def load_forecaster(model_key: str, settings: BacktestSettings) -> Forecaster:
    """Create the model registered as `model_key` in config.MODELS_REGISTRY."""
    if model_key == "persistence":
        return Persistence(min_context=settings.horizon_steps + settings.latency_steps)
    if model_key in config.TRADITIONAL_MODELS:
        from src.models_traditional import load_traditional

        return load_traditional(
            model_key,
            strategy=settings.traditional_strategy,
            horizon_steps=settings.horizon_steps,
            latency_steps=settings.latency_steps,
            issue_time=settings.issue_time,
        )
    if config.MODELS_REGISTRY.get(model_key, {}).get("type") == "tsfm":
        from src.models_tsfm import load_tsfm

        return load_tsfm(model_key)
    raise ValueError(f"Model '{model_key}' is not available in the backtest yet")


def issue_times(settings: BacktestSettings) -> pd.DatetimeIndex:
    """One issue time per day at settings.issue_time UTC, within [test_start, test_end)."""
    start, end = pd.Timestamp(settings.test_start), pd.Timestamp(settings.test_end)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    issues = days + pd.Timedelta(f"{settings.issue_time}:00")
    return issues[(issues >= start) & (issues < end)]


def context_steps_for(forecaster: Forecaster, settings: BacktestSettings) -> int:
    """Configured context, raised to the model's min_context; warns if the model will cut it."""
    steps = max(settings.context_steps, forecaster.min_context)
    if forecaster.max_context is not None and steps > forecaster.max_context:
        logger.warning("%s: context of %d steps exceeds its max context %d, the model keeps the most recent steps",
                       forecaster.name, steps, forecaster.max_context)
    return steps


def _complete(data: pd.Series | pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True if `data` has a value at every step from start to end (inclusive)."""
    window = data.loc[start:end]
    return len(window) == int((end - start) / STEP) + 1 and not window.isna().to_numpy().any()


def _context_end(issue: pd.Timestamp, settings: BacktestSettings) -> pd.Timestamp:
    return issue - (settings.latency_steps + 1) * STEP


def build_request(dataset: Dataset, issue: pd.Timestamp, context_steps: int, settings: BacktestSettings,
                  covariates: list[str] | None) -> ForecastRequest:
    """Request for one issue time: `context_steps` of target ending latency_steps before the issue time."""
    context_end = _context_end(issue, settings)
    context = dataset.target.loc[context_end - (context_steps - 1) * STEP:context_end]
    forecast_index = pd.date_range(issue, periods=settings.horizon_steps, freq=STEP, name="timestamp")
    if not covariates:
        return ForecastRequest(issue, forecast_index, context)
    future_index = pd.date_range(context_end + STEP, forecast_index[-1], freq=STEP)
    return ForecastRequest(issue, forecast_index, context,
                           past_covariates=dataset.weather.loc[context.index, covariates],
                           future_covariates=dataset.weather.loc[future_index, covariates])


def run_backtest(dataset: Dataset, model_keys: list[str], settings: BacktestSettings | None = None) -> BacktestResult:
    """Run requested models and post-process any requested ensembles.

    Ensemble choices automatically run their configured traditional
    dependencies with ``settings.traditional_strategy``. Dependency forecasts
    are used internally and appear in the returned table only when the
    corresponding individual model was also requested.
    """
    settings = settings or BacktestSettings()
    missing = set(settings.covariates) - set(dataset.weather.columns)
    if missing:
        raise ValueError(f"covariates not in the dataset: {sorted(missing)}")

    ensemble_keys = [key for key in model_keys if key in config.ENSEMBLE_MODELS]
    requested_base_keys = [key for key in model_keys if key not in config.ENSEMBLE_MODELS]
    dependency_keys = config.ENSEMBLE_BASE_MODELS if ensemble_keys else []
    run_keys = list(dict.fromkeys([*requested_base_keys, *dependency_keys]))
    tsfm_keys = [key for key in requested_base_keys if key in config.TSFM_MODELS]

    raft = None
    raft_fit_seconds = raft_seconds = 0.0
    if settings.raft is not None:
        if not tsfm_keys:
            raise ValueError("Select at least one TSFM to run RAFT")
        if (settings.context_steps != CONTEXT_HOURS * STEPS_PER_HOUR
                or settings.horizon_steps != HORIZON_HOURS * STEPS_PER_HOUR
                or settings.latency_steps != 0 or not settings.issue_time.endswith(":00")):
            raise ValueError("RAFT requires a 72 h context, 39 h horizon, zero latency and a whole-hour issue time")
        if pd.Timestamp(settings.train_end) > pd.Timestamp(settings.test_start):
            raise ValueError("RAFT training must end before the first evaluation origin")
        start = time.perf_counter()
        raft = RaftRetriever(settings.raft, dataset.target, settings.target_source,
                             int(settings.issue_time.split(":")[0]), pd.Timestamp(settings.train_start),
                             pd.Timestamp(settings.train_end))
        raft_fit_seconds = time.perf_counter() - start

    forecasters = [load_forecaster(key, settings) for key in run_keys]
    forecaster_by_key = dict(zip(run_keys, forecasters))
    context_steps = {f.name: context_steps_for(f, settings) for f in forecasters}
    covariate_contexts = [context_steps[f.name] for f in forecasters if f.uses_covariates and settings.covariates]

    # Same issue times for every model: skip a day if any model would get incomplete data
    valid, skipped = [], []
    raft_results = []
    for issue in issue_times(settings):
        context_end = _context_end(issue, settings)
        last = issue + (settings.horizon_steps - 1) * STEP
        ok = _complete(dataset.target, context_end - (max(context_steps.values()) - 1) * STEP, last)
        if ok and covariate_contexts:
            ok = _complete(dataset.weather[settings.covariates],
                           context_end - (max(covariate_contexts) - 1) * STEP,
                           last)
        if ok and raft is not None:
            start = time.perf_counter()
            retrieval = raft.predict(issue)
            raft_seconds += time.perf_counter() - start
            ok = retrieval is not None
            if ok:
                raft_results.append(retrieval)
        (valid if ok else skipped).append(issue)
    if skipped:
        logger.warning("Skipped %d issue times with incomplete data: %s",
                       len(skipped), [t.strftime("%Y-%m-%d") for t in skipped])
    if not valid:
        raise ValueError("no issue time with complete data in the test period")
    logger.info("Backtest %s: %d issue times %s..%s, models %s", settings.run_name, len(valid),
                valid[0], valid[-1], [f.name for f in forecasters])

    train = slice(pd.Timestamp(settings.train_start), pd.Timestamp(settings.train_end) - STEP)
    tables, all_timings = [], {}
    for forecaster in forecasters:
        covariates = settings.covariates if forecaster.uses_covariates else None
        start = time.perf_counter()
        forecaster.fit(dataset.target.loc[train], dataset.weather.loc[train, covariates] if covariates else None)
        fit_seconds = time.perf_counter() - start

        requests = [build_request(dataset, issue, context_steps[forecaster.name], settings, covariates)
                    for issue in valid]
        start = time.perf_counter()
        results = forecaster.predict(requests)
        all_timings[forecaster.name] = {
            "load_s": getattr(forecaster, "load_seconds", 0.0),
            "fit_s": fit_seconds,
            "predict_s": time.perf_counter() - start,
        }
        tables += [result.to_long() for result in results]
        logger.info("%s done: %d forecasts, timings %s", forecaster.name, len(results),
                    {k: round(v, 1) for k, v in all_timings[forecaster.name].items()})

    base_predictions = pd.concat(tables, ignore_index=True)
    base_predictions.insert(
        2,
        "lead_step",
        ((base_predictions["timestamp"] - base_predictions["issue_time"]) / STEP).astype(int),
    )
    base_predictions["actual"] = dataset.target.reindex(base_predictions["timestamp"]).to_numpy()

    requested_names = [forecaster_by_key[key].name for key in requested_base_keys]
    output_tables = [base_predictions[base_predictions["model"].isin(requested_names)]] if requested_names else []
    timings = {name: all_timings[name] for name in requested_names}
    model_configs = {
        name: getattr(forecaster_by_key[key], "configuration", {})
        for key, name in zip(requested_base_keys, requested_names)
    }

    if raft is not None:
        from src.models_tsfm import blend_tsfm_with_raft

        retrieval = pd.concat([result.to_long() for result in raft_results], ignore_index=True)
        retrieval.insert(2, "lead_step", ((retrieval["timestamp"] - retrieval["issue_time"]) / STEP).astype(int))
        retrieval["actual"] = dataset.target.reindex(retrieval["timestamp"]).to_numpy()
        base_names = [forecaster_by_key[key].name for key in tsfm_keys]
        start = time.perf_counter()
        blended = blend_tsfm_with_raft(base_predictions[base_predictions["model"].isin(base_names)],
                                       retrieval, settings.raft.gamma)
        blend_seconds = time.perf_counter() - start
        output_tables.extend([retrieval, blended])
        retrieval_name = raft_results[0].model
        timings[retrieval_name] = {"load_s": 0.0, "fit_s": raft_fit_seconds, "predict_s": raft_seconds}
        model_configs[retrieval_name] = {"parameters": asdict(settings.raft),
                                         "weather_archive_sha256": settings.raft.archive_digest,
                                         "model_weather": "dataset"}
        for name in base_names:
            blend_name = f"{name} + {retrieval_name}"
            timings[blend_name] = {"load_s": 0.0, "fit_s": 0.0, "predict_s": blend_seconds / len(base_names)}
            model_configs[blend_name] = {"base_model": name, "gamma": settings.raft.gamma}

    ensemble_base_names = (
        [forecaster_by_key[key].name for key in config.ENSEMBLE_BASE_MODELS]
        if ensemble_keys
        else []
    )
    for ensemble_key in ensemble_keys:
        start = time.perf_counter()
        ensemble = build_ensemble(
            ensemble_key=ensemble_key,
            predictions=base_predictions,
            models=ensemble_base_names,
            strategy=settings.traditional_strategy,
            latency_steps=settings.latency_steps,
        )
        elapsed = time.perf_counter() - start
        ensemble_name = ensemble.predictions["model"].iloc[0]
        output_tables.append(ensemble.predictions)
        timings[ensemble_name] = {"load_s": 0.0, "fit_s": 0.0, "predict_s": elapsed}
        model_configs[ensemble_name] = {
            "ensemble_key": ensemble_key,
            "base_models": ensemble_base_names,
            "strategy": settings.traditional_strategy,
            "parameters": ENSEMBLE_PARAMETERS[ensemble_key],
        }
        logger.info("%s done: %d forecasts in %.2fs", ensemble_name,
                    ensemble.predictions["issue_time"].nunique(), elapsed)

    predictions = pd.concat(output_tables, ignore_index=True)
    return BacktestResult(
        settings=settings,
        predictions=predictions,
        skipped=skipped,
        timings=timings,
        model_configs=model_configs,
    )


def save_backtest(result: BacktestResult, directory: Path = config.BACKTEST_RESULTS_DIR) -> Path:
    """Write <run_name>.parquet (predictions) and <run_name>.json (settings, skipped days, timings).

    If the run already exists, the rows of the models in `result` replace theirs and the other models are kept,
    so re-running one model with the same settings does not drop the others.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.settings.run_name}.parquet"
    predictions = result.predictions
    timings = dict(result.timings)
    model_configs = dict(result.model_configs)
    if path.exists():
        old, old_meta = load_backtest(result.settings.run_name, directory)
        kept = old[~old["model"].isin(predictions["model"].unique())]
        if len(kept):
            logger.info("Keeping models %s from the saved run", sorted(kept["model"].unique()))
        predictions = pd.concat([kept, predictions], ignore_index=True)
        timings = {**{m: t for m, t in old_meta["timings"].items() if m in set(kept["model"])}, **timings}
        old_configs = old_meta.get("model_configs", {})
        model_configs = {
            **{m: value for m, value in old_configs.items() if m in set(kept["model"])},
            **model_configs,
        }
    predictions.to_parquet(path)
    meta = {
        "target": result.settings.target_source,
        "settings": asdict(result.settings),
        "models": sorted(predictions["model"].unique().tolist()),
        "skipped": [t.isoformat() for t in result.skipped],
        "timings": timings,
        "model_configs": model_configs,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Backtest saved to %s", path)
    return path


def load_backtest(run_name: str, directory: Path = config.BACKTEST_RESULTS_DIR) -> tuple[pd.DataFrame, dict]:
    """Read back the predictions table and the metadata of a saved run."""
    path = directory / f"{run_name}.parquet"
    return pd.read_parquet(path), json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def saved_runs(directory: Path = config.BACKTEST_RESULTS_DIR) -> dict[str, dict]:
    """Metadata of every saved run, newest first: run_name -> json metadata."""
    paths = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths if p.with_suffix(".parquet").exists()}
