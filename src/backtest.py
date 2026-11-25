"""Rolling-window backtest: one forecast per day of the test period, for every model, saved as one long table.

This is the only place that cuts the data in time. For each issue time it builds the ForecastRequests
(context up to the data latency before the issue time, covariates, forecast timestamps); the models only see those.
An issue time is skipped for every model if any of them would get incomplete data, so all models are scored
on the same days. Metrics are computed afterwards from the saved table (src/metrics.py).
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import pandas as pd

import config
from logger import logger
from src.baselines import Persistence
from src.dataset import Dataset
from src.ensembles import DEFAULT_PARAMETERS as ENSEMBLE_PARAMETERS
from src.ensembles import build_ensemble
from src.forecast import STEP, Forecaster, ForecastRequest


@dataclass
class BacktestSettings:
    """Everything that defines a backtest run; defaults come from config.py."""
    test_start: str = config.TEST_START
    test_end: str = config.TEST_END
    train_start: str = config.TRAIN_START
    train_end: str = config.TRAIN_END
    issue_time: str = config.BACKTEST_ISSUE_TIME
    issue_every_days: int = config.BACKTEST_ISSUE_EVERY_DAYS
    horizon_steps: int = config.BACKTEST_HORIZON_STEPS
    context_steps: int = config.BACKTEST_CONTEXT_STEPS
    latency_steps: int = config.BACKTEST_LATENCY_STEPS
    covariates: list[str] = field(default_factory=lambda: list(config.TSFM_COVARIATES[config.TARGET_SOURCE]))
    traditional_strategy: str = config.TRADITIONAL_STRATEGY
    target_source: str = config.TARGET_SOURCE
    device: str | None = None

    @property
    def run_name(self) -> str:
        """e.g. wind_ctx288_h156_lat0_cov7-1a2b3c: covariate count + short hash of the (sorted) covariate names.

        A run on every N-th day only gets the suffix _everyNd (no suffix for daily runs, the default)."""
        if self.covariates:
            digest = hashlib.md5(",".join(sorted(self.covariates)).encode()).hexdigest()[:6]
            covariates = f"cov{len(self.covariates)}-{digest}"
        else:
            covariates = "nocov"
        every = f"_every{self.issue_every_days}d" if self.issue_every_days != 1 else ""
        return (f"{self.target_source}_ctx{self.context_steps}_h{self.horizon_steps}"
                f"_lat{self.latency_steps}_{covariates}{every}")


@dataclass
class BacktestResult:
    """predictions: one row per (issue_time, timestamp, model) with lead_step, q10, q50, q90 and actual."""
    settings: BacktestSettings
    predictions: pd.DataFrame
    skipped: list[pd.Timestamp]
    timings: dict[str, dict[str, float]]
    model_configs: dict[str, dict] = field(default_factory=dict)
    dataset_fingerprint: str | None = None


def load_forecaster(model_key: str, settings: BacktestSettings, base_revisions: dict | None = None) -> Forecaster:
    """Create the model registered as `model_key` in config.MODELS_REGISTRY."""
    if model_key.startswith("lora:"):
        from src.models_tsfm import TSFM_ADAPTERS
        from src.tsfm_runs import available_adapters, fingerprint

        records = available_adapters()
        if model_key not in records:
            raise ValueError(f"Adapter run is missing or unverified: {model_key}")
        _, family, run_id = model_key.split(":", 2)
        record = records[model_key]
        if settings.target_source != "wind" or record.get("target") != "wind_MW":
            raise ValueError(f"{model_key} was trained on wind_MW; select the wind target")
        trained_with = record.get("covariates") or []
        if trained_with and sorted(trained_with) != sorted(settings.covariates):
            raise ValueError(f"{model_key} was trained with the covariates {trained_with}; "
                             f"select exactly these for the backtest (got {list(settings.covariates)})")
        run_dir = config.BACKTEST_RESULTS_DIR / "finetuning" / family / run_id
        artifact = run_dir / record.get("adapter_path", "finetuned-ckpt" if record.get("backend") == "peft" else "adapter.safetensors")
        files = [artifact] if artifact.is_file() else sorted(p for p in artifact.rglob("*") if p.is_file())
        if not files:
            raise ValueError(f"Adapter files missing: {artifact}")
        if family == "chronos2":
            from tuning.train_chronos2 import load_lora_run
        else:
            from tuning.train_timestral import load_lora_run
        from src.models_tsfm import default_device
        device = settings.device or default_device()
        start = time.perf_counter()
        tuned = load_lora_run(run_dir, device)
        forecaster = TSFM_ADAPTERS[family](device=device, model=tuned, name=f"{family} LoRA ({run_id})")
        forecaster.load_seconds = time.perf_counter() - start
        forecaster.configuration = {"model_key": family, "run_id": run_id,
            "base_checkpoint": record["base_checkpoint"], "base_revision": record["base_revision"],
            "adapter_sha256": {str(p.relative_to(run_dir)): fingerprint(p) for p in files},
            "training": record}
        return forecaster
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
        from src.models_tsfm import load_tsfm, TSFM_ADAPTERS

        if base_revisions and model_key in base_revisions:
            from huggingface_hub import snapshot_download
            revision = base_revisions[model_key]
            options = {"allow_patterns": ["config.json", "model.safetensors"]} if model_key == "timestral_209m" else {}
            checkpoint = config.MODELS_REGISTRY[model_key]["checkpoint"]
            snapshot = snapshot_download(checkpoint, revision=revision, local_files_only=True, **options)
            forecaster = TSFM_ADAPTERS[model_key](device=settings.device, checkpoint=snapshot)
            forecaster.configuration = {"model_key": model_key, "base_checkpoint": checkpoint, "base_revision": revision}
            return forecaster

        forecaster = load_tsfm(model_key, device=settings.device)
        forecaster.configuration = {"model_key": model_key,
                                    "base_checkpoint": config.MODELS_REGISTRY[model_key]["checkpoint"],
                                    "base_revision": None}
        return forecaster
    raise ValueError(f"Model '{model_key}' is not available in the backtest yet")


def issue_times(settings: BacktestSettings) -> pd.DatetimeIndex:
    """One issue time every settings.issue_every_days days (from the first test day) at settings.issue_time UTC,
    within [test_start, test_end)."""
    if settings.issue_every_days < 1:
        raise ValueError(f"issue_every_days must be >= 1, got {settings.issue_every_days}")
    start, end = pd.Timestamp(settings.test_start), pd.Timestamp(settings.test_end)
    days = pd.date_range(start.normalize(), end.normalize(), freq=f"{settings.issue_every_days}D")
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

    from src.tsfm_runs import available_adapters
    adapter_records = available_adapters() if any(k.startswith("lora:") for k in run_keys) else {}
    base_revisions = {}
    for key in run_keys:
        if not key.startswith("lora:"):
            continue
        if key not in adapter_records:
            raise ValueError(f"Adapter run is missing or unverified: {key}")
        family = key.split(":", 2)[1]
        revision = adapter_records[key]["base_revision"]
        if family in base_revisions and base_revisions[family] != revision and family in run_keys:
            raise ValueError(f"Cannot pair one {family} zero-shot model with adapters from different base revisions")
        base_revisions[family] = revision
    forecasters = [load_forecaster(key, settings, base_revisions) if base_revisions else load_forecaster(key, settings) for key in run_keys]
    forecaster_by_key = dict(zip(run_keys, forecasters))
    context_steps = {f.name: context_steps_for(f, settings) for f in forecasters}
    covariate_contexts = [context_steps[f.name] for f in forecasters if f.uses_covariates and settings.covariates]

    # Same issue times for every model: skip a day if any model would get incomplete data
    valid, skipped = [], []
    for issue in issue_times(settings):
        context_end = _context_end(issue, settings)
        last = issue + (settings.horizon_steps - 1) * STEP
        ok = _complete(dataset.target, context_end - (max(context_steps.values()) - 1) * STEP, last)
        if ok and covariate_contexts:
            ok = _complete(dataset.weather[settings.covariates],
                           context_end - (max(covariate_contexts) - 1) * STEP, last)
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
        dataset_fingerprint=hashlib.sha256(
            pd.util.hash_pandas_object(dataset.target, index=True).values.tobytes()
            + (pd.util.hash_pandas_object(dataset.weather[settings.covariates], index=True).values.tobytes()
               if settings.covariates else b"")
        ).hexdigest(),
    )


def save_backtest(result: BacktestResult, directory: Path = config.BACKTEST_RESULTS_DIR, *, preserve_history: bool = False) -> Path:
    """Write <run_name>.parquet (predictions) and <run_name>.json (settings, skipped days, timings).

    If the run already exists, the rows of the models in `result` replace theirs and the other models are kept,
    so re-running one model with the same settings does not drop the others.
    """
    directory.mkdir(parents=True, exist_ok=True)
    name = result.settings.run_name
    if preserve_history:
        name += "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    path = directory / f"{name}.parquet"
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
    if preserve_history:
        from src.metrics import summary
        meta["created_at"] = datetime.now(timezone.utc).isoformat()
        meta["dataset_fingerprint"] = result.dataset_fingerprint
        meta["scores_all"] = json.loads(summary(predictions, "all").to_json(orient="index"))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Backtest saved to %s", path)
    return path


def run_recorded_backtest(dataset: Dataset, model_keys: list[str], settings: BacktestSettings,
                          directory: Path = config.BACKTEST_RESULTS_DIR) -> Path:
    """Run a separate evaluation and record completed, failed and interrupted attempts."""
    from src.tsfm_runs import write_record

    attempt_id = uuid4().hex
    attempt_path = directory / "backtest_attempts" / f"{attempt_id}.json"
    attempt = {"attempt_id": attempt_id, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
               "model_keys": model_keys, "settings": asdict(settings)}
    write_record(attempt_path, attempt)
    try:
        result = run_backtest(dataset, model_keys, settings)
        path = save_backtest(result, directory, preserve_history=True)
        attempt.update(status="completed", result=path.name, finished_at=datetime.now(timezone.utc).isoformat())
        write_record(attempt_path, attempt)
        try:
            from src.experiment_tracking import track_evaluation
            track_evaluation(path)
        except Exception as exc:
            logger.warning("Backtest saved locally, but MLflow tracking failed: %s", exc)
        return path
    except BaseException as exc:
        attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       finished_at=datetime.now(timezone.utc).isoformat())
        write_record(attempt_path, attempt)
        raise


def load_backtest(run_name: str, directory: Path = config.BACKTEST_RESULTS_DIR) -> tuple[pd.DataFrame, dict]:
    """Read back the predictions table and the metadata of a saved run."""
    path = directory / f"{run_name}.parquet"
    return pd.read_parquet(path), json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def saved_runs(directory: Path = config.BACKTEST_RESULTS_DIR) -> dict[str, dict]:
    """Metadata of every saved run, newest first: run_name -> json metadata."""
    paths = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths if p.with_suffix(".parquet").exists()}
