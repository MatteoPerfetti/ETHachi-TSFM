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
from src.forecast import STEP, Forecaster, ForecastRequest
from src.models_tsfm import TSFM_ADAPTERS, load_tsfm


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
    target_source: str = config.TARGET_SOURCE

    @property
    def run_name(self) -> str:
        """e.g. wind_ctx288_h156_lat0_cov7-1a2b3c: covariate count + short hash of the (sorted) covariate names."""
        if self.covariates:
            digest = hashlib.md5(",".join(sorted(self.covariates)).encode()).hexdigest()[:6]
            covariates = f"cov{len(self.covariates)}-{digest}"
        else:
            covariates = "nocov"
        return (f"{self.target_source}_ctx{self.context_steps}_h{self.horizon_steps}"
                f"_lat{self.latency_steps}_{covariates}")


@dataclass
class BacktestResult:
    """predictions: one row per (issue_time, timestamp, model) with lead_step, q10, q50, q90 and actual."""
    settings: BacktestSettings
    predictions: pd.DataFrame
    skipped: list[pd.Timestamp]
    timings: dict[str, dict[str, float]]


def load_forecaster(model_key: str, settings: BacktestSettings) -> Forecaster:
    """Create the model registered as `model_key` in config.MODELS_REGISTRY."""
    if model_key == "persistence":
        return Persistence(min_context=settings.horizon_steps + settings.latency_steps)
    if model_key in TSFM_ADAPTERS:
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
    """Fit every model on the training period, forecast every valid issue time, collect one long table."""
    settings = settings or BacktestSettings()
    missing = set(settings.covariates) - set(dataset.weather.columns)
    if missing:
        raise ValueError(f"covariates not in the dataset: {sorted(missing)}")

    forecasters = [load_forecaster(key, settings) for key in model_keys]
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
    tables, timings = [], {}
    for forecaster in forecasters:
        covariates = settings.covariates if forecaster.uses_covariates else None
        start = time.perf_counter()
        forecaster.fit(dataset.target.loc[train], dataset.weather.loc[train, covariates] if covariates else None)
        fit_seconds = time.perf_counter() - start

        requests = [build_request(dataset, issue, context_steps[forecaster.name], settings, covariates)
                    for issue in valid]
        start = time.perf_counter()
        results = forecaster.predict(requests)
        timings[forecaster.name] = {"load_s": getattr(forecaster, "load_seconds", 0.0), "fit_s": fit_seconds,
                                    "predict_s": time.perf_counter() - start}
        tables += [result.to_long() for result in results]
        logger.info("%s done: %d forecasts, timings %s", forecaster.name, len(results),
                    {k: round(v, 1) for k, v in timings[forecaster.name].items()})

    predictions = pd.concat(tables, ignore_index=True)
    predictions.insert(2, "lead_step", ((predictions["timestamp"] - predictions["issue_time"]) / STEP).astype(int))
    predictions["actual"] = dataset.target.reindex(predictions["timestamp"]).to_numpy()
    return BacktestResult(settings=settings, predictions=predictions, skipped=skipped, timings=timings)


def save_backtest(result: BacktestResult, directory: Path = config.BACKTEST_RESULTS_DIR) -> Path:
    """Write <run_name>.parquet (predictions) and <run_name>.json (settings, skipped days, timings).

    If the run already exists, the rows of the models in `result` replace theirs and the other models are kept,
    so re-running one model with the same settings does not drop the others.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.settings.run_name}.parquet"
    predictions, timings = result.predictions, dict(result.timings)
    if path.exists():
        old, old_meta = load_backtest(result.settings.run_name, directory)
        kept = old[~old["model"].isin(predictions["model"].unique())]
        if len(kept):
            logger.info("Keeping models %s from the saved run", sorted(kept["model"].unique()))
        predictions = pd.concat([kept, predictions], ignore_index=True)
        timings = {**{m: t for m, t in old_meta["timings"].items() if m in set(kept["model"])}, **timings}
    predictions.to_parquet(path)
    meta = {
        "target": result.settings.target_source,
        "settings": asdict(result.settings),
        "models": sorted(predictions["model"].unique().tolist()),
        "skipped": [t.isoformat() for t in result.skipped],
        "timings": timings,
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
