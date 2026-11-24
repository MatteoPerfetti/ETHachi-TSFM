"""Context-length sweep: the shared rolling backtest (src/backtest.py) run once per context length, one table.

Every context length is an ordinary backtest run (results/<run_name>.parquet, visible in the dashboard); this module
only orchestrates the runs and compares them. Persistence does not depend on the context, so it runs once, in the
first run, as reference. All scores are computed on the issue days that every run kept (common days), so a longer
context that skips a few more days because of data gaps is not compared on a different set of days.
"""
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import config
from logger import logger
from src.backtest import BacktestSettings, run_backtest, save_backtest
from src.dataset import Dataset
from src.metrics import summary

PERSISTENCE_KEY = "persistence"
NO_CONTEXT = "-"  # context label of the persistence rows


def model_name(model_key: str) -> str:
    """Name a model writes in the results table, without loading its weights."""
    if model_key == PERSISTENCE_KEY:
        from src.baselines import Persistence
        return Persistence.name
    from src.models_tsfm import TSFM_ADAPTERS
    if model_key not in TSFM_ADAPTERS:
        raise ValueError(f"context sweep supports persistence and the TSFMs {list(TSFM_ADAPTERS)}, got {model_key!r}")
    return TSFM_ADAPTERS[model_key].name


def sweep_settings(days: int, issue_every_days: int = config.BACKTEST_ISSUE_EVERY_DAYS,
                   device: str | None = None, **overrides) -> BacktestSettings:
    """Backtest settings of one context length; everything else comes from config.py unless overridden
    (any other BacktestSettings field, e.g. test_start / test_end)."""
    if days < 1:
        raise ValueError(f"context days must be >= 1, got {days}")
    return BacktestSettings(context_steps=days * config.STEPS_PER_DAY, issue_every_days=issue_every_days,
                            device=device, **overrides)


def _saved_models(path: Path) -> set[str]:
    json_path = path.with_suffix(".json")
    if not (path.exists() and json_path.exists()):
        return set()
    return set(json.loads(json_path.read_text(encoding="utf-8")).get("models", []))


def run_sweep(days_list: list[int], model_keys: list[str], *,
              issue_every_days: int = config.BACKTEST_ISSUE_EVERY_DAYS, device: str | None = None,
              reuse: bool = False, dataset_loader: Callable[[], Dataset] | None = None,
              directory: Path = config.BACKTEST_RESULTS_DIR, overrides: dict | None = None) -> dict[int, Path]:
    """Run the backtest of `model_keys` for every context length in `days_list` (persistence once, in the first).

    reuse=True skips the models already saved in a run with the same settings (only the missing ones run).
    overrides: other BacktestSettings fields for every run (see sweep_settings).
    Returns context days -> path of the saved run. The dataset is loaded once, only if something has to run.
    """
    if not days_list:
        raise ValueError("days_list is empty")
    if dataset_loader is None:
        from src.dataset import load_dataset
        dataset_loader = load_dataset
    dataset = None
    paths = {}
    for i, days in enumerate(sorted(set(days_list))):
        settings = sweep_settings(days, issue_every_days, device, **(overrides or {}))
        path = directory / f"{settings.run_name}.parquet"
        keys = list(dict.fromkeys(([PERSISTENCE_KEY] if i == 0 else []) + list(model_keys)))
        saved = _saved_models(path) if reuse else set()
        missing = [key for key in keys if model_name(key) not in saved]
        if missing:
            logger.info("Context %d days (%s): running %s", days, settings.run_name, missing)
            dataset = dataset if dataset is not None else dataset_loader()
            save_backtest(run_backtest(dataset, missing, settings), directory)
        else:
            logger.info("Context %d days (%s): reusing saved models %s", days, settings.run_name, sorted(saved))
        paths[days] = path
    return paths


def load_sweep(paths: dict[int, Path], model_keys: list[str]) -> tuple[pd.DataFrame, dict[tuple[str, str], dict]]:
    """Predictions of the sweep models in every run, with a `context` label column ("3d", ..., "-" for
    persistence, taken from the first run that has it), and the timings per (context, model) plus the number
    of forecasts the timing refers to."""
    names = [model_name(key) for key in model_keys]
    persistence = model_name(PERSISTENCE_KEY)
    tables, timings = [], {}
    for days, path in sorted(paths.items()):
        predictions = pd.read_parquet(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        wanted = names + ([persistence] if not any(t[1] == persistence for t in timings) else [])
        for name in wanted:
            rows = predictions[predictions["model"] == name]
            if rows.empty:
                continue
            context = NO_CONTEXT if name == persistence else f"{days}d"
            tables.append(rows.assign(context=context, context_days=np.nan if name == persistence else days))
            timings[(context, name)] = {**meta.get("timings", {}).get(name, {}),
                                        "forecasts": int(rows["issue_time"].nunique())}
    if not tables:
        raise ValueError("no sweep model found in the saved runs")
    return pd.concat(tables, ignore_index=True), timings


def common_issue_times(predictions: pd.DataFrame) -> pd.DatetimeIndex:
    """Issue times present for every (context, model) of the sweep."""
    groups = predictions.groupby(["context", "model"])["issue_time"].unique()
    common = set.intersection(*(set(times) for times in groups))
    return pd.DatetimeIndex(sorted(common))


def coverage(predictions: pd.DataFrame) -> float:
    """Share of actual values inside [q10, q90] (nominal 80%); NaN for a model without interval."""
    rows = predictions[predictions["actual"].notna()]
    if rows[["q10", "q90"]].isna().any().any():
        return float("nan")
    return float(((rows["actual"] >= rows["q10"]) & (rows["actual"] <= rows["q90"])).mean())


def sweep_table(predictions: pd.DataFrame, timings: dict[tuple[str, str], dict]) -> pd.DataFrame:
    """One row per (model, context) on the common issue days: RMSE and Winkler 80% on all 156 steps and on D+1,
    RMSE relative to persistence, q10-q90 coverage, inference time."""
    common = common_issue_times(predictions)
    if common.empty:
        raise ValueError("the sweep runs have no issue time in common")
    predictions = predictions[predictions["issue_time"].isin(common)]
    persistence = model_name(PERSISTENCE_KEY)
    rows = []
    for (context, model), group in predictions.groupby(["context", "model"], sort=False):
        scores_all = summary(group, "all").loc[model]
        scores_d1 = summary(group, "d+1").loc[model]
        timing = timings.get((context, model), {})
        predict_s = timing.get("predict_s", np.nan)
        rows.append({
            "model": model,
            "context": context,
            "context_days": group["context_days"].iloc[0],
            "RMSE": scores_all["RMSE"],
            "Winkler80": scores_all["Winkler80"],
            "RMSE_d+1": scores_d1["RMSE"],
            "Winkler80_d+1": scores_d1["Winkler80"],
            "coverage80": coverage(group),
            "days": int(group["issue_time"].nunique()),
            "predict_s": predict_s,
            "s_per_forecast": predict_s / timing["forecasts"] if timing.get("forecasts") else np.nan,
        })
    table = pd.DataFrame(rows)
    if persistence in set(table["model"]):
        reference = table.loc[table["model"] == persistence].iloc[0]
        table["RMSE_vs_persistence"] = table["RMSE"] / reference["RMSE"] - 1
        table["RMSE_d+1_vs_persistence"] = table["RMSE_d+1"] / reference["RMSE_d+1"] - 1
    return (table.sort_values(["model", "context_days"], na_position="first")
            .reset_index(drop=True))


def pivot(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Model x context days of one metric (persistence left out: it has no context)."""
    tsfm = table[table["context"] != NO_CONTEXT]
    out = tsfm.pivot(index="model", columns="context_days", values=metric)
    out.columns = [f"{int(days)}d" for days in out.columns]
    return out


def format_report(table: pd.DataFrame, dropped: int) -> str:
    """Plain-text report: full table plus the RMSE and Winkler pivots."""
    full = table.drop(columns=["context_days"]).copy()
    for column in ("RMSE_vs_persistence", "RMSE_d+1_vs_persistence", "coverage80"):
        if column in full:
            full[column] = full[column].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}" if "vs" in column
                                            else f"{v:.1%}")
    lines = [
        "=" * 100,
        f" Context sweep: {table['days'].max()} common issue days"
        + (f" ({dropped} issue days dropped because missing in at least one run)" if dropped else ""),
        " MW, lower is better. d+1 = next day 00:00-23:45 only. coverage80 = share of actuals in q10-q90.",
        "=" * 100,
        full.to_string(index=False, float_format=lambda v: f"{v:,.1f}"),
        "",
        " RMSE (all 156 steps), model x context:",
        pivot(table, "RMSE").to_string(float_format=lambda v: f"{v:,.1f}"),
        "",
        " RMSE D+1, model x context:",
        pivot(table, "RMSE_d+1").to_string(float_format=lambda v: f"{v:,.1f}"),
        "",
        " Winkler 80% (all 156 steps), model x context:",
        pivot(table, "Winkler80").to_string(float_format=lambda v: f"{v:,.1f}"),
        "",
        " Inference seconds per forecast (CPU/GPU of this run), model x context:",
        pivot(table, "s_per_forecast").to_string(float_format=lambda v: f"{v:,.2f}"),
    ]
    return "\n".join(lines)


def report(paths: dict[int, Path], model_keys: list[str], out_dir: Path = config.CONTEXT_SWEEP_DIR,
           tag: str = "") -> tuple[pd.DataFrame, Path]:
    """Build the sweep table from the saved runs, save it as CSV (+ txt report) and print the report."""
    predictions, timings = load_sweep(paths, model_keys)
    all_issues = predictions["issue_time"].nunique()
    table = sweep_table(predictions, timings)
    dropped = all_issues - int(table["days"].max())
    text = format_report(table, dropped)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "sweep_" + "-".join(f"{d}d" for d in sorted(paths)) + tag
    csv_path = out_dir / f"{name}.csv"
    table.to_csv(csv_path, index=False)
    csv_path.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    logger.info("Context sweep table saved to %s", csv_path)
    return table, csv_path
