"""Evaluate one TSFM with the project's shared rolling backtest."""
import argparse
from pathlib import Path

import pandas as pd

import config
from src.backtest import BacktestSettings, run_recorded_backtest
from src.dataset import Dataset
from src.tsfm_runs import read_record, wind_path


def evaluate(model_key: str, run_dir: Path | None, start: str, end: str,
             context: int, horizon: int, device: str = "cpu", covariates: list[str] | None = None,
             delta_run_id: str | None = None) -> Path:
    """Rolling backtest of the zero-shot model (and the adapter); `covariates` are weather columns, [] = wind-only.

    `delta_run_id` adds a δY output adapter (results/finetuning/delta_y/<id>) on top of its recorded base model.
    """
    covariates = list(covariates or [])
    if delta_run_id:
        from src.tsfm_runs import available_delta_adapters

        record = available_delta_adapters().get(f"delta:{delta_run_id}")
        if record is None:
            raise ValueError(f"delta-Y adapter run is missing or unverified: {delta_run_id}")
        base = record["base_model_key"]
        if base != model_key and not base.startswith(f"lora:{model_key}:"):
            raise ValueError(f"delta-Y adapter {delta_run_id} corrects {base}, not {model_key}")
    if model_key not in config.TSFM_MODELS:
        raise ValueError(f"Unsupported TSFM: {model_key}")
    if context < 1 or horizon < 1:
        raise ValueError("Context and horizon must be positive")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Provide timezone-aware start and end, with start before end")
    if run_dir:
        record = read_record(run_dir)
        if model_key not in ("chronos2", "timestral_209m") or record["base_checkpoint"] != config.MODELS_REGISTRY[model_key]["checkpoint"]:
            raise ValueError("Adapter does not match the requested model")
        expected = config.BACKTEST_RESULTS_DIR / "finetuning" / model_key / run_dir.name
        if run_dir.resolve() != expected.resolve():
            raise ValueError("Adapter must be a saved run for the selected model")
    # A device override is intentionally passed to the shared model loader via settings.
    settings = BacktestSettings(test_start=start.tz_convert("UTC").isoformat(),
                                test_end=end.tz_convert("UTC").isoformat(),
                                context_steps=context, horizon_steps=horizon,
                                latency_steps=0, covariates=covariates, target_source="wind", device=device,
                                pca_components=None)  # the adapters are trained on the raw covariates
    if covariates:
        from src.dataset import load_dataset
        dataset = load_dataset("wind")  # same target and weather as the dashboard and main.py
    else:
        regional = pd.read_parquet(wind_path(), columns=config.ELIA_SOURCES["wind"]["regions"])
        regional.index = pd.to_datetime(regional.index, utc=True)
        target = regional.sum(axis=1, min_count=1).sort_index().rename("wind_MW")
        dataset = Dataset(target=target, weather=pd.DataFrame(index=target.index))
    models = [model_key]
    if run_dir:
        models.append(f"lora:{model_key}:{run_dir.name}")
    if delta_run_id:
        models.append(f"delta:{delta_run_id}")
    return run_recorded_backtest(dataset, models, settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=config.TSFM_MODELS)
    parser.add_argument("--adapter-run-id")
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-01-01T00:00:00Z", help="Exclusive UTC issue-time boundary")
    parser.add_argument("--context-steps", type=int, default=288)
    parser.add_argument("--horizon-steps", type=int, default=156)
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="cpu")
    parser.add_argument("--covariates", choices=("none", "config"), default="none",
                        help="config: pass config.TSFM_COVARIATES['wind'] to the models (needed for covariate adapters)")
    parser.add_argument("--delta-run-id", help="also evaluate this delta-Y output adapter (tuning/train_delta_y.py)")
    args = parser.parse_args()
    for run_id in (args.adapter_run_id, args.delta_run_id):
        if run_id and (Path(run_id).name != run_id or run_id in (".", "..")):
            parser.error("run IDs must be directory names")
    path = (config.BACKTEST_RESULTS_DIR / "finetuning" / args.model / args.adapter_run_id
            if args.adapter_run_id else None)
    covariates = list(config.TSFM_COVARIATES["wind"]) if args.covariates == "config" else []
    print(f"Saved evaluation: {evaluate(args.model, path, args.start, args.end, args.context_steps, args.horizon_steps, args.device, covariates, args.delta_run_id)}")
