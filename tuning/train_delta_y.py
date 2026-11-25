"""Train an output-side delta-Adapter (Ada-Y) on cached forecasts of a frozen Chronos-2 (zero-shot or LoRA).

1. Cache: the shared rolling backtest forecasts every day of 2024 at the issue time with the base model
   (same context, horizon and covariates as the later 2025 evaluation). Saved under
   results/finetuning/delta_y/base_forecasts/ and reused by every adapter with the same base and settings.
2. Train: February–November 2024 origins train the adapter, December 2024 origins select the best epoch.
   The base model never runs during training; only the adapter's few thousand weights change.
3. Save adapter.safetensors and run.json, reload them and check the validation scores match.
"""
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from src.delta_adapter import (ADAPTER_FILE, DeltaYAdapter, build_inputs, load_adapter, pinball_loss,
                               save_adapter, scores)
from src.tsfm_runs import finish_training, start_training, write_record

CACHE_START = "2024-02-01T00:00:00Z"       # first origin: 30 days of context fit after the data start (2023-12-28)
CACHE_END = "2024-12-31T00:00:00Z"         # exclusive: the last origin's horizon ends inside 2024
VALIDATION_START = pd.Timestamp("2024-12-01T00:00:00Z")
CACHE_DIR = config.BACKTEST_RESULTS_DIR / "finetuning" / "delta_y" / "base_forecasts"


def base_identity(base: str) -> tuple[str, str | None]:
    """Checkpoint and revision of the frozen base model (a LoRA base records its own)."""
    from tuning.train_chronos2 import BASE_CHECKPOINT, BASE_REVISION

    if base == "chronos2":
        return BASE_CHECKPOINT, BASE_REVISION
    if base.startswith("lora:chronos2:"):
        from src.tsfm_runs import available_adapters

        record = available_adapters().get(base)
        if record is None:
            raise ValueError(f"Adapter run is missing or unverified: {base}")
        return record["base_checkpoint"], record["base_revision"]
    raise ValueError(f"Unsupported base model {base}; use chronos2 or lora:chronos2:<run-id>")


def cached_forecasts(args, covariates: list[str], dataset) -> tuple[pd.DataFrame, dict]:
    """Base model forecasts for the 2024 origins, from the backtest engine.

    Cached in results/finetuning/delta_y/base_forecasts/<settings hash>.parquet, so several adapters on the same
    base and settings (e.g. MLP and CNN) share one backtest.
    """
    from src.backtest import BacktestSettings, run_backtest

    key = {"base": args.base, "start": CACHE_START, "end_exclusive": CACHE_END, "context_steps": args.context_steps,
           "horizon_steps": config.BACKTEST_HORIZON_STEPS, "covariates": covariates,
           "issue_time": config.BACKTEST_ISSUE_TIME}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"{digest}.parquet"
    meta = cache.with_suffix(".json")
    key["file"] = str(cache)
    if cache.is_file() and meta.is_file():
        print(f"Reusing cached base forecasts {cache}", flush=True)
        return pd.read_parquet(cache), key
    settings = BacktestSettings(test_start=CACHE_START, test_end=CACHE_END, context_steps=args.context_steps,
                                horizon_steps=config.BACKTEST_HORIZON_STEPS, latency_steps=0, covariates=covariates,
                                target_source="wind", device=args.device)
    predictions = run_backtest(dataset, [args.base], settings).predictions
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(cache)
    meta.write_text(json.dumps(key, indent=2), encoding="utf-8")     # written last: marks a complete cache
    return predictions, key


def train(args: argparse.Namespace) -> Path:
    run_dir = config.BACKTEST_RESULTS_DIR / "finetuning" / "delta_y" / args.run_id
    covariates = list(config.TSFM_COVARIATES["wind"]) if args.covariates == "config" else []
    checkpoint, revision = base_identity(args.base)
    record = start_training(run_dir, "delta_y", args, checkpoint, revision, ADAPTER_FILE, covariates=covariates)
    try:
        return _train(args, run_dir, record, covariates)
    except BaseException as exc:
        finish_training(run_dir, record, "failed", f"{type(exc).__name__}: {exc}")
        raise


def _train(args: argparse.Namespace, run_dir: Path, record: dict, covariates: list[str]) -> Path:
    from src.dataset import load_dataset

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("wind")

    predictions, cache_key = cached_forecasts(args, covariates, dataset)
    base_name = predictions["model"].iloc[0]
    inputs = build_inputs(predictions, dataset.target, dataset.weather, covariates, args.context_steps)
    is_validation = np.asarray(inputs.issue_times >= VALIDATION_START)
    training, validation = inputs.subset(~is_validation), inputs.subset(is_validation)
    if not len(training.issue_times) or not len(validation.issue_times):
        raise ValueError("Need training (Feb–Nov 2024) and validation (December 2024) origins")
    print(f"{base_name}: {len(training.issue_times)} training and {len(validation.issue_times)} validation origins, "
          f"{inputs.quantiles.shape[1]} steps, covariates {covariates or 'none'}, {device}", flush=True)

    adapter = DeltaYAdapter(len(covariates), inputs.quantiles.shape[1], arch=args.arch, hidden=args.hidden,
                            delta=args.delta, width_delta=args.width_delta).to(device)
    adapter.fit_normalization(training)
    baseline = scores(validation.quantiles, validation)          # the frozen base model alone
    print(f"Base model on validation: {baseline}", flush=True)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.learning_rate)
    best, best_state, history, waited = None, None, [], 0
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        loss = pinball_loss(adapter(training), training)
        loss.backward()
        optimizer.step()

        adapter.eval()
        with torch.no_grad():
            current = scores(adapter(validation), validation)
        if best is None or current["loss"] < best["loss"]:
            best = {"step": epoch, "training_loss": loss.item(), **current}
            best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
            waited = 0
        else:
            waited += 1
        if epoch % args.log_every == 0 or epoch == 1:
            history.append({"step": epoch, "training_loss": loss.item(), **current})
            print(f"Epoch {epoch}: training loss {loss.item():.4f}, validation {current}", flush=True)
        if waited >= args.patience:
            print(f"Early stopping at epoch {epoch}, best epoch {best['step']}", flush=True)
            break

    adapter.load_state_dict(best_state)
    save_adapter(adapter, run_dir)
    record.update({
        "target": "wind_MW", "base_model_key": args.base, "base_model_name": base_name,
        "context_steps": args.context_steps, "horizon_steps": int(inputs.quantiles.shape[1]),
        "latency_steps": 0, "issue_time_utc": config.BACKTEST_ISSUE_TIME, "covariates": covariates,
        "adapter_config": adapter.config, "train_start": CACHE_START,
        "validation_start": VALIDATION_START.isoformat(), "validation_end_exclusive": CACHE_END,
        "training_origins": len(training.issue_times),
        "validation_origins": [t.isoformat() for t in validation.issue_times],
        "base_forecasts": cache_key,
        "steps": best["step"], "learning_rate": args.learning_rate, "seed": args.seed, "device": device,
        "zero_shot": baseline, "best": best, "validation_history": history,
    })
    write_record(run_dir / "run.json", record)

    reloaded, _ = load_adapter(run_dir, device)
    with torch.no_grad():
        check = scores(reloaded(validation), validation)
    np.testing.assert_allclose([check[k] for k in ("loss", "RMSE", "Winkler80")],
                               [best[k] for k in ("loss", "RMSE", "Winkler80")], rtol=1e-5, atol=1e-5)
    record.update(reloaded_validation=check, elapsed_seconds=time.perf_counter() - started)
    finish_training(run_dir, record, "completed")
    print(f"Base {baseline}\nWith delta-Y {best}", flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base", default="chronos2", help="chronos2 or lora:chronos2:<run-id>")
    parser.add_argument("--context-steps", type=int, default=2880)
    parser.add_argument("--covariates", choices=("none", "config"), default="config")
    parser.add_argument("--arch", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--delta", type=float, default=0.1, help="max median shift, in context standard deviations")
    parser.add_argument("--width-delta", type=float, default=0.5, help="band widths scaled within exp(±width-delta)")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must start with a letter or digit and contain only letters, digits, underscores or hyphens")
    if min(args.context_steps, args.epochs, args.patience, args.hidden, args.log_every) < 1:
        parser.error("context steps, epochs, patience, hidden and log-every must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.delta, args.width_delta, args.learning_rate)):
        parser.error("delta, width-delta and learning rate must be positive and finite")
    print(f"Saved and verified {train(args)}", flush=True)
