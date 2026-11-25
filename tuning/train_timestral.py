"""Train a Timestral-209M LoRA adapter, optionally with known-future weather covariates."""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from timestral import TimestralForecaster
from timestral.scaler import advancing_block_frames

import config
from src.forecast import STEP
from src.metrics import rmse, winkler
from src.tuning_data import VALIDATION_START, load_covariate_inputs, load_windows
from src.tsfm_runs import finish_training, start_training, write_record


BASE_CHECKPOINT = config.MODELS_REGISTRY["timestral_209m"]["checkpoint"]
BASE_REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
TARGET_MODULES = ("wQKV", "wO", "mlp.w12", "mlp.w3")
ATTENTION_MODULES = ("wQKV", "wO")


def load_lora_run(run_dir: Path, device: str = "cpu") -> TimestralForecaster:
    """Attach a saved PEFT adapter to the exact Timestral revision it used."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    forecaster = TimestralForecaster.from_pretrained(
        record["base_checkpoint"], revision=record["base_revision"], device=device, local_files_only=True,
    )
    adapter = PeftModel.from_pretrained(forecaster.model, run_dir / "finetuned-ckpt").eval()
    forecaster.model = adapter.get_base_model()
    return forecaster


def training_loss(forecaster: TimestralForecaster, batch: torch.Tensor, context: int,
                  known_covariates: torch.Tensor | None = None) -> torch.Tensor:
    model = forecaster.model
    patch_size = forecaster.patch_size
    batch = batch.to(forecaster.device)
    inputs, mask = forecaster._as_context(batch[:, :context], None)
    target = batch[:, context:]
    horizon = target.shape[-1]
    forecast_patches = math.ceil(horizon / patch_size)
    known_covariates = forecaster._as_covariates(known_covariates, len(batch), context + horizon, "known_covariates")
    rows, masks, roles, future, future_mask, _ = forecaster._assemble_rows(
        inputs, mask, horizon, past_covariates=None, known_covariates=known_covariates,
    )
    variates = rows.shape[1]
    groups = torch.arange(len(batch), device=batch.device).repeat_interleave(variates)

    # The public forecast() is inference-only. Capture the native head before
    # denormalization so gradients stay in normalized space.
    head_outputs = []
    with model.head.register_forward_hook(lambda _module, _inputs, output: head_outputs.append(output)):
        model.predict_block(rows.flatten(0, 1), masks.flatten(0, 1), groups, forecast_patches,
                            roles, future, future_mask)
    # Covariate mode computes decoder tokens over context and horizon. The target is row 0 in each group.
    quantiles = head_outputs[0][::variates, -forecast_patches * patch_size:]
    history, observed = rows[:, 0], masks[:, 0]

    # Timestral updates the scale of later forecast patches using its earlier
    # predicted medians. Put the actuals in those same frames without differentiating through the frames.
    with torch.no_grad():
        _, loc, scale = model.scale(history, observed)
        first_loc, first_scale = loc[:, -1:].double(), scale[:, -1:].double()
        median = quantiles[..., forecaster.median_index].double().unflatten(-1, (forecast_patches, patch_size))
        if model.config.use_asinh:
            median = median.clamp(-30, 30).sinh()
        patch_loc, patch_scale = advancing_block_frames(history, observed, median, model.config.minimum_scale)
        patch_loc = torch.cat((first_loc, patch_loc[:, 1:]), dim=1)
        patch_scale = torch.cat((first_scale, patch_scale[:, 1:]), dim=1)
        loc = patch_loc.repeat_interleave(patch_size, dim=1)[:, :horizon]
        scale = patch_scale.repeat_interleave(patch_size, dim=1)[:, :horizon]
        valid = torch.isfinite(target)
        normalized = (torch.where(valid, target.double(), loc) - loc) / scale
        if model.config.use_asinh:
            normalized = normalized.asinh()

    error = normalized[valid].to(quantiles.dtype).unsqueeze(-1) - quantiles[:, :horizon][valid]
    levels = quantiles.new_tensor(forecaster.quantile_levels)
    return torch.maximum(levels * error, (levels - 1) * error).mean()


@torch.inference_mode()
def evaluate(forecaster: TimestralForecaster, windows: torch.Tensor, context: int, batch_size: int,
             weather: torch.Tensor | None = None, d1: slice | None = None) -> dict:
    forecaster.model.eval()
    losses, errors, intervals, d1_errors = [], [], [], []
    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size]
        known = None if weather is None else weather[start:start + batch_size]
        actual = batch[:, context:].numpy()
        forecast = forecaster.forecast(batch[:, :context], prediction_length=actual.shape[1], known_covariates=known)
        q10, q50, q90 = (forecast.quantile(q)[:, 0].cpu().numpy() for q in (0.1, 0.5, 0.9))
        losses.extend([training_loss(forecaster, batch, context, known).item()] * len(batch))
        errors.extend(rmse(y, p) for y, p in zip(actual, q50))
        intervals.extend(winkler(y, lo, hi) for y, lo, hi in zip(actual, q10, q90))
        if d1 is not None:
            d1_errors.extend(rmse(y[d1], p[d1]) for y, p in zip(actual, q50))
    scores = {"loss": float(np.mean(losses)), "RMSE": float(np.mean(errors)),
              "Winkler80": float(np.mean(intervals))}
    if d1 is not None:
        scores["RMSE_D1"] = float(np.mean(d1_errors))
    if not all(map(math.isfinite, scores.values())):
        raise RuntimeError(f"Non-finite validation scores: {scores}")
    return scores


def covariate_windows(inputs: dict, covariates: list[str], context: int,
                      horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use aligned rolling views and sample windows with at least one observed context and label value."""
    length = context + horizon
    target = torch.from_numpy(inputs["target"])
    weather = torch.from_numpy(np.stack([inputs["past_covariates"][name] for name in covariates]))
    observed = torch.isfinite(target).to(torch.int64)
    counts = torch.cat((observed.new_zeros(1), observed.cumsum(dim=0)))
    at_forecast_start = counts[context:-horizon]
    past = at_forecast_start - counts[:-length]
    future = counts[length:] - at_forecast_start
    eligible = torch.where((past > 0) & (future > 0))[0]
    if not len(eligible):
        raise ValueError("No training windows with both observed context and forecast targets")
    return target.unfold(0, length, 1), weather.unfold(-1, length, 1).transpose(0, 1), eligible


def train(args: argparse.Namespace) -> Path:
    run_dir = config.BACKTEST_RESULTS_DIR / "finetuning" / "timestral_209m" / args.run_id
    covariates = list(config.TSFM_COVARIATES["wind"]) if getattr(args, "covariates", "none") == "config" else []
    record = start_training(run_dir, "timestral_209m", args, BASE_CHECKPOINT, BASE_REVISION, "finetuned-ckpt",
                            covariates=covariates)
    try:
        return _train(args, run_dir, record)
    except BaseException as exc:
        finish_training(run_dir, record, "failed", f"{type(exc).__name__}: {exc}")
        raise


def _train(args: argparse.Namespace, run_dir: Path, run_record: dict) -> Path:
    start = time.perf_counter()
    device = args.device
    if device == "auto":
        from src.models_tsfm import default_device
        device = default_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment")
    if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise ValueError("XPU is unavailable in this PyTorch environment (install the torch +xpu build)")
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    if config.BACKTEST_LATENCY_STEPS != 0:
        raise ValueError("This training split assumes zero forecast latency")
    context = int(pd.Timedelta(days=args.context_days) / STEP)
    horizon = config.BACKTEST_HORIZON_STEPS

    torch.manual_seed(args.seed)
    covariates = run_record["covariates"]
    validation_scheme = getattr(args, "validation", None) or ("weeks" if covariates else "december")
    weeks = validation_scheme == "weeks"
    validation_context = int(pd.Timedelta(days=30) / STEP) if weeks else context
    train_weather = validation_weather = d1 = None
    if covariates:
        series, held_out, origins = load_covariate_inputs(context, horizon, covariates,
                                                          validation=validation_scheme,
                                                          validation_context=validation_context)
        windows, train_weather, eligible = covariate_windows(series[0], covariates, context, horizon)
        validation = torch.from_numpy(np.stack([window["target"] for window in held_out]))
        validation_weather = torch.from_numpy(np.stack([
            [window["past_covariates"][name] for name in covariates] for window in held_out
        ]))
    else:
        if weeks:
            raise ValueError("Weeks validation requires --covariates config")
        windows, validation, origins = load_windows(context, horizon)
        eligible = torch.arange(len(windows))
    if weeks:
        day = pd.Timedelta(days=1)
        next_day = int((day - pd.Timedelta(f"{config.BACKTEST_ISSUE_TIME}:00")) / STEP)
        if horizon <= next_day:
            raise ValueError("Forecast horizon does not reach D+1")
        d1 = slice(next_day, next_day + int(day / STEP))
    selection_metric = "RMSE_D1" if weeks else "loss"
    validation_interval = args.validation_interval or (100 if weeks else 50)

    forecaster = TimestralForecaster.from_pretrained(
        BASE_CHECKPOINT, revision=BASE_REVISION, device=device, local_files_only=True,
    )
    if max(context, validation_context) > forecaster.max_context or horizon > forecaster.per_shot_horizon_patches * forecaster.patch_size:
        raise ValueError("Training or validation context exceeds the checkpoint limit, or horizon exceeds one forecast block")
    if covariates and not forecaster.supports_known_covariates:
        raise ValueError("The Timestral checkpoint does not support known-future covariates")

    validation_args = (validation, validation_context, args.batch_size, validation_weather, d1)
    baseline = evaluate(forecaster, *validation_args)
    targets = ATTENTION_MODULES if args.lora_targets == "attention" else TARGET_MODULES
    adapter = get_peft_model(forecaster.model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, target_modules=list(targets),
    ))
    forecaster.model = adapter.get_base_model()
    adapter.peft_config["default"].base_model_name_or_path = BASE_CHECKPOINT
    adapter.peft_config["default"].revision = BASE_REVISION
    trainable = [param for param in adapter.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    def lr_factor(step):
        if args.scheduler == "linear":
            return max(0.0, 1 - step / args.steps)
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1, max(0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    sampler = torch.Generator().manual_seed(args.seed)
    print(f"{len(eligible):,} training windows, {len(origins)} {validation_scheme} validation origins, "
          f"{context} training and {validation_context} validation context steps, "
          f"{horizon} forecast steps, {len(covariates)} covariates, {device}", flush=True)
    print(f"Zero-shot validation: {baseline}", flush=True)

    checkpoint = run_dir / "finetuned-ckpt"
    best, reference, history, bad_evaluations = None, None, [], 0
    for step in range(1, args.steps + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        batch_losses = []
        for _ in range(args.gradient_accumulation):
            indices = eligible[torch.randint(len(eligible), (args.batch_size,), generator=sampler)]
            known = None if train_weather is None else train_weather[indices]
            loss = training_loss(forecaster, windows[indices], context, known)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Timestral loss at step {step}")
            (loss / args.gradient_accumulation).backward()
            batch_losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % validation_interval == 0 or step == args.steps:
            scores = {"step": step, "training_batch_loss": float(np.mean(batch_losses)),
                      **evaluate(forecaster, *validation_args)}
            history.append(scores)
            print(f"Step {step}/{args.steps}: {scores}", flush=True)
            # Any improvement saves the checkpoint; only an improvement of at least --min-delta (in the units of
            # the selection metric, MW for RMSE_D1) since the last such improvement resets the patience counter.
            if best is None or scores[selection_metric] < best[selection_metric]:
                best = scores
                adapter.save_pretrained(checkpoint, save_embedding_layers=False)
            if reference is None or scores[selection_metric] < reference - args.min_delta:
                reference = scores[selection_metric]
                bad_evaluations = 0
            else:
                bad_evaluations += 1
            if args.patience and bad_evaluations >= args.patience:
                print(f"Early stopping at step {step}: {bad_evaluations} evaluations without an improvement of "
                      f"at least {args.min_delta} in {selection_metric}", flush=True)
                break

    record = {**run_record, **vars(args), "device": device, "base_checkpoint": BASE_CHECKPOINT, "base_revision": BASE_REVISION,
              "backend": "peft", "target": "wind_MW", "covariates": covariates, "context_steps": context,
              "horizon_steps": horizon, "issue_time_utc": config.BACKTEST_ISSUE_TIME,
              "train_start": config.TRAIN_START, "validation": validation_scheme,
              "validation_start": (origins[0].normalize() if weeks else VALIDATION_START).isoformat(),
              "validation_end_exclusive": config.TRAIN_END,
              "validation_origins": [origin.isoformat() for origin in origins],
              "alpha": 2 * args.rank, "target_modules": list(targets),
              "trainable_parameters": sum(param.numel() for param in trainable),
              "training_window_count": len(eligible), "latency_steps": 0, "optimizer": "AdamW",
              "validation_context_steps": validation_context, "checkpoint_metric": selection_metric,
              "validation_interval_steps": validation_interval,
              "zero_shot": baseline, "best": best, "validation_history": history}
    manifest = run_dir / "run.json"
    run_record.update(record)
    write_record(manifest, run_record)
    reloaded = evaluate(load_lora_run(run_dir, device), *validation_args)
    np.testing.assert_allclose(list(reloaded.values()), [best[key] for key in reloaded], rtol=1e-5, atol=1e-5)
    record.update(reloaded_validation=reloaded, elapsed_seconds=time.perf_counter() - start)
    run_record.update(record)
    finish_training(run_dir, run_record, "completed")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--context-days", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--covariates", choices=("none", "config"), default="none",
                        help="config: use the configured wind weather covariates over context and horizon")
    parser.add_argument("--validation", choices=("december", "weeks"),
                        help="defaults to weeks with covariates and December for target-only training")
    parser.add_argument("--lora-targets", choices=("attention-mlp", "attention"), default="attention-mlp")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--scheduler", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many evaluations without improvement; 0 disables")
    parser.add_argument("--validation-interval", type=int, default=None,
                        help="training steps between validations (default 100 with weeks validation, 50 with December)")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="smallest decrease of the selection metric (MW for RMSE_D1) that counts as an improvement "
                             "for --patience; the best checkpoint is still kept on any improvement")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must start with a letter or digit and contain only letters, digits, underscores or hyphens")
    if min(args.steps, args.batch_size, args.rank, args.context_days) < 1:
        parser.error("steps, batch size, rank and context days must all be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive and finite")
    if args.validation_interval is not None and args.validation_interval < 1:
        parser.error("--validation-interval must be positive")
    if args.gradient_accumulation < 1 or args.warmup_steps < 0 or args.warmup_steps > args.steps or args.patience < 0 or not math.isfinite(args.min_delta) or args.min_delta < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Invalid accumulation, warmup, patience, min delta or weight decay")
    if args.scheduler == "linear" and args.warmup_steps:
        parser.error("warmup is available with --scheduler cosine")
    if args.validation == "weeks" and args.covariates == "none":
        parser.error("--validation weeks requires --covariates config")
    print(f"Saved and verified {train(args)}", flush=True)
