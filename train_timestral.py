"""Train a target-only LoRA adapter for the cached Timestral-209M checkpoint."""

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
from torch.nn import functional as F

import config
from src.forecast import STEP
from src.metrics import rmse, winkler
from src.tuning_data import VALIDATION_START, load_windows


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


def training_loss(forecaster: TimestralForecaster, batch: torch.Tensor, context: int) -> torch.Tensor:
    model = forecaster.model
    patch_size = forecaster.patch_size
    batch = batch.to(forecaster.device)
    history, target = batch[:, :context], batch[:, context:]
    horizon = target.shape[-1]
    forecast_patches = math.ceil(horizon / patch_size)
    padding = (-context) % patch_size
    observed = F.pad(torch.ones_like(history, dtype=torch.bool), (padding, 0), value=False)
    history = F.pad(history, (padding, 0))

    # The public forecast() is inference-only. Capture the native head before
    # denormalization so gradients stay in normalized space.
    head_outputs = []
    with model.head.register_forward_hook(lambda _module, _inputs, output: head_outputs.append(output)):
        model.predict_block(history, observed, torch.arange(len(batch), device=batch.device), forecast_patches)
    quantiles = head_outputs[0]

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
        normalized = (target.double() - loc) / scale
        if model.config.use_asinh:
            normalized = normalized.asinh()

    error = normalized.to(quantiles.dtype).unsqueeze(-1) - quantiles[:, :horizon]
    levels = quantiles.new_tensor(forecaster.quantile_levels)
    return torch.maximum(levels * error, (levels - 1) * error).mean()


@torch.inference_mode()
def evaluate(forecaster: TimestralForecaster, windows: torch.Tensor, context: int, batch_size: int) -> dict:
    forecaster.model.eval()
    losses, errors, intervals = [], [], []
    for batch in windows.split(batch_size):
        actual = batch[:, context:].numpy()
        forecast = forecaster.forecast(batch[:, :context], prediction_length=actual.shape[1])
        q10, q50, q90 = (forecast.quantile(q)[:, 0].cpu().numpy() for q in (0.1, 0.5, 0.9))
        losses.extend([training_loss(forecaster, batch, context).item()] * len(batch))
        errors.extend(rmse(y, p) for y, p in zip(actual, q50))
        intervals.extend(winkler(y, lo, hi) for y, lo, hi in zip(actual, q10, q90))
    scores = {"loss": float(np.mean(losses)), "RMSE": float(np.mean(errors)),
              "Winkler80": float(np.mean(intervals))}
    if not all(map(math.isfinite, scores.values())):
        raise RuntimeError(f"Non-finite validation scores: {scores}")
    return scores


def train(args: argparse.Namespace) -> Path:
    start = time.perf_counter()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment")
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    if config.BACKTEST_LATENCY_STEPS != 0:
        raise ValueError("This training split assumes zero forecast latency")
    context = int(pd.Timedelta(days=args.context_days) / STEP)
    horizon = config.BACKTEST_HORIZON_STEPS
    run_dir = config.BACKTEST_RESULTS_DIR / "finetuning" / "timestral_209m" / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")

    torch.manual_seed(args.seed)
    windows, validation, origins = load_windows(context, horizon)
    forecaster = TimestralForecaster.from_pretrained(
        BASE_CHECKPOINT, revision=BASE_REVISION, device=device, local_files_only=True,
    )
    if context > forecaster.max_context or horizon > forecaster.config.block_size * forecaster.patch_size:
        raise ValueError("Context exceeds the checkpoint limit or horizon exceeds one Timestral forecast block")

    baseline = evaluate(forecaster, validation, context, args.batch_size)
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
    print(f"{len(windows):,} training windows, {len(origins)} validation origins, "
          f"{context} context steps, {horizon} forecast steps, {device}", flush=True)
    print(f"Zero-shot validation: {baseline}", flush=True)

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "finetuned-ckpt"
    best, history, bad_evaluations = None, [], 0
    for step in range(1, args.steps + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        batch_losses = []
        for _ in range(args.gradient_accumulation):
            indices = torch.randint(len(windows), (args.batch_size,), generator=sampler)
            loss = training_loss(forecaster, windows[indices], context)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Timestral loss at step {step}")
            (loss / args.gradient_accumulation).backward()
            batch_losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == args.steps:
            scores = {"step": step, "training_batch_loss": float(np.mean(batch_losses)),
                      **evaluate(forecaster, validation, context, args.batch_size)}
            history.append(scores)
            print(f"Step {step}/{args.steps}: {scores}", flush=True)
            if best is None or scores["loss"] < best["loss"]:
                best = scores
                bad_evaluations = 0
                adapter.save_pretrained(checkpoint, save_embedding_layers=False)
            else:
                bad_evaluations += 1
            if args.patience and bad_evaluations >= args.patience:
                print(f"Early stopping after {bad_evaluations} evaluations without improvement", flush=True)
                break

    record = {**vars(args), "device": device, "base_checkpoint": BASE_CHECKPOINT, "base_revision": BASE_REVISION,
              "backend": "peft", "target": "wind_MW", "covariates": [], "context_steps": context,
              "horizon_steps": horizon, "issue_time_utc": config.BACKTEST_ISSUE_TIME,
              "train_start": config.TRAIN_START, "validation_start": VALIDATION_START.isoformat(),
              "validation_end_exclusive": config.TRAIN_END,
              "validation_origins": [origin.isoformat() for origin in origins],
              "alpha": 2 * args.rank, "target_modules": list(targets),
              "trainable_parameters": sum(param.numel() for param in trainable),
              "zero_shot": baseline, "best": best, "validation_history": history}
    manifest = run_dir / "run.json"
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    reloaded = evaluate(load_lora_run(run_dir, device), validation, context, args.batch_size)
    np.testing.assert_allclose(list(reloaded.values()), [best[key] for key in reloaded], rtol=1e-5, atol=1e-5)
    record.update(reloaded_validation=reloaded, elapsed_seconds=time.perf_counter() - start)
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--context-days", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-targets", choices=("attention-mlp", "attention"), default="attention-mlp")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--scheduler", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many evaluations without improvement; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must start with a letter or digit and contain only letters, digits, underscores or hyphens")
    if min(args.steps, args.batch_size, args.rank, args.context_days) < 1:
        parser.error("steps, batch size, rank and context days must all be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive and finite")
    if args.gradient_accumulation < 1 or args.warmup_steps < 0 or args.warmup_steps > args.steps or args.patience < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Invalid accumulation, warmup, patience or weight decay")
    if args.scheduler == "linear" and args.warmup_steps:
        parser.error("warmup is available with --scheduler cosine")
    print(f"Saved and verified {train(args)}", flush=True)
