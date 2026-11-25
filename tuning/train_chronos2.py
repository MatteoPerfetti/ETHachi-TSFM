"""Train and validate one Chronos-2 LoRA adapter on cached wind data, optionally with the weather covariates."""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch import nn

import config
from src.forecast import STEP
from src.metrics import rmse, winkler
from src.tuning_data import VALIDATION_START, load_covariate_inputs, load_windows
from src.tsfm_runs import finish_training, start_training, write_record


BASE_CHECKPOINT = config.MODELS_REGISTRY["chronos2"]["checkpoint"]
BASE_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
TARGET_MODULES = ("self_attention.q", "self_attention.k", "self_attention.v",
                  "self_attention.o", "output_patch_embedding.output_layer")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        options = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.down = nn.Linear(base.in_features, rank, bias=False, **options)
        self.up = nn.Linear(rank, base.out_features, bias=False, **options)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + 2 * self.up(self.down(inputs))


def attach_lora(model: nn.Module, rank: int) -> dict[str, nn.Parameter]:
    model.requires_grad_(False)
    modules = [(name, layer) for name, layer in model.named_modules()
               if isinstance(layer, nn.Linear) and name.endswith(TARGET_MODULES)]
    if any(not any(name.endswith(suffix) for name, _ in modules) for suffix in TARGET_MODULES):
        raise RuntimeError("The Chronos checkpoint is missing an expected LoRA projection")

    parameters = {}
    for name, layer in modules:
        lora = LoRALinear(layer, rank)
        model.set_submodule(name, lora)
        parameters[f"{name}.lora_A.weight"] = lora.down.weight
        parameters[f"{name}.lora_B.weight"] = lora.up.weight
    return parameters


def load_base(device: str, checkpoint: str = BASE_CHECKPOINT, revision: str = BASE_REVISION) -> Chronos2Pipeline:
    snapshot = snapshot_download(checkpoint, revision=revision, local_files_only=True)
    return Chronos2Pipeline.from_pretrained(snapshot, local_files_only=True, device_map=device, dtype=torch.float32)


def load_lora_run(run_dir: Path, device: str) -> Chronos2Pipeline:
    """Reload a saved adapter onto the exact base revision recorded in the run."""
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    pipeline = load_base(device, metadata["base_checkpoint"], metadata["base_revision"])
    if metadata.get("backend") == "peft":
        from peft import PeftModel

        model = PeftModel.from_pretrained(pipeline.model, run_dir / "finetuned-ckpt")
        return Chronos2Pipeline(model.merge_and_unload().eval())

    parameters = attach_lora(pipeline.model, metadata["rank"])
    weights = load_file(run_dir / "adapter.safetensors")
    if weights.keys() != parameters.keys():
        raise ValueError("The adapter weights do not match the Chronos checkpoint")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name])
    pipeline.model.requires_grad_(False).eval()
    return pipeline


def forward_batch(pipeline: Chronos2Pipeline, batch: torch.Tensor, context: int):
    batch = batch.to(pipeline.model.device)
    horizon = batch.shape[1] - context
    return pipeline.model(
        context=batch[:, :context], future_target=batch[:, context:],
        group_ids=torch.arange(len(batch), device=batch.device),
        num_output_patches=math.ceil(horizon / pipeline.model_output_patch_size),
    )


def covariate_batches(pipeline: Chronos2Pipeline, inputs: list[dict], context: int, horizon: int, batch_size: int):
    """Validation batches with covariates, built by Chronos itself exactly as during fit().

    Yields (model inputs, target-row mask). Every window is one group: the target row first, then one row
    per covariate; `batch_size` counts windows.
    """
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    rows = 1 + len(inputs[0]["past_covariates"])
    dataset = Chronos2Dataset(inputs=inputs, context_length=context, prediction_length=horizon,
                              batch_size=batch_size * rows, output_patch_size=pipeline.model_output_patch_size,
                              mode=DatasetMode.VALIDATION)
    for batch in dataset:
        group_ids = batch["group_ids"]
        first_row = torch.ones_like(group_ids, dtype=torch.bool)
        first_row[1:] = group_ids[1:] != group_ids[:-1]
        yield batch, first_row


@torch.inference_mode()
def evaluate(pipeline: Chronos2Pipeline, windows: torch.Tensor | list[dict], context: int, batch_size: int) -> dict:
    """Validation loss, RMSE and Winkler80 of the target. `windows` is a tensor (wind-only) or Chronos input dicts."""
    pipeline.model.eval()
    levels = [list(pipeline.quantiles).index(q) for q in (0.1, 0.5, 0.9)]
    losses, errors, intervals = [], [], []
    def wind_only(batch):
        out = forward_batch(pipeline, batch, context)
        return out.loss, out.quantile_preds, batch[:, context:].numpy()

    def with_covariates(batch, first_row):
        device = pipeline.model.device
        out = pipeline.model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
        return out.loss, out.quantile_preds[first_row.to(device)], batch["future_target"][first_row].numpy()

    if isinstance(windows, torch.Tensor):
        batches = (wind_only(batch) for batch in windows.split(batch_size))
    else:
        horizon = len(windows[0]["target"]) - context
        batches = (with_covariates(*item)
                   for item in covariate_batches(pipeline, windows, context, horizon, batch_size))
    for loss, quantile_preds, actual in batches:
        q10, q50, q90 = quantile_preds[:, levels, :actual.shape[1]].cpu().numpy().transpose(1, 0, 2)
        losses.extend([loss.item()] * len(actual))
        errors.extend(rmse(y, p) for y, p in zip(actual, q50))
        intervals.extend(winkler(y, lo, hi) for y, lo, hi in zip(actual, q10, q90))
    scores = {"loss": float(np.mean(losses)), "RMSE": float(np.mean(errors)),
              "Winkler80": float(np.mean(intervals))}
    if not all(map(math.isfinite, scores.values())):
        raise RuntimeError(f"Non-finite validation scores: {scores}")
    return scores


def train_with_peft(pipeline, training, validation, context, horizon, run_dir, args):
    from peft import LoraConfig
    from transformers.trainer_callback import TrainerCallback

    class CaptureState(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            self.best_step = int(Path(state.best_model_checkpoint).name.rsplit("-", 1)[-1])
            self.evaluations = [entry for entry in state.log_history if "eval_loss" in entry]

    if isinstance(training, torch.Tensor):
        # Recover the continuous training series from its consecutive overlapping windows.
        inputs = [torch.cat((training[0, :-1], training[:, -1]))]
        validation_inputs = list(validation.unbind())
        fit_batch_size = args.batch_size
    else:
        # Covariate dicts: fit() counts the rows of a batch (target + covariates), so with C covariates
        # args.batch_size windows take batch_size * (1 + C) rows.
        inputs, validation_inputs = training, validation
        fit_batch_size = args.batch_size * (1 + len(training[0]["past_covariates"]))
    state = CaptureState()
    period = min(50, args.steps)
    tuned = pipeline.fit(
        inputs=inputs, validation_inputs=validation_inputs,
        context_length=context, min_past=context, prediction_length=horizon,
        finetune_mode="lora", lora_config=LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, target_modules=list(TARGET_MODULES),
        ),
        learning_rate=args.learning_rate, num_steps=args.steps, batch_size=fit_batch_size,
        output_dir=run_dir, callbacks=[state], eval_steps=period, save_steps=period,
        seed=args.seed, optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
        disable_tqdm=True,
    )
    if not (run_dir / "finetuned-ckpt" / "adapter_config.json").is_file():
        raise RuntimeError("Chronos did not save a PEFT adapter")
    best = {"step": state.best_step, **evaluate(tuned, validation, context, args.batch_size)}
    return best, state.evaluations, fit_batch_size


def training_covariates(args: argparse.Namespace) -> list[str]:
    """Weather columns used in training: none, or config.TSFM_COVARIATES['wind'] (the backtest's default inputs)."""
    return list(config.TSFM_COVARIATES["wind"]) if getattr(args, "covariates", "none") == "config" else []


def train(args: argparse.Namespace) -> Path:
    run_dir = config.BACKTEST_RESULTS_DIR / "finetuning" / "chronos2" / args.run_id
    record = start_training(run_dir, "chronos2", args, BASE_CHECKPOINT, BASE_REVISION,
                            "finetuned-ckpt" if args.backend == "peft" else "adapter.safetensors",
                            covariates=training_covariates(args))
    try:
        return _train(args, run_dir, record)
    except BaseException as exc:
        finish_training(run_dir, record, "failed", f"{type(exc).__name__}: {exc}")
        raise


def _train(args: argparse.Namespace, run_dir: Path, record: dict) -> Path:
    started = time.perf_counter()
    context = int(pd.Timedelta(days=args.context_days) / STEP)
    horizon = config.BACKTEST_HORIZON_STEPS
    if config.BACKTEST_LATENCY_STEPS != 0:
        raise ValueError("The current training split assumes zero forecast latency")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment")
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    covariates = training_covariates(args)
    if covariates and args.backend != "peft":
        raise ValueError("Training with covariates needs --backend peft")
    if covariates:
        training, validation, origins = load_covariate_inputs(context, horizon, covariates)
        window_count = len(training[0]["target"]) - context - horizon + 1
    else:
        training, validation, origins = load_windows(context, horizon)
        window_count = len(training)
    pipeline = load_base(device)
    if context > pipeline.model_context_length:
        raise ValueError("Training history exceeds the base checkpoint context limit")
    baseline = evaluate(pipeline, validation, context, args.batch_size)
    print(f"{window_count:,} training windows, {len(validation)} validation origins, "
          f"{context} context steps, {horizon} forecast steps, covariates {covariates or 'none'}, {device}",
          flush=True)
    print(f"Zero-shot validation: {baseline}", flush=True)
    metadata = {**record, "base_checkpoint": BASE_CHECKPOINT, "base_revision": BASE_REVISION,
                "backend": args.backend, "target": "wind_MW", "context_steps": context,
                "horizon_steps": horizon, "train_start": config.TRAIN_START,
                "validation_start": VALIDATION_START.isoformat(),
                "validation_end_exclusive": config.TRAIN_END,
                "validation_origins": [origin.isoformat() for origin in origins],
                "rank": args.rank, "alpha": 2 * args.rank, "seed": args.seed,
                "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "device": device, "zero_shot": baseline, "training_window_count": window_count,
                "covariates": covariates, "context_days": args.context_days,
                "target_modules": list(TARGET_MODULES), "issue_time_utc": config.BACKTEST_ISSUE_TIME,
                "latency_steps": 0, "optimizer": "AdamW", "weight_decay": 0.0,
                "validation_interval_steps": min(50, args.steps) if args.backend == "peft" else 50}
    if args.backend == "peft":
        best, history, fit_batch_size = train_with_peft(pipeline, training, validation, context, horizon,
                                                        run_dir, args)
        metadata["fit_batch_rows"] = fit_batch_size
    else:
        best, history = train_custom(pipeline, training, validation, context, run_dir, args, generator)

    metadata.update(best=best, validation_history=history)
    manifest = run_dir / "run.json"
    record.update(metadata)
    write_record(manifest, record)
    reloaded = evaluate(load_lora_run(run_dir, device), validation, context, args.batch_size)
    np.testing.assert_allclose(list(reloaded.values()), [best[key] for key in reloaded], rtol=1e-5, atol=1e-5)
    metadata.update(reloaded_validation=reloaded, elapsed_seconds=time.perf_counter() - started)
    record.update(metadata)
    finish_training(run_dir, record, "completed")
    return run_dir


def train_custom(pipeline, training, validation, context, run_dir, args, generator):
    parameters = attach_lora(pipeline.model, args.rank)
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / args.steps)
    history, best, best_weights = [], None, None
    for step in range(1, args.steps + 1):
        pipeline.model.train()
        indices = torch.randint(len(training), (args.batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = forward_batch(pipeline, training[indices], context).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at optimizer step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == args.steps:
            scores = evaluate(pipeline, validation, context, args.batch_size)
            record = {"step": step, "training_batch_loss": loss.item(), **scores}
            history.append(record)
            print(f"Step {step}/{args.steps}: {record}", flush=True)
            if best is None or scores["loss"] < best["loss"]:
                best = record
                best_weights = {name: param.detach().cpu().clone() for name, param in parameters.items()}

    if best is None or best_weights is None:
        raise RuntimeError("Training finished without a validated checkpoint")
    save_file(best_weights, run_dir / "adapter.safetensors")
    return best, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend", choices=("custom", "peft"), default="custom")
    parser.add_argument("--context-days", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="training windows per optimizer step (with covariates each window is 1 + C rows)")
    parser.add_argument("--covariates", choices=("none", "config"), default="none",
                        help="config: train with config.TSFM_COVARIATES['wind'] as known future covariates "
                             "(PEFT backend only)")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must start with a letter or digit and contain only letters, digits, underscores or hyphens")
    if min(args.steps, args.batch_size, args.rank, args.context_days) < 1:
        parser.error("steps, batch size, rank and context days must all be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive and finite")
    if args.covariates != "none" and args.backend != "peft":
        parser.error("--covariates config needs --backend peft")
    print(f"Saved and verified {train(args)}", flush=True)
