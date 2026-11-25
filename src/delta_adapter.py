"""Output-side δ-Adapter (Ada-Y) in quantile form, on top of a frozen forecaster.

Idea from "The Forecast after the Forecast: A Postprocessing Shift in Time Series" (ICLR 2026): keep the forecaster F
frozen and learn a tiny network that makes bounded corrections to its output, Ỹ = F(X) + δ·tanh(g(·)). Here F is
Chronos-2 (zero-shot or with a LoRA adapter) and the correction acts on its q10/q50/q90:

    q50' = q50 + s·δ·tanh(a)                          s = std of the 30-day target context (MW)
    q90' = q50' + (q90 - q50)·exp(δ_w·tanh(b_up))     band widths scaled by a factor in [e^-δ_w, e^δ_w]
    q10' = q50' - (q50 - q10)·exp(δ_w·tanh(b_down))

a, b_up, b_down come from one small network shared by all lead steps (MLP per step, or a 1-D CNN over the steps).
Its output layer starts at zero, so the untrained adapter returns the forecaster's quantiles exactly, and
q10' <= q50' <= q90' always holds. Training uses the pinball loss of the three quantiles, in units of s.

The same `build_inputs` feeds training (`tuning/train_delta_y.py`) and the rolling backtest (`src/backtest.py`),
so the features cannot differ between the two.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.forecast import STEP
from src.metrics import rmse, winkler

QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_COLUMNS = ("q10", "q50", "q90")
SCALE_FLOOR_MW = 1.0          # minimum context std, avoids dividing by ~0 on flat contexts
RECENT_STEPS = 96             # last 24 h of the context, summarized as features


@dataclass
class AdapterInputs:
    """Everything the adapter needs for N forecasts of H steps (one row per issue time)."""
    issue_times: pd.DatetimeIndex
    timestamps: np.ndarray            # (N, H) datetime64, forecast timestamps
    quantiles: torch.Tensor           # (N, H, 3) base q10, q50, q90 in MW
    loc: torch.Tensor                 # (N, 1) context mean, MW
    scale: torch.Tensor               # (N, 1) context std, MW (floored)
    covariates: torch.Tensor          # (N, H, C) future weather, raw units (C may be 0)
    hour: torch.Tensor                # (N, H, 2) sin/cos of the hour of day of each forecast timestamp
    context_summary: torch.Tensor     # (N, 3) last value, mean and std of the last 24 h, in units of the context
    actual: torch.Tensor              # (N, H) observed MW, NaN where unknown

    def subset(self, mask: np.ndarray) -> "AdapterInputs":
        index = torch.as_tensor(np.flatnonzero(mask))
        return AdapterInputs(self.issue_times[mask], self.timestamps[mask],
                             *(t[index] for t in (self.quantiles, self.loc, self.scale, self.covariates,
                                                  self.hour, self.context_summary, self.actual)))


def build_inputs(base: pd.DataFrame, target: pd.Series, weather: pd.DataFrame, covariates: list[str],
                 context_steps: int) -> AdapterInputs:
    """Adapter inputs from one model's rows of a backtest predictions table.

    `base` has the columns issue_time, timestamp, q10, q50, q90 (one row per issue time and forecast step, every
    issue time with the same number of steps). The context is the `context_steps` target values before each issue
    time (zero latency), exactly what the forecaster saw.
    """
    base = base.sort_values(["issue_time", "timestamp"])
    issues = pd.DatetimeIndex(base["issue_time"].unique())
    horizon = len(base) // len(issues)
    if horizon * len(issues) != len(base):
        raise ValueError("Every issue time needs the same number of forecast steps")
    quantiles = base[list(QUANTILE_COLUMNS)].to_numpy(dtype=np.float32).reshape(len(issues), horizon, 3)
    if np.isnan(quantiles).any():
        raise ValueError("The base forecasts need q10, q50 and q90 (no missing quantiles)")
    timestamps = pd.DatetimeIndex(base["timestamp"])

    stats = []
    for issue in issues:
        context = target.loc[issue - context_steps * STEP: issue - STEP].to_numpy(dtype=np.float64)
        if len(context) != context_steps or np.isnan(context).any():
            raise ValueError(f"Incomplete target context before {issue}")
        loc, scale = context.mean(), max(context.std(), SCALE_FLOOR_MW)
        recent = (context[-RECENT_STEPS:] - loc) / scale
        stats.append((loc, scale, recent[-1], recent.mean(), recent.std()))
    stats = np.asarray(stats, dtype=np.float32)

    if covariates:
        future = weather.reindex(timestamps)[covariates].to_numpy(dtype=np.float32)
        if np.isnan(future).any():
            raise ValueError("Missing future covariates for some forecast steps")
        future = future.reshape(len(issues), horizon, len(covariates))
    else:
        future = np.zeros((len(issues), horizon, 0), dtype=np.float32)
    angle = 2 * np.pi * (timestamps.hour + timestamps.minute / 60) / 24
    hour = np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32).reshape(len(issues), horizon, 2)
    actual = target.reindex(timestamps).to_numpy(dtype=np.float32).reshape(len(issues), horizon)

    tensor = lambda array: torch.tensor(np.array(array, dtype=np.float32))     # owned, writable copies
    return AdapterInputs(issues, timestamps.to_numpy().reshape(len(issues), horizon),
                         tensor(quantiles), tensor(stats[:, :1]), tensor(stats[:, 1:2]), tensor(future),
                         tensor(hour), tensor(stats[:, 2:]), tensor(actual))


class DeltaYAdapter(nn.Module):
    """Bounded, monotone correction of q10/q50/q90; one trunk shared by every lead step."""

    def __init__(self, n_covariates: int, horizon: int, arch: str = "mlp", hidden: int = 64, step_dim: int = 8,
                 kernel_size: int = 5, delta: float = 0.1, width_delta: float = 0.5):
        super().__init__()
        if arch not in ("mlp", "cnn"):
            raise ValueError(f"arch must be 'mlp' or 'cnn', got {arch}")
        self.config = {"n_covariates": n_covariates, "horizon": horizon, "arch": arch, "hidden": hidden,
                       "step_dim": step_dim, "kernel_size": kernel_size, "delta": delta, "width_delta": width_delta}
        self.arch, self.delta, self.width_delta = arch, delta, width_delta
        in_dim = 3 + n_covariates + 2 + step_dim + 3
        self.step_embedding = nn.Embedding(horizon, step_dim)
        if arch == "mlp":
            self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        else:   # 1-D convolution over the lead steps: neighbouring steps share information
            padding = kernel_size // 2
            self.trunk = nn.Sequential(nn.Conv1d(in_dim, hidden, kernel_size, padding=padding), nn.GELU(),
                                       nn.Conv1d(hidden, hidden, kernel_size, padding=padding), nn.GELU())
        self.head = nn.Linear(hidden, 3)                     # a, b_up, b_down
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # covariate normalization, set from the training data by `fit_normalization`
        self.register_buffer("covariate_mean", torch.zeros(n_covariates))
        self.register_buffer("covariate_std", torch.ones(n_covariates))

    def fit_normalization(self, inputs: AdapterInputs) -> None:
        flat = inputs.covariates.reshape(-1, inputs.covariates.shape[-1])
        if flat.shape[-1]:
            self.covariate_mean.copy_(flat.mean(0))
            self.covariate_std.copy_(flat.std(0).clamp_min(1e-6))

    def forward(self, inputs: AdapterInputs) -> torch.Tensor:
        """Corrected quantiles (N, H, 3) in MW."""
        device = self.head.weight.device
        q = inputs.quantiles.to(device)
        loc, scale = inputs.loc.to(device)[:, :, None], inputs.scale.to(device)[:, :, None]
        n, horizon, _ = q.shape
        features = torch.cat([
            (q - loc) / scale,
            (inputs.covariates.to(device) - self.covariate_mean) / self.covariate_std,
            inputs.hour.to(device),
            self.step_embedding(torch.arange(horizon, device=device)).expand(n, -1, -1),
            inputs.context_summary.to(device)[:, None, :].expand(-1, horizon, -1),
        ], dim=-1)
        hidden = self.trunk(features) if self.arch == "mlp" else self.trunk(features.transpose(1, 2)).transpose(1, 2)
        a, b_up, b_down = self.head(hidden).unbind(-1)

        q10, q50, q90 = q.unbind(-1)
        center = q50 + scale[..., 0] * self.delta * torch.tanh(a)
        upper = center + (q90 - q50).clamp_min(0) * torch.exp(self.width_delta * torch.tanh(b_up))
        lower = center - (q50 - q10).clamp_min(0) * torch.exp(self.width_delta * torch.tanh(b_down))
        return torch.stack([lower, center, upper], dim=-1)


def pinball_loss(predicted: torch.Tensor, inputs: AdapterInputs) -> torch.Tensor:
    """Mean pinball loss of q10/q50/q90 in units of the context std; steps without an observed value are ignored."""
    actual = inputs.actual.to(predicted.device)
    scale = inputs.scale.to(predicted.device)
    levels = torch.tensor(QUANTILES, device=predicted.device)
    error = (actual[..., None] - predicted) / scale[:, :, None]
    loss = torch.maximum(levels * error, (levels - 1) * error)
    observed = torch.isfinite(actual)[..., None].expand_as(loss)
    return loss[observed].mean()


@torch.no_grad()
def scores(quantiles: torch.Tensor, inputs: AdapterInputs) -> dict:
    """Pinball loss, and RMSE / Winkler80 in MW as the mean over forecasts (as src.metrics.summary)."""
    q = quantiles.cpu()
    actual = inputs.actual.numpy()
    errors, intervals = [], []
    for i in range(len(actual)):
        known = np.isfinite(actual[i])
        errors.append(rmse(actual[i][known], q[i, known, 1].numpy()))
        intervals.append(winkler(actual[i][known], q[i, known, 0].numpy(), q[i, known, 2].numpy()))
    return {"loss": float(pinball_loss(q, inputs)), "RMSE": float(np.mean(errors)),
            "Winkler80": float(np.mean(intervals))}


def apply_to_predictions(adapter: DeltaYAdapter, base: pd.DataFrame, target: pd.Series, weather: pd.DataFrame,
                         covariates: list[str], context_steps: int, name: str) -> pd.DataFrame:
    """The base model's rows of a predictions table with corrected quantiles, renamed `name`."""
    base = base.sort_values(["issue_time", "timestamp"]).reset_index(drop=True)
    inputs = build_inputs(base, target, weather, covariates, context_steps)
    adapter.eval()
    with torch.no_grad():
        corrected = adapter(inputs).cpu().numpy().reshape(-1, 3)
    out = base.copy()
    out[list(QUANTILE_COLUMNS)] = corrected.astype(float)
    out["model"] = name
    return out


ADAPTER_FILE = "adapter.safetensors"


def save_adapter(adapter: DeltaYAdapter, run_dir: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().cpu().contiguous() for k, v in adapter.state_dict().items()}, run_dir / ADAPTER_FILE)


def load_adapter(run_dir: Path, device: str = "cpu") -> tuple[DeltaYAdapter, dict]:
    """Rebuild a saved adapter from its run.json and weights."""
    from safetensors.torch import load_file

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    adapter = DeltaYAdapter(**record["adapter_config"])
    adapter.load_state_dict(load_file(run_dir / ADAPTER_FILE))
    return adapter.to(device).eval(), record


def check_settings(record: dict, settings) -> None:
    """Raise unless the backtest gives the adapter the same inputs it was trained on."""
    expected = {"context_steps": record["context_steps"], "horizon_steps": record["horizon_steps"],
                "latency_steps": record.get("latency_steps", 0), "issue_time": record["issue_time_utc"],
                "target_source": "wind"}
    actual = {key: getattr(settings, key) for key in expected}
    problems = [f"{key} {actual[key]} (trained with {value})" for key, value in expected.items() if actual[key] != value]
    if sorted(record.get("covariates") or []) != sorted(settings.covariates):
        problems.append(f"covariates {list(settings.covariates)} (trained with {record.get('covariates')})")
    if problems:
        raise ValueError(f"delta-Y adapter {record['run_id']} needs the settings it was trained with: " + "; ".join(problems))
