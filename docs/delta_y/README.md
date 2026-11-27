# δY output adapter (Ada-Y): post-hoc correction of the forecasts

Status: **tried and dropped (November 2026).** The code was removed from the project and is kept in git
history. This page explains the method, how to restore or reimplement it, and why we stopped.

## Idea

The method comes from *"The Forecast after the Forecast: A Postprocessing Shift in Time Series"* (ICLR 2026).
The forecaster F stays frozen (zero-shot, or with its LoRA adapter). A tiny network learns a **bounded
correction of the output**:

    Ỹ = F(X) + δ · tanh(g(·))

In our version the correction acts on the three quantiles q10 / q50 / q90 of every forecast step:

    q50' = q50 + s · δ · tanh(a)                          s = std of the target context (MW), δ = 0.1 by default
    q90' = q50' + (q90 − q50) · exp(δ_w · tanh(b_up))      band widths scaled within [e^−δ_w, e^δ_w], δ_w = 0.5
    q10' = q50' − (q50 − q10) · exp(δ_w · tanh(b_down))

- `a, b_up, b_down` come from one small network shared by all 156 lead steps: an MLP per step, or a 1-D CNN
  over the steps (`--arch mlp|cnn`).
- **Inputs per step:**
  - the base quantiles, normalized by the context mean and std;
  - the future weather covariates (standardized);
  - sin/cos of the hour of day;
  - a learned lead-step embedding;
  - a summary of the last 24 h of context.
- The output layer starts at zero, so an untrained adapter returns the base forecast exactly.
  q10' ≤ q50' ≤ q90' always holds.
- The loss is the pinball loss of the three quantiles, in units of the context std.

The idea behind it: the foundation model has systematic biases (level, interval width per lead time and
weather regime), which a few thousand parameters could correct cheaply after the fact.

## How it was implemented

Removed in commit `0e276c5`. The last version with the code is commit `8fb0466`:

| File | Role |
|---|---|
| `src/delta_adapter.py` | `build_inputs` (features, shared by training and backtest), `DeltaYAdapter` (the network), `pinball_loss`, `scores`, `apply_to_predictions`, save/load, `check_settings` |
| `tuning/train_delta_y.py` | Training CLI (see the steps below) |
| `src/backtest.py` | Backtest key `delta:<run-id>`: runs the base model as a dependency, then applies the correction |
| `src/tsfm_runs.py` | `available_delta_adapters()`: the runs in `results/finetuning/delta_y/` |
| `evaluate_tsfm.py` | `--delta-run-id` |
| `app.py` | δY adapters selectable in the dashboard |

How training worked, in `tuning/train_delta_y.py`:

1. **Cache the base forecasts:** the shared rolling backtest forecasts every day of 2024 with the frozen base
   model (`chronos2` or `lora:chronos2:<run-id>`), using the same context, horizon and covariates as the 2025
   evaluation. The forecasts are stored in `results/finetuning/delta_y/base_forecasts/`.
2. **Train:** February–November 2024 origins train the adapter, and December 2024 picks the best epoch (early
   stopping, patience 100). The base model never runs during training.
3. **Save and check:** save `adapter.safetensors` plus `run.json`, then reload them and check the validation
   scores match.

To restore the code:

```bash
git checkout 8fb0466 -- src/delta_adapter.py tuning/train_delta_y.py
# the hooks in src/backtest.py, src/tsfm_runs.py, evaluate_tsfm.py and app.py: see `git show 0e276c5`
python -m tuning.train_delta_y --run-id my-dy --base lora:chronos2:<run-id> --context-steps 2880 --covariates config
python evaluate_tsfm.py --model chronos2 --adapter-run-id <run-id> --delta-run-id my-dy --covariates config --context-steps 2880
```

If you reimplement it, fix these first (found in the code review before removal):

- **Cache key:** add the LoRA adapter fingerprint, the model revision, the dataset fingerprint and the PCA
  setting. Without them, stale base forecasts get reused.
- **PCA:** pass `pca_components=None` (or the setting you intend) to the cache backtest, and check it in
  `check_settings`.
- **Covariate order:** select the covariates by name, in the training order. The adapter's normalization is
  positional.
- **Train/validation boundary:** drop the training origins whose 39-hour horizon reaches into December.
- **Device:** support `--device xpu`.

## Results

2025 test, 30-day inference context, 7 covariates, RMSE in MW:

| Base model | Base | + δY | Note |
|---|---|---|---|
| Chronos-2 zero-shot, 3-day context (`quick-dy-ctx288`) | 547.4 | 532.2 | Helps only a weak base forecast |
| Chronos-2 LoRA `cov30-r8` | 312.2 | 312.2 | Best epoch = 1: nothing learned |
| Timestral LoRA `tm-w-c30` (284.0 / Winkler 1016.7), offline | 284.0 | 284.0 | Leave-one-month-out validation loss rises from epoch 0, so no correction is chosen |
| Timestral LoRA `tm-w-c30`, online (retrained weekly) | 284.0 / 1016.7 | 284.1 / 1021.2 | Worse on intervals (n.s.) |

## Why we didn't go on

1. **No gain on a good base model.** Once the LoRA is fine-tuned with the covariates, its remaining errors
   are mostly unpredictable weather error, not systematic bias. A step-0 diagnostic on the Chronos-2 LoRA
   showed that even an oracle correction could reach at most −0.3% RMSE and −0.8% Winkler.
2. **Little clean training data.** The adapter has to learn from **out-of-sample** base forecasts. On 2024 the
   LoRA is in-sample, so its residuals are too small and the adapter learns nothing (best epoch 1). The only
   clean 2024 forecasts are the 77 held-out validation days, which isn't enough.
3. **It only helps weak forecasts.** The one real gain came on zero-shot Chronos-2 with a 3-day context
   (−15 MW from 547), a setting we no longer use.
4. **Cost:** about 450 lines plus a second model to version, cache and check, for no measurable benefit.
   Fine-tuning the model (LoRA) and a longer context (60 days: 280.5 RMSE) gave much more.

It could be worth revisiting with a model that has a clear systematic bias, such as a capacity jump between
training and test years, or if enough out-of-sample forecasts become available (a full extra year).
