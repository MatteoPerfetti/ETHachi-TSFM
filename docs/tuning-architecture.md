# Chronos-2 tuning and backtest architecture

Keep training and forecasting as separate jobs. **Training creates an adapter and a run record. A backtest reads that record, loads the adapter once, and rolls through forecast dates with fixed weights.** Hydra configures and launches jobs. Streamlit lets the user choose settings, watch a run, and compare saved results. The run records and artifacts live on disk so PowerShell and Streamlit can work with the same files.

The standalone trainer, custom and official PEFT backends, saved adapters, and `load_lora_run()` already exist. The Hydra entry points, run browser, and adapter backtest integration below are proposed work.

The standalone Timestral trainer uses the same wind windows and December validation dates as Chronos-2. Run it from the project root with the existing Timestral-209M checkpoint cached and PEFT installed.

```powershell
$runId = "wind-3d-timestral-" + (Get-Date -Format "yyyyMMdd-HHmmss")
.\.venv\Scripts\python.exe -m tuning.train_timestral --run-id $runId --context-days 3 --device cpu --steps 200 --batch-size 8
```

Timestral has 32-step patches and uses the model's advancing normalization frames for forecast labels. Its trainer computes pinball loss on the normalized output head before conversion back to MW. Chronos-2 has 16-step patches and provides its own training loss through `fit()`. Use the December RMSE and Winkler80 in MW for comparisons between models. The two training losses use different scales and reductions. Timestral saves the same manifest and standard PEFT files under `results/finetuning/timestral_209m/<run-id>/`. Its trainer currently uses wind generation alone. [TabTune Timestral reference](https://github.com/Lexsi-Labs/TabTune/blob/main/tabtune/models/TimeSeries/timestral.py)

## 1. Configure a run with Hydra

Add a thin Hydra entry point over the existing `tuning/train_chronos2.py` training code. Keep training logic callable from Python. Both the command line and Streamlit should go through the same validation and training path.

| Configuration | Proposed fields | Initial preset |
| --- | --- | --- |
| `data` | Source and version or file fingerprint, target, resolution, train and validation bounds | Belgian wind generation, 15 minutes, January–November 2024 for training, December 2024 for validation |
| `context` | History length or policy and its parameters | Contiguous 288 steps (3 days). Other length presets can be 96, 672, and 1,344 steps. |
| `covariates` | Exact columns, source, history-only or known-future availability, and forecast vintage rule | `none`. A weather preset is selectable once its inputs are implemented and verified. |
| `forecast` | Issue time, time zone, latency, horizon, quantiles for reporting | 09:00 UTC, zero latency, 156 steps, q10/q50/q90 |
| `model` and `lora` | Base checkpoint and revision, backend, rank, alpha, and target projections | Cached `amazon/chronos-2` revision, PEFT, rank 8, alpha 16 |
| `training` | Steps, batch size, optimizer, learning rate, seed, device, validation cadence | 200 steps, batch 8, AdamW, `1e-5`, seed 42, CPU, validation every 50 steps |

Start with Hydra config groups for context and covariates so presets can be combined without copying a whole run configuration. Keep training and backtest configs separate. Freeze each job's resolved configuration at launch, including values inherited from repo defaults. Reject unsupported presets, incompatible horizons, and missing required inputs before allocating the model. Keep `hydra.job.chdir=false` and resolve data and output paths from the repo root. Check that the selected Hydra release works with the repo's Python 3.14 environment before adding it. [Hydra config groups](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/) · [Hydra output directories](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) · [Python compatibility](https://hydra.cc/docs/intro/)

## 2. Run and save fine-tuning

**Working PowerShell command today**, from the repo root:

```powershell
$runId = "wind-3d-peft-" + (Get-Date -Format "yyyyMMdd-HHmmss")
.\.venv\Scripts\python.exe -m tuning.train_chronos2 --backend peft --run-id $runId --context-days 3 --device cpu --steps 200 --batch-size 8
```

**Proposed Hydra commands** after adding the `experiments.train` entry point:

```powershell
# Preview the resolved training config without starting a job.
.\.venv\Scripts\python.exe -m experiments.train context=3day covariates=none --cfg job --resolve

# Run one experiment.
.\.venv\Scripts\python.exe -m experiments.train context=3day covariates=none lora.rank=8 training.steps=200

# Sweep step counts. Each combination receives a distinct training run ID.
.\.venv\Scripts\python.exe -m experiments.train -m context=3day covariates=none 'training.steps=200,500,1000'
```

The worker builds training and validation windows, trains with Chronos's quantile loss, selects the best checkpoint on December validation, and verifies that the adapter reloads onto the recorded base revision. Print the run ID, resolved settings, checkpoint metrics, and final path in PowerShell. Keep the same metrics and log on disk so they can be inspected after the job exits. [Hydra multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)

Keep training records in the existing layout:

```text
results/finetuning/chronos2/<training-run-id>/
  .hydra/                    # Configuration and overrides saved by Hydra
  resolved_config.yaml       # Effective app settings with references resolved
  training.log               # Progress, validation metrics, and errors
  run.json                   # Run record consumed by the CLI and Streamlit
  finetuned-ckpt/            # Official PEFT config and adapter weights
```

Keep `run.json` as the app-facing manifest. Extend it with a schema version, unique run ID, status (`running`, `completed`, or `failed`), timestamps, code revision, data fingerprint, training context and covariates, checkpoint scores, and adapter path relative to the run directory. Record the exact forecast issue and availability rules for any weather inputs. Retain support for the older custom `adapter.safetensors` layout and the run records already on disk.

Give each Hydra job its own folder and pass that absolute directory to the worker. A Hydra folder may already contain config snapshots and logs when training starts. Guard against overwriting an existing training record or checkpoint. Write manifest updates atomically, and mark a run completed only after the adapter passes the load check. Show failed and interrupted runs in history, with only loadable completed runs available for forecasting. The on-disk manifests are enough for a first local run browser. For team access, use a shared artifact location because `results/` is Git-ignored.

## 3. Run rolling evaluation independently

Give every backtest its own evaluation ID and configuration. It references a training run ID and its adapter, or the base checkpoint for zero-shot. It also records **inference** context length or policy, exact inference covariates, horizon, evaluation dates, issue time, and metrics. Multiple backtests can reference the same adapter.

```powershell
# Proposed backtest entry point. This job compares the base and the selected adapter.
.\.venv\Scripts\python.exe -m experiments.backtest adapter_run_id=wind-3d-peft-20261119-122355 context=3day covariates=none evaluation=holdout_2025
```

1. Resolve the selected training run and its base revision. Load that exact base for zero-shot, and reuse `load_lora_run()` for the adapter variant.
2. Load each model once at startup. At each 09:00 UTC origin, supply the observed history that was available at that time and produce a 156-step forecast. Both models receive the same input and forecast dates in a paired comparison.
3. Write the predictions and scores in the existing backtest results format. Extend the metadata to include the training run ID, adapter identity, and resolved inference config. Include those settings in the cache key so a different adapter, context policy, weather preset, or date range cannot reuse the wrong result.

The first comparison uses the same 3-day wind-only inputs for both models. A later 7-day or weather-assisted backtest can still reference that same 3-day adapter. Train with January–November data, use December to choose settings, and use 2025 for the final holdout comparison once the settings are fixed.

## 4. Give the user a Streamlit interface

- **Launch tuning.** A form exposes validated presets and numeric parameters. On submit, launch the same Hydra entry point in a separate process with the project Python executable and structured arguments. Persist the run ID and log location so a page refresh can recover the run status. Show the log, current status, and final validation scores.
- **Browse tuning runs.** Read manifests from disk. Show context, covariates, rank, steps, seed, date split, backend, duration, and metrics. Let users compare runs with matching validation dates and see the exact resolved config for any row.
- **Run a rolling backtest.** The user selects zero-shot and a saved adapter, then separately sets the inference context and covariates. Show the training settings next to those selections so it is clear what changed. Send the evaluation to the same runner available in PowerShell and link the resulting forecast and score files to the selected adapter.

## 5. Implement and verify in order

1. Add the Hydra training entry point and presets. Run the existing 3-day configuration through a short smoke test and confirm it writes a resolved config, adapter, and manifest that can be reloaded.
2. Add the independent rolling backtest entry point. Verify on a short validation range that both models predict the same origins and that the same adapter can be reused under a different inference context. Check that the saved records and cache keys distinguish the two evaluations.
3. Add the Streamlit run browser, then connect the launch forms. Verify that a run started in PowerShell appears in the browser and that a job started in Streamlit remains traceable after a refresh. A full training run should be possible without opening Streamlit.

The first delivery is a reproducible 3-day wind-only PEFT run that can be launched from PowerShell or Streamlit, selected by its run ID, and evaluated beside zero-shot Chronos-2 without retraining.
