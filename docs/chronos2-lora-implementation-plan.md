# Chronos-2 LoRA implementation plan

See the [tuning and backtest architecture](tuning-architecture.md) for Hydra configuration, run tracking, and how PowerShell and Streamlit will launch the jobs.

## First run: fixed three-day history

The first experiment uses only a **3-day context (288 steps)** and the unchanged **39-hour horizon (156 steps)**. The 1-day, 7-day, and 14-day settings and retrieval experiments come later. Run this phase directly from the project root.

```powershell
python -m tuning.train_chronos2 --run-id wind-3d-smoke --context-days 3 --device cpu --steps 3 --batch-size 1
python -m tuning.train_chronos2 --run-id wind-3d-r8-seed42 --context-days 3 --device cpu --steps 200 --batch-size 8
```

The original run used Chronos-2's native model and quantile loss with custom PyTorch LoRA matrices. To use Chronos-2's official `fit()` method with PEFT on the same split and horizon, add `--backend peft` and use a new run ID:

```powershell
python -m tuning.train_chronos2 --backend peft --run-id wind-3d-peft-r8-seed42 --context-days 3 --device cpu --steps 200 --batch-size 8
```

Both paths read the existing wind Parquet and the same cached base revision. They save validation scores in `results/finetuning/chronos2/<run-id>/run.json`. The custom path saves `adapter.safetensors`; the official path saves the standard PEFT files in `finetuned-ckpt/`. `load_lora_run()` reloads either format onto the recorded base revision. The official path passed a three-step smoke test. The full 2025 backtest integration is a follow-up.

The sections below describe an optional extension to mixed contexts, dedicated backtest integration, and retrieval experiments. The `main.py` flags below are still planned.

## Goal

Train `amazon/chronos-2` on the existing Belgian wind generation history, save a LoRA adapter, and measure its effect using ETHachi-TSFM's rolling backtest. Training and evaluation are separate commands. Training loads only Chronos-2. Evaluation loads only zero-shot Chronos-2 and the saved Chronos-2 LoRA variant, with Persistence available as an optional reference.

Train one adapter using a reference distribution of contiguous histories. The team owns the context policy used at inference. Each policy can be compared with the same frozen adapter and zero-shot checkpoint. The training distribution describes the windows the adapter saw. It does not decide which history the backtest supplies.

This plan is based on `main` at `dee35ad`. The commands below describe the interface to implement.

## 1. Command line workflow

From the project root, use PowerShell with the project environment active.

```powershell
# Train a small adapter on CPU to check the full training and save path.
python -m tuning.train_chronos2 --run-id smoke --device cpu --steps 3 --batch-size 1

# Check that the saved adapter reloads and forecasts on a short December validation range.
python main.py --chronos-lora-checkpoint results/finetuning/chronos2/smoke/finetuned-ckpt --test-start 2024-12-10 --test-end 2024-12-14

# Train the first candidate on a CUDA machine.
python -m tuning.train_chronos2 --run-id wind-r8-seed42 --device cuda --steps 200 --batch-size 8

# Compare both model variants with one contiguous context setting on December.
python main.py --chronos-lora-checkpoint results/finetuning/chronos2/wind-r8-seed42/finetuned-ckpt --context-steps 672 --test-start 2024-12-01 --test-end 2024-12-30

# Evaluate the selected context policy and the same saved adapter on 2025.
python main.py --chronos-lora-checkpoint results/finetuning/chronos2/wind-r8-seed42/finetuned-ckpt --context-steps 672
```

`tuning/train_chronos2.py` takes a unique `--run-id`, `--device auto|cpu|cuda`, `--steps`, `--batch-size`, `--rank`, `--learning-rate`, and `--seed`. Start with rank 8, learning rate `1e-5`, seed 42, 200 steps, and batch size 8. Treat these as initial settings to validate. In Chronos, batch size counts variates across tasks. With a single target and zero covariates, this is the number of windows per batch.

`main.py --chronos-lora-checkpoint` selects the explicit model list `chronos2`, `chronos2_lora`, and optionally `persistence` via `--include-persistence`. It uses `covariates=[]` for both Chronos variants. `--context-steps` controls the existing contiguous-history policy at evaluation and defaults to `config.BACKTEST_CONTEXT_STEPS`. The values in the commands above are examples. In this mode, date overrides select forecast issue times in UTC, with the end exclusive. The default `python main.py` and its existing `--reuse` workflow continue to use `config.BACKTEST_MODELS`. In checkpoint mode, `--reuse` looks up the matching comparison and rebuilds it if it is missing.

## 2. Data and chronological split

Reuse `src.dataset.load_dataset("wind")`. It returns the regional wind totals aggregated in MW on a 15-minute UTC grid. Read the cached Parquet files already used by the project. Construct reference windows as `float32` arrays. Chronos handles batch sampling, normalization, patching, and the quantile loss.

Use this first experiment to isolate the impact of LoRA. Both training and evaluation use only wind generation. The seven weather features can be evaluated separately after confirming the availability of forecast values at each issue time. The local DWD table has valid timestamps without a separate forecast issue time.

| Partition | Use |
|---|---|
| `config.TRAIN_START` through November 2024 | Build complete windows within this slice, with a 156-step horizon and a reference mix of 96, 288, 672, and 1,344 contiguous history steps. Use the same eligible training origins for each length to give the native sampler equal representation in expectation. |
| December 2024 before `config.TRAIN_END` | Build a fixed validation set with one window per reference length for each common daily `09:00 UTC` origin. Every label must fall inside the December holdout. Context can reach into late November. Use this reference set for LoRA checkpoint selection. |
| `config.TEST_START` through `config.TEST_END` | Run the repo's usual rolling backtest on the untouched 2025 interval. Both models receive the same requests and are scored on the same timestamps. |

Chronos 2.3.2 scores only the last `prediction_length` steps of each validation input. Build the validation inputs from the repo's daily issue schedule and reuse its context and forecast index logic. Validate the UTC grid, data coverage, and split boundaries before training. If a reference validation window has missing data, record and skip that origin across all reference lengths. If the training slice contains gaps, preserve their timestamps and fail with an actionable message for the first implementation. The local check found continuous wind targets in the proposed training period.

Keep the reference lengths in training configuration, independent of `BACKTEST_CONTEXT_STEPS` and the team's context policy. If the team chooses to run the shared-adapter experiment, record the 1-, 3-, 7-, and 14-day reference mix with that checkpoint. Read horizon and issue time from the existing backtest configuration. The current experiment uses zero latency. If the configuration changes to a nonzero latency, fail with a message until training explicitly handles the latency gap. `ForecastRequest.model_horizon` includes that gap, while the reported horizon begins at the issue time.

## 3. Implement standalone training

1. Add `tuning/train_chronos2.py` in the tuning package. Keep its imports to the dataset, shared forecast/request helpers, and Chronos training dependencies. The trainer should never call `run_backtest()` or read `BACKTEST_MODELS`.
2. Add a `chronos-lora` project extra in `pyproject.toml` and update `uv.lock` so `uv sync --extra chronos-lora` installs the tuning dependencies. Use the PEFT compatibility range declared by the installed Chronos release as the starting point (`peft>=0.18.1,<1`) and verify it with the resolved Transformers version. `shapely` is already declared by the project, so syncing the project also restores it if missing locally. Check that PEFT imports successfully before creating a training run. Chronos 2.3.2 otherwise falls back to full fine-tuning when PEFT is missing.
3. Load the base checkpoint from `config.MODELS_REGISTRY["chronos2"]["checkpoint"]` with `Chronos2Pipeline.from_pretrained()`. Record the resolved base revision. The local cache currently has `29ec3766d36d6f73f0696f85560a422f50e8498c`. Preflight the requested device and report the selected device. A requested unavailable `cuda` should fail before loading the weights.
4. Call native `fit()` with the prebuilt training and validation windows, `finetune_mode="lora"`, and `prediction_length=156`. For the installed Chronos sampler, left-pad each context with `NaN` to the largest reference length and append the 156 labels. Set both `min_past` and `context_length` to that largest length. Each array then has a single possible split point, and Chronos's observation mask preserves the actual number of observed history steps. Check all four observed lengths with a dataset batch before running the optimizer. Pass an explicit LoRA config with the requested rank, `lora_alpha=2*rank`, and Chronos's default projection targets (`self_attention.q`, `.k`, `.v`, `.o`, and `output_patch_embedding.output_layer`). Set seed, logging cadence, and matching save/evaluation steps through the trainer arguments. Use a short cadence for the smoke test and checkpoint selection by validation loss for the full run. On CPU use float32 and `adamw_torch` to keep the smoke test independent of fused CUDA kernels.
5. Use `results/finetuning/chronos2/<run-id>/` as the trainer output directory. Reject an existing run ID so the previous checkpoint remains reproducible. Verify the final `finetuned-ckpt/` contains an adapter config and adapter weights. Record the base ID and revision, target, split boundaries, validation origins, reference lengths and window counts, horizon, issue time, latency, seed, LoRA settings, device, package versions, and training runtime in `run.json`. Write completion metadata after the adapter passes the save/reload check so an interrupted run cannot be mistaken for a complete one.

`Chronos2Pipeline.fit()` returns the tuned pipeline and saves it via `save_pretrained()`. Check that the reloaded adapter uses the recorded base revision and that its predictions agree with the in-memory tuned pipeline on the same validation requests. If the native adapter loader resolves a different revision, load the recorded base explicitly and apply the saved adapter to it. Also use that same base revision for the zero-shot side of the comparison.

## 4. Hook the saved adapter into the backtest

| File | Implementation |
|---|---|
| `src/models_tsfm.py` | Allow `Chronos2Adapter` to take an explicit checkpoint and display name. Use `Chronos2Pipeline.from_pretrained()` so a local PEFT checkpoint takes the adapter-aware load path. Name the two results `Chronos-2` and `Chronos-2 LoRA`. Both reuse `_predict_batch()` and `ForecastResult`. A loaded LoRA checkpoint inherits the no-op `fit()` used for inference. Move Timestral and TimesFM imports into their respective `_load()` methods to keep those libraries out of the Chronos path. |
| `src/backtest.py` | Extend `BacktestSettings` with an optional `chronos2_lora_checkpoint` field and handle the `chronos2_lora` key explicitly in `load_forecaster()`. Use the same `build_request()`, common valid origins, timing, and output schema. Move `load_traditional` into its branch so Chronos-only comparison does not import the traditional model implementations. Store the base revision and adapter/run details in the existing `model_configs` metadata. |
| `main.py` | Add the opt-in checkpoint mode, pass its explicit keys and settings to `run_backtest()`, and reuse `save_backtest()`, `print_summary()`, and the existing plotter. Check that the checkpoint exists and verify its base revision, target, forecast horizon, and latency. Evaluation context and dates are independent settings. Inform the evaluator when the context is outside the reference training range, and validate it against Chronos's actual supported limits. |
| `README.md` | Document environment setup, the train and evaluate commands, and where to find the adapter, predictions, and metrics. |

Keep `chronos2_lora` out of the default benchmark list and the dashboard's model picker in the first implementation. The dashboard can already open saved runs through `saved_runs()`.

The context policy belongs in backtest request construction. For contiguous histories, use the existing `BacktestSettings.context_steps` and `build_request()`. When the team adds a new policy, use it to build requests for both models identically. The current `ForecastRequest` expects a regular 15-minute UTC context with no missing target values. If a policy skips timestamps or masks values, agree on the request contract and adapter support first. Concatenating disjoint intervals would imply consecutive readings to Chronos.

`BacktestSettings.run_name` currently depends on target, context, horizon, latency, and covariates. For this comparison mode, also distinguish the adapter run, test date range, issue time, and optional Persistence selection. Include the policy identifier and settings when policies beyond contiguous history are added. This keeps different policies and checkpoints from sharing a cached result. Keep the usual paired Parquet and JSON files directly under `results/` so the existing dashboard finds them. Store model weights and training metadata in the nested training directory.

## 5. Test retrieval independently of LoRA

Keep the reference LoRA training described above while the team develops a retrieval policy. Save one selected adapter and keep it fixed during retrieval experiments. A RAFT-style method searches for a historical pattern similar to the recent query and retrieves the observations that followed that historical pattern. The published RAFT system fuses these values with the query using its own forecasting layers. The first repo experiment can use the retrieval idea while preserving the current Chronos training interface. [RAFT paper](https://proceedings.mlr.press/v267/han25d.html)

1. **Build a shared archive.** Use January–November 2024 and store each contiguous lookback with its observed 156-step continuation. Require the full continuation to end before the archive cutoff and to be available at the current issue time. Rank candidates using only information available at that issue time. Use the same archive, candidates, and retrieval settings for both model variants. A fixed archive gives December and 2025 the same retrieval protocol. Record source timestamps, similarity scores, and candidate IDs.
2. **Start with prediction blending.** Keep the recent contiguous `ForecastRequest` for Chronos. In parallel, build an analog forecast from the continuation of the top matches, aligned to the current origin and level using only observed history. Use an analog-only forecast as a diagnostic, then blend the analog forecast with each Chronos forecast using the same validation-selected weight. For probabilistic evaluation, blend corresponding quantiles at each horizon using a common weight across quantiles and check calibration. This defines a quantile blend rather than the quantiles of a mixture distribution. This option fits after the existing Chronos prediction and before scoring or saving the result. The forecast output keeps the existing `q10`, `q50`, and `q90` columns. [RAFT retrieval method](https://arxiv.org/html/2505.04163#S3)
3. **Test native conditioning if retrieval adds value.** Chronos-2 can take the aligned analog history and its archived continuation as an auxiliary channel via its past and known-future covariates interface. Those future channel values are derived from an already observed historical analog. They are distinct from the current query's unknown targets. Pass the same auxiliary channels to both checkpoints. Keep the query's timeline intact and keep each query separate from other forecast origins. This changes the inputs seen by Chronos, so assess both checkpoints on December before changing any training. [Chronos-2 architecture](https://arxiv.org/html/2510.15821#S3)
4. **Treat retrieval-aware LoRA as a follow-up.** If native conditioning helps but the reference LoRA adapter has trouble using the analog channels, train a separate adapter on the same retrieval-augmented input format used at inference. For each training query, build its retrieval examples from older data whose continuations were fully available at that query's origin. Consider mixing examples with and without retrieval if the adapter should support both modes. Retain the independent adapter for comparison. [RAF study of frozen and retrieval-aware fine-tuning](https://proceedings.mlr.press/v300/tire26a.html)

Compare four cells on common forecast origins and the same recent lookback: zero-shot and LoRA, each with and without retrieval. Add analog-only as a diagnostic. Freeze the checkpoint, archive protocol, and retrieval settings selected on December before scoring on 2025. The paired comparisons show whether retrieval helps either model independently and whether the two effects combine.

## 6. Verify and evaluate

1. **Preflight.** Verify the dataset loader, PEFT import, cached base checkpoint, and device selection. The checked local `.venv` has CPU-only PyTorch, lacks PEFT and `shapely`, and has Accelerate installed. Use that machine for the smoke test after repairing the environment. Use a CUDA-enabled PyTorch environment for the longer run once available.
2. **Smoke test.** Run three optimizer steps, verify finite training and validation loss, confirm LoRA trainable parameters were used, and reload the saved adapter. On a few December origins, verify that both Chronos variants produce the same number of rows and identical `(issue_time, timestamp, lead_step)` keys with finite `q10`, `q50`, and `q90`. Confirm that no other forecast models were loaded or run.
3. **Check the integration.** Verify the default benchmark path still resolves its configured models. Reload one saved adapter at two context lengths and verify both use the same artifact. Check that comparison mode writes its own Parquet/JSON pair and that two context settings, checkpoints, or date ranges resolve to different cache names. Check that an invalid adapter path is reported clearly.
4. **Run and compare.** Select the rank-8 checkpoint on the fixed December reference validation set, then freeze it before comparing context policies. For each candidate policy, evaluate the same zero-shot and LoRA checkpoints using identical inputs. Compare policies on common December forecast origins. Report the LoRA gain within each policy and the policy effect within each model variant. The differences can show that a policy helps one model more than the other. Run the 2025 holdout once the policy and checkpoint are fixed. Report the repo's daily average **RMSE** on `q50` and **Winkler80** on `[q10, q90]` for both `all` and `d+1`. Include paired differences, issue count, and training time. A lower RMSE with a higher Winkler80 is a trade-off to report.

The first milestone is a saved adapter that can be trained and evaluated with the two commands above, with baseline and tuned forecasts available in the existing results format. Improving 2025 scores is the experiment's objective and is evaluated after the plumbing passes.
