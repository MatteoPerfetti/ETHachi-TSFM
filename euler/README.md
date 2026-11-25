# Running the context sweep on Euler

How to run `context_sweep.py` on the ETH Euler cluster. Written for a **guest account on the public share**:
48 CPU cores and 128 GiB RAM in total, **no GPUs** (check yours with `my_share_info`). Official documentation:
https://docs.hpc.ethz.ch

Each context length runs as its own Slurm array task, so all contexts run in parallel on the cluster's CPUs
instead of one after the other on a laptop.

## 1. One-time setup (on the login node)

The login node is for setup and submitting jobs only; never run the models there.

```bash
ssh <username>@euler.ethz.ch                  # from outside the ETH network: VPN first

# uv (installs into ~/.local/bin)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# code: keep everything in $HOME (50 GB, backed up); /cluster/scratch is purged after ~2 weeks
git clone https://github.com/MatteoPerfetti/ETHachi-TSFM.git
cd ETHachi-TSFM
git checkout experiment-context-lenght

# Python 3.14 + all packages from uv.lock (uv downloads Python itself; several GB because of torch)
uv sync
```

Copy the cached data from your PC (Elia and Open-Meteo parquet files are not in git; the DWD file is).
In PowerShell, from the project folder on the PC:

```powershell
scp -r data/wind data/weather data/solar <username>@euler.ethz.ch:~/ETHachi-TSFM/data/
```

Download the model weights once on the login node (about 2.5 GB in `~/.cache/huggingface`). Compute nodes
then load them offline (`HF_HUB_OFFLINE=1` in the job script), so the jobs do not depend on internet access:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download as get; [get(r) for r in ('amazon/chronos-2', 'mistralai/Timestral-209M', 'google/timesfm-3.0-pytorch')]"
```

## 2. Quick test (about 10 minutes)

A short interactive job on a compute node: 3-day context, one forecast every 30 days.

```bash
srun --cpus-per-task=8 --mem-per-cpu=3G --time=00:45:00 \
     env OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 .venv/bin/python context_sweep.py --days 3 --every 30
```

It should end with the sweep table. Delete the `_every30d` test runs afterwards if you like
(`rm results/*_every30d.*`).

## 3. The sweep

```bash
mkdir -p results/euler_logs
sbatch euler/context_sweep.sbatch                # 3, 7, 14, 30, 60 days, in parallel
```

Other selections, see the comments in [context_sweep.sbatch](context_sweep.sbatch):
`sbatch --array=0-1 ...` (3 and 7 days only), `sbatch --array=4 ...` (60 days only), and any extra option is passed
to `context_sweep.py` (e.g. `--every 7` for a quick pass). `--reuse` is always on, so a re-submitted task only
runs what is not saved yet.

Monitoring:

```bash
squeue -u $USER                                  # pending (PD) / running (R)
myjobs -j <job_id>                               # details of one job
tail -f results/euler_logs/ctx-sweep_<job_id>_<task>.out
scancel <job_id>                                 # cancel the whole array (or <job_id>_<task> for one task)
```

Rough times (CPU, full 2025, three TSFMs; extrapolated from runs on a laptop, so only indicative):
3 days 10–30 min, 7 days 20–60 min, 14 days 35–95 min, 30 days 1–2.7 h, 60 days 1.5–4.5 h.
The array tasks run at the same time, so the sweep takes about as long as its longest context.

## 4. Table and results

When every task is finished, build the comparison table (light work, fine on the login node):

```bash
.venv/bin/python context_sweep.py --days 3 7 14 30 60 --table-only
```

It prints the report and writes `results/context_sweep/sweep_3d-7d-14d-30d-60d.csv` (+ `.txt`).
To look at the runs in the dashboard on the PC, copy them back (PowerShell, from the project folder):

```powershell
scp "<username>@euler.ethz.ch:~/ETHachi-TSFM/results/wind_ctx*" results/
scp -r <username>@euler.ethz.ch:~/ETHachi-TSFM/results/context_sweep results/
```

## 5. Chronos-2 LoRA with covariates

[train_chronos2_cov.sbatch](train_chronos2_cov.sbatch) trains one Chronos-2 adapter with the weather covariates and
then evaluates it on 2025 against zero-shot Chronos-2, both with a 30-day context and the covariates.

From PowerShell on the PC (`ssh` and `scp` are built into Windows):

```powershell
ssh <username>@euler.ethz.ch
```

Then on Euler:

```bash
cd ~/ETHachi-TSFM
git fetch && git checkout Chronos-finetuning && git pull
mkdir -p results/euler_logs
sbatch --time=02:00:00 euler/train_chronos2_cov.sbatch cov30-smoke 3 30   # first: time a few steps
sbatch euler/train_chronos2_cov.sbatch cov30-r8 200 30                    # main run, 30-day training context
sbatch euler/train_chronos2_cov.sbatch cov3-r8 200 3                      # comparison: 3-day training context
squeue -u $USER
tail -f results/euler_logs/chronos2-cov_<job_id>.out
exit
```

Back in PowerShell, from the project folder: copy the adapter and its evaluation, then add them to the local MLflow
(the importer skips runs it already has):

```powershell
scp -r "<username>@euler.ethz.ch:~/ETHachi-TSFM/results/finetuning/chronos2/cov30-r8" results\finetuning\chronos2\
scp "<username>@euler.ethz.ch:~/ETHachi-TSFM/results/wind_ctx2880_*" results\
.\.venv\Scripts\python.exe import_experiments.py
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///results/mlflow.db --port 5000
```

## Troubleshooting

- **Out of memory** (`oom-kill` in the log, state `OUT_OF_MEMORY` in `myjobs`): re-submit that task with more memory,
  e.g. `sbatch --array=4 --mem-per-cpu=6G euler/context_sweep.sbatch`.
- **Time limit**: re-submit with a longer `--time`; models already saved for that context are skipped (`--reuse`).
- **A model is downloaded / network error in the job**: the weights are not in the cache, repeat the download step.
- **Data is downloaded in the job**: the parquet cache in `data/` is missing, repeat the `scp` step.
