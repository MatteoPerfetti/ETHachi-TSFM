#!/bin/bash
# Context-length sweep on Euler: one Slurm array task per context length, all running in parallel.
#
#   mkdir -p results/euler_logs
#   sbatch euler/context_sweep.sbatch                    # every context in DAYS below
#   sbatch --array=0-1 euler/context_sweep.sbatch        # only the first two (3 and 7 days)
#   sbatch --array=4 euler/context_sweep.sbatch          # only 60 days
#   sbatch euler/context_sweep.sbatch --every 7          # extra options go to context_sweep.py
#
# Public share limit: 48 cores and 128 GiB RAM in total over all running jobs.
# 5 tasks x 8 cores x 24 GiB = 40 cores, 120 GiB: all five fit at once. If a long context runs out of memory,
# raise --mem-per-cpu for that task (e.g. sbatch --array=4 --mem-per-cpu=6G ...).
#SBATCH --job-name=ctx-sweep
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --output=results/euler_logs/ctx-sweep_%A_%a.out

set -euo pipefail

DAYS=(3 7 14 30 60)                      # context lengths in days, indexed by the array task id
D=${DAYS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"                   # submit from the repository root
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # torch uses the allocated cores only
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1                  # weights were downloaded on the login node (see euler/README.md)

echo "Task $SLURM_ARRAY_TASK_ID: context $D days on $(hostname), $SLURM_CPUS_PER_TASK cores, started $(date)"
.venv/bin/python context_sweep.py --days "$D" --device cpu --reuse "$@"
echo "Finished $(date)"
