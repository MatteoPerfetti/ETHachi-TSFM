#!/bin/bash
# Chronos-2 LoRA with the weather covariates on Euler, then the full-2025 rolling evaluation of the adapter
# against zero-shot Chronos-2 with the same context and covariates.
#
#   mkdir -p results/euler_logs
#   sbatch euler/train_chronos2_cov.sbatch <run-id> [steps] [context-days]
#   sbatch euler/train_chronos2_cov.sbatch cov30-r8 200 30          # main run: 30 days, 200 steps
#   sbatch euler/train_chronos2_cov.sbatch cov3-r8 200 3            # comparison: same, trained on 3 days
#   sbatch --time=01:00:00 euler/train_chronos2_cov.sbatch cov30-smoke 3 30   # smoke run to time the steps
#
# The evaluation always uses a 30-day inference context (2880 steps), so adapters trained on 3 and 30 days are
# compared on the same forecasts. Training and evaluation are recorded in results/ and in results/mlflow.db.
#SBATCH --job-name=chronos2-cov
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=results/euler_logs/chronos2-cov_%j.out

set -euo pipefail

RUN_ID=${1:?usage: sbatch euler/train_chronos2_cov.sbatch <run-id> [steps] [context-days]}
STEPS=${2:-200}
CONTEXT_DAYS=${3:-30}
EVAL_CONTEXT_STEPS=2880

cd "$SLURM_SUBMIT_DIR"                   # submit from the repository root
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # torch uses the allocated cores only
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1                  # weights were downloaded on the login node (see euler/README.md)

echo "Training $RUN_ID: $CONTEXT_DAYS days, $STEPS steps on $(hostname), started $(date)"
.venv/bin/python -u -m tuning.train_chronos2 --run-id "$RUN_ID" --backend peft --covariates config \
    --context-days "$CONTEXT_DAYS" --steps "$STEPS" --batch-size 8 --rank 8 --learning-rate 1e-5 --device cpu

echo "Evaluating $RUN_ID on 2025 with a $EVAL_CONTEXT_STEPS-step context, started $(date)"
.venv/bin/python -u evaluate_tsfm.py --model chronos2 --adapter-run-id "$RUN_ID" --covariates config \
    --context-steps "$EVAL_CONTEXT_STEPS" --device cpu
echo "Finished $(date)"
