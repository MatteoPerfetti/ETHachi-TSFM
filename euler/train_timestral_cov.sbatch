#!/bin/bash
# Submit from the repository root after creating results/euler_logs.
# sbatch --time=03:00:00 euler/train_timestral_cov.sbatch tm-smoke 3 3 1e-4
#SBATCH --job-name=timestral-cov
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=results/euler_logs/timestral-cov_%j.out

set -euo pipefail

RUN_ID=${1:?usage: sbatch euler/train_timestral_cov.sbatch <run-id> [steps] [context-days] [lr] [batch] [rank] [seed]}
STEPS=${2:-300}
CONTEXT_DAYS=${3:-7}
LEARNING_RATE=${4:-1e-4}
BATCH_SIZE=${5:-8}
RANK=${6:-8}
SEED=${7:-42}
VALIDATION=${VALIDATION:-weeks}
EVAL_CONTEXT_STEPS=${EVAL_CONTEXT_STEPS:-2880}
# Early stopping: validate every VALIDATION_INTERVAL steps and stop after PATIENCE validations in a row without
# a D+1 RMSE decrease of at least MIN_DELTA MW (defaults: stop after 250 steps without a 1 MW gain; PATIENCE=0 = off).
# The best checkpoint is kept on any improvement.
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-50}
PATIENCE=${PATIENCE:-5}
MIN_DELTA=${MIN_DELTA:-1}

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1

echo "Training $RUN_ID: $CONTEXT_DAYS days, $STEPS steps on $(hostname), started $(date)"
.venv/bin/python -u -m tuning.train_timestral --run-id "$RUN_ID" --covariates config \
    --validation "$VALIDATION" --context-days "$CONTEXT_DAYS" --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" --rank "$RANK" --learning-rate "$LEARNING_RATE" --seed "$SEED" \
    --validation-interval "$VALIDATION_INTERVAL" --patience "$PATIENCE" --min-delta "$MIN_DELTA" --device cpu

echo "Evaluating $RUN_ID on 2025 with a $EVAL_CONTEXT_STEPS-step context, started $(date)"
.venv/bin/python -u evaluate_tsfm.py --model timestral_209m --adapter-run-id "$RUN_ID" --covariates config \
    --context-steps "$EVAL_CONTEXT_STEPS" --device cpu
echo "Finished $(date)"
