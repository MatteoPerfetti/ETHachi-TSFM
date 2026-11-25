#!/bin/bash
# Chronos-2 LoRA with the weather covariates on Euler, then the full-2025 rolling evaluation of the adapter
# against zero-shot Chronos-2 with the same context and covariates.
#
#   mkdir -p results/euler_logs
#   sbatch euler/train_chronos2_cov.sbatch <run-id> [steps] [context-days] [learning-rate] [batch-size] [rank] [seed]
#   sbatch euler/train_chronos2_cov.sbatch cov30-r8 200 30          # main run: 30 days, 200 steps
#   sbatch euler/train_chronos2_cov.sbatch cov3-r8 200 3            # comparison: same, trained on 3 days
#   sbatch euler/train_chronos2_cov.sbatch c3-lr1e-4-s1000 1000 3 1e-4   # 3 days, 1000 steps, learning rate 1e-4
#   sbatch --time=01:00:00 euler/train_chronos2_cov.sbatch cov30-smoke 3 30   # smoke run to time the steps
#
# Defaults: learning rate 1e-5, batch size 8 (training windows per step), rank 8, seed 42.
# VALIDATION=weeks (exported) validates on days 1-7 of every month of 2024 with a 30-day context and keeps the
# checkpoint with the best D+1 RMSE; the default VALIDATION=december keeps the original December validation:
#   VALIDATION=weeks sbatch euler/train_chronos2_cov.sbatch w-c3-lr1e-4-s500 500 3 1e-4
# The evaluation uses a 30-day inference context (2880 steps) unless EVAL_CONTEXT_STEPS is exported, so adapters
# trained on different contexts are compared on the same forecasts. Training and evaluation are recorded in
# results/ and in results/mlflow.db.
#SBATCH --job-name=chronos2-cov
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=results/euler_logs/chronos2-cov_%j.out

set -euo pipefail

RUN_ID=${1:?usage: sbatch euler/train_chronos2_cov.sbatch <run-id> [steps] [context-days]}
STEPS=${2:-200}
CONTEXT_DAYS=${3:-30}
LEARNING_RATE=${4:-1e-5}
BATCH_SIZE=${5:-8}
RANK=${6:-8}
SEED=${7:-42}
EVAL_CONTEXT_STEPS=${EVAL_CONTEXT_STEPS:-2880}
VALIDATION=${VALIDATION:-december}
# Early stopping (weeks validation only): validate every VALIDATION_INTERVAL steps and stop after PATIENCE
# validations in a row without a relative D+1 RMSE improvement of MIN_IMPROVEMENT (default: 250 steps without a
# 0.4% gain; PATIENCE=0 = off). The best checkpoint is kept on any improvement.
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-50}
PATIENCE=${PATIENCE:-5}
MIN_IMPROVEMENT=${MIN_IMPROVEMENT:-0.004}
EARLY_STOP=()
if [ "$VALIDATION" = "weeks" ]; then
    EARLY_STOP=(--validation-interval "$VALIDATION_INTERVAL" --patience "$PATIENCE" --min-improvement "$MIN_IMPROVEMENT")
fi

cd "$SLURM_SUBMIT_DIR"                   # submit from the repository root
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # torch uses the allocated cores only
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1                  # weights were downloaded on the login node (see euler/README.md)

echo "Training $RUN_ID: $CONTEXT_DAYS days, $STEPS steps, lr $LEARNING_RATE, batch $BATCH_SIZE, rank $RANK," \
     "seed $SEED, validation $VALIDATION on $(hostname), started $(date)"
.venv/bin/python -u -m tuning.train_chronos2 --run-id "$RUN_ID" --backend peft --covariates config \
    --context-days "$CONTEXT_DAYS" --steps "$STEPS" --batch-size "$BATCH_SIZE" --rank "$RANK" \
    --learning-rate "$LEARNING_RATE" --seed "$SEED" --validation "$VALIDATION" ${EARLY_STOP[@]+"${EARLY_STOP[@]}"} --device cpu

echo "Evaluating $RUN_ID on 2025 with a $EVAL_CONTEXT_STEPS-step context, started $(date)"
.venv/bin/python -u evaluate_tsfm.py --model chronos2 --adapter-run-id "$RUN_ID" --covariates config \
    --context-steps "$EVAL_CONTEXT_STEPS" --device cpu
echo "Finished $(date)"
