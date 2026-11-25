#!/bin/bash
# Wave 2: Timestral covariate LoRA (context, validation and learning-rate schedule) plus Chronos-2 at 30 days.
# All runs: 500 steps, batch 8, rank 8, seed 42, 7 covariates, then the full-2025 evaluation at a 30-day context.
# Early stopping on: validate every 50 steps, stop after 5 validations without a 0.4% improvement
# (D+1 RMSE with weeks validation, pinball loss with December); the best checkpoint is kept.
#
# Run from the repository root on the Euler login node:  bash euler/submit_wave2.sh
set -euo pipefail
mkdir -p results/euler_logs

# 1. Timestral, 3-day training context, lr 1e-4, weeks validation
VALIDATION=weeks sbatch --job-name=tm-w-c3 euler/train_timestral_cov.sbatch tm-w-c3-lr1e-4-s500 500 3 1e-4
# 2. Timestral, 30-day training context, lr 1e-4, weeks validation
VALIDATION=weeks sbatch --job-name=tm-w-c30 euler/train_timestral_cov.sbatch tm-w-c30-lr1e-4-s500 500 30 1e-4
# 3. Chronos-2, 30-day training context, lr 1e-4, weeks validation
VALIDATION=weeks sbatch --job-name=ch-w-c30 euler/train_chronos2_cov.sbatch ch-w-c30-lr1e-4-s500 500 30 1e-4
# 4. Timestral, 3-day training context, lr 1e-4, December validation
VALIDATION=december sbatch --job-name=tm-d-c3 euler/train_timestral_cov.sbatch tm-d-c3-lr1e-4-s500 500 3 1e-4
# 5. Timestral, 3-day training context, lr 1e-3 decreasing quadratically to 1e-5, weeks validation
VALIDATION=weeks SCHEDULER=quadratic FINAL_LR=1e-5 sbatch --job-name=tm-w-c3-q \
    euler/train_timestral_cov.sbatch tm-w-c3-lrq1e-3to1e-5-s500 500 3 1e-3

squeue -u "$USER"
