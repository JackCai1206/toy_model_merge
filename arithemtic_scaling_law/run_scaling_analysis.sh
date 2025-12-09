#!/usr/bin/env bash
# Aggregate results across all seeds in a run group. Run this after array jobs finish:
#   MODE=full RUN_GROUP=... Q_KEEP=... MAX_STEPS_PER_BLOCK=... sbatch arithemtic_scaling_law/run_scaling_analysis.sh

#SBATCH --job-name=arith-scale-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=arithemtic_scaling_law/logs/analyze_%A.out
#SBATCH --error=arithemtic_scaling_law/logs/analyze_%A.err

set -euo pipefail

REPO_DIR="/scratch/gpfs/ARORA/zc5794/toy_model_merge"
cd "${REPO_DIR}"

source "${REPO_DIR}/.venv/bin/activate"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

MODE="${MODE:-full}"
TAG="${TAG:-v2}"
ACC_TARGET="${ACC_TARGET:-0.95}"
K_MIN="${K_MIN:-1}"
K_MAX="${K_MAX:-33}"
Q_KEEP=${Q_KEEP:-0.8}
MAX_STEPS_PER_BLOCK=${MAX_STEPS_PER_BLOCK:-2}
METRIC=${METRIC:-eval_acc_expr}
REGIME_SLUG="${REGIME_SLUG:-q${Q_KEEP//./}_b${MAX_STEPS_PER_BLOCK}}"
K_SUFFIX="${K_SUFFIX:-}"
K_SUFFIX="kmin${K_MIN}_kmax${K_MAX}"
RUN_GROUP=${RUN_GROUP:-"run_${MODE}_${REGIME_SLUG}_t${ACC_TARGET}_${K_SUFFIX}_${TAG}"}
RESULTS_ROOT="${REPO_DIR}/arithemtic_scaling_law/results"

PLOTS_DIR="${RESULTS_ROOT}/${RUN_GROUP}/plots"
mkdir -p "${PLOTS_DIR}"
echo "[$(date)] Aggregating ${RUN_GROUP} (metric=${METRIC}; eval-k on x-axis, per-run curves)"
python -u arithemtic_scaling_law/analyze_scaling_results.py \
  --results_parent "${RESULTS_ROOT}/${RUN_GROUP}" \
  --regime "${REGIME_SLUG}" \
  --metric "${METRIC}" \
  --output_dir "${PLOTS_DIR}" \
  --aggregate_levels
