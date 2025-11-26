#!/usr/bin/env bash
# Submit with e.g.:
#   MODE=sanity sbatch arithemtic_scaling_law/run_scaling_experiment.sh
#   sbatch arithemtic_scaling_law/run_scaling_experiment.sh                 # full run defaults
# MODE=sanity dials down dataset sizes/steps for a quick GPU smoke test. Override any
# variable below via env (e.g., TRAIN_EXAMPLES_PER_LEVEL=1000).

#SBATCH --job-name=arith-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli-c
#SBATCH --output=arithemtic_scaling_law/logs/scale_%A.out
#SBATCH --error=arithemtic_scaling_law/logs/scale_%A.err

# salloc --job-name=arith-scale --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=24:00:00 --gres=gpu:1 --partition=pli-c

set -euo pipefail

REPO_DIR="/scratch/gpfs/ARORA/zc5794/toy_model_merge"
cd "${REPO_DIR}"

source "${REPO_DIR}/.venv/bin/activate"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Keep logs clean.
# export TQDM_DISABLE=1
# export DISABLE_PROGRESS_BAR=1
# export DATASETS_DISABLE_PROGRESS_BAR=1
# export HF_HUB_DISABLE_PROGRESS_BARS=1

MODE="full"
SEED=123

# Full-scale defaults match run_cot_scaling_experiment.py arguments.
K_MAX=6
TRAIN_EXAMPLES_PER_LEVEL=20000
EVAL_EXAMPLES_PER_LEVEL=2000
DATA_DIR="${REPO_DIR}/arithemtic_scaling_law/data"
ARTIFACTS_DIR="${REPO_DIR}/arithemtic_scaling_law/artifacts"
RESULTS_DIR="${REPO_DIR}/arithemtic_scaling_law/results"
CONTEXT_LENGTH=1024
PER_DEVICE_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=1024
GRAD_ACCUM=1
EVAL_STEPS=2000
EVAL_REFINE_ROUNDS=5
ROLLBACK_BRANCHES=1
GREEDY_EVAL_BATCH_SIZE=1024
GREEDY_EVAL_MAX_NEW_TOKENS=1024
GREEDY_EVAL_MATCH_TARGET_LENGTH=1
MAX_STEPS=200000
ACC_TARGET=0.9
Q_KEEP=${Q_KEEP:-1.0}
MAX_BLOCK_SIZE=${MAX_BLOCK_SIZE:-1}
FORCE_REGEN=0
## Override Q_KEEP/MAX_BLOCK_SIZE to control the dataset regime; FORCE_REGEN=1 forces dataset regeneration.

if [[ "${MODE}" == "sanity" ]]; then
  # Smaller settings for a fast sanity sweep.
  K_MAX=${K_MAX_SANITY:-3}
  TRAIN_EXAMPLES_PER_LEVEL=${TRAIN_EXAMPLES_PER_LEVEL_SANITY:-2000}
  EVAL_EXAMPLES_PER_LEVEL=${EVAL_EXAMPLES_PER_LEVEL_SANITY:-200}
  CONTEXT_LENGTH=${CONTEXT_LENGTH_SANITY:-256}
  MAX_STEPS=${MAX_STEPS_SANITY:-2000}
fi

CMD=(python -u arithemtic_scaling_law/run_cot_scaling_experiment.py
  --k_max "${K_MAX}"
  --train_examples_per_level "${TRAIN_EXAMPLES_PER_LEVEL}"
  --eval_examples_per_level "${EVAL_EXAMPLES_PER_LEVEL}"
  --data_dir "${DATA_DIR}"
  --artifacts_dir "${ARTIFACTS_DIR}"
  --results_dir "${RESULTS_DIR}"
  --context_length "${CONTEXT_LENGTH}"
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --grad_accum "${GRAD_ACCUM}"
  --eval_steps "${EVAL_STEPS}"
  --eval_refine_rounds "${EVAL_REFINE_ROUNDS}"
  --rollback_branches "${ROLLBACK_BRANCHES}"
  --greedy_eval_batch_size "${GREEDY_EVAL_BATCH_SIZE}"
  --greedy_eval_max_new_tokens "${GREEDY_EVAL_MAX_NEW_TOKENS}"
  --max_steps "${MAX_STEPS}"
  --acc_target "${ACC_TARGET}"
  --seed "${SEED}"
)

CMD+=(--q_keep "${Q_KEEP}")
CMD+=(--max_block_size "${MAX_BLOCK_SIZE}")

if [[ "${GREEDY_EVAL_MATCH_TARGET_LENGTH}" != "0" ]]; then
  CMD+=(--greedy_eval_match_target_length)
fi

if [[ "${FORCE_REGEN}" != "0" ]]; then
  CMD+=(--force_regen)
fi

echo "[$(date)] Mode=${MODE} Running: ${CMD[*]}"
exec "${CMD[@]}"
