#!/usr/bin/env bash
# Submit with e.g.:
#   MODE=sanity sbatch arithemtic_scaling_law/run_scaling_experiment.sh
#   sbatch arithemtic_scaling_law/run_scaling_experiment.sh                 # full run defaults
# MODE=sanity dials down dataset sizes/steps for a quick GPU smoke test. Override any
# variable below via env (e.g., TRAIN_EXAMPLES_PER_LEVEL=1000).

#SBATCH --job-name=arith-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli-c
#SBATCH --output=arithemtic_scaling_law/logs/scale_%A.out
#SBATCH --error=arithemtic_scaling_law/logs/scale_%A.err

# salloc --job-name=arith-scale --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=64G --time=12:00:00 --gres=gpu:1 --partition=pli-c

set -euo pipefail

REPO_DIR="/scratch/gpfs/ARORA/zc5794/toy_model_merge"
cd "${REPO_DIR}"

source "${REPO_DIR}/.venv/bin/activate"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Keep logs clean.
export TQDM_DISABLE=1
export DISABLE_PROGRESS_BAR=1
export DATASETS_DISABLE_PROGRESS_BAR=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

TAG=""
MODE="full"
SEED_LIST=(${SEEDS:-43 44 45 46 47})
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" -lt 0 || "${SLURM_ARRAY_TASK_ID}" -ge "${#SEED_LIST[@]}" ]]; then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range for SEED_LIST (${#SEED_LIST[@]} entries)" >&2
    exit 1
  fi
  SEEDS=${SEED_LIST[${SLURM_ARRAY_TASK_ID}]}
else
  SEEDS="${SEED_LIST[@]}"
fi

# Full-scale defaults match run_cot_scaling_experiment.py arguments.
K_MIN=4
K_MAX=17
TRAIN_EXAMPLES_PER_LEVEL=20000
EVAL_EXAMPLES_PER_LEVEL=2000
FINAL_EVAL_EXAMPLES_PER_LEVEL=500
DATA_DIR="${REPO_DIR}/arithemtic_scaling_law/data"
ARTIFACTS_ROOT="${REPO_DIR}/arithemtic_scaling_law/artifacts"
RESULTS_ROOT="${REPO_DIR}/arithemtic_scaling_law/results"
CONTEXT_LENGTH=1024
PER_DEVICE_BATCH_SIZE=128
PER_DEVICE_EVAL_BATCH_SIZE=1024
GRAD_ACCUM=2
EVAL_STEPS=1000
EVAL_REFINE_ROUNDS=5
ROLLBACK_BRANCHES=3
PREV_LEVEL_MIX_FRACTION=0.15
PREV_LEVEL_MIX_DECAY=0.8
GREEDY_EVAL_BATCH_SIZE=1024
GREEDY_EVAL_MATCH_TARGET_LENGTH=1
MAX_STEPS=200000
RESUME_OPTIMIZER_STATE=${RESUME_OPTIMIZER_STATE:-1}
WARMUP_STEPS=100
ACC_TARGET=0.90
FINAL_EVAL_STOP_THRESHOLD=${FINAL_EVAL_STOP_THRESHOLD:-}
Q_KEEP=${Q_KEEP:-1.0}
MAX_BLOCK_SIZE=${MAX_BLOCK_SIZE:-1}
REGIME_SLUG="q${Q_KEEP//./}_b${MAX_BLOCK_SIZE}"
K_SUFFIX=""
if [[ "${K_MIN}" != "1" ]]; then
  K_SUFFIX="kmin${K_MIN}_kmax${K_MAX}"
fi
RUN_GROUP=${RUN_GROUP:-"run_${MODE}_${REGIME_SLUG}_t${ACC_TARGET}_${K_SUFFIX}_${TAG}"}
## Override Q_KEEP/MAX_BLOCK_SIZE to control the dataset regime.

if [[ "${MODE}" == "sanity" ]]; then
  # Smaller settings for a fast sanity sweep.
  K_MIN=${K_MIN_SANITY:-$K_MIN}
  K_MAX=${K_MAX_SANITY:-3}
  TRAIN_EXAMPLES_PER_LEVEL=${TRAIN_EXAMPLES_PER_LEVEL_SANITY:-2000}
  EVAL_EXAMPLES_PER_LEVEL=${EVAL_EXAMPLES_PER_LEVEL_SANITY:-200}
  FINAL_EVAL_EXAMPLES_PER_LEVEL=${FINAL_EVAL_EXAMPLES_PER_LEVEL_SANITY:-50}
  CONTEXT_LENGTH=${CONTEXT_LENGTH_SANITY:-256}
  MAX_STEPS=${MAX_STEPS_SANITY:-2000}
  RESUME_OPTIMIZER_STATE=${RESUME_OPTIMIZER_STATE_SANITY:-${RESUME_OPTIMIZER_STATE}}
  WARMUP_STEPS=${WARMUP_STEPS_SANITY:-${WARMUP_STEPS}}
fi

for SEED in ${SEEDS}; do
  RUN_NAME=${RUN_NAME_PREFIX:-"seed_${SEED}"}
  ARTIFACTS_DIR="${ARTIFACTS_ROOT}/${RUN_GROUP}/${RUN_NAME}"
  RESULTS_DIR="${RESULTS_ROOT}/${RUN_GROUP}/${RUN_NAME}"

  CMD=(python -u arithemtic_scaling_law/run_cot_scaling_experiment.py
    --k_min "${K_MIN}"
    --k_max "${K_MAX}"
    --train_examples_per_level "${TRAIN_EXAMPLES_PER_LEVEL}"
    --eval_examples_per_level "${EVAL_EXAMPLES_PER_LEVEL}"
    --final_eval_examples_per_level "${FINAL_EVAL_EXAMPLES_PER_LEVEL}"
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
    --prev_level_mix_fraction "${PREV_LEVEL_MIX_FRACTION}"
    --prev_level_mix_decay "${PREV_LEVEL_MIX_DECAY}"
    --resume_optimizer_state "${RESUME_OPTIMIZER_STATE}"
    --greedy_eval_batch_size "${GREEDY_EVAL_BATCH_SIZE}"
    --max_steps "${MAX_STEPS}"
    --warmup_steps "${WARMUP_STEPS}"
    --acc_target "${ACC_TARGET}"
    --seed "${SEED}"
    --run_name "${RUN_NAME}"
    --run_group "${RUN_GROUP}"
  )

  CMD+=(--q_keep "${Q_KEEP}")
  CMD+=(--max_block_size "${MAX_BLOCK_SIZE}")

  if [[ -n "${FINAL_EVAL_STOP_THRESHOLD}" ]]; then
    CMD+=(--final_eval_stop_threshold "${FINAL_EVAL_STOP_THRESHOLD}")
  fi

  if [[ "${GREEDY_EVAL_MATCH_TARGET_LENGTH}" != "0" ]]; then
    CMD+=(--greedy_eval_match_target_length)
  fi

  echo "[$(date)] Mode=${MODE} RunGroup=${RUN_GROUP} Seed=${SEED} Running: ${CMD[*]}"
  "${CMD[@]}"
done

# Run aggregation separately after all array jobs complete:
#   MODE=full RUN_GROUP=... Q_KEEP=... MAX_BLOCK_SIZE=... sbatch arithemtic_scaling_law/run_scaling_analysis.sh
