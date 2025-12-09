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
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli-c
#SBATCH --output=arithemtic_scaling_law/logs/scale_%a.out
#SBATCH --error=arithemtic_scaling_law/logs/scale_%a.err

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

TAG="${TAG:-}"
MODE="${MODE:-full}"
SEED_LIST=(${SEEDS:-42 44 45 46 47})
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
K_MIN=${K_MIN:-4}
K_MAX=${K_MAX:-17}
TRAIN_EXAMPLES_PER_LEVEL=${TRAIN_EXAMPLES_PER_LEVEL:-20000}
EVAL_EXAMPLES_PER_LEVEL=${EVAL_EXAMPLES_PER_LEVEL:-2000}
FINAL_EVAL_EXAMPLES_PER_LEVEL=${FINAL_EVAL_EXAMPLES_PER_LEVEL:-500}
DATA_DIR="${REPO_DIR}/arithemtic_scaling_law/data"
ARTIFACTS_ROOT="${REPO_DIR}/arithemtic_scaling_law/artifacts"
RESULTS_ROOT="${REPO_DIR}/arithemtic_scaling_law/results"
CONTEXT_LENGTH=${CONTEXT_LENGTH:-2048}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-128}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1024}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EVAL_STEPS=${EVAL_STEPS:-500}
EVAL_STEPS_MIN=${EVAL_STEPS_MIN:-5}
EVAL_STEPS_MAX=${EVAL_STEPS_MAX:-1000}
EVAL_JITTER_FRACTION=${EVAL_JITTER_FRACTION:-0.5}
PREV_LEVEL_MIX_FRACTION=${PREV_LEVEL_MIX_FRACTION:-0.15}
PREV_LEVEL_MIX_DECAY=${PREV_LEVEL_MIX_DECAY:-0.8}
GREEDY_EVAL_BATCH_SIZE=${GREEDY_EVAL_BATCH_SIZE:-1024}
GREEDY_EVAL_MATCH_TARGET_LENGTH=${GREEDY_EVAL_MATCH_TARGET_LENGTH:-1}
MAX_STEPS=${MAX_STEPS:-200000}
WARMUP_STEPS=${WARMUP_STEPS:-100}
ACC_TARGET=${ACC_TARGET:-0.90}
FINAL_EVAL_STOP_THRESHOLD=${FINAL_EVAL_STOP_THRESHOLD:-}
Q_KEEP=${Q_KEEP:-1.0}
MAX_STEPS_PER_BLOCK=${MAX_STEPS_PER_BLOCK:-1}
REGIME_SLUG="${REGIME_SLUG:-q${Q_KEEP//./}_b${MAX_STEPS_PER_BLOCK}}"
K_SUFFIX=""
if [[ "${K_MIN}" != "1" ]]; then
  K_SUFFIX="kmin${K_MIN}_kmax${K_MAX}"
fi
RUN_GROUP=${RUN_GROUP:-"run_${MODE}_${REGIME_SLUG}_t${ACC_TARGET}_${K_SUFFIX}_${TAG}"}
## Override Q_KEEP/MAX_STEPS_PER_BLOCK to control the dataset regime.

if [[ "${MODE}" == "sanity" ]]; then
  # Smaller settings for a fast sanity sweep.
  K_MIN=${K_MIN_SANITY:-1}
  K_MAX=${K_MAX_SANITY:-3}
  TRAIN_EXAMPLES_PER_LEVEL=${TRAIN_EXAMPLES_PER_LEVEL_SANITY:-2000}
  EVAL_EXAMPLES_PER_LEVEL=${EVAL_EXAMPLES_PER_LEVEL_SANITY:-200}
  FINAL_EVAL_EXAMPLES_PER_LEVEL=${FINAL_EVAL_EXAMPLES_PER_LEVEL_SANITY:-50}
  CONTEXT_LENGTH=${CONTEXT_LENGTH_SANITY:-256}
  MAX_STEPS=${MAX_STEPS_SANITY:-2000}
  WARMUP_STEPS=${WARMUP_STEPS_SANITY:-${WARMUP_STEPS}}
fi

if (( K_MAX <= K_MIN )); then
  K_MAX=$((K_MIN + 1))
fi

for SEED in ${SEEDS}; do
  RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-"seed_"}
  RUN_NAME="${RUN_NAME_PREFIX}${SEED}"
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
    --eval_steps_min "${EVAL_STEPS_MIN}"
    --eval_steps_max "${EVAL_STEPS_MAX}"
    --eval_jitter_fraction "${EVAL_JITTER_FRACTION}"
    --prev_level_mix_fraction "${PREV_LEVEL_MIX_FRACTION}"
    --prev_level_mix_decay "${PREV_LEVEL_MIX_DECAY}"
    --greedy_eval_batch_size "${GREEDY_EVAL_BATCH_SIZE}"
    --max_steps "${MAX_STEPS}"
    --warmup_steps "${WARMUP_STEPS}"
    --acc_target "${ACC_TARGET}"
    --seed "${SEED}"
    --run_name "${RUN_NAME}"
    --run_group "${RUN_GROUP}"
  )

  CMD+=(--q_keep "${Q_KEEP}")
  CMD+=(--max_steps_per_block "${MAX_STEPS_PER_BLOCK}")

  if [[ -n "${FINAL_EVAL_STOP_THRESHOLD}" ]]; then
    CMD+=(--final_eval_stop_threshold "${FINAL_EVAL_STOP_THRESHOLD}")
  fi

  if [[ "${GREEDY_EVAL_MATCH_TARGET_LENGTH}" != "0" ]]; then
    CMD+=(--greedy_eval_match_target_length)
  fi

  echo "[$(date)] Mode=${MODE} RunGroup=${RUN_GROUP} Seed=${SEED} Running: ${CMD[*]}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY RUN] Skipping execution because DRY_RUN=1"
  else
    "${CMD[@]}"
  fi
done

# Run aggregation separately after all array jobs complete:
#   MODE=full RUN_GROUP=... Q_KEEP=... MAX_STEPS_PER_BLOCK=... sbatch arithemtic_scaling_law/run_scaling_analysis.sh
