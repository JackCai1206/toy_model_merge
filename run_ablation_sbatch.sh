#!/bin/bash
# Submit with, e.g.,: sbatch --array=0-31%8 run_ablation_sbatch.sh
# The array index controls a chunk of seeds (default 16 seeds/job). Within each
# job up to PARALLEL_ABLATE seed pipelines run concurrently (default 2).
# For a given seed we run ablations sequentially on:
#   1) AB pretrain checkpoint (shared_seed{seed} if present, else C_seed{seed} fallback) using A/B eval
#   2) C family checkpoint (prefers c_finetune/C_seed{seed}, falls back to c_scratch/C_seed{seed})
#   3) NC family checkpoint (prefers c_finetune/NC_seed{seed}, falls back to c_scratch/NC_seed{seed})
# Local test without Slurm:
#   ./run_ablation_sbatch.sh 5 6 7        # explicit seed list
#   SEEDS="0 1" ./run_ablation_sbatch.sh  # env var seed list

#SBATCH --job-name=ablate_reverse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=pli-c
#SBATCH --output=logs/ablate_%A_%a.out
#SBATCH --error=logs/ablate_%A_%a.err

set -euo pipefail

SEED_START=${SEED_START:-0}
SEED_COUNT=${SEED_COUNT:-128}
SEEDS_PER_JOB=${SEEDS_PER_JOB:-32}
PARALLEL_ABLATE=${PARALLEL_ABLATE:-4}

NUM_SAMPLES=${NUM_SAMPLES:-1024}
BATCH_SIZE=${BATCH_SIZE:-1024}
MODE=${MODE:-split}
DEVICE=${DEVICE:-cuda}

if ! [[ "${SEED_START}" =~ ^[0-9]+$ ]]; then
  echo "SEED_START must be a non-negative integer, got '${SEED_START}'." >&2
  exit 1
fi
if ! [[ "${SEED_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED_COUNT must be a positive integer, got '${SEED_COUNT}'." >&2
  exit 1
fi
if ! [[ "${SEEDS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEEDS_PER_JOB must be a positive integer, got '${SEEDS_PER_JOB}'." >&2
  exit 1
fi
if ! [[ "${PARALLEL_ABLATE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_ABLATE must be a positive integer, got '${PARALLEL_ABLATE}'." >&2
  exit 1
fi

# Determine seeds to run.
SEED_LIST=()
if [[ -n "${SEEDS:-}" ]]; then
  # User provided explicit list via env var.
  for s in ${SEEDS}; do
    if ! [[ "$s" =~ ^[0-9]+$ ]]; then
      echo "Invalid seed in SEEDS: '$s'." >&2
      exit 1
    fi
    SEED_LIST+=("$s")
  done
elif [[ $# -gt 0 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # Explicit list via CLI args when not under Slurm.
  for s in "$@"; do
    if ! [[ "$s" =~ ^[0-9]+$ ]]; then
      echo "Invalid seed argument: '$s'." >&2
      exit 1
    fi
    SEED_LIST+=("$s")
  done
else
  # Compute chunk from array index.
  JOB_INDEX=${SLURM_ARRAY_TASK_ID:-0}
  if ! [[ "${JOB_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "SLURM_ARRAY_TASK_ID must be an integer, got '${JOB_INDEX}'." >&2
    exit 1
  fi
  JOB_START=$((SEED_START + JOB_INDEX * SEEDS_PER_JOB))
  LAST_SEED=$((SEED_START + SEED_COUNT - 1))
  if (( JOB_START > LAST_SEED )); then
    echo "Array task ${JOB_INDEX} has no seeds to process (start ${JOB_START} > last ${LAST_SEED})."
    exit 0
  fi
  JOB_END=$((JOB_START + SEEDS_PER_JOB - 1))
  if (( JOB_END > LAST_SEED )); then
    JOB_END=${LAST_SEED}
  fi
  for ((s=JOB_START; s<=JOB_END; s++)); do
    SEED_LIST+=("$s")
  done
fi

if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "No seeds to process." >&2
  exit 1
fi

echo "[$(date)] Seeds: ${SEED_LIST[*]}"
echo "Parallel seed pipelines per GPU: ${PARALLEL_ABLATE}"
echo "NUM_SAMPLES=${NUM_SAMPLES}, BATCH_SIZE=${BATCH_SIZE}, MODE=${MODE}, DEVICE=${DEVICE}"

TASKS_AB=("A" "B")
TASKS_C=("A" "B")

run_ablation() {
  local family="$1"
  local ckpt="$2"
  shift 2
  local tasks=("$@")
  local base
  base=$(basename "$ckpt")
  local out="results/ablation/${family}/${base}.json"
  mkdir -p "$(dirname "$out")"
  echo "[$(date)]   -> ${family} @ ${ckpt} -> ${out}"
  python analyze_ablation.py \
    --checkpoint "$ckpt" \
    --family "$family" \
    --tasks "${tasks[@]}" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --mode "$MODE" \
    --output "$out"
}

attempt() {
  local family="$1"; shift
  local -n tasks_ref="$1"; shift
  local paths=("$@")
  for path in "${paths[@]}"; do
    if [[ -f "${path}/model.safetensors" ]]; then
      run_ablation "$family" "$path" "${tasks_ref[@]}"
      return 0
    fi
  done
  echo "[$(date)]   Skipping ${family}: no checkpoint found among: ${paths[*]}" >&2
}

run_seed() {
  local seed="$1"
  echo "[$(date)] Running ablations for seed ${seed}"

  # 1) AB pretrain
  attempt "GENERAL" \
    TASKS_AB \
    "artifacts/ab_pretrain/shared_seed${seed}" \
    "artifacts/ab_pretrain/C_seed${seed}"

  # 2) C family fine-tune (or scratch fallback)
  attempt "C" \
    TASKS_C \
    "artifacts/c_finetune/C_seed${seed}" \
    "artifacts/c_scratch/C_seed${seed}"

  # 3) NC family fine-tune (or scratch fallback)
  attempt "NC" \
    TASKS_C \
    "artifacts/c_finetune/NC_seed${seed}" \
    "artifacts/c_scratch/NC_seed${seed}"

  echo "[$(date)] Done seed ${seed}"
}

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL_ABLATE}" ]; do
    wait -n
  done
}

for seed in "${SEED_LIST[@]}"; do
  wait_for_slot
  run_seed "${seed}" &
done

wait
