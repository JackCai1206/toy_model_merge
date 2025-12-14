#!/usr/bin/env bash
# Launch the full Algorithm Composition 2 pipeline across multiple seeds.
# Submit with e.g.:
#   sbatch run_experiment.sh --array=0-3
# or run locally:
#   bash run_experiment.sh --skip-existing

#SBATCH --job-name=algo-comp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli-c
#SBATCH --output=logs/run_experiment_%A_%a.out
#SBATCH --error=logs/run_experiment_%A_%a.err

# salloc --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=64G --time=4:00:00 --gres=gpu:1 --partition=pli-c

set -euo pipefail

SKIP_EXISTING=${SKIP_EXISTING:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# Keep training logs concise by disabling progress bars.
export TQDM_DISABLE=1
export DISABLE_PROGRESS_BAR=1
export DATASETS_DISABLE_PROGRESS_BAR=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Global hyper-parameters (override via env).
DATASET_SIZE=${DATASET_SIZE:-100000}
EVAL_SAMPLES=${EVAL_SAMPLES:-256}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1024}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-128}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-256}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_STEPS=${MAX_STEPS:-4000}
EVAL_STEPS=${EVAL_STEPS:-100}
EVAL_REFINE_ROUNDS=${EVAL_REFINE_ROUNDS:-3}
EVAL_JITTER_FRACTION=${EVAL_JITTER_FRACTION:-0.5}
ROLLBACK_BRANCHES=${ROLLBACK_BRANCHES:-1}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.95}
GREEDY_EVAL_BATCH_SIZE=${GREEDY_EVAL_BATCH_SIZE:-512}
GREEDY_EVAL_MATCH_TARGET_LENGTH=${GREEDY_EVAL_MATCH_TARGET_LENGTH:-0}
ATOMIC_MIX_FRACTION=${ATOMIC_MIX_FRACTION:-0.0}
PARALLEL_RUNS=${PARALLEL_RUNS:-1}
TRAIN_FULL_STEPS=${TRAIN_FULL_STEPS:-0}

SEED_LIST=($(seq 0 64))
if ! [[ "${PARALLEL_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_RUNS must be a positive integer, got '${PARALLEL_RUNS}'." >&2
  exit 1
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  block_start=$(( SLURM_ARRAY_TASK_ID * PARALLEL_RUNS ))
  block_end=$(( block_start + PARALLEL_RUNS - 1 ))
  if (( block_start >= ${#SEED_LIST[@]} )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} has no seeds to process (start ${block_start} >= total ${#SEED_LIST[@]})." >&2
    exit 0
  fi
  SEED_LIST=("${SEED_LIST[@]:${block_start}:${PARALLEL_RUNS}}")
fi

if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "No seeds to run." >&2
  exit 1
fi

greedy_flags=()
if [[ "${GREEDY_EVAL_MATCH_TARGET_LENGTH}" != "0" ]]; then
  greedy_flags+=(--greedy_eval_match_target_length)
fi

seed_complete() {
  local seed="$1"
  local base="results"
  local expect=(
    "${base}/A_seed${seed}_atomic.json"
    "${base}/B_seed${seed}_atomic.json"
    "${base}/joint_seed${seed}_ab.json"
    "${base}/finetune_from_joint_seed${seed}_finetune.json"
    "${base}/finetune_from_merged_seed${seed}_finetune.json"
    "${base}/compare_seed${seed}.json"
  )
  for path in "${expect[@]}"; do
    if [[ ! -f "${path}" ]]; then
      return 1
    fi
  done
  return 0
}

print_seed_list() {
  if [[ $# -eq 0 ]]; then
    echo "Running seeds: <none>"
    return
  fi
  echo -n "Running seeds:"
  for seed in "$@"; do
    printf "\n%s" "$seed"
  done
  printf "\n"
}

if (( SKIP_EXISTING )); then
  filtered=()
  for seed in "${SEED_LIST[@]}"; do
    if seed_complete "${seed}"; then
      echo "Seed ${seed} already has atomic, joint, and comparison outputs; skipping."
      continue
    fi
    filtered+=("${seed}")
  done
  if (( ${#filtered[@]} == 0 )); then
    echo "All requested seeds already complete. Nothing to run."
    exit 0
  fi
  SEED_LIST=("${filtered[@]}")
fi

print_seed_list "${SEED_LIST[@]}"
echo "Parallel seed pipelines per GPU: ${PARALLEL_RUNS}"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL_RUNS}" ]; do
    wait -n
  done
}

run_seed_pipeline() {
  local seed="$1"

  echo "=== Seed ${seed} :: train A ==="
  python train_atomic_task.py \
    --task A \
    --seed "${seed}" \
    --dataset_size "${DATASET_SIZE}" \
    --eval_samples "${EVAL_SAMPLES}" \
    --context_length "${CONTEXT_LENGTH}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --max_steps "${MAX_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --eval_refine_rounds "${EVAL_REFINE_ROUNDS}" \
    --eval_jitter_fraction "${EVAL_JITTER_FRACTION}" \
    --rollback_branches "${ROLLBACK_BRANCHES}" \
    --success_threshold "${SUCCESS_THRESHOLD}" \
    $( (( TRAIN_FULL_STEPS )) && echo --train_full_steps ) \
    --greedy_eval_batch_size "${GREEDY_EVAL_BATCH_SIZE}" \
    "${greedy_flags[@]}"

  echo "=== Seed ${seed} :: train B ==="
  python train_atomic_task.py \
    --task B \
    --seed "${seed}" \
    --dataset_size "${DATASET_SIZE}" \
    --eval_samples "${EVAL_SAMPLES}" \
    --context_length "${CONTEXT_LENGTH}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --max_steps "${MAX_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --eval_refine_rounds "${EVAL_REFINE_ROUNDS}" \
    --eval_jitter_fraction "${EVAL_JITTER_FRACTION}" \
    --rollback_branches "${ROLLBACK_BRANCHES}" \
    --success_threshold "${SUCCESS_THRESHOLD}" \
    $( (( TRAIN_FULL_STEPS )) && echo --train_full_steps ) \
    --greedy_eval_batch_size "${GREEDY_EVAL_BATCH_SIZE}" \
    "${greedy_flags[@]}"

  echo "=== Seed ${seed} :: joint A&B ==="
  python train_joint_ab.py \
    --seed "${seed}" \
    --dataset_size "${DATASET_SIZE}" \
    --eval_samples "${EVAL_SAMPLES}" \
    --context_length "${CONTEXT_LENGTH}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --max_steps "${MAX_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --eval_refine_rounds "${EVAL_REFINE_ROUNDS}" \
    --eval_jitter_fraction "${EVAL_JITTER_FRACTION}" \
    --rollback_branches "${ROLLBACK_BRANCHES}" \
    --success_threshold "${SUCCESS_THRESHOLD}" \
    --greedy_eval_batch_size "${GREEDY_EVAL_BATCH_SIZE}" \
    "${greedy_flags[@]}"

  echo "=== Seed ${seed} :: fine-tune C from joint vs merged ==="
  python compare_sample_complexity.py \
    --seed "${seed}" \
    --dataset_size "${DATASET_SIZE}" \
    --eval_samples "${EVAL_SAMPLES}" \
    --context_length "${CONTEXT_LENGTH}" \
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --max_steps "${MAX_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --eval_refine_rounds "${EVAL_REFINE_ROUNDS}" \
    --eval_jitter_fraction "${EVAL_JITTER_FRACTION}" \
    --rollback_branches "${ROLLBACK_BRANCHES}" \
    --success_threshold "${SUCCESS_THRESHOLD}" \
    $( (( TRAIN_FULL_STEPS )) && echo --train_full_steps ) \
    --greedy_eval_batch_size "${GREEDY_EVAL_BATCH_SIZE}" \
    --atomic_mix_fraction "${ATOMIC_MIX_FRACTION}" \
    "${greedy_flags[@]}"
}

for seed in "${SEED_LIST[@]}"; do
wait_for_slot
run_seed_pipeline "${seed}" &
done

wait
echo "All requested seeds complete."
