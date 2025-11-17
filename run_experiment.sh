#!/usr/bin/env bash
#SBATCH --job-name=toy-compose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli-c
#SBATCH --output=logs/run_experiment_%A_%a.out
#SBATCH --error=logs/run_experiment_%A_%a.err

# Submit with e.g.:
#   sbatch --array=0-31 run_experiment.sh [--skip-existing]
# Each array task handles SEEDS_PER_JOB seeds (default 32) so 1024 seeds
# would be evenly divided across 32 jobs. Override SEED_START, SEED_COUNT,
# SEEDS_PER_JOB, or provide an explicit SEEDS list to customize. Pass
# --skip-existing (or set SKIP_EXISTING=1) to automatically filter out
# seeds that already produced every expected checkpoint.
#
# Execute the full compositionality pipeline with default parameters.
# Runs the shared A&B pretraining once per seed, then trains scratch C
# models and fine-tunes from the shared checkpoint for both NC and C
# families before aggregating Δ_AB metrics.
#
# Control parallelism by exporting PARALLEL_RUNS (defaults to 4) to run
# multiple seed/family pipelines concurrently. A single GPU allocation
# comfortably supports ~4 concurrent runs on this workload.

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

SEED_START=${SEED_START:-0}
SEED_COUNT=${SEED_COUNT:-128}
SEEDS_PER_JOB=${SEEDS_PER_JOB:-16}
FAMILIES=("NC" "C")
PARALLEL_RUNS=${PARALLEL_RUNS:-4}
SUCCESS_THRESHOLD=0.99

# Keep training logs concise by disabling tqdm/HF progress bars everywhere.
export TQDM_DISABLE=1
export DISABLE_PROGRESS_BAR=1
export DATASETS_DISABLE_PROGRESS_BAR=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

if ! [[ "${SEEDS_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEEDS_PER_JOB must be a positive integer, got '${SEEDS_PER_JOB}'." >&2
  exit 1
fi

if ! [[ "${SEED_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED_COUNT must be a positive integer, got '${SEED_COUNT}'." >&2
  exit 1
fi

if ! [[ "${PARALLEL_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_RUNS must be a positive integer, got '${PARALLEL_RUNS}'." >&2
  exit 1
fi

if [[ -z "${SEEDS:-}" ]]; then
  JOB_INDEX=${SLURM_ARRAY_TASK_ID:-0}
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

  if (( JOB_END < JOB_START )); then
    echo "Computed empty seed range [${JOB_START}, ${JOB_END}]; check SEEDS_PER_JOB." >&2
    exit 1
  fi

  SEEDS=$(seq ${JOB_START} ${JOB_END})
fi

# Convert the whitespace-separated SEEDS value into an array for easier filtering later.
SEED_LIST=()
for seed in ${SEEDS}; do
  SEED_LIST+=("${seed}")
done

seed_complete() {
  local seed="$1"
  local ab_dir="artifacts/ab_pretrain/shared_seed${seed}"
  local ab_metrics="results/shared_seed${seed}_ab.json"

  if [[ ! -d "${ab_dir}" || ! -f "${ab_metrics}" ]]; then
    return 1
  fi

  for family in "${FAMILIES[@]}"; do
    local scratch_dir="artifacts/c_scratch/${family}_seed${seed}"
    local scratch_metrics="results/${family}_seed${seed}_scratch.json"
    local finetune_dir="artifacts/c_finetune/${family}_seed${seed}"
    local finetune_metrics="results/${family}_seed${seed}_finetune.json"

    if [[ ! -d "${scratch_dir}" || ! -d "${finetune_dir}" || \
          ! -f "${scratch_metrics}" || ! -f "${finetune_metrics}" ]]; then
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
      echo "Seed ${seed} already has shared AB + ${FAMILIES[*]} artifacts; skipping."
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
echo "Families: ${FAMILIES[*]}"
echo "Parallel seed pipelines per GPU: ${PARALLEL_RUNS}"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL_RUNS}" ]; do
    # wait -n propagates the exit status; set -e will exit on failure.
    wait -n
  done
}

run_seed_pipeline() {
  local seed="$1"

  echo "=== Shared multitask A&B pretraining (seed ${seed}) ==="
  python train_multitask_AB.py --seed "${seed}" --greedy_eval_match_target_length --success_threshold "${SUCCESS_THRESHOLD}"

  for family in "${FAMILIES[@]}"; do
    echo "=== Family ${family} :: C-from-scratch (seed ${seed}) ==="
    python train_compose_scratch.py --family "${family}" --seed "${seed}" --greedy_eval_match_target_length --success_threshold "${SUCCESS_THRESHOLD}"

    echo "=== Family ${family} :: fine-tune C from shared AB checkpoint (seed ${seed}) ==="
    python finetune_from_AB.py --family "${family}" --seed "${seed}" --greedy_eval_match_target_length --success_threshold "${SUCCESS_THRESHOLD}"
  done
}

for seed in "${SEED_LIST[@]}"; do
  wait_for_slot
  run_seed_pipeline "${seed}" &
done

wait

echo "=== Aggregating Δ_AB metrics for NC & C ==="
python analyze_signal.py --families "${FAMILIES[@]}"
