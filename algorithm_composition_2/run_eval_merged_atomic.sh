#!/usr/bin/env bash
set -euo pipefail

# Evaluate merged (or any) checkpoints on tasks A/B without fine-tuning.
# Usage examples:
#   SEEDS=0,1,2 bash algorithm_composition_2/run_eval_merged_atomic.sh
#   CHECKPOINT_PATTERN="artifacts/merged/merged_seed{seed}" SEEDS=0 bash algorithm_composition_2/run_eval_merged_atomic.sh

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SEEDS_RAW="${SEEDS:-0,1,2,3}"
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN:-artifacts/merged/merged_seed{seed}}"
TASKS="${TASKS:-A,B}"
EVAL_SAMPLES=${EVAL_SAMPLES:-512}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
EVAL_DATA_SEED=${EVAL_DATA_SEED:-}

IFS=',' read -r -a SEED_LIST <<<"${SEEDS_RAW}"

for seed in "${SEED_LIST[@]}"; do
  ckpt="${CHECKPOINT_PATTERN//\{seed\}/${seed}}"
  echo "=== Seed ${seed} :: evaluating ${ckpt} on tasks ${TASKS} ==="
  args=(
    --checkpoint "${ckpt}"
    --tasks "${TASKS}"
    --eval_samples "${EVAL_SAMPLES}"
    --context_length "${CONTEXT_LENGTH}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --seed "${seed}"
  )
  if [[ -n "${EVAL_DATA_SEED}" ]]; then
    args+=(--eval_data_seed "${EVAL_DATA_SEED}")
  fi
  python eval_merged_atomic.py "${args[@]}"
done

echo "Finished evaluating seeds: ${SEEDS_RAW}"
