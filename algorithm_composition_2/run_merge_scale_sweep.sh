#!/usr/bin/env bash
set -euo pipefail

# Sweep merge delta scales for composed-task finetuning using existing atomic/joint checkpoints.
# Usage: bash algorithm_composition_2/run_merge_scale_sweep.sh --merge-scales 0.1,0.5,1.0 --seeds 0,1,2

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MERGE_SCALES="${MERGE_SCALES:-0.1,0.2,0.5,0.8,1.0}"
SEEDS_RAW="${SEEDS:-0,1,2,3}"

# Shared hyperparameters (override via env if needed)
DATASET_SIZE=${DATASET_SIZE:-100000}
EVAL_SAMPLES=${EVAL_SAMPLES:-512}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-256}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-128}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-512}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_STEPS=${MAX_STEPS:-1500}
EVAL_STEPS=${EVAL_STEPS:-100}
EVAL_REFINE_ROUNDS=${EVAL_REFINE_ROUNDS:-3}
ROLLBACK_BRANCHES=${ROLLBACK_BRANCHES:-1}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.95}
GREEDY_EVAL_BATCH_SIZE=${GREEDY_EVAL_BATCH_SIZE:-512}
GREEDY_EVAL_MATCH_TARGET_LENGTH=${GREEDY_EVAL_MATCH_TARGET_LENGTH:-0}
ATOMIC_MIX_FRACTION=${ATOMIC_MIX_FRACTION:-0.0}
EVAL_JITTER_FRACTION=${EVAL_JITTER_FRACTION:-0.0}
TRAIN_FULL_STEPS=${TRAIN_FULL_STEPS:-1}

greedy_flags=()
if [[ "${GREEDY_EVAL_MATCH_TARGET_LENGTH}" != "0" ]]; then
  greedy_flags+=(--greedy_eval_match_target_length)
fi

if [[ -z "${SEEDS_RAW}" ]]; then
  echo "Provide seeds via SEEDS env (comma-separated) or --seeds flag." >&2
  exit 1
fi

IFS=',' read -r -a SEED_LIST <<<"${SEEDS_RAW}"

for seed in "${SEED_LIST[@]}"; do
  echo "=== Seed ${seed} :: merge scale sweep ==="
  results_path="results/compare_seed${seed}.json"
  if [[ -f "${results_path}" ]]; then
    if python - "$results_path" <<'PY'
import json, os, sys

path = sys.argv[1]
if not os.path.isfile(path):
    sys.exit(1)
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
except Exception:
    sys.exit(1)

def has_hist(obj):
    return isinstance(obj, dict) and bool(obj.get("eval_history"))

joint_ok = has_hist(data.get("finetune_from_joint"))
merged = data.get("finetune_from_merged")
merged_ok = False
if isinstance(merged, dict):
    if "eval_history" in merged:
        merged_ok = bool(merged.get("eval_history"))
    else:
        for value in merged.values():
            if has_hist(value):
                merged_ok = True
                break

if joint_ok and merged_ok:
    sys.exit(0)
sys.exit(2)
PY
    then
      echo "  Found existing results with eval history at ${results_path}; plotting only."
      python compare_sample_complexity.py --plot_only --results_path "${results_path}"
      continue
    else
      echo "  Existing results missing eval history; retraining."
    fi
  fi
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
    --greedy_eval_batch_size "${GREEDY_EVAL_BATCH_SIZE}" \
    --atomic_mix_fraction "${ATOMIC_MIX_FRACTION}" \
    --merge_scales "${MERGE_SCALES}" \
    $( (( TRAIN_FULL_STEPS )) && echo --train_full_steps ) \
    "${greedy_flags[@]}"
done

echo "Merge scale sweep complete for seeds: ${SEEDS_RAW}"
