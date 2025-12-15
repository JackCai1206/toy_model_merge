#!/usr/bin/env bash
# Submit the training sweep and queue the analysis to run after it finishes.
# Usage examples:
#   bash arithemtic_scaling_law/run_all_scaling.sh --array=0-2
#   MODE=sanity RUN_GROUP=my_group bash arithemtic_scaling_law/run_all_scaling.sh

set -euo pipefail

REPO_DIR="/scratch/gpfs/ARORA/zc5794/toy_model_merge"
cd "${REPO_DIR}"

EXP_SCRIPT="${REPO_DIR}/arithemtic_scaling_law/run_scaling_experiment.sh"
ANALYSIS_SCRIPT="${REPO_DIR}/arithemtic_scaling_law/run_scaling_analysis.sh"

# Centralize run-group naming inputs so both scripts share the same env.
MODE="${MODE:-full}"
TAG="${TAG:-v2}"
ACC_TARGET="${ACC_TARGET:-0.95}"
K_MIN="${K_MIN:-1}"
# Optional sweep: space-separated list of k_min values.
# If set, we will loop over these values and submit one sweep+analysis per k_min.
# Example:
#   K_MIN_LIST="1 2 4 8" bash arithemtic_scaling_law/run_all_scaling.sh
K_MIN_LIST="${K_MIN_LIST:-1 4 16}"
K_MAX="${K_MAX:-17}"
Q_KEEP=${Q_KEEP:-0.5}
MAX_STEPS_PER_BLOCK=${MAX_STEPS_PER_BLOCK:-1}
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-seed_}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

REGIME_SLUG="${REGIME_SLUG:-q${Q_KEEP//./}_b${MAX_STEPS_PER_BLOCK}}"

# If K_MIN_LIST isn't provided, default to the single K_MIN value.
if [[ -z "${K_MIN_LIST}" ]]; then
  K_MIN_LIST="${K_MIN}"
fi

# If the user explicitly set RUN_GROUP, we respect it only when not sweeping.
USER_PROVIDED_RUN_GROUP="${RUN_GROUP:-}"
IS_SWEEP=0
if [[ "${K_MIN_LIST}" == *" "* ]]; then
  IS_SWEEP=1
fi

# Auto-derive array size from the seeds unless the user provided one.
SEED_LIST=(${SEEDS})
if (( ${#SEED_LIST[@]} == 0 )); then
  echo "SEEDS is empty; provide at least one seed." >&2
  exit 1
fi
ARRAY_SPEC="--array=0-$(( ${#SEED_LIST[@]} - 1 ))"

USER_SBATCH_ARGS=("$@")
HAS_ARRAY=0
for arg in "${USER_SBATCH_ARGS[@]}"; do
  case "${arg}" in
    --array=*|-a=*|--array|-a) HAS_ARRAY=1 ;;
  esac
done

EXP_SBATCH_ARGS=("${USER_SBATCH_ARGS[@]}")
if (( ! HAS_ARRAY )); then
  EXP_SBATCH_ARGS+=("${ARRAY_SPEC}")
fi

for K_MIN in ${K_MIN_LIST}; do
  # Recompute suffix + run-group per K_MIN so artifacts/results don't collide.
  K_SUFFIX="kmin${K_MIN}_kmax${K_MAX}"
  if (( IS_SWEEP == 1 )); then
    RUN_GROUP="run_${MODE}_${REGIME_SLUG}_t${ACC_TARGET}_${K_SUFFIX}_${TAG}"
  else
    RUN_GROUP="${USER_PROVIDED_RUN_GROUP:-run_${MODE}_${REGIME_SLUG}_t${ACC_TARGET}_${K_SUFFIX}_${TAG}}"
  fi

  # Optionally filter seeds before submitting jobs so we don't allocate GPU time to completed seeds.
  EFFECTIVE_SEEDS="${SEEDS}"
  if [[ "${SKIP_EXISTING}" != "0" ]]; then
    FILTERED_SEEDS=""
    for SEED in ${SEEDS}; do
      RESULTS_DIR="${REPO_DIR}/arithemtic_scaling_law/results/${RUN_GROUP}/${RUN_NAME_PREFIX}${SEED}"
      SEED_DONE=1
      if [[ ! -d "${RESULTS_DIR}" ]]; then
        SEED_DONE=0
      else
        for ((k=${K_MIN}; k<${K_MAX}; k++)); do
          JSON_PATH="${RESULTS_DIR}/level_k${k}_${REGIME_SLUG}.json"
          if [[ ! -f "${JSON_PATH}" ]]; then
            SEED_DONE=0
            break
          fi
        done
      fi

      if (( SEED_DONE == 1 )); then
        echo "[pre-submit] Skipping seed ${SEED}: results already exist under ${RESULTS_DIR}"
      else
        FILTERED_SEEDS+="${SEED} "
      fi
    done

    EFFECTIVE_SEEDS="${FILTERED_SEEDS%% }"
    if [[ -z "${EFFECTIVE_SEEDS}" ]]; then
      echo "All seeds are complete for RUN_GROUP=${RUN_GROUP} (K_MIN=${K_MIN}, K_MAX=${K_MAX}); skipping submission."
      continue
    fi

    # Re-derive --array bounds to match the filtered seed list.
    SEED_LIST=(${EFFECTIVE_SEEDS})
    ARRAY_SPEC="--array=0-$(( ${#SEED_LIST[@]} - 1 ))"
    EXP_SBATCH_ARGS=(${USER_SBATCH_ARGS[@]})
    HAS_ARRAY=0
    for arg in "${EXP_SBATCH_ARGS[@]}"; do
      case "${arg}" in
        --array=*|-a=*|--array|-a) HAS_ARRAY=1 ;;
      esac
    done
    if (( ! HAS_ARRAY )); then
      EXP_SBATCH_ARGS+=("${ARRAY_SPEC}")
    fi
  fi

  export MODE TAG ACC_TARGET K_MIN K_MAX Q_KEEP MAX_STEPS_PER_BLOCK REGIME_SLUG K_SUFFIX RUN_GROUP RUN_NAME_PREFIX SKIP_EXISTING
  export SEEDS="${EFFECTIVE_SEEDS}"

  echo "=== K_MIN=${K_MIN} K_MAX=${K_MAX} RUN_GROUP=${RUN_GROUP} ==="
  echo "Seeds: ${SEEDS}"
  echo "Submitting experiments: sbatch ${EXP_SBATCH_ARGS[*]} ${EXP_SCRIPT}"
  EXP_SUBMIT_OUTPUT=$(sbatch --wait "${EXP_SBATCH_ARGS[@]}" "${EXP_SCRIPT}")
  echo "Submitted experiments: ${EXP_SUBMIT_OUTPUT}"
  EXP_JOB_ID=$(echo "${EXP_SUBMIT_OUTPUT}" | awk '/Submitted batch job/ {print $4}')
  if [[ -z "${EXP_JOB_ID}" ]]; then
    echo "Could not parse experiment job ID from sbatch output: ${EXP_SUBMIT_OUTPUT}" >&2
    exit 1
  fi

  # ANALYSIS_SUBMIT_OUTPUT=$(sbatch --dependency=afterok:${EXP_JOB_ID} --kill-on-invalid-dep=yes "${ANALYSIS_SCRIPT}")
  # echo "Submitted analysis (afterok:${EXP_JOB_ID}): ${ANALYSIS_SUBMIT_OUTPUT}"
  ANALYSIS_SUBMIT_OUTPUT=$(bash "${ANALYSIS_SCRIPT}")
  echo "Submitted analysis: ${ANALYSIS_SUBMIT_OUTPUT}"
done
