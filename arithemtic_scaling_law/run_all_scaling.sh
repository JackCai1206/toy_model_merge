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
K_MIN="${K_MIN:-8}"
K_MAX="${K_MAX:-33}"
Q_KEEP=${Q_KEEP:-0.8}
MAX_STEPS_PER_BLOCK=${MAX_STEPS_PER_BLOCK:-2}
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-seed_}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50}"

REGIME_SLUG="${REGIME_SLUG:-q${Q_KEEP//./}_b${MAX_STEPS_PER_BLOCK}}"
K_SUFFIX="${K_SUFFIX:-}"
K_SUFFIX="kmin${K_MIN}_kmax${K_MAX}"
RUN_GROUP="${RUN_GROUP:-run_${MODE}_${REGIME_SLUG}_t${ACC_TARGET}_${K_SUFFIX}_${TAG}}"

export MODE TAG ACC_TARGET K_MIN K_MAX Q_KEEP MAX_STEPS_PER_BLOCK REGIME_SLUG K_SUFFIX RUN_GROUP RUN_NAME_PREFIX SEEDS

# Auto-derive array size from the seeds unless the user provided one.
SEED_LIST=(${SEEDS})
if (( ${#SEED_LIST[@]} == 0 )); then
  echo "SEEDS is empty; provide at least one seed." >&2
  exit 1
fi
ARRAY_SPEC="--array=0-$(( ${#SEED_LIST[@]} - 1 ))%4"

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
