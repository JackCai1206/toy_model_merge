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

EXP_SUBMIT_OUTPUT=$(sbatch "$@" "${EXP_SCRIPT}")
echo "Submitted experiments: ${EXP_SUBMIT_OUTPUT}"
EXP_JOB_ID=$(echo "${EXP_SUBMIT_OUTPUT}" | awk '/Submitted batch job/ {print $4}')
if [[ -z "${EXP_JOB_ID}" ]]; then
  echo "Could not parse experiment job ID from sbatch output: ${EXP_SUBMIT_OUTPUT}" >&2
  exit 1
fi

ANALYSIS_SUBMIT_OUTPUT=$(sbatch --dependency=afterok:${EXP_JOB_ID} "${ANALYSIS_SCRIPT}")
echo "Submitted analysis (afterok:${EXP_JOB_ID}): ${ANALYSIS_SUBMIT_OUTPUT}"
