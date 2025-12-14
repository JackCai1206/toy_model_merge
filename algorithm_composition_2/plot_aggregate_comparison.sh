#!/usr/bin/env bash
set -euo pipefail

# Aggregate compare_sample_complexity results across seeds.
# Usage: bash algorithm_composition_2/plot_aggregate_comparison.sh

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RESULTS_DIR=${RESULTS_DIR:-results}
PLOT_PATH=${PLOT_PATH:-${RESULTS_DIR}/compare_aggregate.png}
STATS_PATH=${STATS_PATH:-${RESULTS_DIR}/compare_aggregate.json}
CURVE_PATH=${CURVE_PATH:-${RESULTS_DIR}/compare_aggregate_curves.png}
GLOB=${GLOB:-"${RESULTS_DIR}/compare_seed*.json"}

python compare_sample_complexity.py \
  --aggregate \
  --results_dir "${RESULTS_DIR}" \
  --aggregate_results_glob "${GLOB}" \
  --aggregate_plot_path "${PLOT_PATH}" \
  --aggregate_stats_path "${STATS_PATH}" \
  --aggregate_curve_path "${CURVE_PATH}"

echo "Aggregated comparison plot: ${PLOT_PATH}"
echo "Aggregated stats:          ${STATS_PATH}"
echo "Aggregated history plot:   ${CURVE_PATH}"
