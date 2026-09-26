#!/usr/bin/env bash
set -euo pipefail

# Re-run the existing four-market ablations while preserving the original
# results and reusing any existing TSFM rollout caches. The updated pipeline
# additionally writes predictions_test.csv for paired peak/error analysis.
#
# Usage:
#   bash scripts/electricity/run_mechanism_reanalysis.sh stage1
#   bash scripts/electricity/run_mechanism_reanalysis.sh stage2
#   bash scripts/electricity/run_mechanism_reanalysis.sh stage3
#   bash scripts/electricity/run_mechanism_reanalysis.sh all

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STAGE="${1:-all}"
export RESULTS_ROOT="${RESULTS_ROOT:-results/mechanism_reanalysis/electricity}"
export TSFM_CACHE_ROOT="${TSFM_CACHE_ROOT:-results/electricity}"

run_stage() {
  local stage="$1"
  echo "========== mechanism reanalysis: ${stage} =========="
  bash scripts/electricity/run_two_dataset_ablation.sh "$stage"
  bash scripts/electricity/run_shanxi_ablation.sh "$stage"
  bash scripts/electricity/run_reale_fr_ablation.sh "$stage"
}

case "$STAGE" in
  stage1|stage2|stage3)
    run_stage "$STAGE"
    ;;
  all)
    run_stage stage1
    run_stage stage2
    run_stage stage3
    ;;
  *)
    echo "Usage: $0 [stage1|stage2|stage3|all]" >&2
    exit 2
    ;;
esac

echo "Results written under: $RESULTS_ROOT"
echo "TSFM cache reused from: $TSFM_CACHE_ROOT"
find "$RESULTS_ROOT" -name "predictions_test.csv" -print
