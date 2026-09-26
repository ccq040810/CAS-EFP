#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAGE="${1:-stage1}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${CHRONOS2_MODEL_PATH:-$ROOT/models/chronos-2}"
SEQ_LEN="${SEQ_LEN:-168}"
PRED_LEN="${PRED_LEN:-24}"
OUT="results/electricity/reale_fr"
CACHE="$OUT/tsfm_cache_qmad_v1"
mkdir -p "$OUT" "$CACHE"

common=(
  --device "$DEVICE" --model_name chronos2-lgbm --seed "${SEED:-42}"
  --data_path data/REALE/FR_aligned.csv
  --regress_cols_path configs/columns/REALE/FR/regress_dayahead_safe.json
  --tsfm_cols_path configs/columns/REALE/FR/tsfm_dayahead_safe.json
  --data_split_path configs/time/data_split_REALE/REALE.json
  --tag reale_fr_chronos2_h${PRED_LEN}_qmad_v1 --save_dir "$OUT/PLACEHOLDER"
  --is_std --enable_tsfm 1 --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN"
  --batch_size "${BATCH_SIZE:-16}" --tsfm_num_samples "${TSFM_NUM_SAMPLES:-100}"
  --tsfm_cache_path "$CACHE" --tsfm_models chronos2
  --tsfm_model_paths "{\"chronos2\":\"$MODEL_PATH\"}"
  --tsfm_use_future_covariates 0 --regression_model lgbm
  --num_boost_round 2000 --early_stopping_rounds 100 --lgbm_learning_rate 0.05
  --num_leaves 63 --max_depth -1 --min_data_in_leaf 40 --feature_fraction 0.9
  --bagging_fraction 0.8 --bagging_freq 5 --n_jobs "${N_JOBS:-8}" --log_period 50
  --use_ar_features --ar_lags 1,24,168 --ar_roll 24 --disable_shap
)

run_one() {
  local name="$1"; shift
  local args=("${common[@]}")
  for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--save_dir" ]]; then args[$((i+1))]="$OUT/$name"; break; fi
  done
  echo "[reale_fr][$name] starting"
  python run_pipeline.py "${args[@]}" "$@"
}

case "$STAGE" in
  stage1)
    run_one ar_lgbm --enable_tsfm 0 --no-use_tsfm_point --no-use_tsfm_uncertainty
    run_one tsfm_point --no-use_tsfm_uncertainty
    run_one tsfm_point_rawU --use_tsfm_uncertainty
    run_one tsfm_point_C --no-use_tsfm_uncertainty --use_tsfm_confidence
    run_one tsfm_point_C_weighted_mse --no-use_tsfm_uncertainty --use_tsfm_confidence --sample_weight_mode confidence --sample_weight_lambda 1.0
    ;;
  stage2)
    run_one tsfm_point_C_fixed_asym --no-use_tsfm_uncertainty --use_tsfm_confidence --use_asymmetric_loss --asymmetric_alpha 2.0 --asymmetric_direction auto
    run_one tsfm_point_C_confidence_asym --no-use_tsfm_uncertainty --use_tsfm_confidence --use_asymmetric_loss --asymmetric_alpha 2.0 --asymmetric_direction auto --sample_weight_mode confidence --sample_weight_lambda 1.0
    ;;
  stage3)
    run_one tsfm_point_fixed_asym_noC --no-use_tsfm_uncertainty --use_asymmetric_loss --asymmetric_alpha 2.0 --asymmetric_direction auto
    ;;
  *) echo "Usage: $0 [stage1|stage2|stage3]" >&2; exit 2;;
esac
