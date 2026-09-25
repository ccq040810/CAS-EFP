#!/usr/bin/env bash
set -euo pipefail

# Run stage 1 first. Review train_direction_diagnostics.csv and
# confidence_diagnostics_test.csv before deliberately starting stage 2.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STAGE="${1:-stage1}"
DEVICE="${DEVICE:-cuda}"
CHRONOS2_MODEL_PATH="${CHRONOS2_MODEL_PATH:-$ROOT/models/chronos-2/chronos-2}"
SEQ_LEN="${SEQ_LEN:-168}"
PRED_LEN="${PRED_LEN:-24}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_JOBS="${N_JOBS:-8}"
SEED="${SEED:-42}"

run_dataset() {
  local name="$1" data="$2" regress="$3" tsfm="$4" split="$5"
  local tag="${name}_chronos2_h${PRED_LEN}_qmad_v1"
  local out="results/electricity/${name}"
  local cache="$out/tsfm_cache_qmad_v1"
  mkdir -p "$out" "$cache"

  local common=(
    --device "$DEVICE" --model_name "chronos2-lgbm" --seed "$SEED"
    --data_path "$data" --regress_cols_path "$regress"
    --tsfm_cols_path "$tsfm" --data_split_path "$split"
    --tag "$tag" --save_dir "$out/PLACEHOLDER"
    --is_std --enable_tsfm 1 --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN"
    --batch_size "$BATCH_SIZE" --tsfm_num_samples 21
    --tsfm_cache_path "$cache" --tsfm_models chronos2
    --tsfm_model_paths "{\"chronos2\":\"$CHRONOS2_MODEL_PATH\"}"
    --tsfm_use_future_covariates 0 --regression_model lgbm
    --num_boost_round 2000 --early_stopping_rounds 100
    --lgbm_learning_rate 0.05 --num_leaves 63 --max_depth -1
    --min_data_in_leaf 40 --feature_fraction 0.9 --bagging_fraction 0.8
    --bagging_freq 5 --n_jobs "$N_JOBS" --log_period 50
    --use_ar_features --ar_lags 1,24,168 --ar_roll 24 --disable_shap
  )

  run_one() {
    local exp="$1"; shift
    local args=("${common[@]}")
    for i in "${!args[@]}"; do
      if [[ "${args[$i]}" == "--save_dir" ]]; then args[$((i+1))]="$out/$exp"; break; fi
    done
    echo "[$name][$exp] starting"
    python run_pipeline.py "${args[@]}" "$@"
  }

  if [[ "$STAGE" == "stage1" ]]; then
    run_one ar_lgbm --enable_tsfm 0 --no-use_tsfm_point --no-use_tsfm_uncertainty
    run_one tsfm_point --no-use_tsfm_uncertainty
    run_one tsfm_point_rawU --use_tsfm_uncertainty
    run_one tsfm_point_C --no-use_tsfm_uncertainty --use_tsfm_confidence
    run_one tsfm_point_C_weighted_mse --no-use_tsfm_uncertainty --use_tsfm_confidence \
      --sample_weight_mode confidence --sample_weight_lambda 1.0
  elif [[ "$STAGE" == "stage2" ]]; then
    # Direction is auto-diagnosed on the training period and remains fixed on test.
    run_one tsfm_point_C_fixed_asym --no-use_tsfm_uncertainty --use_tsfm_confidence \
      --use_asymmetric_loss --asymmetric_alpha 2.0 --asymmetric_direction auto
    run_one tsfm_point_C_confidence_asym --no-use_tsfm_uncertainty --use_tsfm_confidence \
      --use_asymmetric_loss --asymmetric_alpha 2.0 --asymmetric_direction auto \
      --sample_weight_mode confidence --sample_weight_lambda 1.0
  else
    echo "Usage: $0 [stage1|stage2]" >&2
    exit 2
  fi
}

run_dataset guangdong data/GD/guangdong_aligned.csv \
  configs/columns/CH/electricity_price_regress_safe.json \
  configs/columns/CH/electricity_price_tsfm_safe.json \
  configs/time/data_split_CH/guangdong_realtime.json

run_dataset reale_delu data/REALE/DE-LU_aligned.csv \
  configs/columns/REALE/DE/regress_dayahead_safe.json \
  configs/columns/REALE/DE/tsfm_dayahead_safe.json \
  configs/time/data_split_REALE/DE_dayahead.json
