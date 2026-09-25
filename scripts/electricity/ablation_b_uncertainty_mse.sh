#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

device="cuda"

data_path="data/electricity_price/combined_epf.csv"
regress_cols_path="tmp_electricity_regress.json"
tsfm_cols_path="tmp_electricity_tsfm.json"
data_split_path="tmp_electricity_split_ratio.json"

tag="electricity_ablation_b_uncertainty_mse"
model_name="pipeline_electricity_ablation_b_uncertainty_mse"
save_dir="./results/quick_electricity_ablation/${tag}"
tsfm_cache_path="./results/quick_electricity_ablation/tsfm_cache"
mkdir -p "$save_dir" "$tsfm_cache_path"

is_std=1
enable_tsfm=1
pred_len=24
seq_len=168
batch_size=32
tsfm_num_samples=20
tsfm_models="chronos2"
tsfm_use_future_covariates=0

tsfm_model_paths='{"chronos2":"/share/home/beiyou2/FutureBoosting/models/chronos-2/chronos-2"}'

num_boost_round=2000
early_stopping_rounds=100
lgbm_learning_rate=0.05
num_leaves=63
max_depth=-1
min_data_in_leaf=20
min_sum_hessian_in_leaf=1e-3
feature_fraction=0.9
bagging_fraction=0.8
bagging_freq=5
min_gain_to_split=0.0
reg_alpha=0.0
reg_lambda=0.0
n_jobs=8
log_period=50
seed=42

python run_pipeline.py \
  --device "$device" \
  --model_name "$model_name" \
  --seed "$seed" \
  --data_path "$data_path" \
  --regress_cols_path "$regress_cols_path" \
  --tsfm_cols_path "$tsfm_cols_path" \
  --data_split_path "$data_split_path" \
  --tag "$tag" \
  --save_dir "$save_dir" \
  --is_std \
  --enable_tsfm "$enable_tsfm" \
  --seq_len "$seq_len" \
  --pred_len "$pred_len" \
  --batch_size "$batch_size" \
  --tsfm_num_samples "$tsfm_num_samples" \
  --tsfm_cache_path "$tsfm_cache_path" \
  --tsfm_models "$tsfm_models" \
  --tsfm_model_paths "$tsfm_model_paths" \
  --tsfm_use_future_covariates "$tsfm_use_future_covariates" \
  --tsfm_force_current_rerun \
  --use_tsfm_uncertainty \
  --regression_model "lgbm" \
  --num_boost_round "$num_boost_round" \
  --early_stopping_rounds "$early_stopping_rounds" \
  --lgbm_learning_rate "$lgbm_learning_rate" \
  --num_leaves "$num_leaves" \
  --max_depth "$max_depth" \
  --min_data_in_leaf "$min_data_in_leaf" \
  --min_sum_hessian_in_leaf "$min_sum_hessian_in_leaf" \
  --feature_fraction "$feature_fraction" \
  --bagging_fraction "$bagging_fraction" \
  --bagging_freq "$bagging_freq" \
  --min_gain_to_split "$min_gain_to_split" \
  --reg_alpha "$reg_alpha" \
  --reg_lambda "$reg_lambda" \
  --n_jobs "$n_jobs" \
  --log_period "$log_period"
