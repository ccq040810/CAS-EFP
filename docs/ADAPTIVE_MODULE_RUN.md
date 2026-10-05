# Adaptive module experiment

The adaptive method treats confidence awareness and asymmetric loss as two
switchable modules of the same TSFM--LightGBM correction pipeline.  The
rolling experiment selects the switch state from validation data only, locks
it, and evaluates the following test segment once.

## GPU inventory on `gpu36`

Run this after connecting to the host:

```bash
hostname
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
```

The number of lines printed by `nvidia-smi -L` is the number of visible GPUs.
Before launching jobs, inspect free memory and utilization.  Pin independent
jobs with `CUDA_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=1`, etc.; do not let
several large Chronos jobs share one GPU unless memory is known to be safe.

## Run the rolling candidates

```bash
cd /path/to/FutureBoosting
CUDA_VISIBLE_DEVICES=0 python scripts/run_rolling_adaptive_experiment.py \
  --output-root results/rolling_adaptive_v4 \
  --cache-root results/rolling_adaptive_cache_v4 \
  --only guangdong shanxi \
  --windows 3 \
  --device cuda
```

The script writes a training-fitted P95 threshold for each window.  To use
multiple GPUs, start one process per GPU and assign disjoint markets, for
example one process for Guangdong and one for Shanxi.  Do not run the same
market/window twice into the same output directory.

## Summarize module activation and regret

```bash
python scripts/adaptive_module_selector.py \
  --root results/rolling_adaptive_v4 \
  --output results/rolling_adaptive_v4/adaptive_module_summary.csv \
  --risk-weight 0.5 \
  --max-normal-degradation 0.05
```

The output reports the selected switch state, every fixed-mode score, the
post-hoc oracle mode (diagnostic only), and selection regret.  The oracle is
never used to select a test configuration.

## Recommended paper comparison

Report fixed base, fixed confidence, fixed asymmetric, fixed confidence plus
asymmetric, adaptive selection, and the post-hoc oracle.  Use the adaptive
result as the method and the oracle only as an upper bound.
