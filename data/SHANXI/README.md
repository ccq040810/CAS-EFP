# Shanxi electricity market data

Source: [ronaldowzy/price_forecast](https://github.com/ronaldowzy/price_forecast),
file `data/价格预测数据集.csv`.

The checked-in source snapshot is a 15-minute Shanxi spot-market series from
2024-01-01 through 2026-01-14 (including the source's `24:00` labels). The
pipeline adapter `scripts/align_shanxi.py` converts four 15-minute records to
hourly means and writes `shanxi_aligned.csv` for a task-comparable experiment.

Target: real-time price (`y`). Day-ahead price and forecast variables are
available as covariates. Actual physical variables are retained for historical
context only; future actual values must not be passed as prediction-time
covariates.
