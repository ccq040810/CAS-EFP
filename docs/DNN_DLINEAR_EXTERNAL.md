# Additional external baselines

Only DNN and DLinear are trained: two markets, three original windows and three
seeds42/43/44, totaling36 jobs. Existing prepared hourly-origin inputs, flat
row/horizon/time keys, missing-target masks and splits are reused unchanged.

DNN is an explicitly adapted feedforward MLP (256/128 ReLU, output24) using
the existing LEAR historical curves/calendar/exogenous input. Feature and target
median/MAD-asinh transforms fit training only. This is not an exact reproduction
of epftoolbox DNN's tuning or daily recalibration protocol.

DLinear implements the official univariate architecture: historical price168,
moving-average decomposition with kernel25 and replicated endpoints, separate
168-to24 seasonal/trend linear mappings, summed output. It does not consume
exogenous inputs. Architecture reference:
https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
Thus information availability differs from the exogenous-input methods and must
be reported rather than disguised as fully matched inputs.

Both use fixed Adam learning rate0.001, batch256, clip5, raw-unit validation RMSE
early stopping patience20. Maximum epochs are DNN150 and DLinear200. All jobs
save a selected checkpoint and TRAINING_FROZEN.json before loading test arrays.
Training uses eight GPU worker slots. No prior baseline or CAS model is refitted.

The primary reference is explicitly frozen to the pre-major-revision saved
original-window CAS result at results/cas_original_windows_saved_score_20261029:
Guangdong MAE57.379800/RMSE96.190876, DE-LU27.545974/40.015899. The old leaking
legacy_us replay is excluded. Metrics use float64 original prices (GDyuan/MWh,
DEeuro/MWh), first average windows per seed, then mean/std across seeds. Keep
all36 runs irrespective of the rankings. Prior test periods were observed in
development and are not a fresh independent holdout.
