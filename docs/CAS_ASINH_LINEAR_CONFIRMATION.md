# Frozen three-seed linear-head confirmation

The finite seed42 study passed the decision rule recorded before its metrics
were inspected: the three-window c method improved both MAE and RMSE over the
nonlinear seed42 reference (53.180123970647344, 92.69189747620028). Its result is
52.96040797311079, 92.31560099091452. These are validation results only.

Confirmation requires a successful independent pilot audit. It preserves every
head's seed42 training preprocessing, target scale, residual center, calibration,
tree capacity, lambda and validation early-stopping rule. All 360 seed42 models
are reused exactly; 720 models add seeds43/44. No probes or preprocessing are
refitted. Ordinary/fixed/c/shuffled-c/standalone-tree are all retained, yielding
1080 child records and 45 reconstructed window-method-seed records.

The plan snapshots source models, predictions, calibration and preprocessing
before training and checks their hashes afterward. Only training and validation
arrays are indexed. Three-window averages precede seed averages and deviations.
No test scoring occurs here. Further comparisons must disclose that the original
test periods have already been observed during development.
