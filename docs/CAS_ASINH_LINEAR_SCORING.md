# Frozen linear-head development scoring

The three-seed validation c result is MAE52.92616073281138 and
RMSE92.25457354943775. Both improve the prior nonlinear confirmation. Fixed
asymmetry has slightly lower validation MAE52.91249246062054, so these results do
not establish that confidence weighting is uniformly superior to its ablations.

`score_cas_asinh_linear_original_windows.py` requires the independent confirmation
audit to pass. It snapshots all saved models, calibration and training-only
preprocessing before scoring. Each head transforms its original test features
with its unchanged saved preprocessing, predicts its transformed residual, and
inverts the saved target transform. No fitting or calibration takes place.

All original windows and seeds remain included. The German models and existing
external baseline predictions remain unchanged; the Guangdong standalone tree
receives the same linear-head capacity and historical features without upstream
inputs. Market-specific settings must be disclosed. All 162 records, including
ordinary/fixed/c/shuffled-c and the original fusion baseline, are retained.

These original test periods have already been observed in development. Any
rankings are development results, not independent confirmation of generalization.
Previous unsuccessful configurations and subsequent-period results are retained.
