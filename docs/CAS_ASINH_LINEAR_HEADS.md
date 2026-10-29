# Fixed validation-only linear-leaf study

This finite study tests linear leaves on the already selected asinh, full-curve,
per-horizon representation. Prior linear leaves improved validation on the old
representation, but that experiment did not evaluate this combination. The
hypothesis is that a linear leaf can retain local linear relationships in the
historical curves while the confidence-weighted objective corrects residuals.

The only new leaf setting is `linear_tree=True, linear_lambda=1`. No regularizer
search is performed. The original three Guangdong windows, seed42, five matched
methods (ordinary, fixed, c, shuffled c, standalone tree) yield 360 models and 72
fixed 300-round training probes, reconstructed into 15 window-level records.
All controls receive the same linear-leaf capacity and original curve features;
the standalone tree excludes all upstream-derived inputs. Confidence is only in
the transformed-residual training objective, never in prediction features.

Each head fits imputation and normalization on training rows only. The probe
fits its own preprocessing and target transformation on its purged fit80%; the
remaining training20% calibrates direction and strength. No validation or test
statistics fit those transformations. CPU execution uses four parallel heads,
two threads each. Validation RMSE in original price units controls early stopping.
Inverse-transform overflow is an explicit failed candidate, never clipped away.

The immutable plan is saved before training. Only train/validation arrays are
indexed. All windows, methods, numerical failures and earlier studies are kept.
Further seeds require useful validation improvement; no automatic test selection
or repeated parameter search is authorized by this script. Previously inspected
test periods remain development evaluations, regardless of subsequent rankings.
