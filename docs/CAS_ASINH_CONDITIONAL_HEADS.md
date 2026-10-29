# Finite confidence-conditioned cost diagnostic study

The new fixed ordinary linear-head validation residuals retain uncertainty versus
absolute transformed-error rank correlations: per-horizon median ranges about
0.23–0.35 across windows and seeds. Low-confidence minus high-confidence signed
transformed bias is negative in 69/72 horizon-seed combinations in all three
windows. Day-block intervals often cross zero, so this is conditional diagnostic
evidence, not causal proof. No test residuals select the proposed change.

The existing linear-c excess cost has limited numerical variation: median
focused coefficient variation relative to fixed asymmetry is approximately
0.30%, 0.22%, 0.56% in the three training windows. A common direction per horizon
can also obscure distinct confidence-group biases. This motivates one fixed
conditional-cost study on the new representation; the older representation's
unsuccessful conditional study remains preserved.

Each horizon uses its existing linear-tree features, capacity and train-only
preprocessing. A deterministic 300-round probe is reconstructed on purged early
training80%, including its own preprocessing and target scale. Three confidence
groups use thresholds fitted on that early segment. Remaining training20%
estimates each group's signed transformed bias, shrunk to its horizon mean with
100 pseudo samples. Day-block SE softens excess cost, capped at0.5; fewer than10
samples or7 days produces neutral costs. All settings are fixed before fitting.
Confidence affects both the direction and strength of asymmetric loss costs;
this explicitly extends the original common-direction formula. It never enters
prediction features. Validation does not estimate the cost table.

The 360 original seed42 checkpoints are reused exactly, with their source hashes
and predictions retained. Only conditional-c/conditional-mean/conditional-shuffle
add216 models, plus72 probes. All eight methods yield24 three-window records.
Mean and shuffled controls match each horizon's candidate coefficients on each
error side, not the realized residual subsets of successive boosting iterations.

Advance only if conditional-c beats the prior seed42 linear-c result
(52.96040797311079,92.31560099091452) AND conditional-mean AND conditional-shuffle
in both three-window mean validation metrics. Otherwise archive this study;
there is no lambda/bin/pseudo-sample search or automatic test scoring.
