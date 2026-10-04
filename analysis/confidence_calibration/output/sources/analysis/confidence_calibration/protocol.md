# E43S / D67S: one scalar confidence layer for frozen v2

Fixed 2026-10-04 after E42S and user approval. Feedback-informed development,
not a pre-feedback registration. No new features, classifier, quality policy,
annotation, acquisition statistic, label override or confidence cutoff.

Use the actual D52 v2 frozen material pipeline and deletion models (29 inputs).
Its liblinear probabilities are normalised one-vs-rest scores. Transform those
scores as q = softmax(alpha * log(p)), alpha = 1/T > 0. Do not apply softmax
to its raw decision_function: that would fail to reproduce the original scores
at alpha=1. This single monotone scalar preserves the entire class ordering.
No intercepts/class-specific corrections. Fit alpha in [0.05,20] by equal-true-class
multiclass negative log likelihood using bounded scalar minimisation, checking
both endpoints and identity alpha=1 explicitly; numerical ties prefer identity.
This may sharpen as well as soften; neither direction is assumed to be better.

Outer crop-LOO: preserve all 31 frozen deletion-model scores. For each outer
training set of30, generate internal leave-one-crop-out scores from classifiers
trained on29. Each internal fit chooses C from [0.1,0.5,2] on its own training-only
stratified3 folds with seed0, fold-local imputation, scaling and class balancing.
Reuse each of465 unordered paired-deletion fits symmetrically; neither omitted
crop enters that classifier. Fit the scalar only on the other30 internal scores
and labels, then apply it to the untouched outer-held-out score. Persist every
train/validation index and fit. Internal training size29 is close to outer30,
but remains a one-crop difference. A pre-fit reviewer raised the larger mismatch
from the initial3fold draft (20 versus30); that draft is retained as
protocol_initial_threefold.md. No fit/outcome was inspected before this revision. Never calibrate the 31 outer
scores on all 31 labels and then claim held-out calibration. Actual source-group
IDs are unavailable; do not invent them. No IID tests or intervals.

For the full deployment candidate, fit one alpha to the31 frozen crop-LOO scores
from30-training-crop models. This fit is for deployment only; do not evaluate
that scalar on these same31 as held-out evidence. Apply it to the unchanged frozen full classifier.
Save the known evaluation, fits, promotion decision and receipt before reading
the already revealed drop scores/truth. Those three labels are diagnostics only,
not training, hyperparameter, criterion, threshold or promotion selection inputs.

Compare original, fitted scalar and fixed alpha=0.5 softening control; also report
a uniform vector with original bets retained solely as a flattening control.
Report full-label multiclass NLL/Brier and own-bet correctness Brier/binary NLL,
each with ordinary and equal-true-class weighting. All actual candidate bets must
remain identical, so their binary correctness targets remain identical too.
Fixed bins [1/3,.5,.6,.7,.8,1] and thresholds [.5,.6,.7,.8,.9] are descriptive
occupancy/error tables, not an organiser grading rule. Exact payoff remains unknown.

**Promotion rule fixed before fitting:** at least 1% relative reduction in both
equal-class multiclass NLL and equal-class own-bet Brier; no worsening of any of
the other six loss measures by more than 1e-6; all 31 bets unchanged; valid finite
scores and complete provenance/replay. This is a declared practical gate, not
a significance test. User has authorised replacement if this criterion is met.
Only the fitted scalar is eligible; controls cannot be selected. A supported
replacement requires a distinct version and freeze preserving v2. Otherwise
retain v2. No crop-level result certifies new-source calibration or confidence
as a probability of correctness. Preserve the classifier's feature explanations
and missing/quality flags separately; the scalar creates no new material evidence.
