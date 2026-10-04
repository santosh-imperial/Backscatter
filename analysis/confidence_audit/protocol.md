# E42S / D64S: fixed L2 geometry probe and confidence audit

Fixed 2026-10-04 after E39S/E41S and revealed first-drop results. The user now
clarifies that judging rewards correct assignment, confidence quality and
explanations; exact payout/threshold definitions are unavailable. This is
feedback-informed development, not a pre-data/pre-feedback registration.

One additional candidate: complete-gated existing29 + exact E41S eight geometry,
multinomial L2 logistic regression. Reuse E39S's balanced-class lbfgs softmax,
maxiter4000/seed0, training-fold median imputation/StandardScaler and C exactly
[0.1,0.5,2.0] chosen by the same weighted fold-wise balanced log loss on training
stratified3folds. Shared content order, all31 crops, outer crop holdout. No feature,
seed, geometry definition, RF grid, loss-weight search or automatic promotion.
Primary new model is geometry_l2; existing gated29 L2 isolates added inputs, and
geometry L1 is the already-tested logistic comparator. Save known fits/scores and
receipt before loading the existing revealed-drop measurements/truth.

Confidence summaries compare all fixed measurement candidates from E39S and E41S,
deduplicating identical legacy/baseL1 outputs; acquisition controls remain separate
and are not candidates. Evaluate proper full-vector multiclass log loss and Brier
with observed-crop and equal-true-class weighting. Also report forced-bet correctness
binary Brier and NLL for q=max(model scores), z=(bet correct), under both weightings;
q is an uncalibrated candidate confidence, not established correctness probability.
The correctness target changes when a model's bet changes: these binary scores
are diagnostic and cannot replace common-label multiclass scores or assignment
accuracy as a single model-selection criterion.
Use fixed bins[1/3,0.5,0.6,0.7,0.8,1] with all occupancy/correct counts and all-class
composition. These descriptive reliability summaries are not calibration fitting,
independent significance or source-held-out validation. No new IID interval/p-value.

Report separately wrong-bet and correct-bet mean scores, paired changes on reference
correct/incorrect crops, true-class scores and changed bets. Lowering every score
can help wrong bets while weakening right bets; neither alone establishes better
confidence. Include the uniform3-class score vector as a flattening control using
each candidate's original bets for correctness scoring (explicitly not a new
argmax classifier). Include the outer-training class-prior vector as a separate
full-vector/bet baseline; no test label estimates its prior. Fixed threshold
diagnostics0.5/0.6/0.7/0.8/0.9 count high-score
wrong and right bets and coverage; these are sensitivity tables, not organiser
payoffs or a chosen high/low cutoff. Missing low-confidence-correct payout and
confidence semantics make the organiser's expected utility unidentifiable.

Existing known outputs remain immutable, the three revealed drop examples remain
development diagnostics, and no model/calibrator/threshold is chosen from them.
Preserve exact linear explanations and observed/imputed labels. Unknown shared
parents from around15 source images qualify every result. No new SME masks, 34-crop
fit, source group invention, correctness calibration or confidence re-labelling.
Current submission and QC remain unchanged; any future chosen revision needs a new
version and new-source evidence. Log all candidates, negative or positive.
