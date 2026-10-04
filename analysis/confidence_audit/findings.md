# E42S: confidence matters alongside the bet

The user clarifies that judging includes correct batch assignment, confidence
and explanation quality. Examples reward low-confidence mistakes over highly
confident mistakes; the exact rule and low-confidence-correct reward are unknown.
We assess probability quality and explanations without inventing a competition
utility or selecting from three revealed examples (D64S/D65S).

The eight additions were already tested with L1 logistic in E41S. This study adds
one fixed multinomial L2 probe, with the same complete-gated 37 inputs and E39S
training-only scaling/imputation/C selection. All 31 crops remain. Parent IDs are
unknown, so source-related crops can cross holdouts. No new IID p-values/intervals,
calibrator, source-held-out claim or SME labels. Known fits precede drop scoring.

## Model and score results

Lower losses are better; class-balanced means give each true batch equal weight.
Ordinary means reflect the supplied 7/7/17 crop mixture, which is not a verified
deployment prior. Balanced fitting does not calibrate scores automatically.

| Candidate | Balanced accuracy | Ordinary log loss | Equal-class log loss | Ordinary Brier | Equal-class Brier |
|---|---:|---:|---:|---:|---:|
| Preserved v2 L1 procedure | 0.607843 | 0.815916 | 0.889949 | 0.462452 | 0.513636 |
| Gated29 L1 | 0.568627 | 0.908893 | 0.986486 | 0.536502 | 0.592455 |
| Gated29 multinomial L2 | 0.445378 | 0.903811 | 0.961912 | 0.545128 | 0.587896 |
| Gated37 L1 | 0.521008 | 0.985665 | 1.090002 | 0.587521 | 0.661301 |
| **Gated37 multinomial L2, fixed new probe** | **0.397759** | **0.881224** | **0.966066** | **0.536642** | **0.599300** |

Adding eight to L2 improves ordinary log loss/Brier while worsening equal-class
losses and balanced accuracy. Thus “no accuracy gain” alone misses a score tradeoff.
The preserved v2 procedure still has better known-crop assignment and both proper
scores under both weightings; these crop diagnostics do not prove external performance.
All ten deduplicated fixed candidates, including forests, LDA and appearance models,
are in the [confidence report](report.html). The same L1 aliases are not extra models.

Confidence in the forced bet uses q=max(class scores), compared with whether that
bet was correct. Binary Brier/NLL and fixed-bin reliability are descriptive;
q is not established correctness probability. The target changes when a candidate
changes its bet, so these losses cannot by themselves rank classifiers or replace
common-label multiclass losses and accuracy. Reliability-bin and wrong/right mean
scores are observed-crop summaries with changing class composition; balanced losses
are a separate target. No calibration curve is fitted, and small bins are not
independent calibration evidence. Uniform confidence retaining each candidate's
bets checks blanket flattening; uniform full-vector and training-only class-prior
baselines are separate controls. Five fixed high-score thresholds report counts,
never an organiser payoff or chosen deployment threshold.

## Revealed-drop examples: a useful confidence pattern, with limits

| Crop / truth | V2 bet / score | Gated29 L2 bet / score | Gated37 L2 bet / score |
|---|---|---|---|
| 3e122cbj / B2 | B1 / 0.838211 | B1 / 0.583707 | B1 / 0.591778 |
| fn0mhxef / B1 | B2 / 0.495566 | B2 / 0.483763 | B1 / 0.471656 |
| xrv9xvzb / B3 | B3 / 0.621786 | B3 / 0.800000 | B3 / 0.778071 |

Gated29 L2 keeps the same bets yet lowers both wrong scores and raises the correct
score versus v2. That is a favourable developmental score pattern, with no accuracy
change. Gated37 L2 also fixes the second mistake, yielding 2/3; that correct bet
has modest score 0.471656 versus B2 0.388680. The remaining low-contrast B2 crop has
only 12/37 observed inputs; bright-related geometry remains unavailable. This is
already-revealed development evidence, not a new test or a model-selection objective.
Forests lower wrong scores but also weaken the correct B3 score. Gated37 L1 raises
both wrong scores and lowers the right score relative to gated29 L1 on these examples.

Full-vector drop log losses are v2 1.352475, gated29 L2 0.887823 and gated37 L2
0.871985. Their bet-correctness Brier values are 0.363743/0.204913/0.226200; the
last models do not have identical correctness targets because one bet changes.
Do not present any of these as calibrated confidence or an exact judging score.

## Explanation and delivery checks

The L2 probe retains exact standardised linear decision-margin terms plus the
intercept, comparing a non-B3 bet to B3 (B3 to runner-up). These are noncausal model
terms, not probability contributions or a material mechanism. Observed/imputed
terms remain distinct. An inherited E39 `family` metadata field mislabelled new
geometry as image appearance; a separately hash-bound derivative corrects that
label and retains `as_run_family`, without changing any terms, fits or original
files. [Correction receipt](output/explanation_metadata_correction.json).

Independent review replays all 31 folds within 1.1e-16, verifies training-only
medians/scaling, recomputes all 340 prediction rows, 20 score summaries, 100 fixed
reliability rows and 100 threshold rows. Margins reconstruct within 1.3e-15.
All 633 original/prior-study artifacts and 102 raw TIFFs remain unchanged.
The full suite again passes 232 tests, 15 existing warnings, 189.09s. Scientific
plots and report links/images are checked; native browser layout remains unverified.

No candidate, confidence cutoff or calibrator is promoted. Future model comparisons
must evaluate assignment, full-vector score quality, confidence tradeoffs and usable
explanations together. Next is the separately scoped frozen-DINO probe with these
score diagnostics; any calibration experiment must be fixed and fit inside outer
training folds, evaluated against unchanged scores and flattening controls, with
true source grouping if available. Current submission and QC remain unchanged.

[Protocol](protocol.md) · [Score summaries](output/confidence_summary.csv) ·
[Known-stage receipt](output/known_stage_receipt.json) ·
[Audit receipt](output/audit_receipt.json) ·
[Delivery checks](output/delivery_validation.json)
