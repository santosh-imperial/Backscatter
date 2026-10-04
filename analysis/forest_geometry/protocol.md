# E41S / D62S — fixed Random Forest × morphology comparison

Fixed 2026-10-04 before E41S scores, after earlier exploratory work, E39S negative
results and first-drop feedback. Santosh authorises this experiment and parallel
agents. This is feedback-informed development, not a pre-data registration.

## Labels and dependence

The 31 original known crops have labels7/7/17. The organiser constructed artificial
visual batches from crops of around15 electrode source images. Labels identify
visual groups, not verified supplier lots or battery outcomes. Parent IDs and
independent specimen counts are unavailable: crop-LOO can share a source across
folds. E40S finds no recoverable overlaps but cannot identify non-overlapping
related crops. No source-held-out claim, new IID p-value or confidence interval.
The three revealed first-drop labels are development failure diagnostics only.

## Fixed two-by-two and controls

| Candidate | Inputs | Model | Role |
|---|---|---|---|
| base_l1 | complete-gated existing29 | matched L1 | factorial baseline |
| base_rf | same29 | fixed forest | isolates model change |
| geometry_l1 | same29 + fixed8 geometry | matched L1 | isolates added inputs |
| geometry_rf | same37 | fixed forest | primary interaction candidate |
| legacy_l1 | ungated existing29 | matched L1 | preserved v2 procedure reference |
| quality_rf | existing bright/pore bad-or-unknown flags | fixed forest | acquisition control, never bet |
| availability_rf | finite-input indicators of gated37 | fixed forest | coverage/imputation shortcut control, never bet |

No feature selection, architecture/grid search, ensemble, seed selection or calibration.
The primary and all comparisons are fixed before scores; comparator superiority does
not select a new primary or submission. Frame dimensions, raw detector levels,
site IDs, explicit QA flags and coverage counts are not measurement-model predictors.
The last two controls deliberately expose acquisition/coverage information and are
labelled separately. Windows/components/trees are not independent labelled observations.

## Eight added measurements

`bright_aspect_count_iqr`, `bright_circularity_count_iqr`,
`bright_solidity_count_q10`, `void_local_width_d50_px`,
`void_local_width_d90_px`, `graph_edge_length_iqr_ratio`,
`local_bright_std_512px`, `bright_pore_crosscorr_xy_contrast_256px`.

Reuse exact E24/E25/E26G/E30K definitions, nominal caches and coverage gates, with
source hashes and fixed-example raw parity checks before fitting. No winning
univariate descriptor/lag selection. Shapes, graph spacing and local bright-fraction
dispersion are bright-only; widths are pore-only; cross-phase arrangement is joint.
Use E37S data-derived flags (unknown fails closed) plus original computational
coverage gates. Invalid/unavailable inputs become NaN; per-cell reasons are saved.
The new geometry API extracts the same measurements on raw future/drop crops with
no SME labels. Source/phase chemistry and collector direction remain unvalidated;
all widths are pixels and correlations refer to image axes. Crop extent/clipping
can affect these descriptors without being an explicit predictor.

## Models, folds and scores

- RF:500 trees, gini, max_depth3, min_samples_leaf4, max_features=sqrt,
  class_weight=balanced, bootstrap=True, seed0, no OOB score. No RF tuning.
- L1: liblinear, balanced classes, max_iter2000, C exactly[0.1,0.5,2.0]. E37S/E39S
  training-only three-fold stratified balanced-log-loss selection, seed0; same
  weighted fold-loss convention and tie rule. If min class count<2, fixedC0.5;
  missing training class is an explicit error.
- Shared content order from original raw BSE correlation/FFT anchors, held constant
  across variants; outer leave-one-crop-out. Whole aligned image set is one crop.
  Train-fold median imputation, keep_empty_features=True, no missing indicators;
  L1 uses training StandardScaler, RF needs no scaling. Empty training columns
  retain positions with the imputer's declared zero fallback.
- Report every class recall/confusion, balanced/ordinary accuracy, balanced log loss
  and Brier, observed input counts/C choices and quality subgroup counts. Reproduce
  original matched ungated/gated L1 scores before accepting the benchmark.
- Scores are uncalibrated model probabilities under class balancing, not confidence
  of correctness. Source-grouped outer AND inner folds require actual parent IDs;
  no proxy grouping or crop fallback described as grouped validation.

## Prespecified diagnostics

1. Known geometry variants: nominal, joint threshold−5/+5 and bright object-floor100.
   Retain original void floor30; raster window/correlation values are floor-invariant
   by definition, not independent robustness evidence. Fixed original fits score
   held-out altered geometry. Existing29 stay unchanged: **partial-input mask
   sensitivity**, not full-pipeline acquisition invariance or retraining.
   Nominal-unavailable cells cannot become observed in variant scoring; additional
   variant coverage failure stays unavailable. Original quality gate is fixed.
2. Primary forest seeds1 and2 repeat crop holdouts as fixed stochastic sensitivity
   only; seed0 remains primary. No favourable seed is chosen or extra n inferred.
3. Primary forest replaces observed inputs in each fixed feature family by its outer
   training median (zero fallback only for empty training columns), preserving NaNs.
   Report score/bet/margin changes as **fixed-model input sensitivity**. Correlated
   hybrid inputs can be implausible: this is not refit ablation, causal importance,
   material validity or independent testing.
4. RF local explanations average root class probabilities and each decision-path
   node probability change assigned to the splitting feature. The class-difference
   contributions plus root difference must reproduce the score margin. This exact
   path decomposition is order/correlation dependent, noncausal and not SHAP.
   L1 uses exact standardised linear decision-margin contributions plus intercept;
   its units are not probability contributions. Mark imputed inputs unobserved.

Save all known fits/results and a `stage=known_only_complete` receipt before raw
first-drop geometry extraction/reading truth. Only then use known31 fits to diagnose
the three revealed failures. No use of drop labels in extraction, fitting, input
choice or candidate/seed selection. No34-site submission fit. Save forest fold fits
individually/compressed so each artifact stays below20MiB. Sources/raw images,
E39S/E40S and original v1/v2/QC artifacts are protected; never overwrite trial output.

## Delivery and decision

Report all fixed outcomes, negative or positive, with measured coverage and actual
geometry images. Update logs/append-only registry and the existing metric inventory
without expert/QC promotion. Full repository tests, rejected-value and fold-leakage
regressions, replay/attribution/source checks. Current v2/QC remain unchanged pending
a separate model-version decision; three development failures and crop CV cannot
establish new-source or real manufacturing performance.
