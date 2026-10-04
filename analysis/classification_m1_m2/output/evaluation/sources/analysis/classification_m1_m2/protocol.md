# E39S / D59S — bounded classification and mask-independent appearance

Protocol fixed 2026-10-04 before this experiment's scores, after known-data
exploration and first-drop feedback. This is feedback-informed development.

## New source information

Organiser, relayed by Santosh: “The dataset is formed from crops taken from around
15 electrode sample images - we had to backwards engineer them into artificial
batches to ensure there were trends that would actually be possible to extract in
2 days! So the data is real images of electrodes, but chopped up from and organized
into batches that contain shared features”.

Batch labels are constructed visual groups. Batch 3 remains the challenge reference;
labels do not establish real supplier-lot provenance, phase truth or manufacturing
outcomes. Around 15 source images is not 15 confirmed independent electrodes. Parent
IDs are unavailable: crop holdouts can share a parent with training crops. Historical
and new crop/site CV are diagnostics, not source-held-out generalisation estimates.
No new IID permutation p-values or confidence intervals will be reported. The
separate source-overlap audit is label-blind; verified overlaps reveal minimum
dependence, while absence of overlaps cannot establish independence.

## M1: same measurements, bounded model change

Known development data: all 31 original sites (7/7/17), including flagged and cracked
reference cases. Each aligned detector set is one crop/site. Shared content order
uses E37S's finite, unique raw BSE correlation/FFT anchor pairs. No site IDs, frame
dimensions, raw intensity or explicit quality indicators are predictors.

- **M1 primary:** complete-dependency-gated existing 29, L2 multinomial logistic.
- Comparators: gated L1 OvR logistic, gated equal-prior automatic-shrinkage LDA,
  and matched ungated L1 (frozen v2 procedure reference).
- Median imputation (keep empty columns, no indicators) and StandardScaler fit
  only to the fold's training data. E37S invalidates bright, pore and joint-interior
  dependencies using data-derived quality flags. It does not establish valid phase
  boundaries or eliminate imputation-pattern shortcuts.
- Logistic C grid is exactly [0.1, 0.5, 2.0], class_weight=balanced. L2 uses lbfgs,
  multinomial, max_iter=4000; L1 uses liblinear, max_iter=2000. LDA uses lsqr,
  shrinkage=auto, priors=[1/3,1/3,1/3], with no search.
- Outer leave-one-crop/site-out; logistic C selected using training-only shuffled
  stratified three-fold class-balanced log loss, seed 0. All preprocessing is
  refit inside inner folds. Ties select the first/strongest regularisation. With
  fewer than two samples in any training class, use fixed C=0.5, disclosed.
- Class columns always map explicitly to Batch_1, Batch_2, Batch_3.

## M2: fixed new measurements, same bounded L2 procedure

Fixed 24 image-signature descriptors (eight per BSE/ETD/Inlens), without phase masks:
Gaussian band-energy fractions at scales 2/8/32 px; fine/coarse energy-fraction IQR
across a 3×3 grid of 512 px measurement tiles; sigma-8 gradient axial cos(2θ), sin(2θ)
moments and median axial coherence. Tile grid covers the shared BSE-trimmed extent;
context is explicit and measured union coverage is recorded. Exact equations,
border handling and source hashes are stored with extraction. Patches are coverage,
never independent labelled samples. Axes are image axes, not verified coating axes.

- **M2 primary incremental candidate:** gated29 + fixed24, same L2 multinomial/C
  selection and shared outer folds. Compare against M1 primary; no feature search.
- Fixed comparator: new24 alone with the same L2 procedure. Acquisition-sensitive
  appearance is the interpretation, not chemistry or battery harm.
- Extraction is deterministic and label-independent. Positive gain/offset without
  clipping is tested at fixed trim; gamma and quantisation are fixed challenges
  across all 31 sites. Perturbed copies never cross validation folds or count as n.
- For new24-only, use each original outer fit on the withheld site's perturbed
  descriptors to report changed bets and score movement. This tests the descriptor
  family, not whole-pipeline invariance or robustness to arbitrary acquisition.
- Report redundancy with existing BSE FFT/correlation summaries descriptively.
  Response maps illustrate what filters measure, not segmentation ground truth.

## Reporting / promotion

Report confusion matrices, all class recalls, balanced accuracy, ordinary accuracy,
class-balanced log loss/Brier, outer C choices, missingness and linear score-margin
contributions. Probabilities are **uncalibrated model scores under a balanced class
target**, not empirical posteriors or confidence of correctness. No automatic winner
search, label swap, calibration or ensemble.

Fit each fixed candidate on the 31 known sites and save it separately. Only after
the known-data stage is saved, score the three already-revealed first-drop sites
as failure diagnostics. Do not tune using their outcomes; they are development
evidence for these feedback-informed methods. Do not fit 34-site submission models
or replace the declared primary in this experiment. Any future submission version
requires a separate decision and frozen provenance. Existing submissions, QC
thresholds/hash, phase metrics and raw TIFFs are protected and unchanged.

If a complete parent map arrives, replace crop folds with parent-held-out outer and
inner folds. Missing class coverage is an explicit infeasibility, never silent crop
fallback. Verified-overlap grouping can only be a lower-bound dependence stress test.
