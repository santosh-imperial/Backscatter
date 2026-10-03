# Site categoriser and baseline-membership assessment (`polaron_qc.categorise`)

Three **separate** answers per sample (sample = one site = one BSE/ETD/Inlens image set), never conflated:

1. **Site categorisation** — "does this sample resemble Batch 1, 2 or 3?" A 3-class model over the known
   batches. Its outputs are **model probabilities** (normalised classifier scores), not posteriors and not
   confidence, until the calibration assessment in §4 says otherwise.
2. **Baseline OOD assessment** — "is it outside Batch 3's promised distribution?" A robust distance to the
   Batch 3 sites with leave-one-site-out reference percentiles. A 3-class model must pick one of the known
   batches even for an unfamiliar sample, so a high Batch 3 probability cannot establish baseline membership;
   (2) is built independently of (1).
3. **QC hand-off** — per sample: the features that drove (1) and (2) with sign and the Batch 3 median/MAD,
   the data-derived acquisition flags, and a pointer to the batch-level QC verdict (`polaron_qc.report`),
   which is a separate output and is not changed here.

Context from the organisers (relayed in the task brief as decision D49P; that entry is not present in the
`docs/decision_log.md` of the commit this branch starts from, 87ecee3, so it is quoted from the brief):
Batch 3 is the supplier-promised baseline; Batches 1 and 2 arrived later and do differ, showing the kinds of
variation to detect; judging is to categorise held-back samples and to say whether an unknown batch is inside
or outside the baseline distribution.

Provenance: developed using exploratory analysis of Batches 1–3 (all three were inspected long before this
module existed; see docs/problem_and_findings.md); the procedure below is frozen before the unseen batch is
examined. Nothing here feeds `decision.py`, `report.py` thresholds or the frozen notebook.

Script / module: `polaron_qc/categorise.py`; CLI `python -m polaron_qc.categorise`. Every number in §3–§6 is
read from a CSV/JSON in `analysis/categoriser/output/`.

## 1. Pre-registration — written and committed before the first run

This section was written and committed before `polaron_qc/categorise.py` was executed on the real cache.
It is not edited afterwards; departures are recorded under "Deviations" at the end of this section.

### 1.1 Units and data

31 known sites: 17 Batch 3, 7 Batch 1, 7 Batch 2 — one row per site from `features.extract_batch` (cached
tables in `analysis_cache/features`, FEATURE_VERSION 1.1.0), plus the per-site acquisition statistics that
`report.derive_flags` derives from the `images` table. Patches never enter any count. Specimen independence
between sites is unconfirmed (plan §2.0 rule 9), so "31 independent units" is an assumption stated, not a fact.

### 1.2 Feature families (exact column lists; nothing added after this commit)

* **MORPH** (19) — "morphology-only": `MATERIAL_KPIS` (`pore_frac, pore_d50, pore_elong, pore_max_d,
  crack_frac, crack_count_per_Mpx, bright_frac, bright_count_per_Mpx, bright_d50, bright_d90, bright_circ,
  corr_len_px, fft_slope, etd_crack_density_particles`) plus the trusted BSE geometry KPIs of the catalogue not
  in that list: `pore_d90, pore_count_per_Mpx, bright_d10, bright_solidity, bright_max_d`. No intensity
  statistics. Excluded on purpose: `graphite_frac` (near-complement of pore + bright), the 20 `profile_*`
  columns (positional, too many for 31 sites), every battery-secondary KPI (trust "exploratory" by rule).
* **ACQ** (10) — "acquisition-only": `bse_p1, bse_std, bse_empty_bin_frac, H, bright_sep,
  etd_boundary_sharpness, etd_curtain_frac, bse_p50, etd_p50, inlens_p50`.
* **TEXTURE** (10) — the ETD/Inlens texture family, labelled throughout as *"acquisition-sensitive;
  batch-fingerprint evidence, origin (microstructure vs imaging) not established"*:
  `etd_crack_density_graphite` (= `crack_g_0.4`, the T* value), `crack_g_0.05, crack_g_0.1, crack_g_0.2,
  crack_g_0.3` (ridge-density sensitivity grid), `ridge_p97, inlens_particle_texture,
  inlens_particle_texture_p90, inlens_speckled_particle_frac, inlens_grad_energy`.
* **COMBINED** (39) = MORPH + TEXTURE + ACQ.

Known facts at the time of writing: `bright_sep` is NaN on one site (`4ih2ggld`, bright mode unresolved);
no candidate column is constant. Hence a median imputer is part of every fold-local pipeline.

### 1.3 Site categorisation (answer 1)

Per family, the **same procedure**:

* Leave-one-site-out over the 31 known sites. Rows are first put in the canonical content order of
  `ml._canonical_order` (D43) so nothing depends on site ids or input row order.
* Inside every training fold, in this order: `SimpleImputer(median)` → `StandardScaler` →
  `LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", random_state=seed)`
  (one-vs-rest; the three OvR scores are normalised to sum to one — these are the **model probabilities**).
  The regularisation strength `C` is chosen **inside the fold** from the fixed grid {0.1, 0.5, 2.0} by inner
  stratified 3-fold CV over the training sites (shuffled, `random_state=seed`), criterion = class-balanced
  multiclass log-loss (each class weighted to equal total weight); the fold model is then refitted at that `C`
  on the whole training fold. The L1 penalty is the only feature selection; there is no univariate screening.
* Outputs per site: out-of-fold model probabilities for Batch 1/2/3, argmax.
* Metrics: accuracy with a Clopper–Pearson 95 % interval (n = 31), **balanced accuracy** (mean per-class
  recall; chance 1/3), per-class recall, the full 3 × 3 confusion counts. Majority-class accuracy is 17/31 =
  0.548 and is printed next to accuracy.
* Permutation test: **200** site-label permutations; the entire fold-local procedure (imputer, scaler, inner
  C selection, refit) is repeated inside every permutation; statistic = balanced accuracy;
  p = (b + 1)/(n + 1), one-sided, Monte Carlo; seed 0.
* Calibration (assessed, not assumed): multiclass Brier score = mean over sites of Σ_k (p_k − 1[y = k])²,
  compared with the Brier of the training class prior (7/31, 7/31, 17/31 → 0.598); reliability of the
  top-class probability in three fixed bins [1/3, 0.5), [0.5, 0.7), [0.7, 1] (bin mean top-class probability
  vs observed share of correct argmax, with the bin count), and the expected calibration error over those bins.
  With 31 sites the bins are coarse; the result is a statement of what was observed, not a calibration
  guarantee. The word "probability" in the output stays "model probability" unless the reliability table and
  Brier both support calibration, and even then LOO on 31 sites cannot establish it for an unseen batch.

**Primary model (pre-registered): COMBINED.** MORPH and ACQ are secondary; three models are compared, so a
secondary p is read against a Bonferroni-style α of 0.05/3 ≈ 0.017, and the primary against 0.05. A separate
TEXTURE-only model is **not** run (it would be a fourth comparison and the family is already inside COMBINED).

Interpretation rules fixed in advance:
(a) a low p for ACQ or for COMBINED establishes a **batch fingerprint** in the images, not a microstructure
difference — predictive usefulness of acquisition-sensitive features does not settle whether the signature is
microstructure or imaging;
(b) only a low p for MORPH could support a material reading, and even then it is an exploratory classifier
whose features include low-contrast-affected bright-phase KPIs on two Batch 1 sites (those sites are kept, and
flagged in the hand-off; they are not excluded because the procedure must run unchanged on an unseen batch);
(c) an argmax is a resemblance label among known batches; it is never a verdict and never "baseline membership".

### 1.4 Baseline OOD assessment (answer 2)

Reference = the 17 Batch 3 sites (all of them: Batch 3 is the promised baseline *including* its grey-pore and
cracked sub-populations; the sub-population membership of each reference site is printed so a reader can see
which neighbours a query lands on).

* Standardisation: per feature, robust z = (x − median_ref)/(1.4826 · MAD_ref); if MAD_ref = 0 the IQR/1.349
  is used; if that is also 0 the feature is dropped from the distance for that fit and the drop is recorded.
  Query NaNs are imputed with the reference median.
* **Primary score: mean Euclidean distance to the k = 3 nearest reference sites in robust-z space** (chosen
  before any run because the reference is heterogeneous; a centroid distance would penalise a query for
  resembling a Batch 3 sub-population). Secondary scores, reported side by side: RMS robust z (distance to the
  reference medians) and Mahalanobis distance with a Ledoit–Wolf shrunk covariance fitted on the reference z
  (p ≈ n, so the shrinkage is doing real work and this score is listed last).
* Reference leave-one-site-out: each Batch 3 site is scored against the other 16 with median/MAD and the
  neighbour set recomputed on those 16. Query sites (Batch 1, Batch 2, any `--score` folder) are scored
  against all 17.
* Reporting per query site: weak percentile of its score within the 17 reference LOO scores
  (`scipy.stats.percentileofscore`, kind="weak"); `exceeds_ref_loo_max`; rank p = (#{ref LOO ≥ score} + 1)/18,
  whose **floor is 1/18 = 0.056** — no single site can be called outside the reference at 5 % by rank alone.
  Per batch: fraction of sites above the reference LOO maximum; under an i.i.d. null one site exceeds with
  probability 1/18 and at least one of seven with probability 1 − (17/18)^7 ≈ 0.33 — the exceedance is an
  evidence flag, not a defect call (plan §2.0 rule 5).
* Matched-count sensitivity: every query is also scored against each of the 17 LOO reference sets (16 sites
  each) and the median percentile is reported, because the primary compares a 17-site memory for queries with
  16-site memories for the reference.
* Variants, labelled: **MORPH (primary)**, MORPH + TEXTURE ("texture-inclusive; acquisition-sensitive"),
  ACQ ("acquisition-only; session fingerprint").
* Top contributing features per query = the features with the largest mean squared z-difference to its k
  nearest reference neighbours (primary score), with the sign of the query's own robust z.
* The percentile is never called a probability of being defective, out of specification, or "rejected".

### 1.5 Hand-off (answer 3)

Per site row: model probabilities per family and argmax; OOD score, percentile, exceedance flag per variant;
the top three categoriser contributions (coefficient × standardised value for the predicted class, with sign,
Batch 3 median and MAD of that feature); the top three OOD contributions; the acquisition flags from
`report.derive_flags` / `apply_derived_flags` (`acquisition_group, bright_low_contrast, grey_pore,
raised_black_level, contrast_stretched_bse, cracked_known`); the text pointer "batch-level QC verdict:
`python3 -m polaron_qc.report <reference> <batch>` — separate output".

### 1.6 Context only

Kruskal–Wallis H across the three batches per candidate feature (39 tests, unadjusted p, printed as context).
No selection, weighting or model choice reads this table.

### 1.7 Deviations from §1.1–§1.6

(none at the time of the first run — to be filled in if anything above had to change; each line says what,
why, and whether the number it affects was seen before the change)

## 2. Setup and provenance

(filled after the run)

## 3. Site categorisation — results

(filled after the run)

## 4. Calibration of the model probabilities

(filled after the run)

## 5. Baseline OOD assessment — results

(filled after the run)

## 6. Univariate context (Kruskal–Wallis)

(filled after the run)

## 7. Limitations

(filled after the run)

## 8. Checklist applied

(filled after the run)
