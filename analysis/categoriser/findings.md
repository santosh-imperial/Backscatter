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

Recorded after the run; §1.1–§1.6 above are unchanged from commit 691990e.

1. *Post-hoc OOD variant* "morphology without bright-phase KPIs" (§5.3) was added **after** reading the primary
   OOD table, to diagnose why the two low-contrast Batch 1 sites exceed the reference. It is labelled post hoc
   everywhere and does not replace the pre-registered primary.
2. *Seeds 1 and 2* (§3.3) were run after the seed-0 results were seen; disclosed as a range, not averaged (D45).
3. *Implementation detail, numbers unaffected:* the NaN sentinel used to make `ml._canonical_order` NaN-safe was
   first set to −1e300, which overflowed inside its rounding step (a NumPy warning); it was changed to −1e30 and the
   evaluation re-run — every number was identical before and after.
4. *Procedure, no number affected:* a `--score-cache-dir` CLI option was added after the first run so a scored
   folder can use a scratch feature cache without writing into the shared one.

## 2. Setup and provenance

* Branch `worktree-agent-a861a0e9f8558250c` from main 87ecee3. Pre-registration (§1) committed as 691990e before
  `polaron_qc/categorise.py` was first executed on the cache. Site tables: `features.extract_batch` cache hits
  (`Batch_3_21893e9e3a20`, `Batch_1_8d2c001bed2f`, `Batch_2_c5b1475904f3`, FEATURE_VERSION 1.1.0); flags and
  acquisition covariates from `report.derive_flags` → `apply_derived_flags` (data-derived; on these 31 sites the
  groups equal the constant lists: 4 grey-pore, 2 low-contrast, 3 cracked-known).
* Command: `python3 -m polaron_qc.categorise Dataset/Batch_3 Dataset/Batch_1 Dataset/Batch_2 --n-perm 200 --seed 0`
  (34 s on 12 cores; the LOO procedure alone is ≈ 0.5 s per family). Outputs in `output/`: `categoriser_summary.csv`,
  `confusion_<family>.csv`, `reliability_<family>.csv`, `perm_null_<family>.csv`, `ood_batch_summary.csv`,
  `ood_ref_loo_<variant>.csv`, `site_table.csv` (one row per site, all three answers), `kruskal_context.csv`,
  `run_meta.json`, `categoriser_table.md/.html`, and the post-hoc `ood_posthoc_morph_no_bright.csv` (§5.3).
* `--score` path exercised once on a renamed fixture (§3.4): three Batch 1 sites hard-linked under new ids
  (`u156a27e, u2d17d8d, ue93893e` = ffwubibz, f1vzngrs, iv6g2oq0), cold feature extraction into a scratch cache
  (128 s for 3 sites), flags from the data (all three "ordinary"), scored with the models fitted on the 31 known sites.
* Seeds 1 and 2 were run after the seed-0 result was seen (§3.3); they are disclosed as a range, not averaged (D45).

## 3. Site categorisation — results (31 sites, leave-one-site-out, 200 site-label permutations, seed 0)

### 3.1 Headline table

| family | n feat | accuracy [CP 95 %] | majority | balanced acc. (chance 0.33) | recall B1 / B2 / B3 | perm. p (b ≥ obs of 200) | null bal. acc. median / p95 / max | α to read against |
|---|---|---|---|---|---|---|---|---|
| **combined (PRIMARY)** | 39 | **0.677 [0.49, 0.83]** | 0.548 | **0.664** | 0.71 / 0.57 / 0.71 | **0.005 (0)** | 0.333 / 0.473 / 0.608 | 0.05 |
| morph (secondary) | 19 | 0.387 [0.22, 0.58] | 0.548 | 0.347 | 0.57 / 0.00 / 0.47 | 0.159 (31) | 0.333 / 0.471 / 0.560 | 0.017 (Bonferroni, 3 models) |
| acq (secondary) | 10 | 0.548 [0.36, 0.73] | 0.548 | 0.473 | 0.57 / 0.14 / 0.71 | 0.035 (6) | 0.333 / 0.434 / 0.597 | 0.017 (Bonferroni, 3 models) |

Confusion counts (rows = true, columns = predicted B1 / B2 / B3):

| family | true Batch 1 | true Batch 2 | true Batch 3 |
|---|---|---|---|
| combined | 5 / 2 / 0 | 2 / 4 / 1 | 2 / 3 / 12 |
| morph | 4 / 1 / 2 | 5 / 0 / 2 | 4 / 5 / 8 |
| acq | 4 / 3 / 0 | 2 / 1 / 4 | 2 / 3 / 12 |

Reading, with the rules fixed in §1.3:

* The pre-registered primary (combined) separates the three batches above the site-label null (no permutation of
  200 reached its balanced accuracy; p = 1/201 is the Monte Carlo floor). Its accuracy interval [0.49, 0.83] is
  0.34 wide: 31 sites cannot say whether this categoriser is "two-thirds right" or "half right".
* **Morphology alone does not categorise** (balanced accuracy 0.35, p 0.16; Batch 2 recall 0 — every Batch 2 site is
  called Batch 1 or Batch 3). In 9 of the 31 folds the fold-local C = 0.1 zeroed every coefficient, so those sites
  received the uniform 1/3 vector. The one morphology KPI that the model keeps selecting is `pore_count_per_Mpx`
  (selected in 71 % of LOO folds for both B1 and B2 one-vs-rest scores; Batch 2 median 86 /Mpx vs 98 for Batch 3 and
  101 for Batch 1), followed by `pore_max_d`, `bright_d50`, `crack_count_per_Mpx`, `corr_len_px`.
* **Acquisition alone is at the edge** (p 0.035, not below the Bonferroni 0.017 for a secondary): its LOO selections
  are `etd_boundary_sharpness`, `etd_curtain_frac`, `bse_empty_bin_frac`, `etd_p50`, `bse_p50`, `bse_std`,
  `inlens_p50`, `H` (each in ≥ 87 % of folds). Batch 1 vs Batch 3 is carried by boundary sharpness and curtaining,
  Batch 2 by the BSE / Inlens medians (bse_p50 49 vs 57; inlens_p50 higher).
* The primary's gain over morphology comes from the texture family and the acquisition statistics: the features
  selected in ≥ 97 % of its LOO folds are `ridge_p97` (−, B1), `inlens_particle_texture_p90` (+, B1),
  `bse_empty_bin_frac` (−, B1), `pore_count_per_Mpx` (B1 +, B2 −), `pore_max_d` (+, B3), `inlens_p50` (+, B2),
  `corr_len_px` (−, B2), `bse_p50` (B1 +, B2 −), `etd_boundary_sharpness` (+, B1), `bse_std` (−, B3). Full-fit
  (C = 0.5) non-zero coefficients: 8 / 8 / 7 per class.
  **Rule (a) applies: this is a batch fingerprint in the images. Predictive usefulness of ETD/Inlens texture and
  of session statistics does not settle whether the signature is microstructure or imaging; one session per batch
  makes the two inseparable in this dataset.**
* Univariate context agrees (§6): 9 of the 10 texture features and 4 of the 10 acquisition features differ across
  batches at unadjusted p < 0.05, against 1 of the 19 morphology features.

### 3.2 Where the primary goes wrong (LOO rows; model probabilities of the fold that did not see the site)

| true | site | flags | mp B1 / B2 / B3 | argmax |
|---|---|---|---|---|
| Batch 1 | ffwubibz | ordinary | 0.30 / 0.55 / 0.16 | B2 |
| Batch 1 | fzrt2k6r | ordinary | 0.334 / 0.335 / 0.331 | B2 (a three-way tie) |
| Batch 2 | b3esycq1 | ordinary | 0.85 / 0.00 / 0.15 | B1 |
| Batch 2 | epqdaau9 | ordinary | 0.71 / 0.01 / 0.28 | B1 |
| Batch 2 | rxax5ozo | ordinary | 0.11 / 0.35 / 0.54 | B3 |
| Batch 3 | 0grcilhi | cracked | 0.37 / 0.31 / 0.32 | B1 (near tie) |
| Batch 3 | cfe5vt7s | ordinary | 0.39 / 0.59 / 0.01 | B2 |
| Batch 3 | pl8uabbv | ordinary | 0.15 / 0.84 / 0.01 | B2 |
| Batch 3 | ufdvpb81 | cracked | 0.72 / 0.01 / 0.27 | B1 |
| Batch 3 | vc2whyaq | ordinary | 0.06 / 0.50 / 0.44 | B2 |

Two of the three cracked reference sites are called Batch 1; three errors carry a top model probability ≥ 0.71
(b3esycq1 0.85 → B1, pl8uabbv 0.84 → B2, ufdvpb81 0.72 → B1). The errors are not confined to flagged sites.

### 3.3 Seed sensitivity (run after seeing seed 0; disclosed, not averaged)

| seed | combined bal. acc. / p | acq bal. acc. / p | morph bal. acc. / p |
|---|---|---|---|
| 0 (pre-registered) | 0.664 / 0.005 | 0.473 / 0.035 | 0.347 / 0.159 |
| 1 | 0.664 / 0.010 | 0.541 / 0.025 | 0.291 / 0.766 |
| 2 | 0.664 / 0.005 | 0.493 / 0.075 | 0.291 / 0.826 |

The seed moves the inner-CV split (C choice) and the permutation stream, not the LOO folds. The primary's reading
does not change with the seed (balanced accuracy 0.664 on all three; p 0.005–0.010); the acquisition-only model
spans p 0.025–0.075 across seeds (never below the Bonferroni 0.017); morphology stays at chance (p 0.16–0.83).

### 3.4 `--score` on a renamed fixture (three Batch 1 sites under new ids, cold cache)

| fixture id (= site) | flags (data) | mp B1 / B2 / B3 (combined, full fit) | argmax combined / morph / acq | OOD pct knn: morph / morph+texture / acq | exceeds (morph) |
|---|---|---|---|---|---|
| u156a27e (ffwubibz) | ordinary | 0.51 / 0.38 / 0.11 | B1 / B2 / B1 | 29.4 / 47.1 / 47.1 | no |
| u2d17d8d (f1vzngrs) | ordinary | 0.72 / 0.22 / 0.07 | B1 / B1 / B1 | 94.1 / 100.0 / 70.6 | no (texture-inclusive: yes) |
| ue93893e (iv6g2oq0) | ordinary | 0.61 / 0.17 / 0.22 | B1 / B3 / B1 | 11.8 / 41.2 / 70.6 | no |

The OOD percentiles equal those of the same images under their original ids in the known-site LOO table (29.4 /
94.1 / 11.8) — the reference model does not depend on the query's id. The categoriser rows differ from the LOO rows
(e.g. ffwubibz LOO 0.30 / 0.55 / 0.16 → B2; here 0.51 / 0.38 / 0.11 → B1) because the full fit has seen the
original copy of the image: this is a procedure check (C27), not an unseen-data result, and it also shows how
far one site can move between a fold model and the full model.

## 4. Calibration of the model probabilities

| family | Brier (multiclass) | Brier of the class prior | ECE (3 fixed bins) | mean top prob | top-class accuracy |
|---|---|---|---|---|---|
| combined | 0.494 | 0.597 | 0.146 | 0.655 | 0.677 |
| acq | 0.520 | 0.597 | 0.090 | 0.520 | 0.548 |
| morph | 0.685 | 0.597 | 0.063 | 0.434 | 0.387 |

Reliability of the top-class model probability, combined (primary):

| bin | n | mean top prob | observed accuracy of the argmax |
|---|---|---|---|
| [0.33, 0.50) | 6 | 0.42 | 0.50 |
| [0.50, 0.70) | 13 | 0.60 | 0.77 |
| [0.70, 1.00] | 12 | 0.83 | 0.67 |

Result: the primary's Brier beats the class prior (0.49 vs 0.60), but its top bin is over-confident (mean 0.83,
4 of 12 wrong) and its middle bin under-confident; ECE 0.15 on bins of 6–13 sites. Morphology's Brier is worse than
predicting the prior. **The outputs stay labelled "model probabilities"**: they are not posteriors, not a
probability of being correct, and a 0.85 can be wrong (b3esycq1). LOO on 31 sites could not establish calibration
for an unseen batch even if these bins had agreed (assessment.md, "One-class models and uncertainty").

## 5. Baseline OOD assessment — results (reference = all 17 Batch 3 sites; k = 3 nearest-reference distance in robust-z space; percentiles within the reference's own leave-one-site-out scores)

### 5.1 Per-batch summary (`ood_batch_summary.csv`)

| variant | batch | pct min / median / max | n exceeding ref. LOO max (of 7) | rank p at the floor 1/18 | matched-count median pct | RMS-z exceed | Mahalanobis (LW) exceed |
|---|---|---|---|---|---|---|---|
| **morph (PRIMARY)** | Batch 1 | 11.8 / 29.4 / 100.0 | **2** (0.29) | 2 | 29.4 | 2 | 2 |
| morph | Batch 2 | 0.0 / 41.2 / 76.5 | 0 | 0 | 47.1 | 0 | 0 |
| morph + texture (acquisition-sensitive) | Batch 1 | 29.4 / 47.1 / 100.0 | 3 (0.43) | 3 | 64.7 | 3 | 2 |
| morph + texture | Batch 2 | 0.0 / 41.2 / 88.2 | 0 | 0 | 41.2 | 1 | 0 |
| acq (session fingerprint) | Batch 1 | 47.1 / 70.6 / 82.4 | 0 | 0 | 70.6 | 0 | 0 |
| acq | Batch 2 | 23.5 / 47.1 / 70.6 | 0 | 0 | 58.8 | 0 | 0 |

Resolution floor: with 17 reference sites the smallest rank p is 1/18 = 0.056 and the percentile grid is 100/17 ≈
5.9 points; no single site can be placed outside the reference at 5 %. Under an i.i.d. null one site exceeds the
reference LOO maximum with probability 1/18 and at least one of seven does so with probability ≈ 0.33, so one
exceedance per batch is unremarkable and two or three are an evidence flag, not a defect call.

### 5.2 Per-site, primary variant (Batch 1 and Batch 2 sites; `site_table.csv`)

| batch | site | flags | score | pct | rank p | exceeds | 3 nearest reference sites | top contributing features (robust z) |
|---|---|---|---|---|---|---|---|---|
| B1 | 4ih2ggld | low_contrast | 37.8 | 100 | 0.056 | yes | x7u69zsw, tuy3zymq, kbdh4tri (grey-pore) | bright_count_per_Mpx +36.7, bright_frac +8.1, bright_solidity −7.7 |
| B1 | 5n1q8atc | low_contrast | 44.2 | 100 | 0.056 | yes | x7u69zsw, cfe5vt7s, tuy3zymq | bright_count_per_Mpx +39.5, bright_frac +13.6, bright_max_d +9.9 |
| B1 | f1vzngrs | ordinary | 7.9 | 94.1 | 0.111 | no | pl8uabbv, kbdh4tri, 71vgq3fw | pore_count_per_Mpx +6.6, etd_crack_density_particles +2.3, fft_slope +1.7 |
| B1 | ffwubibz | ordinary | 4.5 | 29.4 | 0.72 | no | hawkfj64, x77cy643, kbdh4tri | pore_count_per_Mpx −1.4, corr_len_px −1.1 |
| B1 | fzrt2k6r | ordinary | 4.1 | 17.6 | 0.83 | no | 71vgq3fw, 9luzk4jm, pl8uabbv | bright_max_d +2.1, bright_circ −1.6 |
| B1 | iv6g2oq0 | ordinary | 4.0 | 11.8 | 0.89 | no | utfgcjfa, x77cy643, 9luzk4jm | bright_d50 +2.2, crack_count_per_Mpx +2.0 |
| B1 | uhdslk0o | ordinary | 4.3 | 17.6 | 0.83 | no | utfgcjfa, cfe5vt7s, xgj4xftb | bright_d90 +1.5, bright_count_per_Mpx +1.1 |
| B2 | 3806gxp0 | ordinary | 4.5 | 29.4 | 0.72 | no | cfe5vt7s, 71vgq3fw, vc2whyaq | pore_count_per_Mpx −3.1, pore_d90 −1.1 |
| B2 | avn74qx1 | ordinary | 6.0 | 76.5 | 0.28 | no | vc2whyaq, ptg8lmto, x77cy643 | pore_count_per_Mpx −5.0, bright_circ +1.6 |
| B2 | b3esycq1 | ordinary | 4.9 | 52.9 | 0.50 | no | 71vgq3fw, cfe5vt7s, kbdh4tri | bright_frac +2.9, bright_count_per_Mpx +2.7, bright_d50 +2.0 |
| B2 | epqdaau9 | ordinary | 4.0 | 17.6 | 0.83 | no | 71vgq3fw, cfe5vt7s, x77cy643 | pore_count_per_Mpx +2.5, bright_d50 +1.4 |
| B2 | i9jiqjwl | ordinary | 4.7 | 41.2 | 0.61 | no | 71vgq3fw, tuy3zymq, cfe5vt7s | bright_max_d +2.0, fft_slope +1.7 |
| B2 | r17byphk | ordinary | 5.0 | 52.9 | 0.50 | no | vc2whyaq, tuy3zymq, ptg8lmto | pore_count_per_Mpx −3.0, crack_count_per_Mpx +1.6 |
| B2 | rxax5ozo | ordinary | 3.6 | 0.0 | 1.00 | no | 71vgq3fw, cfe5vt7s, vc2whyaq | pore_count_per_Mpx −2.2, bright_circ −1.1 |

Reference LOO scores (primary): 3.7 (71vgq3fw) … 5.3 (ptg8lmto) for the 12 ordinary + grey-pore sites, then
ufdvpb81 5.8 (cracked), mgxahqnk 6.1, x7u69zsw 6.6, 0grcilhi 7.3 and hzumfsms 8.3 (cracked). **The reference
maximum is set by a cracked Batch 3 site**: "exceeds the reference" therefore means "farther from its three
nearest Batch 3 sites than the most unusual cracked reference site is from its own", and conversely a cracked
unseen site would sit *inside* this reference. Cracks are the business of the batch-level report's localized
path (Check B), not of this percentile.

Reading:

* **The two Batch 1 sites that exceed the morphology reference are the two low-contrast sites, and the feature
  that puts them there is `bright_count_per_Mpx` at z ≈ +37 / +39** (with bright_frac and bright_solidity). On a
  low-contrast BSE image the fallback bright threshold fragments the bright phase into many small components
  (rule 4 in CLAUDE.md: bright-phase KPIs unreliable on these sites). This is a measurement artefact flagged by the
  acquisition flags in the same row, not evidence about the material; C03 applies and the post-hoc check in §5.3
  confirms it.
* f1vzngrs (ordinary) sits at the 94th percentile with `pore_count_per_Mpx` +6.6 z (134 /Mpx vs reference 98 ± 5.5
  MAD) and ETD crack density on particles +2.3 z; it exceeds in the texture-inclusive variant. One such site in
  seven is within what the i.i.d. null produces (≈ 0.33).
* **Batch 2 sits inside the Batch 3 morphology** (median percentile 41, none above the maximum, one site at the
  0th percentile). Its recurring direction is a lower pore count per Mpx (−2 to −5 z on five of seven sites), which
  is also what the morphology categoriser keeps selecting — a shift in the interior of the reference range, not an
  excursion beyond it. Batch 2 is told apart from Batch 3 by texture and session statistics (§3), not by geometry
  outside the reference.
* The matched-count sensitivity (query scored against each 16-site LOO reference set) moves medians by ≤ 18
  points and no exceedance flag.
* Acquisition-only: Batch 1 sits in the upper half (median 71st percentile) without exceeding — the session
  fingerprint is a shift, not an outlier, at this n.

### 5.3 Post-hoc sensitivity (NOT pre-registered; added after reading §5.2): morphology without bright-phase KPIs

`ood_posthoc_morph_no_bright.csv`: the 11 pore / crack / texture-scale KPIs of the morphology family
(`pore_frac, pore_d50, pore_elong, pore_max_d, pore_d90, pore_count_per_Mpx, crack_frac, crack_count_per_Mpx,
corr_len_px, fft_slope, etd_crack_density_particles`), same k-NN score, same reference LOO.

| batch | pct median / max | exceeding | detail |
|---|---|---|---|
| Batch 1 | 52.9 / 100 | 1 | 4ih2ggld 82.4 and 5n1q8atc 88.2 no longer exceed (pore_count_per_Mpx +3.9 / +3.9 z); f1vzngrs exceeds (100; pore_count_per_Mpx +6.6 z) |
| Batch 2 | 64.7 / 82.4 | 0 | avn74qx1 82.4 (pore_count_per_Mpx −5.0), r17byphk and 3806gxp0 76.5 |

Without the bright-phase KPIs the Batch 1 exceedances are no longer artefact-driven: one ordinary site exceeds,
on pore count. This variant was chosen after seeing which feature drove the primary's flags, so it is a diagnosis
of the flags, not a replacement for the pre-registered primary.

## 6. Univariate context (Kruskal–Wallis across the three batches; context only, no selection reads it)

14 of 39 features at unadjusted p < 0.05 (≈ 2 expected by chance): texture 9 of 10, acquisition 4 of 10,
morphology 1 of 19 (`pore_count_per_Mpx`, p 0.022; medians 101 / 86 / 98 per Mpx). At a Bonferroni 0.05/39 =
0.0013 six remain, all texture or acquisition: `crack_g_0.3`, `crack_g_0.2`, `bse_p1` (the grey-pore black level,
Batch 3 only), `etd_crack_density_graphite`, `inlens_grad_energy`, `inlens_particle_texture`. No morphology KPI
survives correction. Next morphology features: `bright_d50` p 0.07, `pore_frac` 0.13, `bright_frac` 0.14.

## 7. Limitations

* **31 sites.** Every accuracy carries a Clopper–Pearson interval ≈ ± 0.17–0.20; per-class recall rests on 7 sites
  (one site = 0.14). Balanced accuracy 0.66 vs 0.47 is not a demonstrated ordering of two models.
* **Batch = session.** Each batch was imaged in its own session(s); texture and acquisition statistics that separate
  batches cannot be attributed to the material. The pre-registered primary is therefore a *batch fingerprint*
  categoriser. The morphology-only model — the only one that could carry a material reading — is at chance.
* **Model probabilities are not calibrated** (§4): over-confident top bin, Brier 0.49; errors at 0.84–0.85.
* **Specimen independence unconfirmed** (plan §2.0 rule 9); if sites share specimens the effective n is smaller and
  every p here is anti-conservative.
* **Reference heterogeneity.** The OOD reference includes 4 grey-pore and 3 cracked sites; its LOO maximum is a
  cracked site. A cracked unseen site is "inside"; a site resembling the grey-pore group is "inside". Membership of
  the promised distribution is thus read against Batch 3 *as delivered*, including its sub-populations, which the
  hand-off shows through the nearest-neighbour column.
* **Low-contrast sites** inflate bright-phase KPIs by one to two orders of magnitude in robust z; they are kept
  (the procedure must run unchanged on an unseen batch) and flagged. Any exceedance whose top contributor is a
  bright-phase KPI on a `bright_low_contrast` site is an acquisition finding (§5.3).
* **Resolution floor** 1/18 for the rank p; percentile steps of 5.9 points; "exceeds the maximum" fires for ≈ 33 %
  of 7-site batches under the null.
* **Multiplicity.** Three categoriser families (one pre-registered primary; secondaries read at 0.017), three OOD
  variants, one post-hoc OOD variant, two extra seeds; all listed, none promoted.
* **Scope.** Nothing here is an accept / reject; no equivalence test and no MDC are computed in this module; the
  batch-level verdict with its drivers, stability and MDC is `polaron_qc.report` (pointer in every row).
* **No unseen data was used.** The `--score` fixture is known data under new ids (procedure check only).

## 8. Checklist applied (docs/decision_log.md Part B)

* **C01** — tables inspected: no constant column; `bright_sep` NaN on one site (imputed in-fold);
  `inlens_speckled_particle_frac` has MAD 0 in Batch 3 (IQR fallback used, nothing dropped); the z ≈ +37/+39 of
  `bright_count_per_Mpx` on the two low-contrast sites was treated as a broken measurement, not a finding.
* **C02** — scaler, imputer and C choice fitted inside every fold and permutation; percentiles come from the
  reference's own LOO, not from a per-batch normalisation; no per-image percentile thresholds.
* **C03** — the primary's separation was tested against an acquisition-only model (p 0.035) and its selected
  features are predominantly texture / session statistics; the OOD flags were traced to a low-contrast artefact.
* **C06 / C09 / C14** — 31 site units, patches never counted; three models pre-registered with Bonferroni for the
  secondaries; the null repeats the whole fold-local procedure under site-label permutation; n after flags stated
  (bright-phase KPIs rest on 5 of 7 Batch 1 sites; the two flagged sites are kept and marked).
* **C07 / C10 / C17** — no acceptance claim from a percentile inside the reference; false-alarm rate of the
  exceedance rule stated (1/18 per site, ≈ 0.33 per 7-site batch); wording "model probability", "percentile of the
  reference LOO", "evidence rank"; no "confidence", no "probability correct".
* **C16 / C21 / C22** — provenance: developed using exploratory analysis of Batches 1–3; definitions committed
  (691990e) before the run; deviations listed in §1.7; the morphology list is asserted against `KPI_TRUST` at
  import and in a test (no flag or confounded column); nothing undefined or constant is reported.
* Also applied: **C27** (renamed fixture through `--score`, cold cache, data-derived flags), **C04** (every
  output named in §1 exists in `output/`).
