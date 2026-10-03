# E20 — Does the per-particle Inlens texture survive local-contrast normalisation?

Status: **§1 pre-registered before the run** (the commit on this branch that adds §1 precedes any result file).
Results are appended in §2+ only after `run_e20.py` has executed on all 31 known sites.

## 0. Question and framing

`inlens_particle_texture` (polaron_qc.features.multichannel_features) is the per-particle standard deviation of
the Inlens image inside BSE bright-particle interiors (bright mask eroded 6 px, components >= 400 px), divided by
the whole-image Inlens IQR; the site value is the median over particles (p90 and a "speckled" fraction with
SD > 0.5 are reported alongside). E03 found that it orders the batches B1 0.32 > B2 0.24 > B3 0.18 (site medians,
Kruskal p ≈ 0.001) and correlates with Inlens median brightness at ρ ≈ 0.77; acquisition-only R² 0.74. It is
therefore trust level **confounded** (D11, CLAUDE.md rule 5). E25 measured a *different* sampled-gradient
quantity under local standardisation and explicitly did not correct this measurand.

E20 tests the **actual** per-particle measurement under prespecified normalisations. Framing fixed in advance
(Santosh's correction):

- If a variant keeps the batch separation **and** loses its brightness association, that establishes robustness
  to *that* transformation only. Charging relief, detector noise, preparation texture and session effects can all
  remain; none of them is removed by a gain/offset or a local-contrast normalisation.
- If a variant **loses** the separation, the normalisation may have removed useful material contrast; loss is not
  proof that the original signal was an artefact.
- Both outcomes are reported neutrally; neither promotes anything above **candidate**.

## 1. Pre-registered protocol

### 1.1 Units, sites, views

- **n = sites**: 7 / 7 / 17 (Batch 1 / 2 / 3). Particles and tiles are never n; per-particle values exist only to
  form site summaries and evidence images.
- **Primary view `all31`**: all sites, to be comparable with the E03 numbers.
- **Sensitivity view `bright_usable`**: sites with `bright_low_contrast == False` from
  `polaron_qc.report.derive_flags` (expected 5 / 7 / 17; the two Batch 1 low-contrast sites have unreliable bright
  masks, CLAUDE.md rule 4). Grey-pore and cracked Batch 3 sites are **kept** in the reference (D49P: the whole of
  Batch 3 is the promised baseline) and marked on plots.
- Reference for shifts and permutation tests: Batch 3 (all 17 usable sites of the view).

### 1.2 Measurement support (reused, not re-implemented)

Per site: `features.load_image` for BSE / ETD-or-SE / Inlens; `features.bright_bands` on the BSE image, the same
row slice applied to all channels (as `features.extract_site`); `features.phase_thresholds(features.smooth_bse(bse))`
for `th_lo`/`th_hi`; `features.segment` for the bright mask; interior = `binary_erosion(bright, iterations=6)`,
`skimage.measure.label`, components with area >= 400 px (exactly `multichannel_features`). Integrity check: the
recomputed site median of the current measurand must equal the cached `inlens_particle_texture` (to 1e-9) for all
31 sites, otherwise the run is invalid.

### 1.3 Variants (all per particle, then site median and site p90 over particles)

Let `I` be the trimmed Inlens image as float, `P` the particle interior pixel set, `IQR_img` the whole-image IQR.

| id | name | definition | invariance (mathematical) | what it measures |
|---|---|---|---|---|
| `cur` | current measurand | `SD(I[P]) / IQR_img` | image-wide affine only | absolute within-particle amplitude relative to image spread |
| `a_cv` | coefficient of variation | `SD(I[P]) / max(mean(I[P]), 1)` | per-particle gain (not offset) | amplitude relative to the particle's own brightness |
| `b_aff` | per-particle affine standardisation | `SD((I[P] − median(I[P])) / max(IQR(I[P]), 1))` = `SD(I[P]) / IQR_p` | per-particle gain **and** offset | **shape** of the within-particle distribution (tail weight), not amplitude; the floor at 1 grey level catches saturated/flat particles, recorded as `b_floor` |
| `c_loc64` | local z-normalisation then SD | `N = (I − G64(I)) / sqrt(max(G64(I²) − G64(I)², 0) + (0.05·IQR_img)²)`, Gaussian σ = 64 px on the whole trimmed image; value = `SD(N[P])` | affine (gain/offset) image-wide and, up to the floor, locally | particle-internal amplitude **relative to its 64-px neighbourhood** (which includes particle edges and graphite); `c_floor_frac` = share of P where local variance < floor² |
| `c_loc32` | sensitivity of c | same with σ = 32 px (E25's scale) | as c | as c, tighter neighbourhood |
| `d_lbp` | LBP non-uniform fraction | rotation-invariant uniform LBP (P = 8, R = 1, skimage `method='uniform'`) on the uint8 image; value = share of P with the non-uniform code (= 9) | any strictly monotone intensity map (sign comparisons only); quantisation/plateaus and saturation break the invariance in practice | roughness / irregularity of local ordering, independent of amplitude; noise-sensitive by construction |
| `d_lbp_ent` | sensitivity of d | normalised entropy (base 10 codes) of the LBP code histogram inside P | as d | diversity of local patterns |

Primary normalised variants for the recommendation: `a_cv`, `b_aff`, `c_loc64`, `d_lbp`. `c_loc32` and `d_lbp_ent`
are sensitivities and may not be used to pick a winner.

### 1.4 Site covariates and flags

From the cached feature tables (`polaron_qc.features.extract_batch`, cache hit, FEATURE_VERSION 1.1.0) and
`polaron_qc.report.derive_flags(sites, images)`: `inlens_p50`, `inlens_std` (untrimmed image statistics as in
`image_quality.csv`), `bse_p1`, `bse_std`, `bright_sep`, `H`, `bright_low_contrast`, `grey_pore`, `cracked`,
`acquisition_group`, `inlens_empty_bin_frac`.

### 1.5 Analyses (fixed)

1. **Brightness association.** Spearman ρ of each variant's site median with `inlens_p50` and with `inlens_std`:
   across the view's sites and within each batch (n 7/7/17 or 5/7/17; within-batch values are descriptive only).
2. **Batch separation.** Kruskal–Wallis across the three batches on site medians; batch ordering by median of site
   medians.
3. **Pairwise vs reference.** `polaron_qc.stats.permutation_test(ref=B3, batch=B1|B2, statistic="hl_shift")`
   (exact enumeration: C(24,7) = 346 104 allocations; min two-sided p 2/346 104) and
   `polaron_qc.stats.robust_shift` (shift in Batch 3 MADs with bootstrap-over-sites interval, n_boot 2000, seed 0).
4. **Acquisition-only control.** Leave-one-site-out ridge (alpha = 1, median imputation and standardisation fitted
   inside each fold) predicting each variant's site median from `inlens_p50, inlens_std, bse_p1, bse_std,
   bright_sep, H`; held-out R² = 1 − SS_res / SS_tot. This is the texture-specific covariate list named in the task;
   it differs from assessment.md's shared geometry control (which uses BSE empty-bin fraction and ETD boundary
   sharpness instead of the Inlens statistics) and is reported as such. Exploratory, not a promotion gate:
   Kruskal on the held-out residuals. A negative or low R² does not demonstrate invariance (C03).
5. **Perturbation robustness** on copies of the real trimmed Inlens image of four prespecified sites, BSE masks
   unchanged: Batch 1 `f1vzngrs` (first ordinary by id), Batch 2 `3806gxp0` (first by id), Batch 3 `9luzk4jm`
   (first by id that is neither grey-pore nor cracked), and the low-contrast `4ih2ggld`. Perturbations: gamma 0.75
   and 1.25 (`round(255·(I/255)^γ)`), and the non-clipping affine remap `round(0.6·I + 40)` (range 40–193).
   Rounding back to uint8 is part of the perturbation and is noted. Reported quantity: change in the site median
   of each variant, divided by the unperturbed between-batch gap |median(B1 site medians) − median(B3 site
   medians)| of that variant (view all31). A movement of ≥ 0.5 gap under a gain/offset remap disqualifies a variant
   from being called gain/offset robust in practice.
6. **Evidence images.** Two particles per batch, prespecified as the largest interior component of the first and
   last site by id: Batch 1 `4ih2ggld`, `uhdslk0o`; Batch 2 `3806gxp0`, `rxax5ozo`; Batch 3 `0grcilhi`, `xgj4xftb`.
   For each: BSE crop with the interior outline, raw Inlens crop on the 0–255 scale, per-particle standardised crop
   (variant b, ±3), local-z crop (variant c, ±3) and the LBP non-uniform mask; the variant values printed in the
   title. Purpose: let a materials expert judge whether what survives looks like particle-internal texture or
   edge/charging relief. These are illustrations, not n.

### 1.6 Decision rule for the recommendation (fixed before the run)

A normalised variant may be proposed as a **candidate** (never primary) for the site categoriser's texture family
only if all hold in the primary view: (i) |ρ| with `inlens_p50` across sites < 0.4 **and** no within-batch |ρ| >
0.7 with n ≥ 7; (ii) Kruskal p < 0.05 with at least one pairwise exact permutation p < 0.05 vs Batch 3; (iii)
acquisition-only held-out R² < 0.3; (iv) perturbation movement < 0.5 gap for the affine remap on all four sites;
(v) the evidence crops do not show the surviving signal to be dominated by edges or charging relief (expert
judgement, recorded as open until Santosh reviews). Failing (v) leaves the status "candidate pending expert
review". Nothing here changes KPI_TRUST, MATERIAL_KPIS or any decision input.

### 1.7 What this cannot show

Gain/offset and local-contrast invariance are properties of the arithmetic, verified on copies of real images.
They say nothing about charging (which changes spatial structure, not only gain), detector noise, polishing
relief, or session differences that co-vary with batch. Expert validation of particle masks and of the texture's
material meaning remains open (A-register), and the unseen batch has not been used.
