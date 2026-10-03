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

---

## 2. Run and integrity

- `run_e20.py --stage all`, 2026-10-03, FEATURE_VERSION 1.1.0, feature cache hit for all three batches
  (`Batch_1_8d2c001bed2f`, `Batch_2_c5b1475904f3`, `Batch_3_21893e9e3a20`). Extraction 965 s for 31 sites.
- **Integrity check passed on all 31 sites**: the recomputed `cur` site median, particle count and both BSE thresholds
  equal the cached `inlens_particle_texture`, `inlens_particles_measured`, `th_lo`, `th_hi` (`sites.csv:matches_cache`).
  The `cur` row below therefore reproduces E03 exactly (ρ 0.77, Kruskal p 0.001, acquisition R² 0.75).
- 2 323 particles over 31 sites (42–135 per site). Floors essentially never acted: `b_floor` (particle IQR < 1 grey
  level) on 10 of 2 323 particles, `c_loc64` floor-dominated share 0, saturated-pixel share ≈ 0. Inlens empty-bin fractions 0.24–0.31 in every batch: most
  Inlens images are contrast-stretched (plateaued histograms), which matters for the rank variant d (ties).
- Code change after the pre-registered commit: figure layout only (panel width, title line breaks, evidence dpi).
  No measurement, variant, view, test or site list changed.

## 3. Results — primary view `all31`, site medians (n = 7 / 7 / 17)

Shifts are Hodges–Lehmann in exact site-label permutation tests (346 104 allocations) and robust shifts in Batch 3
MADs with bootstrap-over-sites 95 % intervals (approximate; the upper bounds are wide because the reference MAD is
re-drawn from 17 sites). Full tables: `batch_tests.csv`, `correlations.csv`, `control_r2.csv`.

| variant | ρ vs Inlens p50, all / within B1 / B2 / B3 | Kruskal p | medians B1 / B2 / B3 | B1 vs B3 shift (B3 MAD) [95 %], exact p | B2 vs B3 shift (B3 MAD) [95 %], exact p | acquisition-only LOO R² |
|---|---|---|---|---|---|---|
| `cur` (current) | **+0.77** / +0.20 / +0.77 / +0.73 | **0.0011** | 0.320 / 0.237 / 0.177 | +2.18 [+0.97, +10.5], **4.6e-5** | +0.92 [−0.00, +9.6], **0.013** | **0.75** |
| `a_cv` | +0.09 / **−0.74** / −0.20 / −0.09 | 0.023 | 0.089 / 0.089 / 0.079 | +0.83 [+0.15, +5.1], 0.015 | +0.87 [−0.06, +3.4], 0.071 | 0.40 |
| `b_aff` | −0.46 / −0.70 / −0.52 / −0.24 | 0.14 | 0.764 / 0.751 / 0.765 | −0.10 [−2.7, +1.7], 0.54 | −1.22 [−3.5, +0.23], 0.031 | 0.40 |
| `c_loc64` | **+0.68** / −0.18 / **+0.94** / **+0.75** | **0.0016** | 0.344 / 0.343 / 0.258 | +1.98 [+0.31, +6.4], **0.0034** | +1.95 [+0.27, +6.0], **0.0092** | **0.71** |
| `c_loc32` (sens.) | +0.56 / −0.20 / +0.29 / +0.72 | 0.018 | 0.411 / 0.418 / 0.346 | +1.94, 0.009 | +2.14, 0.042 | 0.48 |
| `d_lbp` | +0.39 / +0.59 / +0.54 / −0.10 | 0.059 | 0.229 / 0.229 / 0.220 | +2.89 [−0.18, +10.3], 0.017 | +3.17 [−0.32, +9.1], 0.020 | **0.03** |
| `d_lbp_ent` (sens.) | −0.28 / −0.63 / −0.76 / −0.06 | 0.78 | 0.938 / 0.939 / 0.944 | −1.12, 0.45 | −0.81, 0.35 | −0.24 |

Batch 3 MADs (the unit of the shift column): `cur` 0.066, `a_cv` 0.011, `b_aff` 0.012, `c_loc64` 0.044, `d_lbp`
0.0031. The `d_lbp` shifts look large in MADs only because its reference spread is tiny; in absolute terms the
B1 − B3 gap is 0.009 on a value of 0.22 (≈ 4 %).

Other pre-registered readings:

- **Ordering.** The monotone `cur` ordering B1 > B2 > B3 survives in **no** normalised variant: `a_cv` 0.0888 vs
  0.0890, `c_loc64` 0.344 vs 0.343, `d_lbp` 0.2286 vs 0.2294 — Batches 1 and 2 coincide and both sit above
  Batch 3. The B1 > B2 part of the ordering is carried by the amplitude component, which is the part that tracks
  Inlens brightness.
- **Evidence flags (exceeding the reference maximum, not defect calls).** Later-batch sites above the Batch 3
  median / above the Batch 3 maximum — `cur`: B1 7/1, B2 7/1; `a_cv`: 7/2, 6/0; `c_loc64`: 7/1, 7/0; `d_lbp`: 6/0, 5/2.
- **Inlens std.** Within-batch Spearman with image std (n 7/7/17): `cur` +0.29 / −0.82 / −0.18; `c_loc64` +0.18 /
  −0.71 / +0.14; `d_lbp` −0.21 / **−0.96** / −0.37. The rank variant tracks the image noise/contrast level inside
  Batch 2 almost perfectly; it is a roughness-of-ordering measure and detector noise sets that.
- **p90 statistic** (`batch_tests.csv`, statistic = p90): same picture; `cur` and `c_loc64` keep B1 vs B3 (p 5e-4,
  0.007), B2 vs B3 weakens (p 0.12, 0.06); `d_lbp` p90 is acquisition-predictable (R² 0.57) unlike its median.
- **Residual Kruskal** after removing the acquisition-ridge prediction (exploratory): `cur` 0.45, `a_cv` 0.37,
  `b_aff` 0.42, `c_loc64` 0.67, `d_lbp` 0.17 — no variant separates the batches once its acquisition-predictable
  part is removed; for `d_lbp` that is because little was predictable and little was separated to begin with.

### 3.1 Sensitivity view `bright_usable` (n = 5 / 7 / 17; the two Batch 1 low-contrast sites removed)

`cur`: pooled ρ rises to +0.85 and acquisition R² to **0.89**; Kruskal 0.0028; B1 +2.18 MAD p 3e-4, B2 +0.92 p 0.013.
`c_loc64`: ρ +0.80, R² 0.68, Kruskal 0.0052, B1 +1.91 p 0.0087, B2 +1.95 p 0.0092 (ordering B2 > B1 > B3).
`a_cv`: Kruskal 0.077, R² 0.06, B1 vs B3 p 0.12 — the Batch 1 effect in the primary view came largely from the
two low-contrast sites (their `a_cv` 0.119 / 0.113 are the two highest of all 31; the crops in §5 show why).
`d_lbp`: Kruskal 0.015, B1 +3.78 MAD p 0.0016, B2 +3.17 p 0.020, R² 0.24, pooled ρ +0.38.
`b_aff`: Kruskal 0.046 but with ordering B3 > B1 > B2 and shifts of −0.9 / −1.2 MAD on a near-constant index (§3.3).

### 3.2 Perturbation robustness (copies of the real Inlens image, BSE masks fixed; `perturbation.csv`, `figures/perturbation.png`)

Movement of the site median, in units of the unperturbed B1 − B3 gap of that variant (gaps: `cur` 0.143, `a_cv`
0.0093, `b_aff` 0.0012, `c_loc64` 0.086, `d_lbp` 0.0089):

| variant | affine 0.6·I + 40 (4 sites) | gamma 0.75 | gamma 1.25 |
|---|---|---|---|
| `cur` | −0.02 … +0.04 | −0.34 … −0.06 | +0.06 … +0.34 |
| `a_cv` | **−4.7 … −2.3** | −3.2 … −1.8 | +1.8 … +3.3 |
| `b_aff` | +0.05 … +17.7 (gap ill-defined; absolute Δ ≤ 0.021) | −3.4 … +5.3 | −0.2 … +10.7 |
| `c_loc64` | −0.01 … +0.04 | −0.40 … −0.06 | +0.10 … +0.45 |
| `d_lbp` | −0.19 … +0.08 (absolute Δ ≤ 0.002) | −0.54 (9luzk4jm) … +0.06 | −0.42 (9luzk4jm) … +0.05 |

Readings: `cur` and `c_loc64` are gain/offset invariant in practice (≤ 0.04 gap, i.e. rounding only). `a_cv` is
not (offset enters the denominator); its pooled brightness decorrelation is therefore partly arithmetic
(dividing by the particle's brightness) rather than a property of the texture — a C02 concern. `d_lbp` is exactly
invariant to the affine map up to re-quantisation (Δ ≤ 0.002 absolute) but a gamma remap moves it by half a gap at
one site because gamma re-quantisation changes the tie structure of a stretched 8-bit image. **A gamma-type
LUT difference between sessions moves `cur` and `c_loc64` by up to a third to a half of the between-batch gap** on a
single site; the LUT history of the Inlens images is unknown (empty-bin fractions 0.24–0.31), so this is a live
alternative explanation for part, not all, of the gap.

### 3.3 Variant b is a near-constant shape index

`b_aff` = SD / IQR of the within-particle intensity distribution; its site medians are 0.72–0.81 on every site (a Gaussian gives 0.741; individual particles spread 0–11.6,
dominated by a few flat or degenerate particles). It carries no amplitude information by construction and says only that the
within-particle intensity distributions are near-Gaussian everywhere. It is stated here once and is not reported
as a KPI (C22 reasoning).

## 4. Pre-registered decision rule (§1.6) applied

| criterion (all31) | `a_cv` | `b_aff` | `c_loc64` | `d_lbp` |
|---|---|---|---|---|
| (i) pooled \|ρ\| < 0.4 and no within-batch \|ρ\| > 0.7 (n ≥ 7) vs p50 | fail (within B1 −0.74) | fail (−0.46) | fail (+0.68; B2 +0.94) | pass (+0.39; max 0.59) |
| (ii) Kruskal p < 0.05 and a pairwise exact p < 0.05 | pass (0.023; 0.015) | fail (0.14) | pass (0.0016; both < 0.01) | **fail narrowly** (0.059; pairwise 0.017 / 0.020) |
| (iii) acquisition-only LOO R² < 0.3 | fail (0.40) | fail (0.40) | fail (0.71) | pass (0.03) |
| (iv) affine perturbation < 0.5 gap on all four sites | fail | fail / undefined | pass | pass |
| (v) crops not dominated by edge/charging relief | open (expert) | — | open (expert) | open (expert) |

**No variant meets the rule.** `d_lbp` is the closest (three of four numerical criteria) and fails the Kruskal
threshold narrowly; the rule is applied as written and it is not promoted. Any renewed attempt is a new
pre-registration, not a re-reading of this run.

## 5. Evidence images (`figures/evidence_particles.png`; illustrations, not n)

Largest interior component of the first/last site by id per batch. Reading by the analyst, not an expert:

- **Batch 1 `4ih2ggld` (low-contrast)**: the "largest interior" (51 093 px) is a sieve-like mesh — the fallback
  bright threshold fragments one large particle into a holey mask whose "interior" is mostly near mask edges. Its
  per-particle SD, CV and local-z values include mask-edge pixels. This is rule 4 made visible and is why `a_cv`
  puts the two low-contrast sites at the top of all 31.
- **Batch 1 `uhdslk0o`**: fine noise-like grain; faint diagonal striations visible in the standardised and local-z
  crops (possible polishing/curtaining relief or scan artefact), no blotchy internal structure.
- **Batch 2 `3806gxp0`**: mottled dark spots inside the particle are visible in the raw and standardised crops — the
  one crop where resolvable internal structure is apparent (`cur` 0.30, `c` 0.48).
- **Batch 2 `rxax5ozo`, Batch 3 `0grcilhi`, Batch 3 `xgj4xftb`**: smooth, uniform fine grain (`cur` 0.13 / 0.12 /
  0.09); `xgj4xftb` has small internal holes excluded from the interior by the mask; bright rims outside the
  interiors are excluded by the 6 px erosion.
- **LBP panel**: non-uniform pixels are salt-like everywhere, inside and outside particles; `d_lbp` measures pixel
  noise roughness, consistent with its tiny spread and its −0.96 within-Batch-2 correlation with image std.

In five of six crops what survives normalisation looks like fine grain (noise/relief) rather than resolvable
particle-internal texture; one Batch 2 crop shows real internal structure. Criterion (v) stays **open for
Santosh**; this reading does not substitute for it.

## 6. What the run does and does not establish

Established (on these 31 sites):

1. The Batch 1/2 vs Batch 3 contrast in per-particle Inlens texture is **not a pure gain/offset artefact**: the
   measurand moves ≤ 0.04 gap under an affine remap, and the gain/offset-invariant local-z variant keeps both
   later batches ≈ 2 MAD above the reference with exact p < 0.01.
2. That surviving contrast is **as brightness-associated as the original** (ρ 0.68 pooled, 0.94 within Batch 2,
   acquisition R² 0.71). Brightness-linked structure that is not gain — charging relief, detector noise level,
   non-linear LUT/gamma, session — remains the unresolved alternative to material texture. The gamma perturbation
   shows a LUT difference could account for up to a third to a half of the gap.
3. The **monotone B1 > B2 > B3 ordering does not survive any normalisation**; it was carried by the amplitude
   component. Under normalisation the picture is "both later batches differ from the baseline in the same way",
   not a graded series.
4. Variants that remove the brightness association do so by construction (`a_cv`), remove the information
   entirely (`b_aff`), or measure noise roughness with a 4 % effect (`d_lbp`).

Not established, in either direction: a material origin (neither proven nor excluded); that the loss of the B1 > B2
ordering means that ordering was artefactual (the normalisations also remove amplitude, which may be material);
anything about the unseen batch; expert validity of the bright-particle masks (the low-contrast crop shows the
mask failing); specimen independence of the sites.

## 7. Recommendation

- **Do not enter any normalised variant into the site categoriser's texture family as a candidate** on this run;
  the pre-registered rule is not met. Keep `inlens_particle_texture` at trust **confounded** (unchanged); no
  change to KPI_TRUST, MATERIAL_KPIS, decision inputs or thresholds.
- What may be cited as evidence: E20 shows the existing texture contrast is gain/offset-robust (point 1) and
  brightness-linked beyond gain (point 2); the batch-signatures page (L4) should say both, and should drop the
  "monotone ordering" argument for Inlens texture or qualify it as amplitude-carried (point 3).
- If the team still wants a normalised Inlens texture inside the categoriser (L1), the next step is a **new
  pre-registration** of `d_lbp`-type or local-z measures with (a) a detector-noise covariate (e.g. noise SD
  estimated on graphite interiors, or ETD/Inlens high-frequency power) in the acquisition control, (b) selection
  only inside the categoriser's evaluation folds, never by known-folder significance (D49P), and (c) Santosh's
  crop review (criterion v) completed first. A gamma-type LUT sensitivity must be part of that control.
- Open for Santosh: review the six crops; decide whether the Batch 2 `3806gxp0` mottling is a material feature
  worth a dedicated, mask-independent measurement.

## 8. Limitations

- 31 sites, 7 per later batch; 7 variants × 2 statistics × 2 views × several tests on the same units — exploratory
  multiplicity, no correction applied, no winner picked post hoc (the rule was fixed first).
- Variant definitions were designed after knowing E03's result, before this run; provenance is "developed using
  exploratory analysis of Batches 1–3; frozen before the unseen batch arrived", not independent discovery.
- The acquisition covariate list is the task's texture-specific list, not assessment.md's shared geometry list;
  the two controls are not interchangeable.
- Bootstrap intervals for shifts in MADs are approximate and wide at the top (reference MAD re-drawn from 17 sites).
- Perturbations are intensity remaps only; they cannot emulate charging, noise or preparation differences.
- Particle masks come from BSE thresholds that fail on low-contrast sites; the `bright_usable` view is the
  measurement-honest one for anything bright-particle based, and it strengthens rather than weakens the
  acquisition association of the current measurand (R² 0.89).

## 9. Checklist applied (docs/decision_log.md Part B)

C01 (looked at the tables: `b_aff` near-constant and `d_lbp` tiny-spread flagged; wide bootstrap upper bounds noted;
no 3× jumps), C02 (`a_cv` decorrelation and `b_aff` constancy are by construction; views reported separately),
C03 (every variant correlated with p50 and std pooled and within batch; acquisition-only LOO model; perturbation
confound check), C06 (31 units; the 28 test rows are exploratory; one pre-fixed rule), C09 (exact site-label
permutation with the same statistic and n; reference MAD inside the bootstrap), C14 (usable n 7/7/17 and 5/7/17
stated; specimen sharing unknown), C16 (provenance stated in §8), C21 (no trust-list change; nothing enters
MATERIAL_KPIS), C22 (`b_aff` stated once as a fact, not reported as a KPI).
