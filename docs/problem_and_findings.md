# Polaron challenge — problem statement and state of knowledge

_Last updated: 2026-10-03 (baseline status confirmed). Owner: Santosh (narrative, materials review) + Claude (pipeline, statistics)._

## 1. The problem

**Question posed.** Can you detect when a supplier's electrode material has changed before it becomes a manufacturing problem?

**Setting.** Battery manufacturers need incoming electrode material to be consistent batch to batch. Subtle shifts in formulation or processing alter microstructure in ways that only surface as defects later in production. Using electron-microscopy images, build a trustworthy, interpretable, uncertainty-aware QC system that compares incoming batches against an approved baseline and outputs one of **accept / investigate / reject**, with an explanation a materials expert can check.

**What is provided.** A baseline batch plus several incoming batches mixing acceptable and defective variation. A brand-new unseen batch arrives ~8 hours into day one to test generalisation.

**Judging criteria** (in the organisers' words): quality of extracted material KPIs; accuracy on the new batch; interpretability; honest handling of uncertainty; real-world usability for a QC decision. Raw accuracy alone is explicitly not the target.

**Our framing.** This is not a defect classifier. It is a two-sample comparison: does a batch fall inside the reference batch's own site-to-site variation, or outside it? Thresholds are calibrated against the reference's self-similarity (split-half / leave-one-site-out), not hand-tuned. Because the reference (Batch 3) is itself imperfect and heterogeneous, the system must display the reference's own spread and sub-populations, report all pairwise batch differences, and separate "the material changed" from "the microscope or sample preparation changed", because the data contain both.

## 2. The data

| Item | What we have |
|---|---|
| Location | `Dataset/Batch_{1,2,3}/img_<site>_<detector>.tif` |
| Sites | Batch 1: 7, Batch 2: 7, Batch 3: 17 (31 total) |
| Detectors per site | BSE (backscattered electrons, contrast ∝ atomic number), ETD (secondary electrons, topography; labelled `SE` on 4 sites — confirmed same detector, 2026-10-03), Inlens (surface-sensitive, charging-dominated) |
| Geometry | Stitched cross-section strips, ~7000 px wide × 1612–2316 px tall, 8-bit grayscale stored as identical RGB planes, LZW TIFF |
| Alignment | The three channels of a site are pixel-aligned (phase-correlation shift ≤ 0.2 px) |
| Metadata | Microscope settings (magnification, voltage, detector gain) were stripped when files were re-saved by `tifffile`. The TIFF resolution tags survived and give a **nominal 25.0 nm/pixel** on every image (24.9992–25.0005 nm/px). This is export metadata, not verified calibration. We report all lengths in pixels and quote µm as "nominal": strip ≈ 175 µm wide, coating ≈ 40–58 µm thick, bright-phase D50 ≈ 150 px ≈ 3.8 µm, crack-like void cutoff 500 px ≈ 12.5 µm. |
| Material | Porous coating of plate-like graphite with a sparse brighter (higher-Z) particulate phase. Consistent with a silicon / silicon-oxide–graphite anode cross-section after ion polishing. **Chemistry is not confirmed.** |
| Labels / baseline | Confirmed by the problem providers (2026-10-03): the three folders are three supplier batches of the same nominal product. **Batch 3 is a single batch with more samples and is the closest available reference, but not a clean approved baseline.** There is no clear baseline; the task is to differentiate the batches. No per-batch acceptable / defective labels. |

Current-collector or stitching bands appear at one edge of five images (≤ 56 rows) and are trimmed before measurement. 39 images carry a single differing-colour edge column (export artefact); no colour inside the frame. No duplicate images. Some Inlens images saturate at white over 4–7 % of pixels.

### Two analyses exist in this repo

- `notebooks/01_dataset_analysis.ipynb` (Claude, this document's main source): segmentation-based material KPIs, acquisition sub-groups, cross-channel features, confound checks.
- `analysis/` (parallel, label-free audit: `analyze_dataset.py`, `compare_batches.py`, `findings.md`, `report.html`): file-integrity checks (decoding, duplicates, colour borders, resolution tags), per-image intensity/texture proxies with bootstrap intervals for all pairwise batch differences, and an explicit list of what those proxies are *not* (not porosity, not particle size). It is where the nominal 25 nm/px and the Inlens saturation figures come from, and it independently flagged the same four raised-black-level Batch 3 sites, the copper band in `epqdaau9`, and histogram combing. Its pairwise comparisons show Batch 3 has lower BSE gradient RMS and gray-level std than Batch 1 (standardised difference ≈ −1.6 and −1.1), consistent with the softer boundaries and longer correlation length we measure.

## 3. What we know so far

The full analysis, with figures, is `notebooks/01_dataset_analysis.ipynb`. Headlines:

### 3.1 Acquisition is not uniform, and it matters more than batch

1. **Contrast stretching** was applied after acquisition to about half of all images (comb-shaped histograms, up to 65 % of gray levels empty), unevenly across sites and strongest in Inlens. Raw intensity is therefore not a material signal in any channel.
2. **Two Batch 1 sites** (`4ih2ggld`, `5n1q8atc`, both 2316 px tall) were recorded with the bright phase only 25–35 gray levels above graphite instead of the usual 50–70. No threshold can separate additive from particle rims and binder there; their bright-phase fraction and particle count come out 2–8× too high. Flagged `bright_low_contrast`.
3. **Four Batch 3 sites** (`71vgq3fw`, `kbdh4tri`, `tuy3zymq`, `x7u69zsw`, all 2060 px tall) have no black pixels, a grey plateau where open pores should be, the most curtaining and ridge relief in ETD, and a much darker Inlens. We call them the **grey-pore group**. Most likely a different sample preparation (resin-filled pores) or imaging session. Their pore KPIs rest on a fallback threshold and read low.
4. **Frame heights recur across batches** (2080, 2148, 2156, 2272 px each appear in two folders) and cluster within the sub-groups above. Height behaves like a session fingerprint, not a reliable thickness proxy.

### 3.2 Material structure

- Where acquisition is normal, phase fractions are stable: bright-phase area fraction 0.04–0.08 (median ≈ 0.055) in every group, bright-phase D50 ≈ 140–165 px, pore fraction 0.06–0.14. Formulation loading does not differ between batches on this evidence.
- **The one clear material anomaly is cracking.** Three Batch 3 sites (`hzumfsms`, `0grcilhi`, `ufdvpb81`) contain long delamination-style voids running along the coating: crack-like void fraction 0.048–0.062 versus a median of 0.015–0.022 elsewhere; largest pore ⌀ 450–600 px versus ≈ 250–300 px.
- The additive particles are **intact** everywhere: ETD dark-ridge density inside bright-particle interiors (curtaining-corrected) is 0.02–0.5 % in every group.
- Batch 2 has the lowest and tightest pore fraction (0.075 ± 0.008) and the fewest crack-like voids. Ordinary Batch 3 sites have a slightly longer texture correlation length (22 vs 18–19 px). Both differences are small relative to within-batch spread.
- No through-thickness gradient in pore or bright-phase fraction in any group.

### 3.3 Are the batches separable?

Not as folders. In PCA and Ward clustering on standardised site KPIs, the first split is the two low-contrast Batch 1 sites (acquisition), the second is the three cracked Batch 3 sites (material), and below that Batch 1, 2 and 3 sites interleave freely. Using the folder labels as given, only the bright-phase KPIs reach p < 0.05 between batches — and those are the ones contaminated by the Batch 1 contrast problem.

### 3.4 Cross-channel findings

- A **three-channel agreement** check (robust z of per-channel median intensity and per-channel texture energy) separates the two kinds of anomaly: the grey-pore group moves BSE and Inlens intensity in *opposite* directions with texture unchanged (instrument / preparation signature); the cracked sites move texture across channels with intensity unchanged (material signature).
- **Inlens intra-particle texture** orders the groups more strongly than any other feature (B1 0.32 > B2 0.24 > B3 0.18 > grey 0.13; η² ≈ 0.53). It is **confounded**: ρ ≈ 0.8 with Inlens median brightness, ρ ≈ 0.7 with frame height, ~75 % of its variance explained by Inlens acquisition statistics, and it still tracks brightness within each batch. Only a weak group effect (p ≈ 0.02) survives on the residuals. Status: candidate, not trusted.

## 4. KPI catalogue

All structural KPIs are computed per site on the BSE image unless stated. Lengths in pixels. Thresholds come from the smoothed BSE histogram anchored on the graphite mode: the valley to the second (bright) mode where one is resolved, otherwise mode + 3.5 σ with a flag; same logic on the dark side for pores.

| KPI | Channel | Materials meaning | Trust |
|---|---|---|---|
| `pore_frac` | BSE | open-porosity area fraction (calendering density, electrolyte access) | high, except grey-pore group |
| `pore_d50`, `pore_elong`, `pore_max_d` | BSE | pore size, shape, largest void | high |
| `crack_frac`, `crack_count_per_Mpx` | BSE | area / count of voids with long axis > 500 px (delamination-style cracks) | high; most defect-relevant KPI found |
| `bright_frac`, `bright_count_per_Mpx` | BSE | additive loading and number density | high when `bright_low_contrast` is False |
| `bright_d10/d50/d90`, `bright_circ`, `bright_solidity` | BSE | additive particle size distribution and shape | high when `bright_low_contrast` is False |
| `profile_pore_k`, `profile_bright_k` (k = 0..9) | BSE | through-thickness gradients | medium (orientation vs current collector unknown) |
| `fft_slope`, `corr_len_px` | BSE | scale-free texture descriptors | medium |
| `etd_crack_density_particles` | ETD in BSE mask | intra-particle cracking of the additive | high; null on this data |
| `etd_crack_density_graphite`, `etd_curtain_frac`, `etd_curtain_anisotropy`, `etd_boundary_sharpness` | ETD | sample-preparation / focus flags | flags, not material KPIs |
| `inlens_particle_texture`, `inlens_speckled_particle_frac` | Inlens in BSE mask | speckled vs smooth particle interiors | **confounded**; needs local-contrast normalisation |
| `bright_sep`, `bright_low_contrast`, `pore_mode_resolved`, `graphite_mode`, `th_lo`, `th_hi` | BSE | segmentation quality | diagnostic |
| per-image `p1`, `empty_bin_frac`, `gray_levels`, `band_top/bottom` | all | black level, contrast stretching, edge bands | diagnostic |

Cached outputs: `analysis_cache/site_features.csv` (one row per site), `analysis_cache/etd_inlens_features.csv`, `analysis_cache/bright_particles.csv` (one row per bright particle), `analysis_cache/image_quality.csv` (one row per image).

## 4b. Assumption register

Every interpretive assumption, with an annotated example image and a review status, lives in `docs/assumption_register.html` (built by `docs/_build_assumption_register.py`). Reviews are logged on the cards. Status as of 2026-10-03: A3 (bright phase is a distinct composition) confirmed by Santosh; A13 detector-label part (SE = ETD) confirmed; A2 wording corrected (open porosity is the electrolyte pathway of the cell, not "vacuum") and its binder illustration replaced; all other cards unreviewed.

## 5. Decisions taken

- Deliverable is a Jupyter notebook (judge-readable, reproducible), built from a generator script so it can be regenerated end to end.
- **Batch 3 is the working reference** (confirmed closest-to-baseline, 17 sites), used with robust statistics and with its sub-populations (grey-pore group, cracked sites) shown explicitly. The QC notebook still accepts any batch as reference via one config cell, so Batch 1 and Batch 2 can be compared symmetrically.
- All lengths stay in pixels; the nominal 25 nm/px from export tags may be quoted as "nominal" but not used to assert physical sizes until confirmed.
- Intensity statistics of any channel are instrument flags, never material KPIs.
- Batch 3 is one production batch, so the grey-pore group is intra-batch variation (most likely preparation / session). It stays flagged and is shown as a sub-population inside the reference rather than excluded silently.
- Acquisition-quality flags are first-class outputs of the QC report, alongside the verdict.

## 6. Open questions for the organisers

1. ~~Which batch is the approved baseline?~~ Answered: none is clean; Batch 3 is the closest reference. Follow-up: are there known differences between the three supplier batches that we should recover?
2. Are the four grey-pore Batch 3 sites and the two low-contrast Batch 1 sites intentional (preparation / session) or material?
3. Are the three cracked Batch 3 sites considered acceptable variation within the reference, or defects?
4. Is the 25 nm/px in the TIFF tags the true pixel size? Image orientation relative to the current collector?
5. What is the bright phase (Si, SiOx, other)?

## 7. Plan for the QC notebook (next)

1. Load cached features; one config cell selects the reference batch (default Batch 3) and any sites to flag or exclude.
2. Reference self-characterisation: split-half and leave-one-site-out distributions of every KPI and of a multivariate distance, giving a data-derived null for "normal variation" — reported with the reference's own sub-populations visible, and with robust (median / MAD) and classical versions side by side so the reader sees how much the cracked and grey-pore sites widen the null.
3. Pairwise batch comparison (1 vs 3, 2 vs 3, 1 vs 2): effect sizes with bootstrap intervals, KS tests with multiple-testing correction, energy distance / Mahalanobis in standardised KPI space, image-level drift scores. The deliverable is differentiation with evidence, not only a pass/fail against Batch 3.
4. Decision: accept if within baseline self-variation and no large KPI effect; reject if far outside, consistent across sites, and driven by defect-relevant KPIs; investigate otherwise, including when only acquisition flags or confounded KPIs move. Confidence from bootstrap agreement; every verdict lists what would change it.
5. Explanation: KPI table with baseline range and batch value, top drivers in plain language, example images with detected voids / particles painted, acquisition flags, three-channel agreement.
6. Dry-run on the known batches; unseen batch is one command.
