# Polaron challenge — problem statement and state of knowledge

_Last updated: 2026-10-03 (reference status and fresh graphite–Si/SiOx material confirmed). Owner: Santosh (narrative, materials review) + Claude (pipeline, statistics)._

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
| Material | **Fresh, uncycled graphite–Si/SiOx electrode, confirmed by Santosh (2026-10-03).** Porous coating with plate-like graphite and a sparse brighter particulate additive. Exact Si versus SiOx chemistry, formulation fractions, binder identity and pixelwise chemical mapping are unspecified. Material-family confirmation does not validate every thresholded bright fragment or the inferred section/collector orientation. |
| Labels / baseline | Confirmed by the problem providers (2026-10-03): the three folders are three supplier batches of the same nominal product. **Batch 3 is the reference dataset** (hence more images); the provider adds that *reference does not necessarily mean no defects* and that *it is not just the presence of defects that defines the batches — there are many complex morphology features to examine*. No per-batch acceptable / defective labels. Implication: differentiation must weigh morphology (shape, orientation, arrangement, size-distribution shape, through-thickness structure), not only defect KPIs. |

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
- ETD dark-ridge density inside bright-particle interiors (curtaining-corrected) is 0.02–0.5 % in every group. This is low detected ridge coverage in fresh material; it does not establish that every particle is intact or electrically connected. Unresolved, filled or orientation-filtered cracks can be missed.
- Batch 2 has the lowest and tightest pore fraction (0.075 ± 0.008) and the fewest crack-like voids. Ordinary Batch 3 sites have a slightly longer texture correlation length (22 vs 18–19 px). Both differences are small relative to within-batch spread.
- No through-thickness gradient in pore or bright-phase fraction in any group.

### 3.3 Are the batches separable?

Not as folders. In PCA and Ward clustering on standardised site KPIs, the first split is the two low-contrast Batch 1 sites (acquisition), the second is the three cracked Batch 3 sites (material), and below that Batch 1, 2 and 3 sites interleave freely. Using the folder labels as given, only the bright-phase KPIs reach p < 0.05 between batches — and those are the ones contaminated by the Batch 1 contrast problem.

### 3.4 Cross-channel findings

- A **three-channel agreement** check (robust z of per-channel median intensity and per-channel texture energy) separates the two kinds of anomaly: the grey-pore group moves BSE and Inlens intensity in *opposite* directions with texture unchanged (instrument / preparation signature; BSE +1.7…+2.0 z, Inlens −2.3…−2.4 z); the cracked sites show an ETD texture-energy drop (−2.2, −3.0 z) with intensity unchanged — at a |z| ≥ 2 cut-off BSE correlation length does **not** rise on those sites (z ≤ 0.67), so the cross-channel texture reading rests on ETD alone (D27).
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
| `etd_crack_density_particles` | ETD in BSE mask | detected ridge coverage inside additive-mask interiors; candidate crack appearance | measurement/interpretation review pending; no intact-particle fraction |
| `etd_crack_density_graphite`, `etd_curtain_frac`, `etd_curtain_anisotropy`, `etd_boundary_sharpness` | ETD | sample-preparation / focus flags | flags, not material KPIs |
| `inlens_particle_texture`, `inlens_speckled_particle_frac` | Inlens in BSE mask | speckled vs smooth particle interiors | **confounded**; needs local-contrast normalisation |
| `bright_sep`, `bright_low_contrast`, `pore_mode_resolved`, `graphite_mode`, `th_lo`, `th_hi` | BSE | segmentation quality | diagnostic |
| per-image `p1`, `empty_bin_frac`, `gray_levels`, `band_top/bottom` | all | black level, contrast stretching, edge bands | diagnostic |

Cached outputs: `analysis_cache/site_features.csv` (one row per site), `analysis_cache/etd_inlens_features.csv`, `analysis_cache/bright_particles.csv` (one row per bright particle), `analysis_cache/image_quality.csv` (one row per image).

### 3.5 Sensitivity to acquisition adjustment (added after the acquisition module)

Three views of each primary-KPI comparison against Batch 3 (energy distance, HL statistic, site-level permutation, 5 000 resamples): unadjusted · stratified to ordinary acquisition groups (10 reference sites) · adjusted by residualising on bright_sep, bse_p1 and etd_boundary_sharpness with the regression re-fitted inside every permutation.

| batch | unadjusted E (p) | stratified E (p), n | adjusted E (p) | attenuation strat. / adj. |
|---|---|---|---|---|
| Batch 1 | 0.98 (0.40), 17 v 5 | 0.70 (0.83), 10 v 5 | 2.34 (0.078) | +0.28 / −1.39 |
| Batch 2 | 0.89 (0.34), 17 v 7 | 1.53 (0.16), 10 v 7 | 3.56 (0.015) | −0.73 / −3.01 |

Adjustment **amplifies** rather than attenuates: the covariates, fitted on a heterogeneous reference, encode Batch 3's own sub-populations (bse_p1 vs pore_frac ρ = −0.52 on 17 sites, ≤ 0.36 on the ordinary 10). A 7-covariate ridge variant reverses the picture (p 0.97 / 0.27). Negative attenuation is read as "not explained away by acquisition covariates" and nothing more (D26). The ±5-gray-level threshold band, now computed for all 31 sites, is ≈ 33 % relative on pore_frac in every acquisition group (52 % on grey-pore sites) and exceeds every between-batch pore_frac shift.

## 4b. Assumption register

Every interpretive assumption, with an example image and a review log, lives in `docs/assumption_register.html` (built by `docs/_build_assumption_register.py`). Status as of 2026-10-03: A3 (bright phase is a distinct composition) confirmed by Santosh; A13 detector-label part (SE = ETD) confirmed; A15 reference status confirmed by the providers via Santosh; A17 fresh graphite–Si/SiOx material confirmed by Santosh. A2 now uses that material premise, while binder identification and pixelwise phase assignments still need review. A10 states low detected ridge coverage rather than particle intactness. Browser review selections are separate from these persistent logs; confirmation of metadata does not confirm every interpretation on a related card.

## 5. Decisions taken

- Deliverable is a Jupyter notebook (judge-readable, reproducible), built from a generator script so it can be regenerated end to end.
- **Batch 3 is the provider-confirmed working reference** (17 sites), used with robust statistics and with its sub-populations (grey-pore group, long-void sites) shown explicitly. Its reference status does not establish cleanliness or acceptability. The QC notebook still accepts any batch as reference via one config cell, so Batch 1 and Batch 2 can be compared symmetrically.
- All lengths stay in pixels; the nominal 25 nm/px from export tags may be quoted as "nominal" but not used to assert physical sizes until confirmed.
- Intensity statistics of any channel are instrument flags, never material KPIs.
- Batch 3 is one production batch, so the grey-pore group is intra-batch variation (most likely preparation / session). It stays flagged and is shown as a sub-population inside the reference rather than excluded silently.
- Acquisition-quality flags are first-class outputs of the QC report, alongside the verdict.

## 5b. Provider guidance received (2026-10-03)

> "batch 3 is the reference dataset (it should have more images). Note that reference doesn't necessarily mean no defects, and it's not just the presence of defects that define the batches — there's a lot of complex morphology features to examine!"

What follows from it: (1) the long-void reference sites are not a contradiction — the reference may carry defects, so the localized path stays separate and the reference's own anomalies stay visible; (2) the five primary KPIs are defect- and loading-centred; batches may differ in morphology that those do not capture, without that difference establishing a defect.

**Current computed scope (E18/E24/E25/E26G):** size-distribution widths; bright circularity, solidity and aspect ratio; void elongation; axial bright/void orientation histograms and alignment; guarded bright-centroid spacing; linear image-row phase-profile slopes; BSE texture scale; local void widths; bright/void neighbourhood and local homogeneity; long-void burden/location; fixed bright-object graph edge scale/spread, axial strength and neighbouring log-area association. Centroid spacing is not a validated agglomeration index. Image tensors/ETD ridges are not graphite plate-instance orientations, and image-row profiles are not confirmed through-thickness coordinates.

**Explicit next expansion (D40):** [task K](next_steps.md#k-morphology-differences-beyond-defect-counts) proposes shape-distribution variability and directional two-point phase association, with nonlinear image-depth structure conditional on coverage. Task E has computed fixed graph arrangement in E26G; J is an optional texture comparator. Graphite plate orientation and binder-network segmentation remain validation-gated. The K descriptors remain proposed; E26G graph descriptors are computed but expert-unreviewed. All remain secondary unless separately validated and promoted. See plan §2.2b and the live metric register/atlas for statuses.

## 6. Open questions for the organisers

1. ~~Which batch is the reference?~~ Answered: Batch 3. The provider says reference does not necessarily mean no defects; this is not a claim that every batch is defective. Follow-up: are there known differences between the three supplier batches that we should recover?
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

## 8. Implementation review status (2026-10-03, E21)

The `polaron_qc` modules implement site-level comparisons, separate drift/localized paths and conservative verdict wording, and the existing suite passes (89 tests). Four integration/boundary behaviours remain open, reproduced in `analysis/qc_review/reproduce_integration.py`:

- The report passes no acquisition attenuation into `decision.decide`. Missing views can still support a drift reject whose explanation says adjustment did not attenuate the shift.
- The report's provisional grey-pore flag uses known IDs rather than the available acquisition module's data rule. Its derived flag tables are not propagated into the site tables used for decisions. `stats.usable_n` also ignores a grey-pore boolean column when counting fallback sites.
- An explicit negative human image review remains pending, so it cannot close the localized investigation as the report's wording promises.
- MDC sampling fails when the incoming usable count exceeds the reference count; equal counts leave no reference sites for simulation and return infinity with no usable simulations. This is a limitation of the split-reference design, not evidence of infinite detectable change.

Decision stability currently reruns the statistical decision while holding cached full-sample classifier evidence fixed; it is conditional on that evidence. The classifier still needs a separate pre-run for a new batch. These are implementation findings, not revisions to the observed material KPIs or claims about unseen-batch accuracy. See the experiment log E21 and decision-log open items for the review evidence and proposed corrections.

**Status after E22 (same day):** all four behaviours are fixed with regression tests (D30–D32): the acquisition views run inside the pipeline and a drift reject is withheld when none is available; flags are data-derived and written into the site tables before any statistic (five unseen grey-pore sites now flagged 5/5, counted as fallback, and triggering the quality abstention); image review has confirmed / refuted / unreviewed states; MDC reports "not available" instead of raising or returning ∞. The material-only classifier runs inside `build_result` and is refit in every leave-one-site-out fold, so stability is no longer conditional on a cached run and the unseen batch needs no ML pre-run for its verdict. Known-batch verdicts, stability shares and numbers are unchanged (E22).

## 9. Morphology differentiation first pass (E18, 2026-10-03)

The independent report `analysis/morphology/output/report.html` expands the comparison to the existing material panel and 16 morphology descriptors: size-distribution widths, particle/void shapes, axial orientations, bright-centroid arrangement, image-depth gradients and BSE texture. It reports every batch pair, with quality-matched and ordinary-reference sensitivity views. Quality-matched site counts are 5 / 7 / 13 for Batches 1 / 2 / 3; Batch 3's three known cracked sites remain in that view. The ordinary Batch 3 reference has 10 sites.

The expanded panels do not demonstrate robust batch separation at these counts. Quality-matched morphology energy p-values are 0.669 (B1 vs B3), 0.350 (B2 vs B3) and 0.486 (B2 vs B1); no descriptor survives the broad exploratory multiplicity screen. This does not establish that the materials are identical or acceptable.

Bright-particle alignment is more horizontal in Batch 2 descriptively (+0.141 vs quality-matched B3). Its direction survives ±5-level threshold perturbations, but its approximate site-bootstrap interval [−0.034, +0.293] includes zero. The largest standardised difference, centroid spacing between B1 and B2, is sensitive to thresholding and minimum component size. Image overlays reveal small bright fragments near rims; excluding them largely removes the B1–B2 spacing difference. Centroid spacing therefore remains an unvalidated geometric comparator, not a validated agglomeration or defect KPI.

Every mask-derived descriptor has a perturbation envelope, every inferential calculation counts sites, and acquisition checks include within-batch correlations and a held-out-site acquisition-only model. Negative predictability does not prove freedom from acquisition effects. New assumption A16 records the interpretation limits. No primary KPI or QC threshold changes; ETD plate orientation and binder-network segmentation await measurement validation. See E18 and D33 for method and evidence.

## 10. Morphology metric inventory and measurement validation (E24, 2026-10-03)

The canonical inventory `analysis/morphology/metric_register.json` now tracks 78 metric entries plus ten methods (including E25 and E26G additions). Ten-bin profiles and conditional fraction families list all underlying keys. Implementation, evidence, expert review and QC role are distinct statuses. Generated views are `docs/morphology_metrics.md` and `analysis/morphology/output/metric_atlas.html`; the atlas shows real SEM markup/image-derived graphs, full-site values, source coordinates and limitations. Deferred, confounded, conditional and undefined quantities are visible without invented scalars.

An independent expert annotation page at `analysis/morphology/benchmark/review.html` contains eleven purposive crops from distinct known sites: six for development, five held out for method checks. It covers ordinary, low-contrast, grey-pore, long-void and trimmed-band cases. Manual polygons are separate from predictions; unreviewed crops are not scored, uncertain pixels are ignored and unmeasurable images can abstain. There are currently zero expert-reviewed reference masks and no segmentation-accuracy estimate. The held-out-site split is not unseen-batch validation, and crops do not increase material sample count.

Local width is estimated as twice distance-to-solid along a deterministic medial axis of retained, non-edge-clipped voids, pooling centreline samples approximately by length. Across the 31 nominal sites, median site D50 is 8.944 px and P90 29.732 px. The median relative ±5-level D50 band is 0.0308, reaching 0.3675 at one site. These are method sensitivities, not confidence intervals or 3-D pore-throat estimates. Entire large edge-connected cavities are excluded, so this metric does not describe every visible crack-like feature.

The bright hysteresis candidate changes a median 0.0534 of baseline bright-mask area across all sites (maximum 0.2019). Its median old/new IoU is 0.9493, which measures algorithm agreement, not accuracy. Low-contrast image overlays still contain bright rim fragments; no demonstrated correction or restored composition contrast is claimed. Connectivity changes object definitions slightly and remains a sensitivity audit.

Sixteen geometry/benchmark tests and artifact contracts pass. Manual annotation and KPI-error evaluation remain the next step. A16/D35 record interpretation and review limits. This experiment is independent of production QC and does not add a primary verdict driver. The live register is the maintenance entry point for future morphology experiments.

## 11. Battery application hypotheses and confirmed material state (D36, 2026-10-03)

Santosh directly confirmed **fresh, uncycled graphite–Si/SiOx electrode**. This changes the interpretation scope: observations describe manufacturing/preparation structure and possible susceptibility during formation/later cycling. They are not evidence of prior cycling damage, electrochemical SEI or lithium plating. Exact Si versus SiOx, recipe/binder, pixelwise chemical labels, collector orientation and specimen independence remain unspecified.

The specialist research memo `docs/battery_microstructure_review.md` ties six proposed checks to inspected real images and primary-source mechanisms, with retrieval limits in `analysis/battery/sources.json`. The strongest next checks are local additive/void/residual-solid neighbourhoods (B01) and reviewed long-void/confirmed-collector context (B02). Graphite plate orientation (B03), local additive homogeneity (B04), void-width/depth structure (B05) and fresh-particle fracture appearances (B06) remain application hypotheses requiring measurement review. The existing metric register and visual atlas now link these checks without adding measurements, primary KPIs or release thresholds.

Physics wording is tightened in the plan and assumption cards: more visible pore area or larger void sections are not inherently better transport; coarser bright sections alone do not determine lithiation time or total expansion load; residual-solid adjacency is not electrical contact; low ridge coverage is not an intact-particle fraction. Representative spatial sampling is required for conditional volume fractions; plate alignment alone does not invalidate phase-area estimation. D37/E25 applies these production wording fixes in `physics.py` and regenerated known-batch reports. Conditional stereology remains a model assumption, not an independent 3-D measurement.

Prioritise independent phase-boundary annotations and a small reviewed neighbourhood/interface set before new extraction. Application validation would need matched recipe/process information plus adhesion/contact, wetting, formation/impedance or cycling evidence. No numerical performance, unseen-batch accuracy or new batch-separation claim follows from the literature review.


## 12. Battery geometry checks and pipeline secondary measurements (E25 / D37, 2026-10-03)

Three parallel audits cover additive neighbourhood/local homogeneity, internal versus clipped long-void location/width-depth, and graphite-orientation/Inlens feasibility. The combined illustrated report is `analysis/battery/output/report.html`; exact site values, threshold variants, coverage, acquisition screens and measured-source provenance sit alongside it. All31 known sites were used; this is further known-batch development, not held-out batch validation.

**Eight new experimental pipeline KPIs:** `bright_void_distance_d50_px`, `bright_ring_void_frac_16px`, `local_bright_std_512px`, `local_bright_std_1024px`, `local_bright_pore_spearman_512px`, `local_bright_pore_spearman_1024px`, `long_void_internal_area_frac`, and `long_void_y_centroid_norm`. They appear in a separate HTML report section with raw-unit site-median differences, usable n, site-bootstrap intervals, paired threshold ranges and observability counts. Primary/ML lists, five-KPI multiplicity family, decisions and frozen threshold hash are unchanged. Feature version1.1.0 invalidates stale extraction caches; the audit verifies reuse of unchanged primary columns plus secondary masks recomputed from raw BSE, explicitly distinguished from a full cold all-channel run.

**Descriptive neighbourhood result:** B2−B3 minimum-distance difference +3.19px, site-bootstrap95% [−0.071,4.777], usable7v13. The paired threshold endpoint envelope remains positive, but sampling uncertainty includes zero. Mask-ring void fraction is lower in B2, also with an interval spanning zero. Bright dispersion shows little batch differentiation and correlates with loading/size, so is not a pure agglomeration index. Rings include threshold holes inside the mask silhouette and cannot represent physical expansion space. Exact raster bright/void boundary adjacency is zero at every usable site; it stays diagnostic and is excluded from batch comparisons.

**Long-void context:** the three previously selected long-void reference examples have greater internal burden than the ten ordinary B3 sites: median difference0.01559fraction, bootstrap95% [0.01095,0.02111]. This subgroup was chosen using earlier morphology and is not independent defect validation. Their internal centroids0.299/0.447/0.537 do not show shared lower-frame localisation. Edge-clipped components contain37.5–47.9% of their long-void area; local widths exclude them. No collector interface is independently confirmed; the visible bottom band in `epqdaau9` remains a candidate, and interface-gap fields are unavailable. Width-depth analysis is standalone: all31 nominal profiles and paired sensitivity on11 prespecified examples.

**Orientation/Inlens feasibility:** residual-solid masks form essentially one connected packing region (median largest-component share0.99569), not individual graphite plates. Fine/coarse and detector image directions differ and some resultants are weak. Plate orientation remains deferred until instance/axis review. Local standardisation of a separate sampled Inlens-gradient diagnostic reduces pooled brightness rho0.564→0.119 but leaves/reverses within-batch associations (B1−0.600/B2−0.631/B3+0.598 among bright-usable sites at n5/7/17); it does not fix or promote existing per-particle texture.

**Quality and uncertainty:** all31 `pore_mode_resolved` flags are false. This common fallback-method flag does not isolate the four grey-pore sites; use the separately data-derived `grey_pore` flag for quality filtering, disclose unresolved thresholds and retain expert-label uncertainty. Missing quality flags/measurements cannot imply reliability or zero. Coupled threshold offsets are not a full grid or accuracy interval. Sites are the statistical units; physical specimen independence is unresolved. No battery-performance, equivalence/acceptance, expert segmentation accuracy or unseen-batch claim follows.

**Verification:** the current full pre-commit suite passes: 142 tests in 153.99s using `/opt/anaconda3/bin/python3 -m pytest -q`, including real extraction/cache regressions, geometry contracts and the final missing-quality guards. Real default `build_result` warm replay for both incoming batches preserves all historical primary columns and returns the same 'consistent with working reference (within detectable limits)' outcome, stability1.0 and threshold hash`b4f4da2e357c`. Cache replay mode and source snapshots are explicit in `pipeline_cache_provenance.json`. The live morphology inventory, atlas and A18 record implementation separately from expert validation and QC role. Pre-commit review also checks C04/C13/C16/C21/C28/C29, valid artifact syntax, preserved registry history and exclusion of raw TIFFs/oversized files.


## 13. Additional ML methods feasibility (D38, 2026-10-03)

A dedicated methods agent assessed CNNs/fixed convolutions, object graphs/GNNs, frozen/self-supervised encoders, segmentation and one-class/reconstruction alternatives against this dataset and the existing ML code. See `analysis/ml_options/assessment.md` for the bounded shortlist and two proposed experiment protocols, `sources.json` for primary-source retrieval scope, and `candidate_register.json` for recommendation/deferred statuses. This is a source/code review, not a new performance experiment.

Prioritise deterministic bright-object graph/spatial descriptors and an equal-site-weighted audit of existing frozen DINOv2 novelty. The graph representation should expose arrangement and be checked against node-only size/loading summaries, topology/object-floor/edge sensitivity and acquisition controls. It does not establish physical contact or graphite/electrical connectivity. The encoder audit should give each reference site equal influence during PCA/memory fitting as well as evaluation, refit all learned preprocessing in held-out-site folds and preserve nearest-reference evidence. Existing novelty currently refits PCA in each site fold but uses all reference patches, so variable patch counts can affect representation/memory influence.

Fixed Gabor convolution is an optional bounded texture comparator if it adds evidence beyond existing FFT/image tensors. Supervised U-Net measurement improvement needs independent phase/instance annotations and site-held-out KPI-error evaluation. Training a new CNN/GNN defect model, in-house self-supervised encoders or deep reconstruction/density models remains deferred: only batch identity is known, not accept/reject outcomes; there are few independent material units, uncertain specimens, imperfect masks, acquisition confounds and a heterogeneous reference. More nodes/patches do not increase independent graph/site n. These are project-specific priorities, not claims that those architectures can never work with small labelled datasets.

No new model was trained, no architecture advantage/generalisation was measured and no KPI/classifier/verdict input changed. The plan now matches implemented behaviour: embedding novelty remains evidence only; it cannot independently force investigate. New protocols must be fixed before unseen results are viewed. If the unseen batch is used for method selection, it becomes development evidence. Next training gates are reviewed instances/phase labels, specimen/session grouping and task-valid outcome data; graph learning also needs an edge/no-edge ablation against simple geometry.


## 14. Next working session (D39)

`docs/next_steps.md` provides selectable unclaimed tasks for independent annotation, specimen/process/tolerance metadata, battery-context review, annotation-based measurement evaluation, two bounded ML audits, explicit morphology expansion (K, D40), notebook/report consistency, a current cold drop rehearsal and the judging narrative. Prioritise measurement validity and delivery; new morphology and ML branches remain experimental. These planning updates change no measurements or verdicts and start no new experiments.

## 15. Fixed particle-neighbourhood graph audit (E26G / D41G)

Task E is complete as an exploratory known-site measurement/control audit in an
isolated Codex worktree. Four registered descriptors summarise edge scale,
spread, axial organisation and endpoint log-area association for a fixed
3-nearest-neighbour centroid graph. All known sites and the prespecified
threshold/object-floor variants are measured; object/edge counts are coverage,
not independent sample n. The bright-only quality view is distinct from
grey-pore-excluded and ordinary-reference sensitivities.

`analysis/ml_options/e_graph/report.html` explains the graph on real BSE images;
`findings.md`, `comparisons.csv`, `sensitivity_summary.csv` and
`control_predictions.csv` preserve site uncertainty, threshold/floor/deletion
sensitivity and acquisition/node-loading redundancy. Proximity is not physical
or electrical contact. Fixed ridge controls do not prove causal attribution or
an architecture advantage. Retain for visual review; defer material/QC use until
independent component/specimen/process validation. The primary/classifier and
decision paths remain unchanged; no unseen or defect-accuracy claim follows.

**Measured qualification:** all 12 nominal pairwise site-bootstrap intervals
include zero in both the bright-usable and grey-pore-excluded views. Ten of the
12 bright-usable comparison directions reverse across the six fixed threshold/
floor settings; the two stable directions also have intervals including zero.
These are correlated descriptive comparisons, not independent tests. Typical
edge length is partly predicted by the node/loading control (whole-site LOO
R² 0.496). Poor acquisition prediction does not establish invariance. These
results support exploratory review and deferral, not equivalence or acceptance.

G integration should reconcile additive register/log changes with parallel
H/F/E18 work; see the graph `handoff.md`. Independent expert review remains open.
