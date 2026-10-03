# E25 — graphite-orientation and Inlens feasibility audit

**Automated graphite plate orientation remains deferred.** The fresh graphite–Si/SiOx metadata makes plate alignment an application-relevant question, but it does not turn directional image texture into graphite instances. The audit extracts image texture diagnostics only; no primary KPI, material classifier input, release threshold or verdict is changed.

## What the real images show

The [matched central crops](orientation_matched_crops.png) cover ordinary `Batch_1/f1vzngrs`, low-contrast `Batch_1/4ih2ggld`, grey-pore `Batch_3/71vgq3fw` and long-void site `Batch_3/hzumfsms`. The same raw coordinates are used for BSE, ETD and Inlens, with display limits fixed at 0–255. Recognizable plate sections coexist with fine striations crossing broad particle faces. ETD makes both particle edges and those fine striations more conspicuous. Yellow marks show fine image-tensor directions and are explicitly not plate outlines or plate axes.

The illustration samples tensor directions across all phases at local coherence > 0.35; the numeric site summaries instead use residual-solid centre pixels. This display threshold does not select the numeric measurand.

The three-class BSE mask combines graphite, unresolved fine domains and other residual solid. Where thin interparticle gaps are unresolved, residual-solid regions join. Taking the major axis of that mask would measure a connected packing region, not an individual graphite plate. A contour inferred from a bright particle or a pore is also not a graphite instance. These are specific measurement problems to resolve before adding `reviewed_graphite_plate_alignment` or reviving `etd_plate_orientation`.

Across the 31 sites, the median of the per-site median largest residual-solid component share is **0.99569** in the fixed windows. This establishes extensive mask connectivity under this three-class definition, not that graphite physically constitutes one particle. It directly rules out treating each connected residual-solid region as a graphite plate instance.

## Bounded, reproducible measurement

The script `analysis/battery/run_orientation_audit.py` uses eight fixed 512 × 512 px windows per site, with common coordinates across the three aligned channels. The windows are within-site subsamples, not independent material samples. The raw coordinates and analysis guards are saved in `orientation_patch_manifest.csv`; only the production BSE-derived bright edge bands are trimmed. No extra four-pixel border is silently removed.

Gaussian gradient tensors use derivative sigma 3 or 12 px and integration sigma 12 or 48 px respectively. An eroded residual-solid mask identifies centre pixels, while tensor neighbourhoods may cross phase boundaries. Trace-weighted axial vectors summarize horizontal image-line alignment, directional strength and mean local coherence per site. Positive alignment denotes horizontal lines; negative denotes vertical lines. The angles are relative to image horizontal because collector direction is unconfirmed. Constant or empty support returns unavailable values.

Thresholds are perturbed jointly by ±5 grey levels to audit support-mask sensitivity. These ranges are method envelopes, not confidence intervals. Fine/coarse disagreement and detector disagreement remain visible; no best scale is selected using batch separation. Site-wide axial resultants can be weak because local directions cancel, so nominal angle differences must be read with their directional-strength values; they are not plate-angle errors or a reliability score. No statistical test or classifier interprets this exploratory panel as a material change.

## Local-contrast Inlens check

The optional experiment measures fine-gradient RMS on the same eroded bright-mask support with at least 400 sampled pixels per eligible window. It compares patch-wide IQR scaling against local standardisation:

`N = (I − Gaussian32(I)) / sqrt(max(Gaussian32(I²) − Gaussian32(I)², 0) + (0.05 × patch IQR)²)`.

The floor scales with patch IQR, so affine gain/offset invariance is a meaningful mathematical check. This does not remove saturation, non-linear contrast LUTs, charging relief, preparation texture or changes in noise. A floor-dominated fraction and saturated bright-support fraction are saved alongside the diagnostic. The [real normalization examples](orientation_inlens_normalisation_examples.png) show that conspicuous edges and preparation relief can persist after local standardisation.

This is a new sampled gradient diagnostic, **not a corrected estimate of the existing `inlens_particle_texture` per-particle standard deviation**. Normalisation changes the measurand. A lower brightness correlation would establish only reduced association with that particular covariate, not a material origin or valid discrimination. The gradient comparison uses the nominal bright support; its phase-mask uncertainty is not represented by the directional-texture threshold bands. Low-contrast bright supports retain their quality flag; image-wide normalisation cannot recover composition contrast missing in BSE.

The measured brightness association illustrates that limit. Across all 31 sites, Spearman correlation with Inlens median brightness falls from **+0.564 to +0.119**. Within each batch, locally standardised gradient RMS still associates with brightness: **−0.847 / −0.631 / +0.598**, with **7 / 7 / 17 sites** for Batches 1 / 2 / 3. Excluding the two low-contrast bright supports changes Batch 1 to **−0.600 (n = 5)**; Batch 2/3 bright-usable counts remain 7/17. Restricting Batch 3 to its ordinary source group gives **+0.442 (n = 10)**. These descriptive correlations have small within-batch counts and are not causal or accuracy estimates. The pooled reduction therefore does not justify promoting normalized Inlens texture to a material KPI. All quality views are saved in `orientation_inlens_quality_sensitivity.csv`.

## Interpretation and next measurement

For B03, the useful next step is a small independent outline/axis review of clearly recognizable graphite plate sections across ordinary, low-contrast and grey-pore sites. Preserve unmeasurable/ambiguous instances, record the section and collector orientation, and compare image directions with reviewed plate axes. Then quantify instance coverage and angle error on held-out reviewed sites before attempting automated extraction. Pixel agreement between detectors is corroboration, not an expert label.

The battery reason for asking about alignment is supported by the tomography work cited in [the application review](../../../docs/battery_microstructure_review.md), particularly [Pietsch et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5052642/) and [Ebner et al.](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aenm.201301278). They do not calibrate transport from these 2-D image-tensor values. No tortuosity, wetting, capacity, fast-charge or lifetime prediction follows from this audit.

## Outputs and checks

`orientation_summary.json` is the machine-readable result/provenance entry point. Per-site values, method sensitivities, detector angle differences, group summaries, residual-mask connectivity and acquisition-covariate screens are separate CSVs. Correlation screens use sites and report usable counts for all sites, each batch and the ordinary subset. The correlation subset labelled `ordinary` uses the source acquisition-group label and excludes known cracked sites; the group-summary row labelled `ordinary` denotes normal acquisition and retains them. Individual pixels/patches never become `n`.

The completed audit contains **31 sites, 248 within-site windows and 558 detector/scale/threshold rows**. Median directional strengths range from 0.102 to 0.192 across detector/scale combinations, confirming that nominal global angles often summarize weak resultants. All six ±5-support envelope medians are below 0.009 alignment units; small support sensitivity does not validate the graphite interpretation.

Synthetic checks cover horizontal/vertical angle sign, constant/empty abstention, affine orientation invariance and affine local-standardisation invariance. Raw TIFFs, cached tables and source hashes are checked for stability across extraction. Real crop layout and overlay semantics are inspected separately; no expert instance labels or unseen-batch accuracy are fabricated.

The first complete extraction was withheld because its final source guard detected a concurrent production `features.py` edit. The parent confirmed that this edit changed only the feature version and added a secondary-extraction call; the loader, trim, smoothing, thresholds and segmentation arithmetic were unchanged. The published rerun loads `orientation_features_snapshot.py`, an exact copy from the observed clean commit `79bb912`, as the actual helper module and records its hash. It checkpoints completed-site tables but declares success only after the final input/source check. This private snapshot permits production work to continue without weakening the integrity guard.

The rerun's complete measurements passed that final guard. A subsequent duplicate-column error in the metadata summary join was repaired by selecting only the required covariates. `orientation_extraction_source_snapshot.py` preserves the exact measurement source; `orientation_extraction_manifest.json` records recovery after that summary error. `--from-tables` rechecks the manifest and renders the completed tables without re-extraction. The current summary-source hash is recorded separately. No numeric extraction choices or measurements changed during this repair.

Checklist applied: C01/C02 (nonconstant values, no by-construction material claims), C03 (acquisition screens and residual limits), C04/C05/C13/C21 (actual measurand and no texture-to-plate/transport promotion), C06/C14 (site-level counts and unresolved specimen independence), C16/C17 (known-batch exploratory provenance and method-envelope language), C28 (no predictions treated as independent labels).
