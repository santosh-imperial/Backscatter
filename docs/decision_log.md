# Decision log

Two purposes. **Part A** records every consequential decision: what was decided, why, what else was considered, what would reverse it. **Part B** is a review checklist built from the mistakes that external reviewers caught and we did not; it is run before any plan, result or verdict is presented, and it grows whenever a reviewer catches something new.

Conventions: one entry per decision, newest at the bottom of Part A. `Caught by` names who found a problem with a decision (self, Santosh, external review). Entries are never deleted; reversed decisions get a `Superseded by` line.

---

## Part A — Decisions

### D01 · 2026-10-03 · Deliverable is a Jupyter notebook, generated from a build script
- **Decision:** judge-facing artefacts are executed notebooks; each is produced by `notebooks/_build_*.py` and executed in place.
- **Why:** reproducible end to end; judges can read code and prose together; a generator keeps prose and code in one reviewable file.
- **Alternatives:** CLI + HTML report only (more robust, less narrative); interactive dashboard (demo polish, more failure risk).
- **Reverses if:** judging format requires a live app.

### D02 · 2026-10-03 · Work split by stage
- **Decision:** Claude: loader, KPIs, statistics, decision logic, notebook skeleton. Santosh: data inspection, materials sanity-check, narrative, organiser liaison.
- **Why:** Santosh's choice; keeps materials judgement with the person who has it.

### D03 · 2026-10-03 · All lengths in pixels; µm only as "nominal"
- **Decision:** KPIs in px; µm quoted only as nominal at 25 nm/px.
- **Why:** microscope metadata stripped; only the TIFF resolution tag survived and it is export metadata, not verified calibration.
- **History:** first written as "no metadata survived"; the parallel audit in `analysis/` found the 25 nm/px tag. **Caught by:** external audit. Lesson → checklist C12.

### D04 · 2026-10-03 · Raw intensity of any channel is never a material KPI
- **Decision:** intensity statistics are acquisition flags only.
- **Why:** ~half of images were contrast-stretched after acquisition (comb histograms); Inlens is charging-dominated.
- **Reverses if:** raw, unprocessed exports become available.

### D05 · 2026-10-03 · BSE segmentation by graphite-mode-anchored valley thresholds, with fallback and flag
- **Decision:** per image: graphite mode → valley to the bright mode (if resolved) else mode + 3.5 σ with `bright_low_contrast = True`; same on the dark side for pores.
- **Why:** plain multi-Otsu placed the upper threshold inside the graphite peak on low-contrast images and inflated bright fraction 3×.
- **Alternatives:** fixed thresholds (fail under stretching); multi-Otsu (failed); learned segmentation (no labels).
- **Caught by:** self, from a Batch 1 outlier in the first KPI table. Lesson → C01.

### D06 · 2026-10-03 · Texture scale KPI = autocorrelation 1/e length, replacing FFT peak
- **Decision:** `corr_len_px` replaces `fft_peak_px`.
- **Why:** the FFT peak was constant across all sites (argmax at the lowest frequency) — a broken feature that also broke the ANOVA.
- **Caught by:** self, from a constant column in the summary table. Lesson → C01.

### D07 · 2026-10-03 · Two Batch 1 sites flagged low-contrast; bright-phase KPIs unreliable there
- **Decision:** `4ih2ggld`, `5n1q8atc` flagged; excluded from pooled particle statistics; still used for pore/crack KPIs.
- **Why:** bright phase only 25–35 gray levels above graphite vs 50–70 elsewhere; identical frame height; no threshold can separate additive from rims/binder.

### D08 · 2026-10-03 · Batch 3 grey-pore group kept as a flagged sub-population, not excluded
- **Decision:** `71vgq3fw`, `kbdh4tri`, `tuy3zymq`, `x7u69zsw` analysed as a group inside the reference.
- **Why:** no black pixels, grey pore plateau, identical height, more curtaining, lowest bright-phase separation: a preparation/session fingerprint. Batch 3 is confirmed to be one production batch, which strengthens but does not prove that reading.
- **Reverses if:** organisers say these sites were prepared identically (then it is a material anomaly inside the reference).

### D09 · 2026-10-03 · Crack-like void KPI = pores with long axis > 500 px
- **Decision:** `crack_frac`, `crack_count_per_Mpx`, `pore_max_d` added after the PCA exposed three Batch 3 sites with large elongated voids.
- **Why:** the only clear material anomaly in the data; a materials engineer would want it shown, not scored.
- **Open:** 500 px (≈ 12.5 µm nominal) is a judgement call; asked in the assumption register (A8).

### D10 · 2026-10-03 · ETD ridge threshold is one absolute value across sites
- **Decision:** `T_STAR` = grid value nearest the median per-site 97th percentile of normalised ridge strength.
- **Why:** a per-image percentile threshold would make every density equal by construction.
- **Caught by:** self, while reviewing the prototype. Lesson → C02.

### D11 · 2026-10-03 · Inlens intra-particle texture labelled confounded; cannot drive a verdict
- **Decision:** kept as candidate KPI with explicit confound warning.
- **Why:** strongest batch ordering of any feature (η² ≈ 0.53) but ρ ≈ 0.8 with Inlens median brightness, ρ ≈ 0.7 with frame height; ~75 % of variance explained by acquisition statistics; weak residual effect (p ≈ 0.02).
- **Caught by:** self, by testing the feature *because* it looked too good. Lesson → C03.

### D12 · 2026-10-03 · ETD and Inlens features added inside BSE masks; three-channel agreement check
- **Decision:** intra-particle crack density, curtaining and boundary-sharpness flags, Inlens particle texture, per-channel z-score agreement.
- **Why:** reviewer noted every structural KPI was BSE-only and the stated ETD role was a promise the code had not kept.
- **Caught by:** external review. Lesson → C04.

### D13 · 2026-10-03 · Open porosity wording; binder network illustrated, not segmented
- **Decision:** "open porosity (empty in the section, electrolyte pathway in the cell)", never "vacuum"; the binder/carbon-black network is shown with insets and acknowledged as unsegmented (falls into pore or graphite class).
- **Caught by:** Santosh, via the assumption register (A2). Lesson → C05.

### D14 · 2026-10-03 · Batch 3 is the working reference, not a clean baseline; the task is differentiation
- **Decision:** reference = Batch 3 with sub-populations visible; all pairwise comparisons reported; reference selectable in one cell.
- **Why:** confirmed by the problem providers: three supplier batches of one product; Batch 3 is one batch with more samples; no clean baseline exists.
- **Supersedes:** "baseline unknown, baseline-agnostic by default".

### D15 · 2026-10-03 · ML corroboration limited to two members
- **Decision:** grouped-CV L1-logistic regression (site-level permutation null, equal site weight) + exploratory patch-embedding novelty. GMM density and isolation forest deferred.
- **Why:** four models on seven independent sites would mostly agree on shared confounders; isolation forest has no out-of-bag score with default settings.
- **Caught by:** external review (first plan review). Lesson → C06.

### D16 · 2026-10-03 · Top verdict is "consistent with the working reference, within detectable limits"; "accept" needs tolerances
- **Decision:** four verdicts; accept reserved for an equivalence test against practical tolerances that do not yet exist; release provisional.
- **Why:** a null result from a weak test with 7 sites is not acceptance.
- **Caught by:** external review. Lesson → C07.

### D17 · 2026-10-03 · Separate decision paths for batch-wide drift and localized defects
- **Decision:** Check A (drift, primary KPIs) and Check B (localized, per-site maxima and extreme patches) run in parallel.
- **Why:** a reject rule requiring cross-site consistency would miss one severe delamination.
- **Caught by:** external review. Lesson → C08.

### D18 · 2026-10-03 · Calibration must match the real comparison
- **Decision:** site-level permutation nulls for the actual pair; MDC at α and 80 % power by simulation with the usable n; "decision stability" instead of "confidence"; reference splits are internal diagnostics only.
- **Caught by:** external review (both rounds). Lesson → C09, C10.

### D19 · 2026-10-03 · "Sensitivity to acquisition adjustment", not "instrument share"
- **Decision:** unadjusted / stratified / adjusted views; attenuation described, no causal percentage.
- **Caught by:** external review. Lesson → C11.

### D20 · 2026-10-03 · Physics layer is qualitative and relative; stereology secondary
- **Decision:** direction-of-change statements only; 2-D tortuosity is a section index; 2-D connection ≠ electronic continuity; matched-quantile size comparison; capacity proxy dropped; primary KPIs are observed 2-D measurements.
- **Caught by:** external review (both rounds). Lesson → C12, C13.

### D21 · 2026-10-03 · Small-sample discipline (plan §2.0)
- **Decision:** five primary KPIs; usable site counts after flags; permutation at site level with all reference-dependent fitting inside the loop; jackknife stability; patches never count as n; exceedance of the reference maximum is an evidence flag with severity/reliability/image-review conditions; provenance wording "developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived".
- **Caught by:** Santosh's question ("are we taking the small dataset into account?") and the second external review. Lessons → C09, C10, C14, C15.

### D22 · 2026-10-03 · Decision thresholds (polaron_qc/decision.py `Thresholds`)
- **Decision:** α = 0.05 on Holm-adjusted primary p-values; a primary KPI "carries" drift only if |robust shift| ≥ 1.0 reference MAD; cross-site consistency = ≥ 50 % of usable batch sites beyond the ordinary-reference range in the shift direction; minimum usable sites per KPI = 5 (below → that KPI abstains; all primary abstaining → quality abstention); Check B severity margin = 1.0 MAD of ordinary-reference per-site maxima; strong attenuation = multivariate statistic dropping by > 50 % under stratified/adjusted views → investigate, not reject; reject from Check B alone needs ≥ 2 credible sites each ≥ 2.0 MAD beyond the reference max; image review is a required human step for "credible" (status `credible_pending_review` until then); grey-pore is a *soft* reliability flag (fallback threshold), low-contrast is a hard one for bright-phase KPIs.
- **Why:** round numbers chosen from the plan's logic, not fitted to any batch. Thresholds are hashed and the hash is asserted before the unseen batch runs.
- **Provenance:** developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived.
- **Alternatives:** tuning on Batch 1/2 verdicts (rejected — no labels, would make the unseen batch the only test of a fitted rule); α = 0.10 for more power (rejected — would raise the drift-alert rate on a heterogeneous reference).
- **Reverses if:** organisers supply tolerances (then "accept" via equivalence test replaces "consistent") or the reference-split diagnostics show the drift-alert rate far from α.
- **Superseded in part by:** D29 (severity margin 1.0 → 2.0 MAD; promotion restricted to crack_frac and pore_max_d).

### D23 · 2026-10-03 · MDC simulation draws without replacement; verdict statistic is Hodges–Lehmann; MDC on the full usable reference
- **Decision:** `stats.mdc` draws the pseudo-batch from the reference *without* replacement (an exact random split); the permutation statistic for verdict p-values is `hl_shift` (Hodges–Lehmann median of pairwise differences), with median-difference / reference-MAD kept as the reported effect size; MDC is computed on the full usable reference (17 sites), never on the 10 ordinary sites.
- **Why:** the with-replacement design I specified rejected **11 %** at zero shift (α = 5 %): duplicated values land only in the pseudo-batch while exactly those sites leave the reference, breaking exchangeability. The exact difference-of-medians statistic has only ~56 attainable values at 7 v 17; 21 % of null p-values are exactly 1 and power is lost (bright_d50 MDC 3.75 MAD vs 2.50 with HL; crack_frac 2.50 vs 1.75). On 10 ordinary sites with 7 incoming, 3 remain and power never reaches 0.8.
- **Caught by:** stats build agent (self-test of the null at zero shift; null-uniformity test on the statistic). Lessons → C19, C20.
- **Reverses if:** a KPI has heavy ties where HL misbehaves (then fall back to mean_diff with the same enumeration).

### D24 · 2026-10-03 · Classifier corroboration uses material KPIs only; boundary sharpness is a flag
- **Decision:** Check A(iv) reads the c2st run on material KPIs only (pore_frac, pore_d50, pore_elong, pore_max_d, crack_frac, bright_frac, bright_count_per_Mpx, bright_d50, bright_circ, corr_len_px, etd_crack_density_particles); the flag-inclusive run is reported as acquisition evidence. `etd_boundary_sharpness` is removed from every "trusted material KPI" list.
- **Why:** with it included, Batch 1 vs Batch 3 separates (AUC 0.79, p 0.03) entirely through boundary sharpness; without it AUC = 0.53. The findings doc already filed it as a prep/focus flag; my brief to the ML agent listed it as trusted — a leak.
- **Caught by:** ML build agent (C03). Lesson → C21.

### D25 · 2026-10-03 · Novelty is an evidence flag; no 2-D tortuosity index; threshold band per site
- **Decision:** patch-embedding novelty can only raise an evidence flag (same severity logic as C4; i.i.d. exceedance ≈ 29 % per 7-site batch); the report prints its acquisition correlates. The section tortuosity index is not presented (undefined on all 31 sites); `fraction_connected = 0` is stated once. The ±5-gray-level threshold band is computed for every site in the features stage and printed next to every pore-based shift; physics transport wording is emitted only when |shift| exceeds the band.
- **Why:** novelty's top correlates are bse_std, curtaining, low-contrast flag and black level; its site ranking changes with input resolution (ρ 0.43). No site has a 2-D macro-pore path top-to-bottom at 6–14 % area fraction. The band moves pore_frac by 28–50 % relative, as large as the between-batch differences.
- **Caught by:** ML and physics build agents.

### D26 · 2026-10-03 · Acquisition adjustment is shown as two variants and read as "not explained away" when it amplifies
- **Decision:** the adjusted view uses three a-priori covariates (bright_sep, bse_p1, etd_boundary_sharpness; OLS, re-fitted inside every permutation) and is always shown next to a 7-covariate ridge variant; negative attenuation is reported as "the shift is not explained away by acquisition covariates", never as evidence of a material change; fixed-residual p-values are kept as a column to show they are anti-conservative.
- **Why:** on this reference the adjustment *amplifies* the shift (attenuation −1.39 for Batch 1, −3.01 for Batch 2): covariates fitted on the heterogeneous reference encode its own grey-pore and cracked sub-populations (bse_p1 vs pore_frac ρ = −0.52 on all 17 sites, |ρ| ≤ 0.36 on the 10 ordinary ones), predict higher pore_frac for every black-level-0 site and shrink the residual MAD. Fixed residuals gave p 0.002 where in-loop refitting gives 0.021. The 7-covariate ridge variant reverses the conclusion (p 0.97 / 0.27), which is itself the plan §2.6 point: the conclusion depends on the covariate set.
- **Caught by:** acquisition build agent (C03, C09, C19). Lesson → C23.
- **Reverses if:** covariates are fitted on the ordinary subset only (an option to evaluate) or a clean reference becomes available.

### D27 · 2026-10-03 · Three-channel agreement does not support "BSE correlation length rises on cracked sites"
- **Decision:** the findings doc and register wording for the cracked sites is reduced to what the agreement check supports at |z| ≥ 2: ETD texture energy drops (−2.2, −3.0 z); BSE correlation length does not cross the threshold (z ≤ 0.67). The grey-pore "intensity-only shift" reading stands (BSE +1.7…+2.0 z, Inlens −2.3…−2.4 z).
- **Caught by:** acquisition build agent porting notebook-01 §6d with fixed thresholds. Lesson → C24.

### D28 · 2026-10-03 · Localized path restricted to extreme-semantics KPIs; a missing classifier run never counts as corroboration; unreviewed severe flags route to review
- **Decision:** Check B considers only KPIs with per-site *extreme* semantics (`LOCAL_KPIS`: crack_frac, crack_count_per_Mpx, pore_max_d, bright_max_d, patch maxima of crack area and pore size, intra-particle crack density). Exceedances on batch-mean KPIs (pore_frac, bright_frac, bright_d50) are reported as "outlying site" descriptive flags and feed the per-site drift score, never a localized verdict. Reject via Check A requires the material-only classifier to be present *and* positive; `c2st=None` yields "classifier corroboration not available" and caps the verdict at investigate. A severe, reliable, unreviewed flag yields "investigate — localized anomaly (image review pending)" with its crop, because routing to a reviewer is the investigation; Check A's cross-site consistency counts usable sites only.
- **Why:** the first end-to-end run returned "investigate — localized anomaly" for Batch 2 on a single-site bright_frac exceedance (+2.1 MAD) — a high-loading site, not a defect — and would have allowed a reject with no classifier result because `None` was treated as satisfied. The report agent caught both.
- **Caught by:** report build agent (end-to-end assembly). Lesson → C25.
- **Reverses if:** organisers define per-site tolerances for batch-mean KPIs.

### D29 · 2026-10-03 · Localized path calibrated: 2.0-MAD severity margin, promotion only on crack_frac and pore_max_d
- **Decision:** `Thresholds.severity_margin_mad` = 2.0 (was 1.0) and only `LOCAL_KPIS_PROMOTE = [crack_frac, pore_max_d]` can turn an exceedance flag into *pending review / credible*; the other extreme-semantics KPIs (crack count, bright_max_d, patch maxima, intra-particle crack density) are still flagged and shown but never drive a verdict. The measured false-alarm rate of the localized path is printed in the notebook self-test next to the rule.
- **Why (calibration, not tuning on verdicts):** under a Gaussian i.i.d. null with 7 incoming vs 10 ordinary reference sites, P(≥ 1 pending flag) at a 1.0-MAD margin is 13.5 % per KPI and 42 % across four KPIs; at 2.0 MAD it is 4.0 % per KPI, ≈ 8 % for two and 15 % for four. On real 5-v-5 splits of the ten ordinary reference sites the pending rate was 83 % at 1.0 MAD and still 57 % at 2.5 MAD across five KPIs (heavy-tailed KPIs, five-site MADs). The notebook's first self-test showed 60 % pending on ordinary-only splits — unusable as a review trigger. The three known cracked sites keep margins 5.8 / 5.4 / 2.6 MAD on crack_frac and remain credible; a synthetic 2× crack fraction on Batch 2 gives one site at 3.4 MAD (caught), 1.5× gives 0.6 MAD (missed, as the MDC predicts).
- **Caught by:** the notebook's ordinary-only reference-split diagnostic (C10, C19). Lesson → C26.
- **Provenance:** calibrated on null simulations and reference splits only, before the unseen batch; thresholds hash changes accordingly.
- **Open:** even at 2 MAD on two KPIs the expected review rate on a clean batch is ≈ 8 % Gaussian and higher with heavy tails. Whether that review load is acceptable is a QC-usability question for Santosh / the organisers; the alternative is to require two sites or two KPIs to agree. — **resolved by D33 (tiered rule), agreed with Santosh 2026-10-03.**

---

### D30 · 2026-10-03 · Verdict inputs that are missing default to the conservative verdict; classifier and acquisition views run inside the pipeline
- **Decision:** (a) `decision.decide` requires `attenuation.available = True` (at least one finite stratified / adjusted share) before a Check A reject; `att=None`, `{}` or all-NaN yields *investigate — drift* with the reason "acquisition sensitivity views not available — reject withheld", and the reject reason prints the shares it relied on. (b) `acquisition.three_views` is computed inside `report._run_pipeline` on the primary KPIs (full run and every jackknife fold; a failure is recorded as `{error}` and leaves the reject withheld). (c) The material-only classifier that drives Check A(iv) is run inside `build_result` on the site tables (`report.material_c2st`, MATERIAL_KPIS, 200 site-label permutations; 100 per jackknife fold), so an unseen batch needs no pre-computed ML cache for the verdict; the cached run is kept as `material_cached` for comparison.
- **Why:** E21 reproduced a *reject (provisional)* with all Check A gates positive and no acquisition view computed, whose explanation claimed the shift had not been attenuated. The report had passed `att=None` since the first end-to-end run (D28 fixed the same class of defect for the classifier). The views cost 3–15 s and the classifier 2 s, so there is no reason to keep either outside the pipeline. On Batches 1 and 2 the in-pipeline classifier reproduces the cached numbers (AUC 0.51 / p 0.458 and 0.61 / 0.264), the attenuation shares equal the notebook's (+0.28 / −1.39 and −0.73 / −3.01) and both verdicts and stability shares (1.00) are unchanged.
- **Alternatives rejected:** treating a missing view as "no attenuation" (the previous behaviour; optimistic in the direction of reject); requiring both views (the stratified view legitimately refuses below four sites per group, so one finite view is the usable minimum); keeping the classifier as a cached pre-run with a README instruction (one more thing to forget under time pressure on the unseen batch).
- **Caught by:** E21 implementation review (checklist C25 "every missing-input default against the conservative direction" was written after D28 but had not been re-applied to `att`). Lesson → C27 (exercise the unseen-batch boundaries) stays; C25 now explicitly lists `att`.
- **Consequence for stability:** nothing is held fixed across leave-one-site-out folds any more; `stability.refit_per_fold` / `held_fixed` are printed on the verdict card.

### D31 · 2026-10-03 · Acquisition flags are data-derived and written into the site tables before any statistic runs
- **Decision:** `report.derive_flags` delegates to `acquisition.derive_flags` (grey_pore = BSE p1 > 10 OR known reference list; bright_low_contrast from features; cracked = known reference list only). `report.apply_derived_flags` writes grey_pore / bright_low_contrast / raised_black_level / acquisition_group back into `sites_ref` and `sites_batch` before `compare_kpis`, Check B and the quality abstention run, and `ordinary_ref_sites` is now the set of reference sites with `acquisition_group == "ordinary"`. `stats.usable_n` ORs a boolean `grey_pore` column with the list for the fallback count, as it already did for `bright_low_contrast`.
- **Why:** E21 showed five new site ids with BSE p1 = 24 receiving `raised_black_level` on all five but `grey_pore` on none in the report path (known-ID lookup), while `acquisition.derive_flags` flagged all five; and even a merged boolean column gave zero fallback sites in `usable_n`. On an unseen batch that would have let grey-pore sites pass as ordinary for the pore KPIs and could have hidden a quality abstention (more than half the sites raised black level). After the change the probe gives 5 / 5 / 5, fallback 5, abstention True; on Batches 1–3 the derived groups are identical to the lists (test `test_real_cache_flags_match_known_groups`), so no known-batch number moves.
- **Alternatives rejected:** keeping the constant lists plus a reminder to extend them for the unseen batch (nobody will know the new ids); deriving `cracked` from the data (that is the material signal Check B measures — it stays a reference-only label).
- **Caught by:** E21. Lesson: a "provisional, to be replaced" helper that still returns plausible output on the known data is the most dangerous kind; C27.

### D32 · 2026-10-03 · Three image-review states; MDC infeasibility is reported, not raised
- **Decision:** `check_b` reads `image_reviewed[(site, kpi)]` as confirmed (True → credible), refuted (False → closed: listed under `refuted`, never pending or credible) or unreviewed (key absent / dict None → pending); the report's local-anomaly table shows "refuted by image review (closed)" and the *consistent* reason counts refuted flags. `stats.mdc` returns `feasible=False`, `mdc_mad = NaN`, `n_sim_used = 0` and a `reason` whenever the split design cannot leave ≥ 3 reference sites (n_incoming > n_ref − 3, which includes equal and larger batches) or the reference MAD is zero; the report prints "not available" in that cell, the notebook lists the affected KPIs and the comparison still runs.
- **Why:** E21: an explicit `False` review left the same one pending site as no review at all, so the human gate could open an investigation but never close it; `mdc(17 ref, 18 incoming)` raised `ValueError` and `mdc(7, 7)` returned ∞ from zero simulations, which a reader could mistake for "infinitely large change needed". The provider says the unseen batch should have fewer images than Batch 3, but the code must not depend on that.
- **Alternatives rejected:** a capped fallback design (draw n_ref // 2 and label the MDC an upper bound) — defensible but adds a second design to explain on the first screen; can be added later as a labelled option. Treating a refuted flag as if it never existed — it stays in the table so the review is auditable.
- **Caught by:** E21 (C27).

---

### D33 · 2026-10-03 · Morphology expansion stays an exploratory measurement screen
- **Decision:** E18 adds an independent morphology report in `analysis/morphology`, covering the existing material panel and a pre-specified 16-descriptor morphology panel. It reports all pairwise batch comparisons, quality-matched and ordinary-reference sensitivity views, site-bootstrap intervals, site-label permutation results, acquisition checks and ±5-level mask perturbations. No descriptor is promoted into the five primary QC KPIs and no decision threshold changes.
- **Measurement choices:** orientations are axial and relative to image horizontal; only non-edge-clipped components with aspect ratio ≥1.5 enter orientation summaries. Nearest-centroid spacing is compared against simulated uniform points with the same count/window and the same boundary censoring; it is labelled a geometric comparator, not a validated Clark–Evans agglomeration index. Image-depth gradient magnitude enters the panel because collector orientation is unconfirmed. Descriptor families get equal total weight in the exploratory multivariate comparison.
- **Why:** the provider asked for complex morphology, but the known dataset has few sites and acquisition heterogeneity. Selecting a descriptor or size floor because it separates these folders would reuse the development data as validation. The image overlays also show that tiny thresholded bright fragments near rims can dominate centroid counts; a post-screen 50/500/2,000-pixel area-floor audit is reported without choosing a preferred floor or adding significance tests.
- **Alternatives deferred:** ETD ridge angles as plate orientations (the existing ridge descriptor is a preparation flag); binder-network fraction without a validated fourth class; morphology-driven release decisions without unseen-batch evidence and material tolerances. Revisit after expert measurement review and a separate validation set. New assumption A16 records the geometry interpretation limits.

### D33 · 2026-10-03 · Tiered localized rule: route single sites to review, flip the verdict on 3 MAD or agreement
- **Decision (agreed with Santosh):** a single pending site (promotable KPI, severity ≥ 2.0 MAD, reliable, unreviewed) between 2.0 and 3.0 MAD beyond the ordinary-reference maximum is **routed to image review** with its crop and its own outcome value (`localized = review_routed`), and the batch verdict stays *consistent*. The verdict flips to *investigate — localized* when one site reaches **3.0 MAD** (`Thresholds.single_site_escalate_mad`), when **two sites** or **both promotable KPIs** are pending at ≥ 2.0 MAD (`agreement_min_sites`, `agreement_min_kpis`), or when a reviewer **confirms** a routed crop; a refuted crop closes it (D32). Implemented as `decision.escalation`; the rule text is printed on the verdict card.
- **Why (E19 numbers, clean 7-site batch vs the 10 ordinary reference sites, lognormal fitted to the ordinary values, crack_frac + pore_max_d jointly with Spearman ρ 0.67):** verdict-flip rate single-site@2.0 (D29 rule) **0.18**; tiered(3.0) **0.12** flips + **0.07** routed-only; tiered(2.5) 0.14 + 0.04; two-site or two-KPI agreement alone 0.015–0.03. Re-drawing the reference changes nothing (design B: 0.18 / 0.11 / 0.14). On real 5-v-5 splits of the ordinary reference (five-site MADs, inflated): 0.50 / 0.33 / 0.45. The flip rate stays near 11 % even at 3 MAD because the crack_frac tail is long (σ 0.46; the real ordinary maximum sits at the 92nd percentile of the fit) — the agreement rules add little on clean batches but are what escalates genuine multi-site damage without waiting for a review.
- **Sensitivity (what each rule does with the real anomalies):** cracked reference sites 0grcilhi 5.8 MAD and hzumfsms 5.4 MAD flip under every rule; ufdvpb81 at 2.6 MAD flips under 2.5 but is *routed* under 3.0 and flips only on confirmation — accepted, because routing is the cheap step and the crop is exactly what a reviewer looks at. Synthetic crack_frac ×2 on Batch 2 (3.4 MAD) flips; ×1.5 (0.6 MAD) is missed by every rule, as the MDC predicts.
- **Why 3.0 and not 2.5:** the difference on clean batches is 2–3 points of flip rate, and the only real case between the two margins (2.6 MAD) still reaches a reviewer. The cost asymmetry decides it: a routed crop costs one person a minute; a false batch-level *investigate* costs credibility on every clean batch; a missed single-site crack is why the path exists, and the tier never drops the crop.
- **Alternatives rejected:** keep single-site@2.0 as the verdict trigger (18 % false flips); agreement only (would have missed single-site cracks at 5+ MAD until a second site appeared); a per-KPI α-style threshold on the fitted tail (depends on a parametric fit of ten points — shown as context, not used as the rule).
- **Provenance:** calibrated on null simulations, reference splits and the known reference anomalies only, before the unseen batch; thresholds hash changes; frozen with the notebook (FROZEN_HASH) after the rehearsal (E23).

### D34 · 2026-10-03 · Unseen-batch drop procedure rehearsed and frozen
- **Decision:** the drop runs exactly as README "Unseen batch — the drop procedure": one command per batch, no edits to site lists or thresholds; `FROZEN_HASH = 99d2bbcae6f3` set in the notebook so the §0 assert guards any later change. The hash covers reference, α, power, statistic, KPI lists, flag handling and the thresholds hash (b4f4da2e357c) but deliberately **excludes `CONFIG["compare"]`** — the first freeze included it, which would have made adding the unseen folder trip the assert; caught during the rehearsal and verified both ways (adding a folder keeps the hash, changing α changes it). Human image reviews enter through `--review SITE:KPI=yes|no` (CLI) or `config["image_reviewed"]`.
- **Why:** E23 — a hard-linked copy of Batch 1 under new site ids (`Dataset/Batch_X`) ran end to end in 123 s cold with the two low-contrast sites flagged from the data, the same verdict, attenuation, MDC and stability as Batch 1, and five integration notes, all expected. One difference surfaced: the material-only classifier gave AUC 0.41 / p 0.66 instead of 0.51 / 0.46 on the identical images, because the grouped CV folds and the site-label permutation depend on site-id order; both runs say "not corroborating", but the p-value carries that much Monte Carlo / fold noise at 200 permutations. Recorded as a limitation in the ML block and Part C.
- **Caught by:** C27 (exercise the unseen-batch boundaries). The stale note "Check A(iv) ran with c2st=None" from the cache reader was also found and corrected in the same run.

---

### D35 · 2026-10-03 · Maintain a live morphology inventory; validate candidate methods against independent human annotations
- **Decision:** E24 adds a canonical JSON metric register with stable IDs, exact keys, units, definitions, limitations, artifact provenance, history and separate implementation/evidence/expert-review/QC-role fields. Markdown and a self-contained visual atlas are generated from it. The experiment adds a stratified known-site annotation pack, local void-width estimates, a single histogram-anchored hysteresis candidate and connectivity sensitivity. Only the existing five primary KPIs carry verdicts; no automated status update promotes a candidate.
- **Human-reference contract:** masks shown by algorithms are predictions. Manual polygons are stored separately; unreviewed crops are never scored and unlabelled/uncertain pixels are ignored. Fully labelled masks are required for connected-object and local-width error; reviewed unmeasurable images count as abstentions. Six development and five held-out sites are kept separate, but all known batches were explored previously and specimen independence is unknown.
- **Measurement choices:** deterministic medial axis, 2×Euclidean distance-to-solid, ≥30 px² non-edge-clipped voids, nominal/±5 thresholds; approximate centreline-length sampling, no pruning and no 3-D pore-throat interpretation. Bright hysteresis seeds at site threshold +5 grow inside −5 with 4-connectivity then existing opening. Parameters use the established threshold perturbation scale, not folder separation. Preview compression preserves lossless annotation references and exact masks/arrays.
- **Why:** the E18 spacing audit showed sensitivity to tiny rim fragments; overlay review and independent measurement checks matter before adding more descriptors. Local width may describe narrow/broad voids but excludes edge-connected cavities. Hysteresis does not restore absent contrast and currently retains rim fragments in low-contrast examples. Algorithm agreement cannot substitute for accuracy.
- **Alternatives deferred:** immediate watershed splitting, new verdict drivers, training a segmentation network on pseudo-labels, choosing size floors/thresholds by known-batch separation, or pretending unreviewed predicted masks are ground truth. Revisit after expert pixel/instance review and independent batch evidence. A16 retains the interpretation limits; C28 makes the reference-label gate explicit.
- **Book sources:** distance/local-width and reconstruction/hysteresis chapters of https://bioimagebook.github.io/; SEM conventions and validation gates are adaptations. IDs E24/D35 avoid IDs concurrently used by the QC work. Existing duplicate D33 entries are retained as historical entries and must be identified by title when cited.

### D36 · 2026-10-03 · Use confirmed fresh graphite–Si/SiOx metadata to prioritise battery hypotheses without promoting unvalidated metrics
- **Human evidence:** Santosh directly replied “Fresh graphite–Si/SiOx electrode confirmed”. Fresh/uncycled state and material family are confirmed; exact chemistry, recipe/binder, pixelwise phase labels, collector direction and physical specimen IDs are not supplied. A17 records this metadata; A2/A3 and the persistent review logs distinguish it from mask validation.
- **Domain review:** `docs/battery_microstructure_review.md` and `analysis/battery/sources.json` link inspected real image examples to verified primary studies, retaining full-text versus abstract/indexed retrieval limits. B01–B06 cover additive neighbourhood, long-void/interface context, graphite plate orientation, local additive homogeneity, void/depth structure and fresh-particle fracture appearance. The live metric inventory/atlas links existing entries to these checks; proposed companion measurements remain hypothesis-only and excluded from verdicts.
- **Priority rationale:** review local void/residual-solid relationships and long-void/confirmed-collector context first. Residual-solid adjacency is not electrical contact; tight enclosure and sparse contact can pose different application questions. Pore fraction/equivalent diameter cannot rank transport, and the existing bright/void alignment does not measure graphite alignment.
- **Wording corrections:** low ETD ridge coverage is not intact-particle share; larger section sizes alone do not establish slower lithiation or greater total expansion/binder load. Fresh sections do not demonstrate cycling-induced fracture, electrochemical SEI or lithium plating. Plan/finding/assumption wording is revised; the memo lists production-physics wording still pending an implementation correction. Representative sampling, rather than plate alignment alone, governs area-to-volume interpretation; no stereological output is promoted.
- **Alternatives deferred:** fitting a battery-performance predictor to these few unlabelled sites; adopting published recipe/contact/size cutoffs as product tolerances; inferring binder or 3-D connectivity from greyscale adjacency; adding literature hypotheses as verdict drivers. Revisit after independent measurement review, specimen/process metadata and matched adhesion, wetting, formation or cell evidence. The five primary KPIs and frozen decision configuration are unchanged.
- **Validation:** inventory contracts and regenerated assumption/atlas coverage checked, including unchanged measurement values/statuses. This is a qualitative application review, not a new image-extraction experiment or unseen-batch accuracy result. Applied C04/C05/C06/C11/C13/C16/C18/C21/C28.

### D37 · 2026-10-03 · Add battery geometry as experimental secondary measurements; retain frozen decisions and honest observability
- **Decision:** integrate8 geometry scalars from E25 through `features.extract_site` and a separate `battery_secondary` report object. Original5 primary KPIs, original material classifier list, multiplicity/statistical family, decision logic and threshold hash remain frozen. Exact bright/void boundary adjacency is extracted for observability but constant0 on usable known sites and excluded from comparisons. Width-depth, graphite image tensors and normalised Inlens gradients remain standalone diagnostics. Development used known batches after the primary freeze, not independent unseen evidence.
- **Why / evidence:** real neighbourhoods and internal/clipped void context add explanations a materials expert can inspect, but small-sample intervals, threshold envelopes, loading/size redundancy and acquisition associations prevent confident mechanism or batch-separation claims. The residual-solid connected mask is not a graphite-instance segmentation. Normalisation changes a sampled gradient measurand and does not repair existing per-particle texture.
- **Reliability contract:** known false data-derived low-contrast/grey-pore flags are required for dependent quantities; missing flags/values abstain. All nominal pore-mode flags are unresolved, counted separately rather than excluding every site or inferring ordinary-mask accuracy. Zero internal area is a measurement; absent centroid stays unavailable. Site-bootstrap intervals need at least2 usable sites per side. Counts of windows/components and medial-axis pixels never increase sample n. Paired threshold shifts are measurement sensitivity, not confidence or a complete independent grid.
- **Observability:** exclude edge-clipped objects for new internal location/width summaries while retaining original crack fraction and clipped-area shares. Mask rings include threshold holes and are not a physically validated outer boundary or expansion volume. Image rows do not locate a collector; gap fields remain unavailable without independent collector geometry. Independent annotations/repeat specimens remain the gate for material use.
- **Production wording fixes:** pore/void geometry cannot rank transport; observed section-size ratios do not determine lithiation time; particle coarsening alone does not determine total expansion load; ridge-pixel coverage is not an intact-particle share or electrical continuity. Representative spatial sampling/phase labels gate conditional stereology; plate alignment alone does not invalidate phase-area estimation. Numeric consequence weights remain unchanged.
- **Verification/provenance:** full140tests plus final55focused checks; default known-batch reports replayed with unchanged original columns, unchanged outcomes/stability/hash. Cache version bumped1.1.0. Explicit verified-primary-cache reuse plus raw-BSE secondary extraction, not full cold all-channel validation. Measured-source snapshots and rendering recoveries remain separate; failed guarded extraction is not published as success. See E25, A18 and the combined illustrated report.
- **Alternatives deferred:** promoting descriptors by known-folder separation, reusing windows as independent n, claiming detector/algorithm agreement as expert truth, graphite axes from connected residual masks, tuning scales after seeing differences, or claiming battery performance from section geometry. Revisit with reviewed labels, specimen/process metadata and independent batches.

## Part B — Pre-presentation review checklist

Run this before presenting a plan, a result, a figure or a verdict. Each item names the failure it exists to prevent and the decision where it was learned. Add an item whenever a reviewer catches something not covered here.

**Data and measurement**
- [ ] **C01 — Look at the table before believing it.** Any constant column, any value 3× its neighbours, any std larger than its mean is a broken feature until proven otherwise. (D05, D06)
- [ ] **C02 — Does the statistic vary by construction or by data?** Per-image percentiles, per-batch normalisation and rank transforms can fix the answer before the data speak. (D10)
- [ ] **C03 — If a feature looks too good, test it for a confound before reporting it.** Correlate it with every acquisition variable and session fingerprint; fit an acquisition-only model; check within-group. (D11)
- [ ] **C04 — Does the code keep every promise the prose makes?** Each channel, feature or capability named in markdown must have a corresponding extracted quantity. (D12)
- [ ] **C05 — Would a domain expert accept the wording?** "Vacuum" vs "open porosity"; "crack" vs "large elongated void"; "Si" vs "bright higher-Z phase". Say what was observed, then what it is assumed to be. (D13)
- [ ] **C12 — Have all metadata and tags actually been read, not assumed absent?** Verify claims like "no metadata survived" by dumping every tag. (D03)
- [ ] **C13 — Is every physics statement scoped to what a 2-D section at nominal scale can support?** No performance percentages; no 2-D-bounds-3-D claims; geometry is not continuity; "consistent with" is not "proves". (D20)

**Statistics**
- [ ] **C06 — Count independent units, then decide how many models the data can support.** More models on the same few units mostly agree on confounders. (D15)
- [ ] **C07 — Does a null result get read as acceptance?** "Not detected" needs the minimum detectable change written next to it; acceptance needs tolerances and an equivalence test. (D16)
- [ ] **C08 — Can one severe local event slip past a batch-average rule?** Keep a separate localized path. (D17)
- [ ] **C09 — Does the null match the comparison?** Same statistic, same sample sizes, same quality flags, every reference-dependent fit repeated inside the resampling loop. (D18, D21)
- [ ] **C10 — Compute the false-alarm rate of any threshold rule under an i.i.d. null before adopting it.** "Exceeds the max of 10" fires 41 % of the time for 7 sites on one KPI. Separate alerts from abstentions. (D21)
- [ ] **C14 — Is n the usable n?** After quality flags, not the folder count; state whether units share specimens. (D21)
- [ ] **C15 — Is "exact" actually exact, and affordable?** Enumerate only simple statistics; label Monte Carlo as Monte Carlo with its count. (D21)
- [ ] **C19 — Simulate the null of every simulation design before trusting its power.** Run it at zero shift and check the rejection rate equals α; resampling schemes that treat the two arms asymmetrically (with-replacement draws from one side) break exchangeability. (D23)
- [ ] **C20 — Check the discreteness of an exact test statistic at the actual n.** Count attainable values and P(p = 1) under the null; prefer a near-continuous robust statistic (Hodges–Lehmann) when the median difference is too coarse. (D23)
- [ ] **C21 — Audit every "trusted / material KPI" list against the catalogue's trust column.** A flag that leaks into a material list will dominate a classifier and look like a discovery. (D24)
- [ ] **C22 — A derived index that is undefined or constant on all real sites is not reported, even with caveats.** State the underlying fact once. (D25)
- [ ] **C23 — Covariate adjustment on a heterogeneous reference can manufacture signal.** Check covariate–KPI correlations on the ordinary subset vs the full reference; show at least two covariate sets; read negative attenuation as "not explained away", never as material evidence. (D26)
- [ ] **C25 — Run the decision logic end to end on real data before trusting it.** Unit tests on synthetic components passed while `None` counted as corroboration and a batch-mean KPI triggered the localized path; only the integrated run showed it. Also check every "missing input" default (None, NaN) against the conservative direction — the list so far: classifier `c2st` (D28), acquisition attenuation `att` (D30), image review (D32), MDC feasibility (D32). (D28, D30–D32)
- [ ] **C27 — Exercise the unseen-batch boundaries, not just the known folders.** Use new site IDs with data-derived quality flags, smaller/equal/larger usable batch counts than the reference, and explicit confirmed/refuted/unreviewed human reviews. Verify that infeasible simulations are labelled unavailable and that the report still renders. Passing tests on the known cardinalities do not cover these contracts. (E21 implementation review)
- [ ] **C26 — Every alert rule gets its false-alarm rate measured two ways before it ships:** on an i.i.d. null and on splits of the homogeneous part of the real reference. Count KPIs: k rules at rate r give ≈ k·r alarms. (D29)
- [ ] **C24 — Prose written from a figure must be re-checked against a thresholded number.** "Rises" / "drops" claims need a stated |z| or effect cut-off that the data actually cross. (D27)

- [ ] **C28 — Are reference labels independent and explicitly reviewed?** Algorithm masks and agreement between algorithms are not ground truth. Never score unreviewed crops or uncertain pixels; require full masks for object/width error, retain unmeasurable abstentions, and keep development vs held-out-site annotations separate. Verify raw/trimmed coordinates and identify edge-excluded objects on overlays. (D35, E24)

**Language and provenance**
- [ ] **C11 — Is any causal word justified?** "Share explained by the instrument" is a regression adjustment, not attribution. Show unadjusted, stratified and adjusted. (D19)
- [ ] **C16 — Does the provenance claim match what actually happened?** We looked at the data before fixing choices; say so. A hash proves stability after a freeze, not independence before it. (D21)
- [ ] **C17 — Is "confidence" being used for something that is not a probability of being right?** Use "decision stability", "interval", "percentile of the null". (D18)
- [ ] **C18 — Are the organisers' answers and reviewers' corrections written into the register, the findings doc and this log in the same change?** (process)

---

- [ ] **C29 — Does a normalisation or new geometry descriptor preserve the intended measurand and disclose observability?** A patch gradient is not per-particle texture; image-tensor directions are not plate axes; a mask ring may include internal holes; clipped area and weak directional strength need explicit coverage. Never rename an acquisition diagnostic into a material KPI after reducing one pooled correlation. (D37/E25)


## Part C — Open items that a future decision must close
- D36: apply the source-backed wording corrections in `docs/battery_microstructure_review.md` to production `polaron_qc/physics.py` and regenerate/review judge-facing reports. — **resolved D37/E25**; notebook/report consistency remains pending.
- D36: independently review B01 additive neighbourhood and B02 long-void/confirmed-collector context; establish chemical, orientation and preparation metadata before application claims.
- Practical tolerances for the primary KPIs (needed for any "accept").
- Whether sites within a batch come from distinct specimens.
- Whether the binder / carbon-black network becomes its own segmentation class (affects pore fraction and the porosity sanity check).
- 500 px crack-like-void cutoff: confirm or replace with a physics-motivated length.
- Inlens texture: local-contrast normalisation to test whether the confound can be removed.
- Owner for `KPI_TRUST` (trust level per KPI in code); proposed `polaron_qc/__init__.py`.
- `polaron_qc.acquisition` (stratified / adjusted views) has no owner yet.
- Deduplicate edge-band trimming (features, ml, physics) by importing from features.
- E23: `ml.c2st` fold assignment and permutation stream depend on site-id order (AUC 0.51 → 0.41 on identical images under relabeling, n_perm 200). Options: sort rows by a content key before CV, or raise n_perm and report the band. Decision-neutral on the known batches.
- E19: a labelled parametric tail estimate (lognormal, ten ordinary sites) is shown as context for the review load; it is not a rule input.
- E21: missing acquisition views must not support a drift reject or a claim that adjustment was checked; wire the available acquisition module into the report and decision. — **resolved D30 (E22)**
- E21: derive unseen grey-pore flags from image measurements, propagate them into the decision/statistics inputs, and honour the boolean column when counting fallback measurements. — **resolved D31 (E22)**
- E21: distinguish an explicitly refuted image anomaly from an unreviewed one; a negative review must be able to close a pending investigation. — **resolved D32 (E22)**
- E21: make MDC feasibility explicit for incoming counts at or above the reference count; keep the rest of the report usable when MDC cannot be estimated by the chosen design. — **resolved D32 (E22)**; a labelled capped-draw upper-bound design remains an option if the unseen batch is as large as the reference
- E21: refit classifier evidence in leave-one-site-out re-verdicts, or describe stability as conditional on the fixed full-sample classifier. — **resolved D30 (E22)**: classifier and acquisition views are refit per fold; `stability.refit_per_fold` / `held_fixed` are printed
