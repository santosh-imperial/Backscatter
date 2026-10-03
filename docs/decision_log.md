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
- **Open:** even at 2 MAD on two KPIs the expected review rate on a clean batch is ≈ 8 % Gaussian and higher with heavy tails. Whether that review load is acceptable is a QC-usability question for Santosh / the organisers; the alternative is to require two sites or two KPIs to agree.

---

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
- [ ] **C25 — Run the decision logic end to end on real data before trusting it.** Unit tests on synthetic components passed while `None` counted as corroboration and a batch-mean KPI triggered the localized path; only the integrated run showed it. Also check every "missing input" default (None, NaN) against the conservative direction. (D28)
- [ ] **C26 — Every alert rule gets its false-alarm rate measured two ways before it ships:** on an i.i.d. null and on splits of the homogeneous part of the real reference. Count KPIs: k rules at rate r give ≈ k·r alarms. (D29)
- [ ] **C24 — Prose written from a figure must be re-checked against a thresholded number.** "Rises" / "drops" claims need a stated |z| or effect cut-off that the data actually cross. (D27)

**Language and provenance**
- [ ] **C11 — Is any causal word justified?** "Share explained by the instrument" is a regression adjustment, not attribution. Show unadjusted, stratified and adjusted. (D19)
- [ ] **C16 — Does the provenance claim match what actually happened?** We looked at the data before fixing choices; say so. A hash proves stability after a freeze, not independence before it. (D21)
- [ ] **C17 — Is "confidence" being used for something that is not a probability of being right?** Use "decision stability", "interval", "percentile of the null". (D18)
- [ ] **C18 — Are the organisers' answers and reviewers' corrections written into the register, the findings doc and this log in the same change?** (process)

---

## Part C — Open items that a future decision must close
- Practical tolerances for the primary KPIs (needed for any "accept").
- Whether sites within a batch come from distinct specimens.
- Whether the binder / carbon-black network becomes its own segmentation class (affects pore fraction and the porosity sanity check).
- 500 px crack-like-void cutoff: confirm or replace with a physics-motivated length.
- Inlens texture: local-contrast normalisation to test whether the confound can be removed.
- Owner for `KPI_TRUST` (trust level per KPI in code); proposed `polaron_qc/__init__.py`.
- `polaron_qc.acquisition` (stratified / adjusted views) has no owner yet.
- Deduplicate edge-band trimming (features, ml, physics) by importing from features.
