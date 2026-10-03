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
