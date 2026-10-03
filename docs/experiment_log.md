# Experiment log and progress tracker

Two parts. **Part A** is a ledger: one entry per experiment, with the question, the setup, the numbers, what we concluded, and what it changed. Numbers live in `experiments/registry.csv` (one row per metric) so later notebooks can append and plot progress. **Part B** is the scoreboard: where each judged capability stands against the target, updated whenever an entry in Part A moves it.

Conventions: ids `E01…`; every entry states reference, batches, n (usable), statistic, seed where relevant; outcomes are numbers first, interpretation second; `Changed` names the decision-log entry or file it altered. Entries are never deleted; a superseded result gets a `Superseded by` line.

---

## Part A — Experiments

### E01 · 2026-10-03 · BSE phase thresholds: multi-Otsu vs graphite-mode valley
- **Question:** does a per-image 3-class multi-Otsu segment pore / graphite / bright phase reliably across contrast-stretched images?
- **Setup:** 31 sites, BSE, notebook 01.
- **Result:** multi-Otsu placed the upper threshold inside the graphite peak on low-contrast images → bright fraction 0.19–0.20 on two Batch 1 sites vs 0.05–0.09 elsewhere (3× inflation). Valley thresholds anchored on the graphite mode: bright fraction 0.04–0.08 on 29/31 sites; the two low-contrast sites remain inflated (0.12, 0.17) and are flagged.
- **Conclusion / changed:** valley thresholds + `bright_low_contrast` flag (D05, D07).

### E02 · 2026-10-03 · Texture-scale KPI
- **Question:** is the FFT dominant-period KPI informative?
- **Result:** constant 170.67 px on all 31 sites (argmax at the lowest frequency). Replaced by autocorrelation 1/e length: 14–28 px, batch medians 18 / 19 / 22 px (B1 / B2 / B3 ordinary).
- **Changed:** D06.

### E03 · 2026-10-03 · Inlens intra-particle texture: material or acquisition?
- **Question:** the strongest batch-ordered feature (η² 0.53; B1 0.32 > B2 0.24 > B3 0.18 > grey 0.13) — is it confounded?
- **Setup:** Spearman vs Inlens acquisition statistics; OLS on Inlens median, std, stretch fraction; ANOVA on residuals.
- **Result:** ρ = +0.77 with Inlens median, +0.68 with frame height, +0.95 with Inlens texture energy; acquisition-only R² = 0.74; raw group F = 10.3 (p 0.0001) → residual F = 3.7 (p 0.023); within-batch ρ with Inlens median 0.7 in B2 and B3.
- **Conclusion / changed:** confounded; candidate only (D11).

### E04 · 2026-10-03 · ETD intra-particle crack density
- **Result:** 0.02–0.5 % of bright-particle interior area in every group after curtaining removal; particles intact in all batches.
- **Conclusion:** null KPI here; retained for the unseen batch (register A10).

### E05 · 2026-10-03 · Crack-like voids (major axis > 500 px)
- **Result:** three Batch 3 sites (hzumfsms, 0grcilhi, ufdvpb81) at 0.048–0.062 void fraction vs medians 0.015–0.022; largest void ⌀ 450–600 px vs ≈ 250–300; delamination index 1.17–1.86 vs ordinary median 0.71; `columns_interrupted_frac` 0.79–0.95 but overlaps four ordinary sites (0.73–0.80).
- **Conclusion:** the only clear material anomaly; `delamination_index` discriminates better than `columns_interrupted_frac`; 500 px cutoff still a judgement call (Part C).

### E06 · 2026-10-03 · Classifier two-sample test (grouped CV, site-level permutation, n_perm 200)
| comparison | KPI set | AUC | null 95 % | p |
|---|---|---|---|---|
| B1 (7) vs B3 (17) | 12 "trusted" incl. etd_boundary_sharpness | 0.790 | 0.19–0.78 | 0.030 |
| B1 vs B3 | material only | 0.529 | 0.14–0.77 | 0.398 |
| B1 w/o low-contrast (5) vs B3 | material only | 0.506 | 0.11–0.87 | 0.408 |
| B2 (7) vs B3 | material only | 0.639 | 0.15–0.85 | 0.239 |
- **Conclusion / changed:** the whole B1–B3 separation is `etd_boundary_sharpness` (a prep flag); material KPIs do not separate either batch (D24, C21).

### E07 · 2026-10-03 · DINOv2 patch-embedding novelty (518 px, PCA 32, kNN 5, LOO per site)
- **Result:** B1 sites at 65–100th percentile of the reference LOO distribution (three at 100: f1vzngrs and the two low-contrast sites); B2 12–82, none exceed. Top correlates all acquisition: bse_std +0.44, etd_curtain_frac −0.44, low-contrast flag +0.43, bse_p1 −0.40; best KPI bright_d50 +0.32 (p 0.08). Ranking at 224 px correlates ρ 0.43 with 518 px. Known Batch 3 sub-populations not novel.
- **Conclusion / changed:** exploratory evidence flag only (D25).

### E08 · 2026-10-03 · MDC simulation design
- **Question:** does the planned with-replacement pseudo-batch draw hold its false-alarm level?
- **Result:** at zero shift, 17 normal sites, 400 sims, α 0.05: with replacement **11.0 %** rejections; without replacement 5.5 %.
- **Changed:** draw without replacement (D23, C19).

### E09 · 2026-10-03 · Permutation statistic at 7 v 17 (exact enumeration, 300 null seeds)
| statistic | KS p vs uniform | P(p ≤ .05) | note |
|---|---|---|---|
| mean_diff | 0.33 | 0.053 | |
| hl_shift | 0.28 | 0.050 | |
| cliffs_delta | 0.003 | 0.043 | ties |
| median_diff | 4e-12 | 0.043 | 56 attainable values; P(p = 1) = 0.21 |
- **MDC (α .05, power .8, 7 v remaining 10, HL / median_diff):** crack_frac 1.75 / 2.50; pore_max_d 1.75 / 2.00; pore_frac 2.00 / 2.25; bright_frac 1.50 / 1.50; bright_d50 2.50 / 3.75 reference MADs.
- **Changed:** HL is the verdict statistic (D23, C20).

### E10 · 2026-10-03 · Primary-KPI comparison, reference Batch 3 (17), HL statistic, exact permutation
| batch | KPI | n_batch usable | shift (MAD) | 95 % CI (approx) | p_perm | p_holm |
|---|---|---|---|---|---|---|
| B1 | crack_frac | 7 | +0.12 | −2.09, 2.18 | 0.91 | ≥ 0.36 |
| B1 | pore_max_d | 7 | +0.04 | −1.32, 1.01 | 0.26 | |
| B1 | pore_frac | 7 | −0.05 | −1.86, 1.94 | 0.59 | |
| B1 | bright_frac | **5** | +0.80 | −1.40, 2.32 | 0.40 | |
| B1 | bright_d50 | **5** | +1.45 | −0.11, 7.84 | 0.072 | 0.36 |
| B2 | crack_frac | 7 | −0.40 | −2.78, 0.51 | 0.22 | 0.65 |
| B2 | pore_max_d | 7 | −0.24 | −1.16, 0.41 | 0.11 | 0.55 |
| B2 | pore_frac | 7 | −0.96 | −1.69, 0.10 | 0.14 | 0.57 |
| B2 | bright_frac | 7 | +0.22 | −2.15, 1.33 | 0.97 | 0.97 |
| B2 | bright_d50 | 7 | +1.34 | −0.51, 5.69 | 0.31 | 0.65 |
- Energy distance on the 5 primary KPIs: B1 E 0.98 p 0.40 (5 sites after NaN-ing bright KPIs); B2 E 0.89 p 0.34.
- **Conclusion:** no primary KPI differs from the reference after Holm for either batch; the largest raw signal is B2 pore_frac (−0.96 MAD). All shifts are below the MDCs in E09 — "not detectable", not "no change".

### E11 · 2026-10-03 · Local exceedance vs ordinary-reference maximum (10 sites)
- **Result:** cracked reference sites exceed on crack_frac by 5.8 / 5.4 / 2.6 MAD (credible); hzumfsms also on pore_max_d (2.7). Batch 1: one crack_frac exceedance (4ih2ggld, 0.27 MAD, low-contrast site → flag only). Batch 2: none. Reference-split diagnostic (crack_frac, 100 splits 7 v 10): 4 % alerts.
- **Conclusion:** the localized path recovers the known cracked sites; the i.i.d. flag rate (7/17 ≈ 41 %) is printed with every flag.

### E12 · 2026-10-03 · ±5-gray-level threshold sensitivity (6 sites)
- **Result:** pore_frac band 28–50 % of nominal (grey-pore site 50 %); bright_frac band 10–17 %, 48 % on the low-contrast site.
- **Conclusion / changed:** the band is as large as the between-batch pore_frac differences; computed per site in the features stage and printed next to pore-based shifts; physics wording gated on it (D25).

### E13 · 2026-10-03 · 2-D macro-pore connectivity
- **Result:** `fraction_connected` = 0 on all 31 sites (0 of 50 seeds reach bottom through pores); tortuosity index undefined everywhere.
- **Changed:** index dropped; single dataset-level statement (D25, C22).

### E14 · 2026-10-03 · Reference characterisation prototype (Batch 3, primary KPIs, HL statistic, MDC without replacement)
| KPI | median (17) | MAD (17) | median (ordinary 10) | MAD (ordinary 10) | MDC n=7 (MAD) | MDC n=7 (abs) | MDC n=5 (MAD) |
|---|---|---|---|---|---|---|---|
| crack_frac | 0.0201 | **0.0139** | 0.0201 | **0.0046** | 1.75 | 0.024 | 2.00 |
| pore_max_d | 278 px | 105 | 308 | 66 | 1.75 | 183 px | 2.00 |
| pore_frac | 0.093 | 0.018 | 0.094 | 0.014 | 2.00 | 0.037 | 2.00 |
| bright_frac | 0.053 | 0.0086 | 0.056 | 0.0068 | 1.50 | 0.013 | 1.75 |
| bright_d50 | 146 px | 13.5 | 149 | 31.7 | 2.50 | 34 px | 3.00 |
- **Reading:** the three cracked sites triple the reference MAD of crack_frac (0.0046 → 0.0139), so the absolute MDC on crack fraction (0.024) exceeds the reference median itself; with the full reference as null, only a batch whose crack fraction more than doubles would be detectable as batch-wide drift. This is the "reference heterogeneity swallows the null" risk made concrete, and the reason the localized path (E11) must stand on its own. bright_d50 shows the opposite: the ordinary subset is *wider* (two coarse ordinary sites), so excluding anomalies does not always tighten the null.
- **Runtime:** ≈ 14 s per KPI for two MDC runs (n_sim 400, n_mc 999).
- **Changed:** notebook 02 §2.3 will show both nulls side by side and print absolute MDC next to the reference median.

### E15 · 2026-10-03 · Sensitivity to acquisition adjustment (acquisition module, Batch 3 reference, primary KPIs, HL, n_mc 5000)
| batch | unadjusted E (p) | stratified E (p) n | adjusted 3-cov OLS E (p) | adjusted 7-cov ridge E (p) | attenuation strat / adj |
|---|---|---|---|---|---|
| B1 | 0.981 (0.404), 17 v 5 | 0.704 (0.830), 10 v 5 | 2.343 (0.078) | 0.68 (0.97) | +0.28 / −1.39 |
| B2 | 0.886 (0.343), 17 v 7 | 1.533 (0.156), 10 v 7 | 3.555 (0.015) | 2.12 (0.27) | −0.73 / −3.01 |
- Reference R² of the 3 covariates (17 sites; adj. R²): crack_frac 0.60 (0.51), pore_max_d 0.50 (0.38), pore_frac 0.75 (0.69), bright_frac 0.06 (−0.16), bright_d50 0.17 (−0.02).
- Fixed-residual vs in-loop-refit p for pore_frac: 0.002 vs 0.021 (B1), 0.001 vs 0.014 (B2) — fixed residuals are anti-conservative.
- Null of the adjusted design on 60 random 7-of-17 reference splits: mean p 0.53, share ≤ 0.05 = 0.05 — level holds.
- Threshold band, all 31 sites (median pore_frac / bright_frac relative band): ordinary 0.34 / 0.11, cracked 0.32 / 0.14, grey-pore 0.52 / 0.22, low-contrast 0.31 / 0.40.
- Three-channel agreement at |z| ≥ 2: intensity-only shift on all 4 grey-pore sites and 2 ordinary Batch 2 sites; cracked sites show ETD texture drop only.
- **Conclusion / changed:** adjustment amplifies on this reference (D26, C23); the findings-doc claim about BSE correlation length on cracked sites withdrawn (D27, C24); hazard 4 confirmed on all sites.

### E16 · 2026-10-03 · First end-to-end verdicts (report.build_result; reference Batch 3; HL statistic; thresholds D22; D28 semantics)
| batch | verdict | drift alert (energy p) | localized | quality abstention | stability (LOO) | material c2st AUC (p) | flag-incl. c2st AUC (p) |
|---|---|---|---|---|---|---|---|
| Batch 1 | consistent within detectable limits | no (0.404; 17 v 5) | flag only (4ih2ggld crack_frac +0.27 MAD, low-contrast site) | no | 1.00 (7/7) | 0.513 (0.458) | 0.790 (0.030) |
| Batch 2 | consistent within detectable limits | no (0.343; 17 v 7) | none on LOCAL_KPIS; descriptive: b3esycq1 bright_frac +2.12 MAD (outlying site) | no | 1.00 (7/7) | 0.613 (0.264) | 0.613 (0.244) |
- MDC printed on the first screen (MAD / KPI units): B1 crack_frac 1.75 / 0.024, pore_max_d 1.75 / 183 px, pore_frac 2.00 / 0.037, bright_frac 1.75 / 0.015 (n = 5), bright_d50 3.00 / 41 px (n = 5); B2 identical except bright_frac 1.50 / 0.013, bright_d50 2.50 / 34 px.
- Before the D28 fixes the same run returned "investigate — localized anomaly" for Batch 2 on the bright_frac exceedance, and `c2st=None` would have permitted a reject. Runtime ≈ 50 s per batch warm (MDC 400 sims ≈ 40 s of it).
- **Reading:** on the five primary KPIs neither known batch is distinguishable from the reference at the sensitivity this reference allows; every shift is below its MDC. The report says exactly that, with the MDC next to it. Whether this is "right" cannot be judged without labels or tolerances; what can be judged is that the known cracked reference sites are recovered by the localized path (E11) and that the one Batch 1–3 separation the classifier finds is acquisition (E06).
- **Changed:** D28; report sections → sources table in the workflow diagram; Part B scoreboard.

---

## Part B — Progress toward the target

Target: an interpretable, uncertainty-aware QC system that returns accept / investigate / reject with an explanation a materials expert can verify, and generalises to the unseen batch. Status scale: ☐ not started · ◔ built · ◑ tested · ◕ integrated · ● demoed on known batches.

| Judged capability | Component | Status | Evidence / key number | Gap to target |
|---|---|---|---|---|
| Quality of material KPIs | BSE segmentation + site KPIs | ◑ | E01; bit-identical regression; 29/31 sites clean | binder network unsegmented; 500 px cutoff unvalidated |
| | ETD / Inlens KPIs | ◑ | E04 (null), E03 (confounded) | Inlens local-contrast normalisation untried |
| | per-site threshold band | ◑ | E15: all 31 sites; pore_frac band ≈ 33 % | gate physics wording; print beside pore shifts |
| Accuracy on the new batch | comparison engine | ◕ | E16 end to end; one command per batch (`python -m polaron_qc.report ref batch out`) | notebook 02 sections 3–9 |
| | localized path | ◑ | E11 recovers the 3 cracked sites | image-review step is manual |
| | classifier corroboration | ◕ | E16: material-only run cached and read by Check A(iv) | run inside the pipeline for the unseen batch (currently a cached pre-run) |
| | novelty catch-all | ◑ | E07 tracks acquisition | exploratory; renderer for novelty map |
| Interpretability | decision layer | ● | E16: both known batches; 14 tests; D28 fixes from the end-to-end run | image-review loop is manual |
| | physics reading | ◑ | E05, E13; caveats mandatory | gate on threshold band; wording review by Santosh |
| | per-batch HTML report | ● | reports/qc_Batch_{1,2}.html, 0.4 MB, 10 sections, MDC on first screen | acquisition block to wire; novelty map margins |
| Honest uncertainty | MDC, permutation nulls, stability | ◑ | E08, E09 | reference-split diagnostics on full pipeline |
| | acquisition sensitivity views | ◑ | E15: adjustment amplifies; two covariate sets shown | wire into decision/report |
| Real-world usability | one-command unseen-batch path | ☐ | — | notebook 02 + README demo path |
| | assumption register / decision log | ● | 15 cards; D01–D25; C01–C22 | keep current |

**Known-batch dry run (target: all three verdicts with drivers, stability and reports):** Batch 1 and Batch 2 vs Batch 3 run end to end (E16); Batch 1 vs Batch 2 not yet.

**Unseen batch:** not yet available.
