# Proposed log text for the site categoriser (not applied to the shared logs — paste after review)

Every number below is read from `analysis/categoriser/output/*.csv` and `analysis/categoriser/findings.md`; the
pre-registration is commit 691990e (before the first run). The experiment log's last entry is **E30K**, so this is
proposed as **E31**; the decision-log text is proposed as **D50** (D49P is referenced in the task brief but is not in
the checkout this branch starts from, 87ecee3 — number it after D49P when applying).

---

## docs/experiment_log.md — Part A

### E31 · 2026-10-03 · Site categoriser (3-class, model probabilities), baseline OOD percentiles and hand-off — `polaron_qc.categorise`
- **Question:** the organisers judge by (i) categorising held-back samples among the known batches and (ii) saying whether an unknown batch is inside or outside the promised Batch 3 distribution. Can a per-site categoriser do (i) above a site-label null on the 31 known sites, which feature family carries it, are its outputs usable as probabilities, and what does a robust distance to Batch 3 say about Batch 1 / Batch 2 sites — kept as three separate answers?
- **Setup:** `polaron_qc/categorise.py`, pre-registered in `analysis/categoriser/findings.md` §1 (commit 691990e) before the first run. 31 sites (17 / 7 / 7), one row per site from the feature cache + `report.derive_flags`. Three families as separate models: morph (19 trusted BSE geometry KPIs = MATERIAL_KPIS + pore_d90, pore_count_per_Mpx, bright_d10, bright_solidity, bright_max_d), acq (10 session statistics), combined (39 = morph + 10 ETD/Inlens texture features labelled "acquisition-sensitive; batch-fingerprint evidence, origin not established" + acq); **primary pre-registered = combined**. Leave-one-site-out; inside every fold: median imputer → scaler → L1 logistic (liblinear OvR, balanced class weights), C from {0.1, 0.5, 2} by inner stratified 3-fold class-balanced log-loss; 200 site-label permutations of the whole fold-local procedure, statistic = balanced accuracy, p = (b+1)/(n+1); seed 0; canonical content order (D43). Calibration: multiclass Brier vs class-prior Brier, reliability in fixed bins [1/3, .5), [.5, .7), [.7, 1], ECE. OOD: robust z (Batch 3 median / 1.4826·MAD, IQR fallback), primary score = mean distance to k = 3 nearest reference sites, reference percentiles by leave-one-site-out over the 17 Batch 3 sites (median/MAD and neighbours refitted on 16); variants morph (primary), morph + texture, acq; secondaries RMS-z and Ledoit–Wolf Mahalanobis; matched-count sensitivity. Hand-off: top-3 categoriser contributions (coef × z for the predicted class, sign, value, Batch 3 median ± MAD), top-3 OOD contributions, nearest reference sites with their acquisition group, data-derived flags, pointer to `polaron_qc.report`. 34 s on 12 cores. `--score` exercised on three Batch 1 sites hard-linked under new ids (cold extraction into a scratch cache, 128 s).
- **Results (categoriser, LOO, 31 sites, 200 permutations, seed 0):**

| family | accuracy [CP 95 %] (majority 0.548) | balanced acc. (chance 0.333) | recall B1 / B2 / B3 | confusion (true B1; B2; B3 → pred B1/B2/B3) | perm p | Brier (prior 0.597) | ECE |
|---|---|---|---|---|---|---|---|
| **combined (primary)** | 0.677 [0.49, 0.83] | **0.664** | 0.71 / 0.57 / 0.71 | 5/2/0; 2/4/1; 2/3/12 | **0.005** (0 of 200 ≥ obs) | 0.494 | 0.146 |
| morph (secondary, α 0.017) | 0.387 [0.22, 0.58] | 0.347 | 0.57 / 0.00 / 0.47 | 4/1/2; 5/0/2; 4/5/8 | 0.159 | 0.685 | 0.063 |
| acq (secondary, α 0.017) | 0.548 [0.36, 0.73] | 0.473 | 0.57 / 0.14 / 0.71 | 4/3/0; 2/1/4; 2/3/12 | 0.035 | 0.520 | 0.090 |

  Seeds 1–2 (post hoc, disclosed): combined 0.664 / p 0.010 and 0.005; acq 0.541 / 0.025 and 0.493 / 0.075; morph 0.291 / 0.77 and 0.83. Primary's LOO-stable selections (≥ 97 % of folds): ridge_p97, inlens_particle_texture_p90, bse_empty_bin_frac, pore_count_per_Mpx, pore_max_d, inlens_p50, corr_len_px, bse_p50, etd_boundary_sharpness, bse_std. Morphology's only recurrent selection: pore_count_per_Mpx (71 % of folds; medians 101 / 86 / 98 per Mpx); 9 of 31 morphology folds zeroed every coefficient (uniform 1/3 output). Calibration of the primary: top bin [0.7, 1] mean 0.83 vs observed 0.67 (n 12), middle bin 0.60 vs 0.77 (n 13); three errors at top model probability 0.72–0.85 (b3esycq1 → B1 0.85, pl8uabbv → B2 0.84, ufdvpb81 → B1 0.72); two of the three cracked reference sites are called Batch 1.
- **Results (OOD, reference = all 17 Batch 3 sites, k-NN robust-z, LOO percentiles; floor 1/18 = 0.056):**

| variant | Batch 1 pct min / median / max; n > ref max | Batch 2 pct min / median / max; n > ref max |
|---|---|---|
| **morph (primary)** | 11.8 / 29.4 / 100; **2 of 7** (4ih2ggld, 5n1q8atc — both low-contrast; driver bright_count_per_Mpx z +37 / +39) | 0 / 41.2 / 76.5; 0 |
| morph + texture | 29.4 / 47.1 / 100; 3 (+ f1vzngrs) | 0 / 41.2 / 88.2; 0 |
| acq | 47.1 / 70.6 / 82.4; 0 | 23.5 / 47.1 / 70.6; 0 |
| post hoc: morph without bright-phase KPIs | 52.9 median; 1 (f1vzngrs, pore_count_per_Mpx +6.6 z; the two low-contrast sites fall to 82 / 88) | 64.7 / 82.4 max; 0 |

  The reference LOO maximum is a cracked Batch 3 site (hzumfsms 8.3; then 0grcilhi 7.3); 12 ordinary / grey-pore reference sites lie between 3.7 and 5.3. Under an i.i.d. null ≥ 1 of 7 sites exceeds with probability ≈ 0.33. Matched-count sensitivity moves medians ≤ 18 points and no flag. Univariate context: 14 of 39 features at unadjusted Kruskal p < 0.05 (texture 9/10, acq 4/10, morph 1/19); none of the morphology KPIs survives 0.05/39.
- **Reading:** (1) the pre-registered primary categorises above the null, but it is a **batch fingerprint** built on texture and session statistics; morphology alone is at chance, so nothing here is a microstructural categorisation and the acquisition-only model is itself at the edge (p 0.025–0.075 across seeds). (2) The model probabilities are **not calibrated** (over-confident top bin, Brier 0.49) and stay labelled "model probabilities". (3) Against the Batch 3 morphology, Batch 2 sits inside (median 41st percentile, none beyond the maximum; its recurrent direction is a lower pore count per Mpx inside the reference range) and Batch 1's two exceedances are a low-contrast segmentation artefact (bright_count_per_Mpx), leaving one ordinary site (f1vzngrs) beyond the maximum on pore count — one of seven, which the null produces a third of the time. (4) With 17 reference sites no single site can be called outside at 5 % by rank (floor 1/18); a cracked unseen site would be *inside* this reference because the reference's own maximum is a cracked site — cracks remain the business of the report's localized path.
- **Not changed:** `decision.py`, `report.py` thresholds, the frozen notebook, CONFIG / FROZEN_HASH. New: `polaron_qc/categorise.py`, `tests/test_categorise.py` (12 tests, one real-cache smoke test guarded), `analysis/categoriser/` (findings, outputs, this file). Deviations from the pre-registration are listed in findings §1.7 (post-hoc OOD variant, extra seeds, a NaN-sentinel fix with identical numbers, a `--score-cache-dir` option). Checklist: C01–C04, C06/C09/C14, C07/C10/C17, C16/C21/C22, C27.

### Figures elsewhere that this touches
- None of the committed verdicts, reports or notebook cells change. The findings doc §4 KPI catalogue could gain one line under "ML corroboration": "a per-site 3-class categoriser exists (E31); it is a batch fingerprint, not a material classifier; its outputs are model probabilities".

---

## docs/experiment_log.md — Part B scoreboard
- Add / update row **per-sample categorisation (judging item: categorise held-back samples)**: "E31: 3-class LOO categoriser, pre-registered primary (morph + texture + acq) balanced accuracy 0.66, p 0.005 (200 site-label perms), accuracy 0.68 [0.49, 0.83]; morphology-only at chance (0.35, p 0.16); acquisition-only at the edge (0.47, p 0.035); model probabilities not calibrated (top-bin 0.83 vs 0.67) — reported as a batch fingerprint, never as a verdict."
- Add / update row **baseline membership (judging item: inside / outside the promised distribution)**: "E31: k-NN robust-z distance to all 17 Batch 3 sites with LOO percentiles (floor 1/18): Batch 2 inside (median 41st pct, 0 of 7 beyond the max); Batch 1 2 of 7 beyond the max, both low-contrast artefacts (bright_count_per_Mpx), 1 of 7 on pore count without bright-phase KPIs; the batch-level verdict path (report) is unchanged and remains the accept/investigate/reject output."

---

## experiments/registry.csv — rows to append (experiment,date,metric,reference,batch,kpi,statistic,n_ref,n_batch,value,unit,note)

```
E31,2026-10-03,categoriser_loo_balanced_accuracy,known-3-batches,all,combined,balanced_accuracy,31,,0.664,fraction,"primary (pre-registered); LOO; fold-local imputer/scaler/C; batch fingerprint, not material"
E31,2026-10-03,categoriser_loo_accuracy,known-3-batches,all,combined,accuracy,31,,0.677,fraction,"Clopper-Pearson 95% [0.49, 0.83]; majority 0.548"
E31,2026-10-03,categoriser_perm_p,known-3-batches,all,combined,p_value,31,,0.004975,p,"200 site-label permutations of the whole LOO procedure; (b+1)/(n+1); b = 0"
E31,2026-10-03,categoriser_loo_balanced_accuracy,known-3-batches,all,morph,balanced_accuracy,31,,0.347,fraction,"secondary; morphology-only; chance 0.333"
E31,2026-10-03,categoriser_perm_p,known-3-batches,all,morph,p_value,31,,0.159,p,"secondary; read against 0.017"
E31,2026-10-03,categoriser_loo_balanced_accuracy,known-3-batches,all,acq,balanced_accuracy,31,,0.473,fraction,"secondary; acquisition-only"
E31,2026-10-03,categoriser_perm_p,known-3-batches,all,acq,p_value,31,,0.0348,p,"secondary; read against 0.017; seeds 1-2 give 0.025 / 0.075"
E31,2026-10-03,categoriser_recall,known-3-batches,Batch_1,combined,recall,31,7,0.714,fraction,"5 of 7"
E31,2026-10-03,categoriser_recall,known-3-batches,Batch_2,combined,recall,31,7,0.571,fraction,"4 of 7"
E31,2026-10-03,categoriser_recall,known-3-batches,Batch_3,combined,recall,31,17,0.706,fraction,"12 of 17"
E31,2026-10-03,categoriser_brier,known-3-batches,all,combined,brier_multiclass,31,,0.494,score,"class-prior Brier 0.597; ECE 0.146; top bin mean 0.83 vs observed 0.67 (n 12) - model probabilities not calibrated"
E31,2026-10-03,ood_knn_pct_median,Batch_3,Batch_1,morph,percentile,17,7,29.4,percentile,"k=3 robust-z; reference LOO percentiles; floor 1/18"
E31,2026-10-03,ood_knn_n_exceed_ref_max,Batch_3,Batch_1,morph,count,17,7,2,sites,"4ih2ggld, 5n1q8atc (low-contrast); driver bright_count_per_Mpx z +37/+39 - acquisition artefact"
E31,2026-10-03,ood_knn_pct_median,Batch_3,Batch_2,morph,percentile,17,7,41.2,percentile,"inside the reference"
E31,2026-10-03,ood_knn_n_exceed_ref_max,Batch_3,Batch_2,morph,count,17,7,0,sites,""
E31,2026-10-03,ood_knn_n_exceed_ref_max,Batch_3,Batch_1,morph_no_bright,count,17,7,1,sites,"post hoc, not pre-registered; f1vzngrs on pore_count_per_Mpx +6.6 z"
E31,2026-10-03,ood_knn_n_exceed_ref_max,Batch_3,Batch_1,morph_texture,count,17,7,3,sites,"texture-inclusive; acquisition-sensitive"
E31,2026-10-03,kruskal_n_p_below_0.05,known-3-batches,all,all_39_features,count,31,,14,features,"context only; texture 9/10, acq 4/10, morph 1/19; none of morph survives 0.05/39"
```

---

## docs/decision_log.md — Part A (proposed D50; place after D49P)

### D50 · 2026-10-03 · Per-sample categorisation and baseline membership are two separate answers, delivered beside — not inside — the verdict
- **Decision:** `polaron_qc.categorise` answers the organisers' two judging items per site as separate outputs: (1) a 3-class categoriser over the known batches whose outputs are called **model probabilities** (not posteriors, not confidence) — pre-registered primary = morphology + texture + acquisition, with morphology-only and acquisition-only reported beside it; (2) a baseline OOD percentile from a k-NN robust-z distance to all 17 Batch 3 sites with leave-one-site-out reference percentiles (floor 1/18) — never a probability of defect; (3) a hand-off row with drivers, Batch 3 median ± MAD, data-derived flags and a pointer to the report's verdict. The verdict path (`decision.py`, `report.py`, the frozen notebook) is unchanged and remains the only accept / investigate / reject output.
- **Why:** a 3-class model must pick a known batch even for an unfamiliar sample, so a high Batch 3 probability cannot establish membership of the promised distribution; and the model that categorises (p 0.005) does so on texture and session statistics while morphology alone is at chance (p 0.16) — so the categoriser is a batch fingerprint and must not be read as a material classification or folded into the verdict. Calibration was assessed and failed (top bin 0.83 vs 0.67), hence the vocabulary.
- **Alternatives rejected:** one combined "membership score" from the categoriser's Batch 3 probability (conflates resemblance with membership); a morphology-only primary (would have been chosen after seeing it is at chance — the primary was fixed before the run); excluding the low-contrast sites from the OOD reference comparison (the procedure must run unchanged on an unseen batch; the artefact is instead exposed by the flags and a labelled post-hoc variant); calling percentiles "p-values of being in distribution" (rank floor 1/18, heterogeneous reference including cracked sites).
- **Provenance:** developed using exploratory analysis of Batches 1–3; definitions committed (691990e) before the first run; deviations listed in findings §1.7. E31.
- **Caught by / lesson:** the pre-registered primary's first OOD table flagged the two low-contrast Batch 1 sites through `bright_count_per_Mpx` at z ≈ +37 — a C03 catch by the hand-off's own driver column. Proposed checklist addition **C32 — When a per-site distance or novelty flag fires, read its top contributing feature against that site's acquisition flags before reporting it; a bright-phase driver on a low-contrast site (or a pore driver on a grey-pore site) is an acquisition finding.**

## docs/decision_log.md — Part C (open items)
- Whether the organisers want held-back samples categorised by resemblance (the fingerprint categoriser) or by material state — the two can disagree (two cracked Batch 3 sites are called Batch 1 by the primary).
- Specimen independence of sites (still unconfirmed) bounds every p in E31.
- If an unseen batch arrives with its own session statistics, the acquisition / texture fingerprint will place it away from all three known batches; the categoriser will still return an argmax. The hand-off row's OOD column and flags are what the reader must look at first.
