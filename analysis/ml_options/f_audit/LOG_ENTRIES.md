# Proposed log text for the Protocol F pilot (parent to renumber and paste)

All numbers are from `analysis/ml_options/f_audit/output/*.csv` and are quoted in `findings.md`.

## docs/experiment_log.md Part A

### E25-F · 2026-10-03 · Protocol F: balanced frozen-encoder (DINOv2) novelty audit
- **Question:** does equalising reference-site influence during PCA fitting and k-NN memory construction change the E07 novelty ranking, and does the ranking survive resolution, coverage and acquisition sensitivities (assessment.md Protocol F; D38; C30)?
- **Setup:** `analysis/ml_options/f_audit/run_f_audit.py`; cached ViT-S/14 CLS vectors of complete 512-px BSE tiles (`n1c3`), 518 px primary / 224 px sensitivity; 32 seeded coordinate-selected tiles per eligible reference site (`default_rng([20261003, crc32(site)])`) feed both the PCA (32 components, rank-capped) and the k = 5 memory; all complete query tiles scored; reference LOO holds every tile of the held-out site out of PCA and memory (verified: 0 own tiles in 2,104 fit rows; 32 per site in every balanced fit). Views: all sites (17/14), quality-matched (10/12), ordinary-reference (10/14). Definitions committed before the first run (a32a0ba). Unbalanced arm reproduces `ml.novelty` and the committed E07 table to 1e-5. Fold-local acquisition-only ridge (alpha 1; bright_sep, bse_p1, bse_std, bse_empty_bin_frac, etd_boundary_sharpness, H), LOO over 31 sites. Challenge: gamma 0.75/1.25 and affine 0.8x+20 copies of f1vzngrs, 4ih2ggld, 3806gxp0, 9luzk4jm, 71vgq3fw, embeddings recomputed with the cached model.
- **Results (518, all-sites view, balanced):** Batch_1 sites at the 59–100th percentile of the reference LOO medians; 5n1q8atc, 4ih2ggld (low-contrast) and f1vzngrs exceed the reference maximum; Batch_2 at 18–82, none exceed — identical exceedance pattern to the unbalanced/E07 arm. Balanced vs unbalanced Spearman ρ 0.97 (31 sites), 0.90 (14 query sites); 0 flag flips in the all-sites and quality-matched views, 1 in ordinary-reference (4ih2ggld 90th vs exceeding). Coverage: memory 819 → 544 rows, per-site share 4.8–6.3 % → 5.9 %; max query NN-hit share 12.6 % → 12.7 %, Gini 0.32 → 0.31; the 39-tile sites still receive 1.7–2.7 % of hits with equal rows. Resolution: ρ(518, 224) 0.53 (31 sites), 0.84 (query sites); f1vzngrs 100 → 65th percentile; quality-matched ρ 0.45. Matched-count LOO sensitivity: medians +0.01–0.10, percentiles ≤ 11.8 points, no flag change. Acquisition-only held-out R²: −0.07 (balanced median), +0.11 (P90), −0.45 ordinary sites; pooled correlates still acquisition-first (bse_std +0.43, low-contrast flag +0.43, etd_curtain_frac −0.37, bse_p1 −0.34; best KPI bright_d50 +0.32). Challenge: affine remap absorbed (median shift ≤ 0.05, top-1 neighbour unchanged 89–100 %); gamma moves site medians up to 0.43 (the size of the between-batch gap, 0.46) and percentiles up to 35 points, top-1 neighbour changes for 13–48 % of tiles. Nearest-reference panels: top query tiles are single-large-flake fields or low-contrast noisy texture; neighbours match coarse field composition, not contrast; nothing beyond existing KPIs.
- **Conclusion / changed:** retain the encoder as a retrieval/review assistant (interpretable neighbours, sensitivities disclosed); stop material interpretation (resolution-dependent exceedance flags, acquisition-flagged top sites, gamma sensitivity). Unequal site influence was not the driver of E07. `candidate_register.json` entry `equal_site_frozen_novelty` → `run_pilot`; qc_role unchanged (`excluded_from_verdict`); no primary/threshold/notebook/report change. Checklist: C01–C03, C06/C09/C14, C07/C10/C17, C16/C21/C22, C30.

## docs/experiment_log.md Part B scoreboard

| | novelty catch-all | ◑ | E07 tracks acquisition; E25-F balanced audit: ranking unchanged by equal site influence (ρ 0.97), unstable across resolution; retained as retrieval aid only | exploratory; excluded from verdict |

## experiments/registry.csv rows

```
E25-F,2026-10-03,f_audit.balanced_vs_unbalanced_spearman_median,Batch_3,,novelty_median,spearman_rho,17,14,0.969,rho,"all-sites view, 518 px, 31 sites; output/agreement.csv"
E25-F,2026-10-03,f_audit.balanced_vs_unbalanced_spearman_median_query,Batch_3,,novelty_median,spearman_rho,17,14,0.899,rho,"all-sites view, 518 px, 14 query sites"
E25-F,2026-10-03,f_audit.resolution_518_vs_224_spearman_median,Batch_3,,novelty_median,spearman_rho,17,14,0.529,rho,"balanced, all-sites view, 31 sites"
E25-F,2026-10-03,f_audit.resolution_518_vs_224_spearman_median_query,Batch_3,,novelty_median,spearman_rho,17,14,0.837,rho,"balanced, all-sites view, 14 query sites"
E25-F,2026-10-03,f_audit.resolution_518_vs_224_spearman_median,Batch_3,,novelty_median,spearman_rho,10,12,0.449,rho,"balanced, quality-matched view, 22 sites"
E25-F,2026-10-03,f_audit.exceeds_flag_flips_balanced_vs_unbalanced,Batch_3,,novelty_median,count,17,14,0,sites,"all-sites view, 518 px"
E25-F,2026-10-03,f_audit.exceeds_flag_flips_518_vs_224,Batch_3,,novelty_median,count,17,14,1,sites,"balanced, all-sites view; f1vzngrs 100 -> 64.7 pct"
E25-F,2026-10-03,f_audit.pct_median_balanced_518,Batch_3,Batch_1,novelty_median,percentile_range,17,7,59-100,percentile,"3 of 7 exceed reference LOO max (5n1q8atc, 4ih2ggld low-contrast; f1vzngrs)"
E25-F,2026-10-03,f_audit.pct_median_balanced_518,Batch_3,Batch_2,novelty_median,percentile_range,17,7,18-82,percentile,"0 of 7 exceed"
E25-F,2026-10-03,f_audit.pct_median_balanced_518_quality_matched,Batch_3,Batch_1,novelty_median,percentile_range,10,5,40-100,percentile,"f1vzngrs exceeds"
E25-F,2026-10-03,f_audit.pct_median_balanced_518_quality_matched,Batch_3,Batch_2,novelty_median,percentile_range,10,7,20-80,percentile,"none exceed"
E25-F,2026-10-03,f_audit.memory_rows_unbalanced,Batch_3,,,count,17,,819,tiles,"39-52 per site"
E25-F,2026-10-03,f_audit.memory_rows_balanced,Batch_3,,,count,17,,544,tiles,"32 per site in PCA fit and memory; verified in balance_holdout_verification_summary.csv"
E25-F,2026-10-03,f_audit.loo_own_tiles_in_fit_or_memory,Batch_3,,,max_count,17,,0,tiles,"every LOO fold, every view/resolution/method"
E25-F,2026-10-03,f_audit.nn_hit_share_max,Batch_3,,,share,17,14,0.127,fraction,"balanced 518 (unbalanced 0.126); Gini 0.313 vs 0.321"
E25-F,2026-10-03,f_audit.acquisition_only_heldout_r2,Batch_3,,novelty_median,heldout_r2,,31,-0.072,r2,"balanced 518; fold-local ridge alpha 1, 6 covariates"
E25-F,2026-10-03,f_audit.acquisition_only_heldout_r2,Batch_3,,novelty_p90,heldout_r2,,31,0.113,r2,"balanced 518"
E25-F,2026-10-03,f_audit.acquisition_only_heldout_r2_ordinary,Batch_3,,novelty_median,heldout_r2,,22,-0.447,r2,"balanced 518, ordinary sites"
E25-F,2026-10-03,f_audit.top_correlate,Batch_3,,novelty_median,spearman_rho,,31,0.426,rho,"bse_std (acquisition); low-contrast flag also 0.426; best KPI bright_d50 0.315"
E25-F,2026-10-03,f_audit.matched_count_pct_shift_max,Batch_3,,novelty_median,percentile_points,17,14,11.765,percentile,"balanced 518 all-sites; medians shift +0.01 to +0.10; no flag change"
E25-F,2026-10-03,f_audit.challenge_affine_max_abs_median_delta,Batch_3,,novelty_median,max_abs_delta,,5,0.044,novelty,"518; top-1 neighbour unchanged 0.885-0.962"
E25-F,2026-10-03,f_audit.challenge_gamma_max_abs_median_delta,Batch_3,,novelty_median,max_abs_delta,,5,0.428,novelty,"518, gamma 1.25 on f1vzngrs; between-batch gap B1-B3 0.46"
E25-F,2026-10-03,f_audit.challenge_gamma_max_abs_pct_shift,Batch_3,,novelty_median,percentile_points,,5,35.294,percentile,"224: 71vgq3fw 59->94 (gamma 0.75), 9luzk4jm 41->6 (gamma 1.25)"
E25-F,2026-10-03,f_audit.unbalanced_reproduces_E07_max_abs_diff,Batch_3,,novelty_median,max_abs_diff,17,14,8.7e-06,novelty,"31 sites vs analysis_cache/ml/novelty_per_site.csv"
E25-F,2026-10-03,f_audit.runtime,,,,seconds,,,80,s,"core audit 15 s + challenge; single BLAS thread"
```

## analysis/ml_options/candidate_register.json (already edited in place in this branch)

`equal_site_frozen_novelty`: `status` → `run_pilot`; `evidence_path` → `analysis/ml_options/f_audit/findings.md`; added `pilot` block (date, script, outputs, finding, validation_state). `qc_role` stays `excluded_from_verdict`; `review_status` stays `unreviewed_by_materials_expert`.

## docs/next_steps.md row F

| **F** | P1 / medium | Implement the balanced frozen-DINO audit, Protocol F; deliver balanced/unbalanced comparisons, nearest-reference crops and resolution/acquisition sensitivity. | Claude (engineering) | **Done 2026-10-03** (pilot, exploratory). Artifact `analysis/ml_options/f_audit/` (`findings.md`, `run_f_audit.py`, `output/`). Finding: equal site influence leaves the E07 ranking unchanged (ρ 0.97); ranks unstable across 518/224 and under gamma; top sites are acquisition-flagged. Retained as retrieval/review aid; material interpretation stopped; excluded from verdict. Santosh materials review of the eight nearest-reference panels still open. |

## docs/decision_log.md (suggested Part A line, if the parent wants one)

D-next · Keep the frozen-encoder novelty out of material interpretation after the balanced audit (E25-F): equal site influence was tested and found immaterial to the ranking; resolution and tone-curve sensitivity, not coverage, limit the method. Alternatives considered: matched-reference-count calibration (shown as a sensitivity only, not substituted); conformal-style ranks (floor 1/18 with 17 reference sites, not pursued).
