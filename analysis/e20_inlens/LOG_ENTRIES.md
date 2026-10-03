# E20 — proposed entries for the shared logs (not applied; the shared logs are append-only and owned by the main checkout)

## A. docs/experiment_log.md Part A — proposed entry

### E32 · 2026-10-03 · E20: the actual per-particle Inlens texture under local-contrast normalisation (31 known sites)
- **Question:** does `inlens_particle_texture` (per-particle Inlens SD inside BSE bright-particle interiors, trust *confounded* since E03: ρ 0.77 with Inlens brightness, acquisition R² 0.74) keep its batch separation and lose its brightness association under prespecified normalisations? Framing fixed first: surviving = robustness to that transformation only; losing = possibly removed material contrast; neither promotes.
- **Setup:** `analysis/e20_inlens/run_e20.py`; §1 of `findings.md` pre-registered and committed (d68ccbd) before the run. Loader/trim/thresholds/segmentation/interior erosion imported from `polaron_qc.features`; integrity check reproduces the cached measurand, particle counts and thresholds on all 31 sites. Variants per particle: a CV = SD/mean; b SD/(particle IQR) (gain+offset invariant); c SD after local z-normalisation (Gaussian σ 64 px; 32 px sensitivity); d LBP (P8 R1 uniform) non-uniform fraction (monotone-map invariant; entropy sensitivity). Site median/p90 over particles; sites are n (7/7/17; `bright_usable` view 5/7/17). Spearman with Inlens p50/std pooled and within batch; Kruskal; exact site-label permutation (hl_shift, 346 104 allocations) and robust shift in Batch 3 MADs; LOO ridge acquisition control (inlens p50/std, bse p1/std, bright_sep, H); intensity-remap perturbations (γ 0.75/1.25, affine 0.6·I+40) on four prespecified sites; six prespecified evidence crops.
- **Result (all31 site medians):** `cur` ρ +0.77, Kruskal p 0.0011, B1/B2/B3 0.320/0.237/0.177, B1 vs B3 +2.18 MAD p 4.6e-5, B2 vs B3 +0.92 MAD p 0.013, R² 0.75 (reproduces E03). `a_cv` ρ +0.09 pooled but −0.74 within B1, Kruskal 0.023, +0.83 MAD p 0.015 / +0.87 p 0.071, R² 0.40, and it moves 2.3–4.7 gaps under the affine remap (offset-sensitive by construction). `b_aff` is a near-constant shape index (0.72–0.81; Gaussian 0.741), Kruskal 0.14, not reportable. `c_loc64` ρ +0.68 (within B2 +0.94, B3 +0.75), Kruskal 0.0016, +1.98 MAD p 0.0034 / +1.95 MAD p 0.0092, R² 0.71, affine-invariant (≤ 0.04 gap) but γ moves it up to 0.45 gap. `d_lbp` ρ +0.39, Kruskal 0.059, +2.89 MAD p 0.017 / +3.17 MAD p 0.020 on a B3 MAD of 0.003 (absolute gap 0.009 ≈ 4 %), R² 0.03, within-B2 ρ −0.96 with Inlens std, γ moves it 0.54 gap at one site. The monotone B1 > B2 > B3 ordering survives in no variant: B1 and B2 coincide in a, c and d. Residual Kruskal after the acquisition ridge > 0.05 for every variant. Low-contrast crop shows the interior mask as a holey mesh; five of six crops show fine grain rather than resolvable internal texture, one Batch 2 crop shows mottling.
- **Conclusion / changed:** pre-registered rule met by no variant; nothing enters the categoriser texture family; `inlens_particle_texture` stays *confounded*. Established: the contrast is not a gain/offset artefact (affine-robust; gain/offset-invariant c keeps ≈ 2 MAD, p < 0.01) but is brightness-linked beyond gain (c ρ 0.68, R² 0.71) and partly LUT-sensitive (γ up to 0.45 gap); the B1 > B2 ordering is amplitude-carried. Next gate: new pre-registration with a detector-noise covariate and γ-LUT control, selection inside L1 folds only (D49P), Santosh's crop review first. Checklist C01–C03, C06, C09, C14, C16, C21, C22.

## B. docs/experiment_log.md Part B — proposed scoreboard line change

- ETD / Inlens KPIs row: status unchanged (◑). Replace the E20 note with: "E32 (E20 run): per-particle texture is gain/offset-robust but brightness-linked beyond gain and γ-LUT sensitive; monotone ordering amplitude-carried; no normalised variant meets the pre-registered candidate rule; expert crop review open."

## C. experiments/registry.csv — proposed rows (append verbatim)

```
E32,2026-10-03,e20_spearman_inlens_p50,known-data,all31,inlens_particle_texture,rho,,31,0.77,rho,"cur site median vs Inlens p50; reproduces E03"
E32,2026-10-03,e20_kruskal_p,known-data,all31,inlens_particle_texture,p,,31,0.00105,p,"cur site median; 3 batches"
E32,2026-10-03,e20_shift_mad,Batch_3,Batch_1,inlens_particle_texture,robust_shift,17,7,2.18,B3 MAD,"cur; bootstrap 95% [0.97, 10.52]; exact hl_shift p 4.6e-5"
E32,2026-10-03,e20_shift_mad,Batch_3,Batch_2,inlens_particle_texture,robust_shift,17,7,0.92,B3 MAD,"cur; [−0.00, 9.59]; exact p 0.0126"
E32,2026-10-03,e20_loo_r2_acquisition,known-data,all31,inlens_particle_texture,r2,,31,0.754,R2,"LOO ridge alpha 1; inlens p50/std, bse p1/std, bright_sep, H"
E32,2026-10-03,e20_spearman_inlens_p50,known-data,all31,inlens_texture_a_cv,rho,,31,0.09,rho,"within B1 −0.74 (n 7)"
E32,2026-10-03,e20_kruskal_p,known-data,all31,inlens_texture_a_cv,p,,31,0.0228,p,
E32,2026-10-03,e20_shift_mad,Batch_3,Batch_1,inlens_texture_a_cv,robust_shift,17,7,0.83,B3 MAD,"exact p 0.0151"
E32,2026-10-03,e20_shift_mad,Batch_3,Batch_2,inlens_texture_a_cv,robust_shift,17,7,0.87,B3 MAD,"exact p 0.0707"
E32,2026-10-03,e20_loo_r2_acquisition,known-data,all31,inlens_texture_a_cv,r2,,31,0.400,R2,
E32,2026-10-03,e20_perturb_affine_delta_over_gap,known-data,4 sites,inlens_texture_a_cv,max_abs,,4,4.70,gap,"affine 0.6x+40; offset-sensitive by construction"
E32,2026-10-03,e20_kruskal_p,known-data,all31,inlens_texture_b_aff,p,,31,0.142,p,"near-constant shape index (site medians 0.72–0.81); not reportable (C22)"
E32,2026-10-03,e20_spearman_inlens_p50,known-data,all31,inlens_texture_c_loc64,rho,,31,0.68,rho,"within B1/B2/B3 −0.18/+0.94/+0.75"
E32,2026-10-03,e20_kruskal_p,known-data,all31,inlens_texture_c_loc64,p,,31,0.00161,p,
E32,2026-10-03,e20_shift_mad,Batch_3,Batch_1,inlens_texture_c_loc64,robust_shift,17,7,1.98,B3 MAD,"[0.31, 6.42]; exact p 0.00344"
E32,2026-10-03,e20_shift_mad,Batch_3,Batch_2,inlens_texture_c_loc64,robust_shift,17,7,1.95,B3 MAD,"[0.27, 5.97]; exact p 0.00921"
E32,2026-10-03,e20_loo_r2_acquisition,known-data,all31,inlens_texture_c_loc64,r2,,31,0.714,R2,
E32,2026-10-03,e20_perturb_affine_delta_over_gap,known-data,4 sites,inlens_texture_c_loc64,max_abs,,4,0.04,gap,"gain/offset invariant up to rounding"
E32,2026-10-03,e20_perturb_gamma_delta_over_gap,known-data,4 sites,inlens_texture_c_loc64,max_abs,,4,0.45,gap,"gamma 1.25 on f1vzngrs"
E32,2026-10-03,e20_perturb_gamma_delta_over_gap,known-data,4 sites,inlens_particle_texture,max_abs,,4,0.34,gap,"gamma 0.75/1.25 on f1vzngrs"
E32,2026-10-03,e20_spearman_inlens_p50,known-data,all31,inlens_texture_d_lbp,rho,,31,0.39,rho,"within B2 rho with Inlens std −0.96 (n 7)"
E32,2026-10-03,e20_kruskal_p,known-data,all31,inlens_texture_d_lbp,p,,31,0.0586,p,"fails pre-registered threshold 0.05"
E32,2026-10-03,e20_shift_mad,Batch_3,Batch_1,inlens_texture_d_lbp,robust_shift,17,7,2.89,B3 MAD,"B3 MAD 0.0031; absolute gap 0.009; exact p 0.0171"
E32,2026-10-03,e20_shift_mad,Batch_3,Batch_2,inlens_texture_d_lbp,robust_shift,17,7,3.17,B3 MAD,"exact p 0.0201"
E32,2026-10-03,e20_loo_r2_acquisition,known-data,all31,inlens_texture_d_lbp,r2,,31,0.025,R2,
E32,2026-10-03,e20_perturb_gamma_delta_over_gap,known-data,4 sites,inlens_texture_d_lbp,max_abs,,4,0.54,gap,"gamma 0.75 on 9luzk4jm; re-quantisation of a stretched image"
E32,2026-10-03,e20_integrity_sites_matching_cache,known-data,all31,inlens_particle_texture,count,,31,31,count,"recomputed cur median, particle count, th_lo, th_hi equal cache"
```

## D. docs/decision_log.md Part A — proposed entry

### D50 · 2026-10-03 · No normalised Inlens texture variant enters the categoriser texture family (E20 / E32)
- **Decision:** apply the pre-registered rule of `analysis/e20_inlens/findings.md` §1.6 as written: no variant (CV, per-particle affine, local-z σ 64 px, LBP non-uniform fraction) is proposed as a candidate; `inlens_particle_texture` keeps trust *confounded*; KPI_TRUST, MATERIAL_KPIS, decision inputs and thresholds are unchanged. The nearest miss (LBP fraction: Kruskal 0.059, three of four numerical criteria) is not re-read as a pass.
- **Rationale:** the run shows the existing contrast is gain/offset-robust (affine remap ≤ 0.04 gap; the gain/offset-invariant local-z variant keeps both later batches ≈ 2 MAD above Batch 3, exact p < 0.01) but as brightness-linked as before (ρ 0.68, acquisition R² 0.71) and sensitive to a gamma-type LUT (up to 0.45 gap on one site). The monotone B1 > B2 > B3 ordering is amplitude-carried and vanishes under every normalisation. The variants that lose the brightness link do so by construction (CV), by discarding the information (SD/IQR ≈ 0.74 everywhere) or by measuring pixel-noise roughness with a 4 % effect (LBP).
- **Alternatives rejected:** promoting local-z texture as a "brightness-robust" candidate (it is not — the association survives the normalisation); promoting LBP on its pairwise p-values (selection by known-folder significance, contrary to D49P; Kruskal threshold failed; noise-driven); dropping the Inlens texture from the signatures page (its gain/offset robustness is a real finding that belongs on the page alongside the confound).
- **Consequences:** `docs/batch_signatures.md` should state both findings and qualify the Inlens "monotone ordering" argument as amplitude-carried; any renewed attempt requires a new pre-registration with a detector-noise covariate and γ-LUT control, selection inside the L1 categoriser folds only, and Santosh's review of the six evidence crops (open item). Add to Part C: expert review of `analysis/e20_inlens/figures/evidence_particles.png`, in particular whether the Batch 2 `3806gxp0` mottling is a material feature worth a mask-independent measurement.
- **Review applied:** C01–C03, C06, C09, C14, C16, C21, C22.

## E. Proposed checklist addition (Part B), only if the reviewers agree it is not covered

- **C32 — A normalisation that removes a confound's correlation must be shown not to do so by construction.** Dividing by, or ranking against, the confound itself (CV against brightness; per-image percentiles) can remove the correlation arithmetically while leaving the confound in the measurement; test with an intensity-remap perturbation and a within-group correlation before calling the variant "robust". (E32)
