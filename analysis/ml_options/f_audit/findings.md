# Protocol F: balanced frozen-encoder novelty audit

Exploratory audit of the existing DINOv2 patch-embedding novelty (E07 / D25) with equal reference-site
influence, per `analysis/ml_options/assessment.md` "Protocol F". Nothing here feeds a verdict; novelty ranks
are not defect probabilities; tiles never count as n. Developed on Batches 1–3 after inspecting them; frozen
before any unseen batch is examined.

Script: `run_f_audit.py` (this folder). Every number quoted below is read from a CSV/JSON in `output/`.

## 1. Definitions, registered before any output was computed

This section was written and committed before `run_f_audit.py` was first executed. It is not edited
afterwards; any departure is recorded under "Deviations" at the end of this section.

**Representation (unchanged from E07).** Cached DINOv2 ViT-S/14 CLS vectors (384-d) of complete,
non-overlapping 512-px tiles of the band-trimmed, per-image median/IQR-normalised BSE image (clip ±3, cache tag
`n1c3`), produced by `polaron_qc.ml.patch_embeddings`. Files
`analysis_cache/ml/emb_<batch>_<site>_BSE_dinov2_vits14_p512_s512_r{518,224}_n1c3.npz`. Input 518 px is
primary; 224 px is a declared sensitivity. No encoder fine-tuning, no new backbone, no new crop scale. The
only new embeddings are those of the perturbed copies in the optional acquisition challenge (same model,
same pipeline).

**Eligibility.** A site is eligible when all its cached vectors are finite and it has ≥ 32 complete tiles.
An ineligible site is reported as "not eligible / abstained" in every table; it is never resampled with
replacement or scored under a different rule.

**Balanced tile selection.** For each eligible reference site, tiles are ordered row-major by (y0, x0); a
NumPy `default_rng([20261003, crc32(site_id)])` draws 32 tile indices without replacement. The draw depends
only on the site id string and the tile grid, not on labels, intensity, batch or any novelty value. The same
32 tiles are used at 518 and 224 (the tile grids are identical).

**Balanced fit (the audited method).** PCA with `n_components = min(32, rank)` (`sklearn.decomposition.PCA`,
`random_state = 0`) fitted on the balanced reference rows only; the k = 5 nearest-neighbour memory is the same
balanced rows projected into that space. Tile novelty = mean Euclidean distance to its 5 nearest memory rows.
All complete tiles of a query site are scored; a site contributes exactly two summaries, the median and the
P90 (`numpy.percentile`, linear) of its tile novelties.

**Reference leave-one-site-out (balanced).** For each reference site s, PCA and memory are refitted on the
balanced rows of the other reference sites (every tile of s excluded from fitting and memory); all complete
tiles of s are then scored. This is the reference's own novelty distribution.

**Unbalanced (existing) method.** `polaron_qc.ml.novelty` as committed: all complete tiles of all reference
sites enter PCA (32 components, seed 0) and the k = 5 memory; its LOO refits PCA per held-out site on all
remaining tiles. Reproduced here from the same cache, not re-read from `analysis_cache/ml/novelty_per_site.csv`
(agreement with that file at 518 is checked and reported).

**Percentile within the reference.** Weak percentile (`scipy.stats.percentileofscore`, `kind="weak"`) of a
query site's median (and separately its P90) within the reference LOO site distribution. Resolution is
1 / n_ref_sites. `exceeds_ref_loo_max` = query median above the largest reference LOO median. These are
evidence flags, not probabilities of defect and not acceptance.

**Full-reference-query vs LOO-reference sensitivity.** Primary: queries are scored against the memory built
from all reference sites; reference sites are scored LOO (one site smaller). Sensitivity: every query site is
additionally scored against each of the n_ref LOO memories and the across-fold median of its site summary and
percentile is reported as "matched-count sensitivity". It is shown as a method sensitivity, not substituted
for the primary, and not presented as a calibrated tail probability.

**Views.** (a) *all sites*: reference = all Batch_3 sites, queries = all Batch_1 and Batch_2 sites;
(b) *quality-matched*: reference = Batch_3 sites with `acquisition_group == "ordinary"`, queries = query sites
with `acquisition_group == "ordinary"`; (c) *ordinary-reference*: reference as in (b), queries = all query
sites. `acquisition_group` comes from `polaron_qc.acquisition.derive_flags` applied to the
`features.extract_batch` sites/images tables (feature cache v1.0.1, cache hit). Views are prespecified, not
chosen by score.

**Acquisition-only control (shared with Protocol G).** For every site summary (method × resolution × median /
P90, all-sites view; query sites carry their full-reference score, reference sites their LOO score, as in
E07), a leave-one-site-out ridge over the 31 sites: `SimpleImputer(median)` → `StandardScaler` →
`Ridge(alpha = 1)` fitted on the training sites of each fold only. Covariates: `bright_sep`, `bse_p1`,
`bse_std`, `bse_empty_bin_frac`, `etd_boundary_sharpness`, `H` (frame height, px). Held-out R² = 1 −
Σ(y − ŷ_oof)² / Σ(y − ȳ)². Also reported: the same on the 22 ordinary sites; Spearman correlations of each
summary with each covariate pooled and within each batch. Negative held-out R² does not demonstrate
invariance; positive R² is not causal attribution (covariates can also encode material).

**Nearest-reference evidence.** Balanced method, all-sites view, 518 primary. For each compared batch the 4
highest-novelty query tiles, plus the highest tile of every query site (table only): tile y0/x0 in the
original frame (the cached y0 already includes `band_top`), novelty, the 3 nearest memory tiles (reference
site, patch id, y0, x0, distance). PNG panels show the raw (un-normalised) BSE and ETD crops of the query tile
and its 3 nearest reference tiles with the acquisition group and flags of each site in the caption. ETD is
read through the `SE` alias where needed; channels are pixel-aligned. At 224 the nearest neighbours are saved
as a table only. A tile localises a 512-px field, not the pixels responsible for anything.

**Coverage audit.** Per reference site: tiles contributed to PCA/memory before (all complete tiles) and after
(32); share of memory rows; share of all query 5-NN hits that land on each reference site before/after; the
maximum share and Gini coefficient of the hit shares.

**Agreement statistics.** Spearman ρ of site medians over all 31 sites (queries full-reference, reference
LOO) between 518 and 224 for each method, and between balanced and unbalanced at each resolution; the same
over query sites only; the number of query sites whose `exceeds_ref_loo_max` flag differs between the two
conditions.

**Optional acquisition challenge (item 5 of the protocol).** Sites fixed by rule before scoring: the
alphabetically first `ordinary` site of each batch (Batch_1 `f1vzngrs`, Batch_2 `3806gxp0`, Batch_3
`9luzk4jm`) plus the low-contrast example `4ih2ggld` and the grey-pore example `71vgq3fw`. Perturbations on
the raw uint8 BSE image before trimming and normalisation: gamma 0.75 and 1.25 (`255·(x/255)^γ`, rounded to
uint8) and one non-clipping affine remap `x' = round(0.8·x + 20)` (maximum 224, no clipping). Embeddings are
recomputed for the copies with the same cached model at 518 and 224; normalisation and bands are recomputed on
the copy. Query-batch copies are scored against the balanced all-sites memory; reference-site copies against
the balanced LOO memory that excludes that site. Reported: change in median/P90 and percentile, fraction of
tiles whose top-1 neighbour is unchanged, mean Jaccard of the 5-NN sets. Synthetic intensity changes are
controls, not material labels and not generalisation validation. Skipped only if budget runs out, and then
stated.

**Go/stop reading rules (fixed now so the reading is not chosen after the fact).** Retain the encoder as a
retrieval/review assistant if the nearest-reference panels are interpretable to a reviewer and all
sensitivities are disclosed. Stop material interpretation if any of: (i) Spearman ρ of site medians between
518 and 224 < 0.5 for the balanced method; (ii) any query site's `exceeds_ref_loo_max` flag flips between
balanced and unbalanced or between resolutions; (iii) held-out acquisition R² > 0.25 for the primary summary
(balanced, 518, median); (iv) the top-ranked query tiles' nearest reference tiles differ from them mainly in
contrast/black level/preparation on inspection. These cut-offs are reporting rules for this audit, not QC
tolerances, and either outcome leaves the method outside the verdict.

**Seeds and settings.** `F_SEED = 20261003`; tiles per site 32; PCA 32 capped by rank; k = 5; resolutions
518 (primary) and 224 (sensitivity); reference `Batch_3`; compared batches `Batch_1`, `Batch_2`.

### Deviations

None at registration time. (Appended after the run if any occurred.)

**Execution notes (not method deviations).** (1) OpenBLAS on this machine spun for ~15 s per 544×384 SVD when
multi-threaded; the script pins BLAS to one thread before importing NumPy (0.03 s per SVD). No numerical
definition depends on this. The first attempt was stopped for this reason before it wrote any site output.
(2) The script was executed twice: core audit, then core audit plus the optional challenge (`--challenge`).
All core outputs are deterministic (seeded selection, `PCA(random_state=0)`) and the manifest checks were
identical in both runs. (3) The 4ih2ggld caption prints `sep nan` because its bright-phase separation is
unresolved in the features table (that is what the low-contrast flag means).

## 2. Inputs and coverage (`output/eligibility.csv`, `output/manifest.json`)

All 31 sites are eligible: every cached vector is finite and every site has 39 or 52 complete tiles (7 sites
with 39: `iv6g2oq0`, `uhdslk0o`, `0grcilhi`, `hawkfj64`, `mgxahqnk`, `ptg8lmto`, `xgj4xftb`; 24 with 52).
No site was abstained. Tile grids are identical at 518 and 224. The acquisition groups from `derive_flags` are
Batch_1 5 ordinary + 2 low-contrast, Batch_2 7 ordinary, Batch_3 10 ordinary + 4 grey-pore + 3 cracked
(`output/site_acquisition_flags.csv`). Views: all sites 17 reference / 14 query; quality-matched 10 / 12;
ordinary-reference 10 / 14.

Memory sizes: all-sites view 819 rows unbalanced → 544 balanced; ordinary views 468 → 320. PCA kept 32
components in every fit (rank 384 ≫ 32); explained variance 0.76 balanced / 0.75 unbalanced at 518, 0.73 / 0.72
at 224, 0.78 / 0.77 in the ordinary-reference views at 518.

Reproduction check: the unbalanced arm, recomputed here from the cache, matches `polaron_qc.ml.novelty` to
3.1e-6 (query medians) / 5.4e-6 (reference LOO medians) and the committed E07 table
`analysis_cache/ml/novelty_per_site.csv` to 8.7e-6 on all 31 sites (float32 vs float64 arithmetic). So the
unbalanced numbers below are the E07 computation, not an approximation of it.

## 3. Balance and holdout verification (`output/balance_holdout_verification.csv`, `_summary.csv`)

Requested by Santosh: three facts, with the numbers that prove them. The per-fold table has 2,104 rows (one
row per reference site per fit: the full-reference fit plus every LOO fold, for 3 views × 2 resolutions ×
2 methods). The PCA-fit matrix and the memory are counted from two separately stored site arrays
(`Memory.pca_fit_sites`, `Memory.memory_sites`); `pca.n_samples_` and `NearestNeighbors.n_samples_fit_` are
asserted equal to those arrays' lengths.

1. **Equal site influence during PCA fitting and during memory construction (balanced method).** In every
   balanced fit, full-reference and LOO, in all three views and at both resolutions, the minimum and maximum
   number of tiles per reference site in the PCA-fit matrix are 32 and 32, and in the k = 5 memory 32 and 32
   (`min/max_tiles_per_site_in_pca_fit_full`, `min/max_tiles_per_site_in_memory_full`,
   `min/max_tiles_per_remaining_site_in_loo_fits`, all = 32). No site contributes more rows. The unbalanced
   method, by contrast, contributes 39–52 tiles per site (its documented behaviour).
2. **Whole-site holdout in reference LOO.** For every held-out reference site in every balanced and
   unbalanced LOO fold (17 folds in the all-sites view, 10 in the ordinary views; 12 condition sets), the
   number of its own tiles in the PCA-fit matrix is 0 and in the memory is 0
   (`max_own_tiles_in_pca_fit_when_held_out = 0`, `max_own_tiles_in_memory_when_held_out = 0`). The script
   asserts this inside the loop and would abort otherwise. All complete tiles of the held-out site are then
   scored: 819 held-out tiles in the all-sites view (= every reference tile), 468 in the ordinary views.
3. **Fold-local refitting for any C2ST/permutation on embeddings.** No C2ST and no label permutation on
   embeddings was run in this audit (none is required by Protocol F), so there is nothing to refit. If one is
   added later it must use the `Memory` constructor inside each fold/permutation exactly as the LOO does;
   the committed `polaron_qc.ml.c2st` already refits its scaler/model per fold and per permutation for KPI
   inputs.

`pca_fit_rows_equal_memory_rows = True` for every condition (the memory is the projected PCA-fit matrix;
nothing else enters it). Ineligible/abstained reference sites: 0 in every view.

## 4. Per-site results, all-sites view (`output/site_table.csv`)

Site medians of tile novelty (mean Euclidean distance to the 5 nearest memory rows in PCA space), 518 px:

| site | batch | group | balanced median | pct within ref LOO | exceeds | unbalanced median | pct | exceeds |
|---|---|---|---|---|---|---|---|---|
| 5n1q8atc | Batch_1 | low_contrast | 17.33 | 100 | yes | 16.99 | 100 | yes |
| 4ih2ggld | Batch_1 | low_contrast | 17.02 | 100 | yes | 16.97 | 100 | yes |
| f1vzngrs | Batch_1 | ordinary | 16.86 | 100 | yes | 16.61 | 100 | yes |
| fzrt2k6r | Batch_1 | ordinary | 15.79 | 82 | no | 15.49 | 82 | no |
| uhdslk0o | Batch_1 | ordinary | 15.65 | 76 | no | 15.57 | 82 | no |
| ffwubibz | Batch_1 | ordinary | 15.48 | 76 | no | 15.25 | 76 | no |
| iv6g2oq0 | Batch_1 | ordinary | 15.34 | 59 | no | 15.11 | 65 | no |
| avn74qx1 | Batch_2 | ordinary | 16.23 | 82 | no | 15.76 | 82 | no |
| b3esycq1 | Batch_2 | ordinary | 15.78 | 82 | no | 15.80 | 82 | no |
| epqdaau9 | Batch_2 | ordinary | 15.70 | 76 | no | 15.12 | 65 | no |
| i9jiqjwl | Batch_2 | ordinary | 15.65 | 76 | no | 15.21 | 76 | no |
| 3806gxp0 | Batch_2 | ordinary | 15.59 | 76 | no | 15.43 | 82 | no |
| rxax5ozo | Batch_2 | ordinary | 14.92 | 18 | no | 14.63 | 35 | no |
| r17byphk | Batch_2 | ordinary | 14.92 | 18 | no | 14.49 | 12 | no |

Reference LOO medians (Batch_3, n = 17): balanced 14.73 (`ptg8lmto`) to 16.77 (`mgxahqnk`), unbalanced 14.25 to
16.59. In both methods the three highest reference LOO sites are `mgxahqnk`, `hawkfj64` and `0grcilhi`, all
39-tile strips (H 1904 px); the grey-pore and cracked groups sit inside the ordinary range. Percentile
resolution is 1/17 ≈ 5.9 points; "100" means above all 17 reference LOO medians, nothing more.

Batch medians of site medians (bars in `figures/strip_all_sites_r518.png`): balanced Batch_1 15.79, Batch_2
15.65, Batch_3 LOO 15.33; unbalanced 15.57, 15.21, 14.78. Balanced values are uniformly higher (a 544-row
memory is sparser than an 819-row one), which is why absolute novelty is never compared across methods; ranks
and percentiles are.

At 224 px (sensitivity) the picture changes for the ordinary Batch_1 sites: `f1vzngrs` falls to the 65th
percentile (no longer exceeds; unbalanced 82), the other four ordinary Batch_1 sites sit at 24–53 (unbalanced
18–59), Batch_2 at 24–65 (unbalanced 18–71). The two low-contrast sites stay at 100 and exceed at both
resolutions and in both methods. At 224 the highest reference LOO medians are the cracked sites, `xgj4xftb`
and `hawkfj64` (17.42–17.49).

## 5. Quality-matched and ordinary-reference views

*Quality-matched (10 ordinary reference sites; queries 5 ordinary Batch_1 + 7 Batch_2), 518 px:* Batch_1
ordinary sites at the 40–100th percentile (balanced; `f1vzngrs` 100, exceeds in both methods, median 17.60 vs
reference LOO maximum 17.34 `hawkfj64`), Batch_2 at 20–80, none exceed. At 224: Batch_1 10–80 (`f1vzngrs`
80, no exceedance), Batch_2 10–70, none exceed. Percentile resolution here is 10 points.

*Ordinary-reference (10 ordinary reference sites; all 14 queries), 518 px:* the two low-contrast sites and
`f1vzngrs` are at or above the reference maximum in the unbalanced arm; in the balanced arm `4ih2ggld` is at
the 90th percentile (17.27 < 17.34) and does not exceed, while `5n1q8atc` and `f1vzngrs` do. That is the one
method-dependent exceedance flag in the whole audit (`agreement.csv`, ordinary_reference / method / 518:
1 flip of 14). At 224 both low-contrast sites exceed in both methods and `f1vzngrs` does not.

## 6. Balanced vs unbalanced: ranking and coverage audit

`output/agreement.csv`: Spearman ρ of site medians between balanced and unbalanced memories, all 31 sites,
0.97 at 518 and 0.90 at 224 (all-sites view); query sites only 0.90 / 0.89; percentiles of query sites 0.93 /
0.78. Exceedance flags: 0 flips of 14 in the all-sites view at both resolutions, 0 of 12 in the
quality-matched view, 1 of 14 in the ordinary-reference view at 518 (above). `figures/scatter_balanced_vs_unbalanced_r518.png`.

`output/coverage_audit.csv`, `_summary.csv` (518): before balancing the 17 reference sites contributed 39–52
tiles each (memory shares 4.8–6.3 %), after balancing 32 each (5.9 %). Where the 728 query tiles find their 5
nearest neighbours barely moves: the maximum share of query NN hits on one reference site is 12.6 %
(`pl8uabbv`) unbalanced and 12.7 % (`vc2whyaq`) balanced; Gini of the hit shares 0.32 → 0.31. The same four
ordinary sites (`vc2whyaq`, `utfgcjfa`, `pl8uabbv`, `cfe5vt7s`) receive 9–13 % each in both arms, and the
39-tile sites `hawkfj64`, `0grcilhi`, `mgxahqnk` receive 1.7–2.7 % in both — their low hit rate is not a
consequence of contributing fewer rows, because it persists when they contribute exactly as many rows as
everyone else. `figures/coverage_audit_r518.png`.

Share of each query site's 5-NN hits by reference acquisition group
(`output/query_nn_reference_group_shares_r518.csv`): ordinary 66–78 %, cracked 10–21 %, grey-pore 6–23 %
(balanced), against memory shares of 59 / 18 / 24 %. Neighbour group is close to the memory composition for
every query site; it carries no information about the query.

Reading: unequal site influence in PCA/memory construction, the specific concern behind this protocol (D38,
C30), was not what produced the E07 ranking. Balancing leaves the ranking essentially unchanged (ρ 0.97).

## 7. Resolution sensitivity (518 primary vs 224)

Balanced method, all 31 sites: Spearman ρ 0.53 (medians), 0.51 (P90); query sites only 0.84 (medians), 0.83
(percentiles); one exceedance flip of 14 (`f1vzngrs`, 100 → 65). Unbalanced: 0.43 / 0.53; query sites 0.82;
one flip. Quality-matched view: ρ 0.45 balanced (0.37 unbalanced) over 22 sites, 0.57 over the 12 query
sites, `f1vzngrs` flips. Ordinary-reference view: 0.57 (0.52), two flips (`f1vzngrs` and `4ih2ggld`).
`figures/scatter_r518_vs_r224_balanced.png` shows the reference LOO sites reorder almost completely between
resolutions while the two low-contrast sites stay on top.

Against the registered reading rules: rule (i) ρ < 0.5 is met in the quality-matched view (0.45) and is at the
margin in the all-sites view (0.53); rule (ii) flag flips between resolutions occur in every view.

## 8. Full-reference query vs LOO-reference (matched-count sensitivity)

`output/loo_vs_full_reference_sensitivity.csv`, `_summary.csv`. Scoring each query site against the 17
one-site-smaller LOO memories and taking the across-fold median changes its median by −0.002 to +0.10 at 518
(balanced, all sites; mean +0.05) and 0.00–0.12 at 224, i.e. the primary full-reference scores are slightly
lower than a same-count reference would give (conservative direction, as `ml.novelty` documents). Percentiles
move by at most 11.8 points at 518 (`r17byphk` 17.6 → 23.5, `rxax5ozo` 17.6 → 29.4) and no exceedance flag
changes; in the 10-site ordinary views the coarser 10-point resolution lets single sites move up to 30 points
at 224. This is a method sensitivity; neither version is a calibrated tail probability.

## 9. Acquisition-only control (`output/acquisition_control_r2.csv`, `_correlations.csv`, `novelty_correlates_*`)

Held-out R² of the fold-local ridge (alpha 1; imputer and scaler fitted on the 30 training sites of each
fold; covariates bright_sep, bse_p1, bse_std, bse_empty_bin_frac, etd_boundary_sharpness, H):

| summary | balanced 518 | unbalanced 518 | balanced 224 | unbalanced 224 |
|---|---|---|---|---|
| site median, 31 sites | −0.07 | −0.01 | −0.24 | −0.38 |
| site P90, 31 sites | +0.11 | +0.07 | −0.20 | −0.27 |
| site median, 22 ordinary sites | −0.45 | −0.55 | −0.98 | −0.47 |

Rule (iii) (R² > 0.25 on the primary summary) is not met. As registered, this does not demonstrate
invariance: the pooled Spearman correlates of the balanced 518 median are still acquisition descriptors first
(bse_std +0.43 p 0.017, low-contrast flag +0.43 p 0.017, etd_curtain_frac −0.37 p 0.044, bse_p1 −0.34
p 0.062) with the best KPI `bright_d50` at +0.32 (p 0.085) — the E07 pattern reproduced with equal site
influence. The within-batch correlations change sign between batches (H: +0.92 in Batch_1 (n = 7), +0.07 in
Batch_2, −0.36 in Batch_3; etd_boundary_sharpness: −0.50, +0.68, −0.49; bse_empty_bin_frac −0.51 on the 22
ordinary sites), which is why a single pooled linear model cannot predict held-out sites, not evidence that
the embedding ignores these variables. The two sites that top every ranking are the two sites flagged as
acquisition anomalies by `derive_flags`.

## 10. Nearest-reference evidence (`output/nearest_reference_top_tiles_r518.csv`, `output/crops/*.png`)

Eight panels (4 per compared batch), each the raw BSE and ETD crop of the query tile and of its 3 nearest
balanced-memory tiles, with site, group, flags, bse_p1, bright_sep and H in the caption; tile y0/x0 are in
the original frame. The highest-novelty Batch_1 tiles are three tiles of `5n1q8atc` and one of `4ih2ggld`
(the low-contrast sites, novelty 22.0–24.0); the Batch_2 tiles are two of `epqdaau9`, one of `avn74qx1`, one
of `3806gxp0` (22.3–24.2). Nearest distances are 15.8–24.6, in most panels barely below the query's own
novelty: these tiles sit in a sparse region of the embedding rather than close to any wrong neighbour.

What the panels show on inspection: (a) seven of the eight query tiles are dominated by one large graphite
flake section or a near-featureless flake face filling most of the 512-px field; their nearest reference
tiles are also large-flake tiles (`xgj4xftb`, `hzumfsms`, `ufdvpb81`, `utfgcjfa`, `vc2whyaq`), so the
retrieval is matching coarse field composition, and "nearest to a cracked site" in two Batch_2 panels means a
flake face resembles a flake face, not that a crack is present; (b) the `4ih2ggld` tile (#3, Batch_1) shows a
finely porous, visibly noisier region whose nearest tile (`pl8uabbv`, 15.8) contains a bright particle and no
comparable texture — the match is not on content, consistent with the low-contrast/noise character of that
site driving its distance; (c) nothing in the panels suggests a morphology the KPIs do not already measure.
Per-site top tiles at both resolutions are in `output/nearest_reference_top_tile_per_site.csv`.

Reading against rule (iv): for the low-contrast sites the retrieved difference is contrast/noise
(acquisition); for the ordinary top tiles it is rare field content (a single large particle), which is neither
an acquisition artefact nor a defect. In both cases the neighbours are interpretable to a reviewer — a
reviewer can see at once what the tile resembles and why it is far from everything — which is the retrieval
use the gate keeps.

## 11. Acquisition challenge (`output/acquisition_challenge.csv`)

Done. Five prespecified sites (`f1vzngrs`, `4ih2ggld`, `3806gxp0`, `9luzk4jm`, `71vgq3fw`), three
perturbations, two resolutions; 52 tiles matched by coordinates in every case (no band change). The
non-clipping affine remap is almost absorbed by the per-image median/IQR normalisation: site medians move by
≤ 0.05 (518: 0.01–0.04; 224: 0.01–0.05), the top-1 neighbour is unchanged for 89–100 % of tiles, mean 5-NN
Jaccard 0.87–0.97, and no percentile moves by more than one step (5.9). Gamma 0.75 / 1.25 is not absorbed:
site medians move by up to +0.43 (`f1vzngrs`, γ 1.25, 518) and −0.39 (`9luzk4jm`, γ 1.25, 224), the top-1
neighbour changes for 13–48 % of tiles (mean 23–35 %), mean Jaccard 0.57–0.77, and percentiles move by up to
+17.6 / −23.5 (518: `9luzk4jm` 12 → 29, `71vgq3fw` 71 → 47) and +35.3 / −35.3 (224: `71vgq3fw` 59 → 94 under γ 0.75; `9luzk4jm` 41 → 6 under γ 1.25). The
gamma-induced median shifts (0.1–0.4) are the size of the between-batch differences in §4 (Batch_1 − Batch_3
LOO medians 0.46 balanced). The exceedance flags of the low-contrast sites and of `f1vzngrs` at 518 did not
change under any perturbation. These are synthetic controls: they show that a tone-curve difference of this
size can move a site across most of the reference percentile range; they are not material labels and do not
validate generalisation.

## 12. Finding against the go/stop gate

**Retain as a retrieval/review assistant; stop material interpretation.** Evidence:

- Retain: the nearest-reference panels are directly interpretable (§10); the method is cheap (15 s from cache),
  deterministic, and its sensitivities are now disclosed in numbers (§6–§9, §11).
- Stop material interpretation: rule (ii) is met — exceedance flags flip between 518 and 224 in every view
  (`f1vzngrs`), and rule (i) is met in the quality-matched view (ρ 0.45) and marginal in the all-sites view
  (0.53); the only sites that exceed the reference at both resolutions and under both memories are the two
  sites `derive_flags` marks as low-contrast acquisition anomalies (§4, §9, rule (iv)); a gamma change moves
  percentiles by up to 35 points (§11); the pooled correlates remain acquisition-first (§9).
- Coverage, the specific concern of Protocol F, is cleared: equal site influence changes neither the ranking
  (ρ 0.97) nor where neighbours are found (§6). E07's acquisition reading stands with the balanced method.

What this does not say: that Batch_1 or Batch_2 is or is not different from Batch_3 (the primary KPIs and
their site-level tests do that), that any site is defective, or that the encoder is useless — it localises
rare field content and acquisition-atypical sites, which is a review aid. No threshold here is a tolerance;
nothing feeds the verdict; a credible new morphology observation from the panels would need independent image
review and cannot alter a verdict automatically.

## 13. Checklist items applied (docs/decision_log.md Part B)

- **C01** table read before belief: no constant column; balanced values uniformly higher than unbalanced by a
  known mechanism (memory size), ranks compared instead. **C02** nothing varies by construction: all-tile query
  scoring and 32-tile memories were fixed in §1; percentiles are relative to a reference LOO distribution
  computed the same way for every reference site. **C03** confound tested: acquisition-only ridge, pooled and
  within-batch correlations, gamma/affine challenge (§9, §11); the top sites are acquisition-flagged.
- **C06** sites are the units (31; 17/7/7; 10/12 and 10/14 in the ordinary views); tiles never counted as n.
  **C09** every reference-dependent fit (tile selection, PCA, memory) is inside the LOO fold; verified in §3.
  **C14** usable n stated per view; no site excluded for coverage (all ≥ 32 tiles).
- **C07** no null read as acceptance: "does not exceed the reference LOO maximum" is reported as a percentile
  with its 1/17 resolution, never as acceptance. **C10** no threshold rule adopted; `exceeds_ref_loo_max` is an
  evidence flag whose instability is itself reported. **C17** no "confidence"; percentiles, ρ, R² and flag-flip
  counts only.
- **C16** provenance: developed on Batches 1–3 after inspecting them; definitions registered and committed
  (a32a0ba) before the first run; the unbalanced arm reproduces E07 to 1e-5. **C21** no acquisition variable
  enters a material list; the covariates are used only as a control. **C22** nothing promoted: the candidate
  stays `excluded_from_verdict`; no primary KPI, threshold, notebook or report was touched.
- **C30** architecture vs representation: same frozen encoder, same cache; the only change is equal site
  influence in fitting/memory, which was isolated and shown to be immaterial to the ranking (§6); whole-site
  holdout verified (§3); more tiles were not treated as more evidence.
