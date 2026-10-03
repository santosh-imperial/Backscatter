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
