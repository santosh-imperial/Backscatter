# Proposed log text for the task-H rehearsal (not applied to the shared logs — paste after review)

All numbers below are taken from `analysis/rehearsal_h/receipt.md`, which holds the commands that reproduce them.
The experiment log already contains an **E24** (morphology benchmark), so this rehearsal is proposed as **E25**.

---

## docs/experiment_log.md — Part A

### E25 · 2026-10-03 · Cold unseen-folder rehearsal repeated on the current path; c2st made independent of site-id order
- **Question:** (a) does `ml.c2st` return the same answer for the same images whatever the site ids (E23 / D34 open item: AUC 0.51 → 0.41 on a renamed copy of Batch 1)? (b) Does the documented one-command drop (README) behave on folders nobody has seen — new ids, more than half the sites flagged, three sites, three cracked sites — including the `--review` step and the notebook path, with no edit to CONFIG, thresholds or `FROZEN_HASH`?
- **Setup:** worktree branch `worktree-agent-a668ba1e55c78cc65` from 79bb912. Fixtures = hard links to the raw TIFFs under new 8-char ids outside `Dataset/` (`analysis/rehearsal_h/make_fixtures.py`; deterministic ids `r/q/s/c` + sha1 prefix): **Batch_R** = the seven Batch 1 sites; **Batch_Q** = the four grey-pore Batch 3 sites + the two low-contrast Batch 1 sites; **Batch_S** = three ordinary Batch 1 sites; **Batch_C** = the three cracked Batch 3 sites. Fresh feature cache (`--cache-dir _scratch/cache`), reference extracted once into it (360.8 s for 17 sites, CPU shared with the test suite), each fixture cold. One command per fixture via `analysis/rehearsal_h/run_rehearsal.py`, which also records the JSON snapshot written by the new `--summary` flag.
- **c2st fix (ml.py):** rows and sites are put in a canonical content-based order before CV — rows sorted inside each site by their rounded feature vector, sites sorted by (sha1 of the rounded block, exact bytes, label) — and `StratifiedGroupKFold`, the site-label permutation stream and the liblinear solver (now seeded from `seed`; it was unseeded, so coefficients differed run to run at 1e-5) all see the same arrays for the same content. Site-level weighting and site-level permutation semantics (plan §2.0 rule 6) unchanged; `per_site_scores` keeps the caller's ids. Tests: renamed ids / permuted rows (within and across blocks) / both, each at patch level and one row per site, plus determinism-and-sensitivity; the 30-simulation null-uniformity test still passes.
- **Results (c2st, material-only, 200 site permutations, seed 0, warm cache):**

| pair | before (E22) | before, same images renamed + rows shuffled | after | after, renamed |
|---|---|---|---|---|
| Batch 1 vs 3 | AUC 0.513, p 0.458 | 0.395, 0.697 | **0.588, 0.299** | 0.588, 0.299 (bit-identical null) |
| Batch 2 vs 3 | 0.613, 0.264 | 0.588, 0.264 | **0.588, 0.294** | 0.588, 0.294 (bit-identical) |

  Seeds 0–9 after the fix: Batch 1 AUC 0.34–0.59, p 0.30–0.73; Batch 2 AUC 0.37–0.66, p 0.21–0.68; no seed below α — "classifier does not corroborate" is robust to the fold / permutation realisation, the point estimate is not. Full one-command re-runs of Batch 1 and Batch 2 (73 s / 80 s warm): same verdicts, stability 1.00 (7/7), attenuation +0.28 / −1.39 and −0.73 / −3.01, MDC unchanged; only the classifier cell moved.
- **Results (rehearsal, reference warm, fixture cold):**

| fixture | wall | verdict | drift alert / localized / quality abstention | derived flags | attenuation (strat / adj, available) | MDC feasible | stability |
|---|---|---|---|---|---|---|---|
| Batch_R (7) | 151 s | consistent within detectable limits | no / flag (0.27 MAD, not severe) / no | low contrast on the two renamed Batch 1 sites, from the data | +0.28 / −1.39, yes | 5/5 (1.75–3.00 MAD) | 1.00 (7/7) |
| Batch_Q (6) | 124 s | investigate — drift, reason "quality abstention: raised black level / grey pores on more than half of the sites" | no / flag / **yes** | grey pore ×4 (BSE p1 rule, source data), low contrast ×2 | n/a / +0.71, yes | 5/5 | 1.00 (6/6) |
| Batch_S (3) | 57 s | investigate — drift, "quality abstention: only 3 sites in batch (< 5); fewer than min usable sites on every primary KPI" | no / none / **yes** | none (all ordinary) | n/a / −2.89, yes | 5/5 | 1.00 (3/3) |
| Batch_C (3) | 65 s | investigate — drift, same n < 5 abstention | **yes** / **pending_review** (3 sites: crack_frac 5.4, 5.8, 2.6 MAD; pore_max_d 2.7) / **yes** | none | n/a / +0.40, yes | 5/5 | 0.00 (0/3) — every 2-site fold returns *investigate — localized* (drift alert cannot fire at n = 2); a label switch between two investigate paths under abstention |

  Image review on Batch_C: `--review c20a68de:crack_frac=yes` → localized **credible** (1 confirmed, 2 still pending); `=no` on all four severe flags → **refuted ×3 sites, localized "flag"**, escalation off, stability 3/3. The verdict label stays *investigate — drift* in both because the abstention branch prefers the drift label when the drift alert is positive (see Part C item below). Notebook path: `Batch_R` added to `CONFIG["compare"]` in a scratch copy of the generated notebook, executed from a scratch root with the same layout — zero error cells (36 cells, 370 s), §0 assert passed with `99d2bbcae6f3`, Batch_R *consistent*, flags from the data, classifier 0.588 / 0.299 (= CLI).
- **Failures found and fixed (all drop-only, none visible on the known folders):** (1) `--review SITE:KPI=yes|no` crashed in `report._config_hash` (`json.dumps` refuses the `(site, kpi)` tuple keys) before any result was assembled — README step 4 had never run end to end; tuple keys now serialise as `site:kpi`, test added. (2) The abstention "what would move it" line rewrote "only 3 sites in batch (< 5)" into "at least 3 sites in batch (< 5)"; it now quotes the recorded reasons. (3) Classifier site-id order dependence, above. Added for the procedure: `--cache-dir` (cold runs without touching the shared cache) and `--summary OUT.json` (diff-able record of verdict, outcome columns, flags with source, MDC feasibility, local flags with review state, per-fold verdicts, notes, hashes).
- **Not changed / not covered:** CONFIG, thresholds (b4f4da2e357c) and `FROZEN_HASH` (99d2bbcae6f3) untouched; config hash of a no-review run efdc7a00eb0b (a run with reviews hashes differently, by design). The fixtures are known data under new names: this is a procedure rehearsal, not validation on unseen material. The MDC-infeasible case (n_incoming > n_ref − 3) is not exercised by these fixtures (covered by the E22 probe). Tests: 96 → 107.
- **Reading:** the drop procedure works unattended for the verdict path, the abstentions come from the data alone, and the classifier is now a function of content and seed. Three defects were found only by exercising the documented steps on renamed folders — the argument for repeating the rehearsal after the last core change stands.
- **Changed:** `polaron_qc/ml.py` (`_canonical_order`, seeded liblinear), `polaron_qc/report.py` (`main()`, `--cache-dir`, `--summary`, `summary()`, `_str_keys` in `_config_hash`), `polaron_qc/decision.py` (abstention wording), tests (+11), `analysis/rehearsal_h/` (receipt, fixture and runner scripts, this file), `.gitignore` (`_fixtures/`, `_scratch/`). Checklist applied: C04, C06–C09, C14, C16–C19, C21, C25, C27.

### Figures elsewhere that this changes
- E22 (Part A) and D30 quote the in-pipeline material c2st as AUC 0.51 / p 0.458 (Batch 1) and 0.61 / 0.264 (Batch 2); E23 / D34 quote 0.41 / 0.657 for the renamed copy. With the canonical order the current numbers are 0.588 / 0.299 and 0.588 / 0.294, and the renamed copy equals the original. The log entries stay as the record of what was measured then; a one-line pointer "superseded by E25 (order-independent c2st)" after each is suggested.
- The committed `reports/qc_Batch_1.html`, `reports/qc_Batch_2.html` and the executed `notebooks/02_batch_qc.ipynb` still show the old classifier cells (0.458 / 0.264). Regenerating them is a one-command / one-build job; proposed but not done in this worktree (shared artefacts).
- Part B scoreboard row "classifier corroboration": add "order-independent since E25".

---

## docs/experiment_log.md — Part B scoreboard
- Row **classifier corroboration**: `E22: … (D30); reproduces the cached numbers` → append "; E25: canonical content order, bit-identical under site relabelling and row permutation (AUC 0.59 / p 0.30 and 0.59 / 0.29); seeds 0–9 never below α".
- Row **unseen-batch procedure** (if present; else add): "E25: rehearsed cold on four renamed fixtures (consistent / quality abstention from data flags / n < 5 / localized + n < 5), `--review` path fixed and exercised both ways, notebook path zero errors with Batch_R added to the compare list, §0 assert passed."

---

## docs/decision_log.md

### Part A — D36 · 2026-10-03 · Classifier two-sample test runs on a canonical content order; liblinear seeded
- **Decision:** `ml.c2st` sorts rows inside each site by their (rounded) feature vector and sites by a content hash before grouped CV, derives the integer group ids from that order, and seeds the liblinear solver with `seed`. Results are bit-identical for identical content under any site relabelling or row permutation; site-level weights and site-level permutation are unchanged. Known-batch material-only numbers move from AUC 0.51 / p 0.458 (Batch 1) and 0.61 / 0.264 (Batch 2) to 0.59 / 0.30 and 0.59 / 0.29; both verdicts, stability shares, attenuation and MDC are unchanged.
- **Why:** E23 showed AUC 0.51 → 0.41 on the same images under new ids because fold assignment and the permutation stream followed site-id sort order. The unseen batch arrives with ids nobody chose, so the classifier cell would otherwise carry an arbitrary component. Seeds 0–9 span AUC 0.34–0.59 / p 0.30–0.73 on Batch 1: the realisation noise is real and now at least reproducible; the corroboration decision (p vs α) did not flip for any seed on either batch.
- **Alternatives rejected:** averaging over several seeds (hides the noise rather than removing the id dependence; can be added as a reported band later); sorting by raw feature values (deterministic but makes fold membership a function of feature magnitude); leave-one-site-out everywhere (removes fold randomness but changes the statistic's power, a method change after the freeze).
- **Caught by:** E23 (C27); closed by E25.

### Part A — D37 · 2026-10-03 · The one drop command gains `--cache-dir` and `--summary`; `--review` fixed
- **Decision:** `python3 -m polaron_qc.report` accepts `--cache-dir DIR` (default unchanged) and `--summary OUT.json` (machine-readable snapshot of the verdict inputs); `_config_hash` serialises the review keys. No change to what the verdict reads.
- **Why:** the rehearsal needs a cold cache without writing into the shared one, and a JSON record lets the drop be diffed against the rehearsal; `--review` had crashed on first use.
- **Caught by:** E25 (C27, C04).

### Part B — checklist addition
- [ ] **C29 — Run every documented operator step, not only the happy path, on a renamed fixture before freezing a procedure.** The `--review` flag was added with the procedure (E23) and documented in README, but first executed in E25, where it crashed before producing a result. A step that is written down and untested is a promise the code does not keep (C04 at the procedure level). (E25)

### Part C — open items
- **Close:** "E23: `ml.c2st` fold assignment and permutation stream depend on site-id order …" → closed by D36 / E25.
- **Open:** under a quality abstention with a positive drift alert, `decide` labels the verdict *investigate — batch-wide drift* even when the localized column is `credible` / `pending_review` (Batch_C: three cracked sites at 2.6–5.8 MAD). Both are "investigate" and the three columns carry the full state, but the label a reader sees first points at the less specific path. Options: prefer the localized label when it is credible or escalated; or print "quality abstention" as the verdict label and keep the paths in the columns. Decision for Santosh / engineering before the drop.
- **Open:** decision stability is printed for abstained batches (Batch_C 0.00 because 2-site folds cannot raise a drift alert). Consider printing "not meaningful under abstention (n < 5)" instead of a share.
- **Open:** regenerate `reports/qc_Batch_1.html`, `reports/qc_Batch_2.html` and notebook 02 so the committed artefacts show the order-independent classifier numbers.

---

## experiments/registry.csv — rows to append

```
E25,2026-10-03,c2st_material_in_pipeline_auc,Batch_3,Batch_1,material14,200 site perms canonical order,17,7,0.588,AUC,p 0.299; before (site-id order) 0.513 / 0.458; renamed copy identical
E25,2026-10-03,c2st_material_in_pipeline_auc,Batch_3,Batch_2,material14,200 site perms canonical order,17,7,0.588,AUC,p 0.294; before 0.613 / 0.264; renamed copy identical
E25,2026-10-03,c2st_material_in_pipeline_auc,Batch_3,Batch_1_renamed_before_fix,material14,200 site perms,17,7,0.395,AUC,p 0.697; same images as Batch_1 under new ids and shuffled rows; pre-fix symptom
E25,2026-10-03,c2st_auc_seed_range_min,Batch_3,Batch_1,material14,seeds 0-9,17,7,0.336,AUC,max 0.588; p 0.299-0.726; no seed below alpha
E25,2026-10-03,c2st_auc_seed_range_min,Batch_3,Batch_2,material14,seeds 0-9,17,7,0.370,AUC,max 0.655; p 0.209-0.677; no seed below alpha
E25,2026-10-03,drop_runtime_cold_s,Batch_3,Batch_R,one command,fixture cold / reference warm,17,7,151.1,s,verdict consistent; stability 1.00 (7/7); low-contrast flags from data on 2 sites
E25,2026-10-03,drop_runtime_cold_s,Batch_3,Batch_Q,one command,fixture cold / reference warm,17,6,123.8,s,quality abstention: grey pore on 4 of 6 sites from BSE p1 rule
E25,2026-10-03,drop_runtime_cold_s,Batch_3,Batch_S,one command,fixture cold / reference warm,17,3,57.2,s,quality abstention: n < 5
E25,2026-10-03,drop_runtime_cold_s,Batch_3,Batch_C,one command,fixture cold / reference warm,17,3,65.1,s,drift alert + localized pending_review (3 sites) + n < 5 abstention
E25,2026-10-03,reference_cold_extraction_s,Batch_3,,features 1.0.1,fresh cache,17,,360.8,s,CPU shared with pytest during the run
E25,2026-10-03,review_confirmed_localized_status,Batch_3,Batch_C,crack_frac,--review c20a68de:crack_frac=yes,17,3,1,credible sites,verdict label unchanged under abstention; 2 sites still pending
E25,2026-10-03,review_refuted_sites,Batch_3,Batch_C,crack_frac+pore_max_d,--review x4 =no,17,3,3,refuted sites,localized column flag; escalation off; stability 3/3
E25,2026-10-03,tests_passed,,,pytest,tests/,,,107,count,96 before this change
```

---

## docs/next_steps.md — row H

| **H** | P0 / medium | Repeat the cold unseen-folder rehearsal with current extraction/quality guards; test new IDs, missing/unavailable evidence and the image-review workflow. Record a delivery receipt and reproducible commands. | Engineering | **Done 2026-10-03 (Claude / engineering), branch `worktree-agent-a668ba1e55c78cc65`, pending merge.** Receipt `analysis/rehearsal_h/receipt.md`; fixtures `analysis/rehearsal_h/make_fixtures.py`; runner `analysis/rehearsal_h/run_rehearsal.py`; log text `analysis/rehearsal_h/LOG_ENTRIES.md` (E25, D36–D37, C29, registry rows). c2st site-id order dependence resolved (D36). Three drop-only defects fixed (`--review` crash, abstention wording, classifier order). Open for decision: verdict label precedence under abstention; stability wording under abstention; regenerate committed reports / notebook. |
