# Delivery receipt — task H: cold unseen-folder rehearsal on the current path (experiment log: proposed E25)

Date 2026-10-03 · owner Claude / engineering · branch `worktree-agent-a668ba1e55c78cc65` (from `79bb912`) · `git describe` at run time `79bb912-dirty` (the commit that carries this receipt is named in the hand-back message).
Scope: README "Unseen batch — the drop procedure" exercised on four renamed known-data fixtures with a fresh feature cache, plus the open item from E23 / D34 (classifier depends on site-id order). Nothing in `CONFIG`, `Thresholds` or `FROZEN_HASH` was changed; nothing was tuned on the fixtures. The fixtures are **known images under new names**: this is a procedure rehearsal, not validation on unseen material.

Hashes: thresholds `b4f4da2e357c` (unchanged) · notebook configuration hash `99d2bbcae6f3` = `FROZEN_HASH` (unchanged; §0 assert passed) · report config hash of a no-review run `efdc7a00eb0b` (a run with `--review` hashes differently: `9c1ce0c1dc83` / `7f495b36c233`, by design — the review is part of the run's configuration) · `FEATURE_VERSION 1.0.1` · feature cache key of the reference in the fresh cache `Batch_3_391781f9150e` (identical to the shared cache's key: the key is content-based, not path-based).

Tests: **96 → 107 passed** (`/opt/anaconda3/bin/python3 -m pytest tests -q`); +6 c2st invariance, +1 c2st determinism, +2 report CLI / summary, +1 config hash with reviews, +1 abstention wording.

## 1 · Reproducible commands (run from the repository root; `python3` = `/opt/anaconda3/bin/python3`)

```bash
# fixtures: hard links to the raw TIFFs under new 8-char ids, outside Dataset/ (git-ignored _fixtures/)
python3 analysis/rehearsal_h/make_fixtures.py                 # Batch_R, Batch_Q, Batch_S, Batch_C (+ <fixture>_map.csv)
mkdir -p _scratch/cache _scratch/out                           # fresh feature cache + outputs (git-ignored)

# reference once into the fresh cache (the real drop has the reference warm and only the new folder cold)
python3 -c "from polaron_qc import features; features.extract_batch('Dataset/Batch_3', cache_dir='_scratch/cache', verbose=False)"

# the one command per fixture (= README step 2 with two optional flags added by this task):
python3 -m polaron_qc.report Dataset/Batch_3 _fixtures/Batch_R _scratch/out/qc_Batch_R.html --cache-dir _scratch/cache --summary _scratch/out/qc_Batch_R.json
#   … same for Batch_Q, Batch_S, Batch_C; or all four, timed and tabulated:
python3 analysis/rehearsal_h/run_rehearsal.py --fixtures _fixtures/Batch_R _fixtures/Batch_Q _fixtures/Batch_S _fixtures/Batch_C --cache _scratch/cache --out _scratch/out

# image review (README step 4) on the fixture that has routed / pending sites
python3 -m polaron_qc.report Dataset/Batch_3 _fixtures/Batch_C _scratch/out/qc_Batch_C_yes.html --cache-dir _scratch/cache --review c20a68de:crack_frac=yes
python3 -m polaron_qc.report Dataset/Batch_3 _fixtures/Batch_C _scratch/out/qc_Batch_C_no.html  --cache-dir _scratch/cache \
        --review c20a68de:crack_frac=no --review c7e5fd06:crack_frac=no --review cb8b735c:crack_frac=no --review c20a68de:pore_max_d=no

# known batches after the c2st change (shared cache, warm)
python3 -m polaron_qc.report Dataset/Batch_3 Dataset/Batch_1 _scratch/out/qc_Batch_1.html --summary _scratch/out/qc_Batch_1.json
python3 -m polaron_qc.report Dataset/Batch_3 Dataset/Batch_2 _scratch/out/qc_Batch_2.html --summary _scratch/out/qc_Batch_2.json

# notebook path (README step 5), without touching the committed notebook or the shared cache: a scratch root with the
# repository layout (polaron_qc symlinked; Dataset/ = symlinks to Batch_1..3 and to _fixtures/Batch_R; analysis_cache/ =
# the committed cache files symlinked + features -> _scratch/cache), generated notebook copied there, "Batch_R" added to
# CONFIG["compare"] with nbformat (the only edit), executed:
python3 notebooks/_build_02_batch_qc.py            # (generator and committed .ipynb verified identical cell by cell first)
cd _scratch/nbroot/notebooks && PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 02_batch_qc.ipynb

# tear-down (originals keep their data; only the links go)
python3 analysis/rehearsal_h/make_fixtures.py --clean
```

Fixture map (deterministic: prefix + sha1(`"<fixture>:<old id>"`)[:7]):

| fixture | new id ← source | | |
|---|---|---|---|
| Batch_R | rff418f5 ← B1/4ih2ggld · rb40a716 ← B1/5n1q8atc · ra878556 ← B1/f1vzngrs · r10cea9a ← B1/ffwubibz | r705e8f8 ← B1/fzrt2k6r · r4a45a9d ← B1/iv6g2oq0 · rdfba6c1 ← B1/uhdslk0o | 7 sites, 21 links, 421 MB shared |
| Batch_Q | qe6c50d6 ← B3/71vgq3fw · q96fa3e1 ← B3/kbdh4tri · q5f9eb48 ← B3/tuy3zymq · qba48513 ← B3/x7u69zsw | qb2fd221 ← B1/4ih2ggld · q464e00b ← B1/5n1q8atc | 6 sites, 18 links |
| Batch_S | s2d0db2f ← B1/f1vzngrs · s365d1e9 ← B1/ffwubibz · sb93bfb7 ← B1/fzrt2k6r | | 3 sites, 9 links |
| Batch_C | c20a68de ← B3/hzumfsms · c7e5fd06 ← B3/0grcilhi · cb8b735c ← B3/ufdvpb81 | | 3 sites, 9 links |

Hard links verified by inode and link count; `Dataset/` listing unchanged before and after (21 / 21 / 51 files).

## 2 · Task 1 — `ml.c2st` independent of site-id order

**Symptom reproduced before any change** (material-only c2st exactly as `build_result` runs it, 200 site permutations, seed 0): Batch_1 vs Batch_3 AUC 0.513 / p 0.458 (= E22); the same seven site tables with new ids and shuffled rows: **0.395 / 0.697**. Batch_2: 0.613 / 0.264 → renamed 0.588 / 0.264. Cause: `np.unique(groups)` fixed the site order by id, and `StratifiedGroupKFold(shuffle=True)` and `rng.permutation(site_y)` consume that order; in addition `LogisticRegression(solver="liblinear")` had no `random_state`, so coefficients differed run to run at the 1e-5 level even for identical inputs.

**Fix** (`polaron_qc/ml.py`): `_canonical_order` sorts rows inside each site by their rounded feature vector (ties: exact values) and sites by (sha1 of the rounded block, exact bytes, label); `c2st` reorders `X, y` and uses the canonical integer site codes as groups, so folds, the permutation stream and the solver see identical arrays for identical content; `_lr_pipeline` is seeded with `seed`. API, site-level weighting and site-level permutation (plan §2.0 rule 6) unchanged; `per_site_scores` carries the caller's ids. The 30-simulation null-uniformity test still passes.

**Tests** (`tests/test_ml.py`): (1) same rows, ids renamed to random 8-char ids that sort differently; (2) same ids, rows permuted within each block, reversed, and (patch shape) interleaved across sites; (3) both at once — each parametrised over patch-level input (4–8 rows per site, repeated group ids) and one row per site; each asserts bit-identical `auc`, `p`, `auc_null`, `cv`, coefficients, selection stability, SHAP and per-site scores. Plus: same call twice is bit-identical and a shifted batch is not a no-op. The first test fails on the pre-fix code (AUC 0.417 vs 0.528).

**Real-data confirmation, before → after** (warm cache; "renamed" = same site tables under new ids, rows shuffled):

| pair | before | before, renamed | after | after, renamed | after via the full one command |
|---|---|---|---|---|---|
| Batch_1 vs Batch_3 | 0.513 / 0.458 | 0.395 / 0.697 | **0.588 / 0.299** | 0.588 / 0.299, `auc_null` bit-identical | 0.588 / 0.299 (73 s) — and Batch_R (the hard-linked copy) 0.588 / 0.299 |
| Batch_2 vs Batch_3 | 0.613 / 0.264 | 0.588 / 0.264 | **0.588 / 0.294** | 0.588 / 0.294, bit-identical | 0.588 / 0.294 (80 s) |

Realisation noise after the fix (seeds 0–9): Batch_1 AUC 0.336–0.588, p 0.299–0.726; Batch_2 AUC 0.370–0.655, p 0.209–0.677; no seed below α on either batch. The corroboration decision is stable; the point estimate carries fold / permutation noise of that size at n = 7 vs 17 with 200 permutations, which the ML block should keep saying.

Everything else in the Batch_1 / Batch_2 one-command runs is unchanged: verdicts *consistent within detectable limits*, stability 1.00 (7/7), attenuation +0.28 / −1.39 and −0.73 / −3.01, MDC 1.75 / 1.75 / 2.00 / 1.75 / 3.00 and 1.75 / 1.75 / 2.00 / 1.50 / 2.50 MAD.

**Figures in the docs that this changes** (not edited here; proposed text in `LOG_ENTRIES.md`): E22 and D30 quote 0.51 / 0.458 and 0.61 / 0.264; E23 / D34 and registry row E23 quote 0.41 / 0.657 for the renamed copy; the committed `reports/qc_Batch_1.html`, `reports/qc_Batch_2.html` and executed `notebooks/02_batch_qc.ipynb` show the old classifier cells (the generator has no hard-coded figure, so a rebuild updates them).

## 3 · Task 2 — rehearsal results

Timings are wall time of the one command, fixture cold, reference warm in the fresh cache, no other heavy job running (the review runs and the known-batch runs overlapped with the notebook execution). Reference cold extraction into the fresh cache: 360.8 s for 17 sites with the test suite running concurrently (E23 measured 55 s for 7 sites cold, i.e. ≈ 8 s / site uncontended; the fixture runs below are consistent with that).

| fixture (sites) | wall / reported | verdict (reason) | drift alert · localized · quality abstention | derived flags (source) | attenuation strat / adj (available) | MDC feasible (MAD) | stability | integration notes |
|---|---|---|---|---|---|---|---|---|
| **Batch_R** (7) | 151.1 s / 144.8 s, 0.34 MB | consistent within detectable limits | no · flag (rff418f5 crack_frac +0.27 MAD, not severe) · no | low contrast on rb40a716, rff418f5 (data); usable n 7 / 7 / 7 / **5** / **5** | +0.28 / −1.39 (yes) | 5/5: 1.75 / 1.75 / 2.00 / 1.75 / 3.00 | 1.00 (7/7) | low contrast on 2 sites → bright KPIs excluded there; no cached material c2st; no novelty rows; drift score not computed for the 2 low-contrast sites; threshold band for 2 of 17 ref sites (known gap) |
| **Batch_Q** (6) | 123.8 s / 117.8 s, 0.30 MB | investigate — batch-wide drift, **"quality abstention: raised black level / grey pores on more than half of the sites"** | no · flag (qb2fd221 +0.27) · **yes** | grey pore ×4 from the BSE p1 rule (source "data"; raised_black_level on the same four), low contrast ×2; usable n 6 / 6 / 6 / 4 / 4 (pore KPIs on fallback threshold for 4) | n/a (no ordinary batch sites → stratified view refuses) / +0.71 (yes) | 5/5: 1.75 / 1.75 / 2.00 / 2.00 / 3.25 | 1.00 (6/6) | grey-pore note on the four new ids; low-contrast note on two; no cached c2st / novelty; drift score not computed for the 2 low-contrast sites |
| **Batch_S** (3) | 57.2 s / 51.5 s, 0.33 MB | investigate — batch-wide drift, **"quality abstention: only 3 sites in batch (< 5); fewer than min usable sites on every primary KPI"** | no · none · **yes** | none, all ordinary; usable n 3 on every KPI | n/a / −2.89 (yes) | 5/5: 2.25 / 2.00 / 2.50 / 2.00 / 3.50 | 1.00 (3/3) | no cached c2st / novelty; threshold-band gap |
| **Batch_C** (3) | 65.1 s / 51.6 s, 0.69 MB (4 anomaly crops) | investigate — batch-wide drift, same n < 5 abstention | **yes** · **pending_review** (3 sites) · **yes** | none, all ordinary | n/a / +0.40 (yes) | 5/5: 2.25 / 2.00 / 2.50 / 2.00 / 3.50 | **0.00 (0/3)** | as Batch_S |

Batch_C local-anomaly table: crack_frac c20a68de +5.39, c7e5fd06 +5.76, cb8b735c +2.57 MAD beyond the ordinary-reference max (all severe, reliable, unreviewed); pore_max_d c20a68de +2.72 (severe), the other two 0.47 / 0.57 (flags only). Escalation: by single site (≥ 3.0), by site agreement (3 ≥ 2) and by KPI agreement (2 ≥ 2) all true. The 0/3 stability: each 2-site fold returns *investigate — localized* (`drift_alert` False at n = 2, localized still pending), the full run *investigate — drift* (`drift_alert` True); a label switch between two investigate paths under abstention, not a change in evidence.

MDC: feasible for every fixture (n_incoming ≤ 7 ≤ n_ref − 3 = 14). The infeasible case (n_incoming > 14 or zero reference MAD) is **not** covered by these fixtures; it was covered by the E22 probe (17 / 18 and 7 / 7 → "not available"). Attenuation: available on all four (the stratified view is NaN whenever a group has < 4 sites or the batch has no ordinary sites; the adjusted view was finite every time, which is what D30 requires for a reject to be considered). Reports: all four render every section (verdict card, primary KPIs with MDC, drivers, evidence images, local-anomaly table, ML, acquisition flags, physics, secondary, limits), 3–7 images each.

### Image-review workflow (README step 4)

Batch_R has no routed or pending site (its only flag is +0.27 MAD, below the 2.0 severity margin, so a review cannot promote it — by design, D32/D33), so Batch_C was used as the task specifies.

| run | wall | verdict | drift · localized · abstention | review states | escalation | stability |
|---|---|---|---|---|---|---|
| no review | 65.1 s | investigate — drift (abstention n < 5) | yes · pending_review · yes | 4 severe flags unreviewed; n_pending 3 | on (single ≥ 3.0, 3 sites, 2 KPIs) | 0.00 (0/3) |
| `--review c20a68de:crack_frac=yes` | 48.1 s | investigate — drift (abstention n < 5) | yes · **credible** · yes | c20a68de crack_frac **confirmed** (n_credible 1), 2 sites still pending | on | 0.00 (0/3): folds → *investigate — localized* with localized credible / credible / pending |
| `--review …=no` on all four severe flags | 41.1 s | investigate — drift (abstention n < 5) | yes · **flag** · yes | **refuted ×3 sites** (listed, closed), 2 non-severe pore_max_d flags unreviewed; n_pending 0 | **off** | 1.00 (3/3) |

The localized outcome column moves exactly as D32 / D33 specify (pending → credible on confirmation; refuted flags closed and still listed; escalation switches off; "what would move it" no longer offers a localized route). The verdict **label** does not move on this fixture because `decide` is in the abstention branch (n = 3) with a positive drift alert, and that branch prefers the drift label; the D33 flip to *investigate — localized* on confirmation therefore cannot be demonstrated with three cracked sites — see "needs a decision" below. The first attempt of both review runs **crashed** (fixed, see §5).

### Notebook path (README step 5)

Generator output and the committed `02_batch_qc.ipynb` are identical cell by cell (36 cells). The scratch copy with `"Batch_R"` appended to `CONFIG["compare"]` executed with **zero error cells in 370 s (36 cells)**; Batch_R built in 63 s → *consistent within detectable limits*, low-contrast flags on rb40a716 / rff418f5 from the data, in-notebook material classifier 0.588 / 0.299 (= CLI), §3c drift plot rendered with the unseen colour, `reports/qc_Batch_R.html` written (0.34 MB). A first attempt failed at §6 on a file missing from my scratch root, see §5 item 7. `FROZEN_HASH` untouched; the §0 print shows configuration hash `99d2bbcae6f3`, the assert passed (the compare list is excluded from the hash, D34).

## 4 · Checklist items run (docs/decision_log.md Part B)

- **C04** code vs prose: README step 4 (`--review`) did not work — fixed with a test; the README's other promises (one command, data-derived flags, abstention on > ½ flagged, n < 5 abstention, notebook compare-list edit) verified by execution.
- **C06** independent units: sites only; patches never counted; the c2st invariance tests cover the patch shape explicitly with site-level weights unchanged.
- **C07** null not read as acceptance: wording stays "consistent within detectable limits" with the MDC on the card; abstentions carry their reason; never "accept".
- **C08** localized path separate: Batch_C shows the localized column and crops independent of the drift column; review changes it independently.
- **C09** null matches the comparison: the canonical order keeps the site-label permutation with the whole CV refit inside each permutation; the null-uniformity test (30 simulations) still passes.
- **C14** usable n: reported per KPI after flags (Batch_R 5 of 7 for bright KPIs; Batch_Q 4 of 6; pore KPIs on fallback for the four grey-pore sites), never the folder count.
- **C16** provenance: thresholds and `FROZEN_HASH` unchanged; fixtures are known data — a rehearsal, stated as such; the c2st change is a determinism fix, not a method change, but it moves two logged numbers, which is recorded.
- **C17** stability, not confidence: shares printed as "n of n re-verdicts"; the Batch_C 0/3 is explained as a label switch under abstention and flagged as misleading wording.
- **C18** register / logs: proposed text for the experiment log, decision log (D36, D37, C29, Part C), registry rows and the next_steps H row in `LOG_ENTRIES.md`; the shared docs were not edited from the worktree.
- **C19** null of the design: no new simulation design was introduced; the existing c2st null test re-run under the new ordering.
- **C21** material KPI list: `MATERIAL_KPIS` untouched; the classifier input is the same 14 columns; no flag column enters.
- **C25** end to end on real data: eight full runs (four fixtures, two reviews, two known batches); missing-input defaults exercised: stratified attenuation NaN → `available` from the adjusted view; no cached c2st → in-pipeline run; no novelty rows → empty section.
- **C27** unseen boundaries: new ids on all fixtures; smaller counts (3, 6, 7 vs 17); confirmed / refuted / unreviewed reviews; infeasible-MDC case not reached by these fixtures (noted); every report rendered.

## 5 · Failures, surprises and fixes

1. **`--review` crashed** (`TypeError: keys must be str … not tuple` in `report._config_hash`, from `json.dumps` on `image_reviewed`'s `(site, kpi)` keys) before any result was assembled — README step 4 had not been executed end to end since it was added in E23. Fix: `_str_keys` serialises tuple keys as `site:kpi`; test `test_config_hash_accepts_image_reviews_and_is_stable_without_them`. A no-review run keeps its hash.
2. **Abstention wording**: "what would move it" read "would become decidable with **at least 3** sites in batch (< 5)" (a blanket `only → at least` rewrite). Now "would become decidable once resolved: only 3 sites in batch (< 5); …"; test added.
3. **Classifier order dependence** (task 1) — fixed; two logged figures change (E22 / D30 / E23 / D34 / registry; committed reports and notebook outputs).
4. **liblinear unseeded** — found by the invariance test (coefficients differed at 1e-5 between identical calls); seeded from `seed`.
5. **Stability share under abstention** can read 0.00 for a batch whose evidence is unchanged fold to fold (Batch_C) — not a defect in the pipeline, but a card a reviewer would misread; listed for decision.
6. **Verdict label precedence under abstention** — drift label shown although three sites sit at 2.6–5.8 MAD beyond the reference max with a confirmed review; the columns carry the state; listed for decision.
7. Notebook rehearsal attempt 1 failed at §6 with `FileNotFoundError: analysis_cache/acquisition_sites.csv` — a defect of my scratch root (I had mirrored only `features`, `ml` and `physics_sites.csv`), not of the procedure; all drop-relevant cells (§0 assert, §1 extraction of Batch_R, §3 three `build_result` calls, §3c drift plot with the unseen colour) had already executed. The committed cache files were linked and the notebook re-executed (result above). Worth knowing for the real drop: notebook 02 reads `analysis_cache/acquisition_sites.csv`, `physics_sites.csv` and `analysis_cache/ml/*` from the repository; the one command reads only the last two and degrades to "not available" without them.
8. The CLI had no way to use a fresh cache (`cache_dir` existed only on `build_result`); `--cache-dir` added so a cold run never writes into the shared cache. `--summary` added so a run leaves a JSON record (used for every table in this receipt).
9. Timing caveat: the reference pre-warm overlapped with the test suite (360.8 s); the four fixture runs did not overlap with anything; the review and known-batch runs overlapped with the notebook execution, so their wall times are upper bounds.

## 6 · What was not done / needs a decision

- Shared docs (`experiment_log.md`, `decision_log.md`, `registry.csv`, `next_steps.md`, README) not edited; text proposed in `LOG_ENTRIES.md`. The log already has an E24, so this is E25.
- `reports/qc_Batch_1.html`, `reports/qc_Batch_2.html` and the committed executed notebook still show the pre-fix classifier numbers; regenerating them is a one-command / one-build job once the branch is merged.
- Decision needed (Santosh / engineering): verdict-label precedence under a quality abstention when the localized column is credible or escalated; whether to print a stability share for abstained batches; whether the ML block should quote the seed range (0.34–0.59 AUC on Batch_1) next to the point estimate.
- Not covered by these fixtures: an incoming batch with n_incoming > n_ref − 3 (MDC "not available" path, covered by the E22 probe), an incoming batch with a genuinely new acquisition signature (no known data can provide one), and the optional `python3 -m polaron_qc.ml` step (exploratory only).

## 7 · Artefacts

Committed: `analysis/rehearsal_h/receipt.md` (this file), `make_fixtures.py`, `run_rehearsal.py`, `LOG_ENTRIES.md`; code and tests listed in §2 and §5. Not committed (git-ignored, left in the worktree for inspection): `_scratch/out/qc_Batch_{R,Q,S,C}.{html,json,log}`, `qc_Batch_C_review_{yes,no}.*`, `qc_Batch_{1,2}_known.*`, `rehearsal_rows*.json`, `material_c2st_{before,after}.json`, `seed_sweep.json`, `nb_exec*.log`, `_scratch/nbroot/notebooks/02_batch_qc.ipynb` (executed scratch copy), `_scratch/cache/` (fresh feature cache). Fixture hard links removed at the end with `make_fixtures.py --clean`; the script stays.
