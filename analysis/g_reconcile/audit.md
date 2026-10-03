# Task G — documentation-vs-code reconciliation audit (2026-10-03)

Scope: every judge-facing or contributor-facing text that states a number, a vocabulary term, a behaviour or a KPI list, checked against the code (`polaron_qc/`), the executed `notebooks/02_batch_qc.ipynb` outputs and the committed `reports/qc_Batch_{1,2}.html`. No code behaviour, threshold, Part A log entry or `.ipynb` cell was changed. Raw data are not available in this worktree, so the notebook and reports were **not** regenerated: the generator markdown and `report.py` prose strings are fixed at source and the owner must rebuild + execute once (see "Follow-up").

Sources of truth used: `polaron_qc/__init__.py` (PRIMARY_KPIS = crack_frac, pore_max_d, pore_frac, bright_frac, bright_d50; MATERIAL_KPIS 14 entries; KPI_TRUST incl. `exploratory` for the eight BATTERY_SECONDARY_KPIS; BATCH_COLORS / UNSEEN_COLOR), `decision.py` (Thresholds α 0.05, min_effect 1.0, consistency 0.5, min usable 5, severity 2.0, strong attenuation 0.5, reject ≥ 2 credible sites ≥ 2.0 MAD, escalate 3.0, agreement 2 / 2; hash `b4f4da2e357c`; verdict strings; localized ∈ none / flag / review_routed / pending_review / credible; D45 abstention label rule; `attenuation.available` gate), `report.py` (DEFAULT_CONFIG with c2st_in_pipeline, c2st_seed_sensitivity 3, acquisition_views, image_reviewed; 11 sections; CLI `--review / --no-images / --cache-dir / --summary`), `features.FEATURE_VERSION 1.1.0`, `stats.mdc` (min_remaining 3 → feasible / reason), `stats.usable_n` (ORs boolean flag columns), `ml.c2st` (canonical content order, seeded liblinear), `pytest --collect-only` = **174 tests** (acquisition 9, battery_metrics 19, decision 20, features 11, ml 17, physics 15, report 14, secondary 5, stats 23, void_metrics 6), notebook outputs (sites table 150 cols; config hash 99d2bbcae6f3 = FROZEN_HASH; thresholds b4f4da2e357c; c2st in-pipeline 0.588 / 0.299 and 0.588 / 0.294; cached 0.513 / 0.458 and 0.613 / 0.264; MDC 1.75 / 1.75 / 2.00 / 1.75 / 3.00 (B1) and 1.75 / 1.75 / 2.00 / 1.50 / 2.50 (B2); attenuation +0.28 / −1.39 and −0.73 / −3.01; split diagnostics drift 0.050, flag 0.917, routed 0.083, pending-or-credible 0.700; ordinary 5-v-5: 0.033 / 0.800 / 0.167 / 0.333; build 60 s / 70 s), report text (footer `config hash ad1f277afcc8`, `git a05860f-dirty`, features v1.1.0, seed sensitivity 4 seeds).

## 1. Discrepancy table

Severity: **J** = judge-visible (README, findings, notebook, reports, plan), **C** = contributor-only. Action: fixed / proposed / historical-kept.

| file:line (before edit) | claim as written | current truth (reference) | action | sev |
|---|---|---|---|---|
| README.md:3 | "return an interpretable **accept / investigate / reject** verdict against an approved baseline" | top outcome is "consistent with working reference (within detectable limits)"; `decision.CONSISTENT`; CLAUDE.md style rule | fixed: verdict vocabulary stated, task framing kept in a parenthesis | J |
| README.md:37 | step 2 lists the first screen; no mention of `--summary` / `--cache-dir` / `--no-images`; "thresholds hash" only | `report.main()` has all four flags (D44); footer prints two hashes | fixed: flags added; two-hash explanation added | J |
| README.md:55 | "rehearsed on a renamed known batch (E23) … Next: … a fresh delivery rehearsal" | E28 cold rehearsal done on five fixtures (next_steps H, receipt.md); 174 tests | fixed | J |
| README.md:67 | "recommends two unrun audits" | E26G and E27 both run and integrated (next_steps E/F rows) | fixed | J |
| README.md:70 | "Tasks remain unclaimed until selected" | E, F, G, H, J, K claimed in next_steps table | fixed | C |
| CLAUDE.md:17–37 | repository layout omits `polaron_qc/`, `tests/`, `reports/`, notebook 02 and its generator, `analysis_cache/features`, `analysis/` sub-folders; "ids A1–A15" | all exist; register cards A1–A18 (`_build_assumption_register.py`) | fixed | C |
| CLAUDE.md:65 | "`bright_bands()` in the build script does this" | `features.bright_bands` (also duplicated in `ml.py:242`, `physics.py:827`, workflow hazard 6) | fixed | C |
| CLAUDE.md:39–50 | only notebook 01 build/run instructions | notebook 02 has its own generator; generator alone overwrites the executed .ipynb | fixed: one bullet added | C |
| docs/problem_and_findings.md:90–93 (§3.5 table) | energy statistics 0.98 / 0.70 / 2.34 and 0.89 / 1.53 / 3.56 | pipeline statistics 1.700 / 1.220 / 4.059 and 1.535 / 2.654 / 6.157 (notebook §5, reports); same p and attenuation; ratio ≈ 1.73 = consequence weighting | fixed: note added that the table holds unweighted E15 values | J |
| docs/problem_and_findings.md:69–82 (§4 catalogue) | no row for the eight battery-secondary KPIs; `pore_d90`, `pore_count_per_Mpx`, `bright_max_d` absent; no statement of which lists the code uses | `KPI_TRUST` exploratory for BATTERY_SECONDARY_KPIS; those three are `high`; PRIMARY/MATERIAL/TRUSTED lists in code | fixed: row + list paragraph added | J |
| docs/problem_and_findings.md:78 | `etd_crack_density_particles` trust "measurement/interpretation review pending; no intact-particle fraction" | `polaron_qc.KPI_TRUST` = `high`; `report.KPI_TRUST` = "high; null on this data"; in MATERIAL_KPIS and LOCAL_KPIS (non-promoting) | **proposed** (which wording stands; code change after freeze needs a D-entry) — noted inline and in decision_log Part C | J |
| docs/problem_and_findings.md:128–135 (§7) | "Decision: accept if …; Confidence from bootstrap agreement"; "KS tests" | D16 (never "accept"), D18 ("decision stability"), D21/D23 (HL permutation, Holm) | fixed: section retitled "superseded — kept for provenance" with a note; body kept | J |
| docs/problem_and_findings.md:139 | "existing suite passes (89 tests)" | 174 now (E21 wording is a dated status) | historical-kept; current count added at end of §8 | J |
| docs/problem_and_findings.md:146 | "Decision stability currently reruns … holding cached classifier fixed; the classifier still needs a separate pre-run" (present tense) | refit per fold, in-pipeline since D30 (`stability.refit_per_fold`, `held_fixed = []`) | fixed: past tense, pointer to the fix | J |
| docs/problem_and_findings.md:148 | E22 numbers only (AUC 0.51 / 0.61) | E28 / D43 moved them to 0.59 / 0.30 and 0.59 / 0.29 | fixed: sentence added | J |
| docs/problem_and_findings.md:197 | "142 tests" | 174 | historical-kept (E25 verification record inside a dated section) | C |
| docs/qc_plan.md:3 | "Status: agreed; build not started" | implemented | fixed | J |
| docs/qc_plan.md:30 (§2.0 rule 5) | severity "at least the ordinary-reference MAD of per-site maxima"; reliability "stable under the ±5-level threshold band" | `severity_margin_mad = 2.0` (D29), tiered D33; `check_b` applies only low-contrast (hard) / grey-pore (soft) flags, no band condition | fixed (2.0 MAD, D33); band condition marked **not wired** and listed as proposed | J |
| docs/qc_plan.md:64 (§2.4) | local KPIs named in prose only | `decision.LOCAL_KPIS`; promotion only crack_frac, pore_max_d (D29) | fixed: implemented lists named | J |
| docs/qc_plan.md:71 (§2.5) | "GroupKFold" | `StratifiedGroupKFold`, canonical content order (D43) | fixed | C |
| docs/qc_plan.md:82 (§2.6) | adjusted on "bright-phase separation, black level, stretch fraction, curtaining, boundary sharpness" | `three_views(covariates=['bright_sep','bse_p1','etd_boundary_sharpness'])` + 7-covariate ridge variant (D26) | fixed | J |
| docs/qc_plan.md:109–118 (§2.7) | Check A has four gates; Check B accepts novelty-map flags; verdict table: localized needs "credible"; reject "A(i)–(iv) and §2.6 views agree" | gate (v) acquisition views available (D30); novelty not a Check B input (D38); pending + escalated also yields investigate-localized (D33); abstention label rule (D45); reject via B needs ≥ 2 credible sites ≥ 2.0 MAD | fixed | J |
| docs/qc_plan.md:146,148 (§3) | "false-alarm rate", "empirical false-alarm rate" | reclassified as reference-split diagnostics (plan review log, notebook §9) | fixed | J |
| docs/qc_plan.md:131 (§2.7) | "Provisional tolerances derived from the physics layer are proposed for discussion" | no tolerances exist anywhere in code or docs | **proposed**: either delete the sentence or record the proposed values; not a mechanical fix | J |
| docs/workflow.md:21 | "Assumption register A1–A15" | A1–A18 | fixed | C |
| docs/workflow.md:77 (M1) | no mention of canonical order / seed range | D43, D45 | fixed: label extended | C |
| docs/workflow.md:144 | `M2 -.->|"flag only; forces investigate only if no trusted KPI moves"| DB` | novelty never enters `check_b` (D38; plan §2.5) | fixed: label now "exploratory evidence only; not a Check B input (D38)" | C |
| docs/workflow.md:131 (DB) and edge `A5 -.->|"reliability condition"| DB` | Check B reliability includes "inside ±5-level band" | not implemented in `check_b` | **proposed** (same decision as plan §2.0 rule 5b); diagram left as the plan's promise | C |
| docs/workflow.md:229 | "(owner to assign) KPI_TRUST … proposed home polaron_qc/__init__.py" | lives in `polaron_qc/__init__.py` with `exploratory` level and assertions | fixed | C |
| docs/workflow.md:233–244 | report-sections table lacks "Battery geometry candidates" | section id `battery-secondary` exists (11 sections) | fixed: row added | C |
| docs/workflow.md:250 (hazard 4) | "Must be computed per site and printed next to every pore-based shift" | computed for all 31 sites in `acquisition_sites.csv`; report's physics table reads `physics_sites.csv` (6 of 31 with a band) → "available for 4 of 24 sites" | fixed wording; code change **proposed** (point the report at `acquisition_sites.csv`) | J (the note is printed in every report) |
| docs/workflow.md:257–262 | hazards numbered 13–16 before 11–12 | — | fixed: reordered, numbers kept (code cites "hazard 1/4/6") | C |
| docs/workflow.md:275 | "Feature version1.1.0" | typo | fixed | C |
| docs/workflow.md:209 | `sites` 150 cols = 87 original + secondary block | notebook: (17, 150); report docstring agrees | verified, no change | — |
| docs/workflow.md:219 | mdc infeasible when n_incoming > n_ref − 3 | `stats.mdc(min_remaining=3)` | verified | — |
| docs/workflow.md:216, 222, 228 | three_views / c2st / verdict contracts | signatures via `inspect.signature` match (covariates default, variant name, localized values, escalation keys) | verified | — |
| docs/experiment_log.md Part B | "comparison engine ◕ … gap: notebook 02 sections 3–9" | notebook executed through §11 | fixed | C |
| docs/experiment_log.md Part B | decision layer "14 tests" | 20 (`test_decision.py`) | fixed | C |
| docs/experiment_log.md Part B | "96 tests" | 174 | fixed | C |
| docs/experiment_log.md Part B | "Unseen batch: … rehearsed (E23)" | + E28 cold rehearsal | fixed | C |
| docs/decision_log.md Part C | "Owner for KPI_TRUST"; "polaron_qc.acquisition has no owner yet" | both resolved | fixed: marked resolved; four G-audit items appended | C |
| docs/next_steps.md "What is already done"; G row | no mention of H completion or the audit | H done (E28); this audit | fixed | C |
| notebooks/_build_02_batch_qc.py:6 | "All ten sections" | sections 0–11 | fixed | C |
| notebooks/_build_02_batch_qc.py:122–124 (§2b) | "*(Filled in after §4 is wired; placeholder.)*" + "# placeholder — wired after the decision pipeline function exists" | split diagnostics run in §9 (outputs present: 60 splits, rates tabulated) | fixed: markdown and comment point to §9; `SPLIT_DIAG = None` kept (no code change) | J |
| notebooks/_build_02_batch_qc.py:172 (§3d) | "severity margin ≥ 1 MAD" | 2.0 MAD (`Thresholds.severity_margin_mad`, D29); tiered D33; report prints "≥ 2.0 MAD" | fixed | **J (most visible numeric error found)** |
| notebooks/_build_02_batch_qc.py:191 (§4) | cached run shown "for comparison" without saying why its numbers differ | cached 0.513 / 0.458 predates D43; in-pipeline 0.588 / 0.299 | fixed: note added | J |
| notebooks/_build_02_batch_qc.py:259 (§8) | report contents listed without the battery-geometry block or the hash caveat | reports have a `battery-secondary` section; footer hash ≠ FROZEN_HASH | fixed | J |
| notebooks/02_batch_qc.ipynb §3d markdown (executed copy) | same "≥ 1 MAD" text | — | **not edited** (generated file); regenerate + execute | J |
| polaron_qc/report.py:7–8 (docstring) | "ml and physics caches read from disk" | classifier + acquisition views in-pipeline (D30) | fixed | C |
| polaron_qc/report.py:99 | "TO BE MOVED to polaron_qc/__init__.py once the owner is assigned" | categorical dict exists there; report keeps labels | fixed | C |
| polaron_qc/report.py:492 (integration note printed in reports) | "features.threshold_band(site) for every site is still 'to add' in workflow.md" | `acquisition.threshold_bands` exists, 31 sites cached | fixed wording | J |
| polaron_qc/report.py:1238 (physics section) | "a per-site band for every site is still to be added to features" | same | fixed wording | J |
| polaron_qc/report.py:1154 (c2st None placeholder) | "not available in analysis_cache/ml for this pair (no material-only run has been cached…)" used for the in-pipeline block too | material block is in-pipeline; absence means it did not run | fixed: generic wording + "never counts as corroboration" | J |
| polaron_qc/report.py:1303 (footer) | "config hash ad1f277afcc8" reads as *the* config hash | it is `_config_hash(cfg_public)` of build_result settings; the frozen notebook hash is 99d2bbcae6f3; E28 saw efdc7a00eb0b before D45 added a key | fixed label ("run-config hash … not the notebook's frozen configuration hash"); whether to print FROZEN_HASH too is **proposed** | J |
| polaron_qc/report.py:1390 (main docstring) | omits `--summary` | flag exists | fixed | C |
| reports/qc_Batch_{1,2}.html | carry the three stale strings above and the old footer label | — | **not edited** (generated); regenerate | J |
| docs/_build_assumption_register.py | card texts on pipeline behaviour (A7 thresholds, A10 ridge coverage, A16/A18 secondary status) | match code; confirmed statuses limited to A3, A13, A15, A17 as findings §4b states; no report/notebook text claims a mask/phase/defect confirmation beyond these | verified, no change | — |
| README.md:51, notebook §0/§7/§10, report verdict card | verdict strings, five localized values, "decision stability", "routed to review", MDC "not available", unavailable ≠ zero | match `decision.py` / `render_report` | verified | — |
| CLAUDE.md Plots; `__init__.BATCH_COLORS` / `UNSEEN_COLOR`; notebook §2/§3c; report strip plot | colours #2a78d6 / #eb6834 / #1baf7a / #eda100 / #e34948 ring / #6f42c1 | match | verified | — |

Counts: **48 rows**; 40 fixed (mechanical), 5 proposed (judgement), 3 historical-kept; 8 verified-no-change rows listed for completeness.

## 2. Proposed (judgement calls for the coordinating owner)

1. **±5-level threshold band as a Check B reliability condition** — promised in plan §2.0 rule 5(b) and workflow node DB / edge A5→DB; `decision.check_b` does not implement it. Either wire it (behaviour change after the freeze: D-entry, tests, re-run both known batches) or remove the promise. Plan text now says "not wired".
2. **Report band table source** — `_read_physics` reads `physics_sites.csv` (6 of 31 sites with a band) while `acquisition_sites.csv` has all 31; every report prints "available for 4 of 24 sites". Reading the acquisition cache is a small code change with no verdict impact.
3. **`etd_crack_density_particles` trust wording** — findings catalogue says "review pending", code says `high`, report says "high; null on this data". It is a classifier input (MATERIAL_KPIS) and a non-promoting local KPI. Pick one; a code-level change after the freeze needs a D-entry.
4. **Two "config hash" values** — footer now says "run-config hash … not the notebook's frozen configuration hash". Decide whether the report should additionally print FROZEN_HASH (it would couple report.py to the notebook constant) or whether the README note suffices.
5. **qc_plan §2.7 "Provisional tolerances derived from the physics layer are proposed for discussion"** — none exist in code or docs; delete or record.

## 3. Coherence — where the judge-facing story is told more than once

The same story is currently told four times with different emphasis:

- **README** leads with the one command and the drop procedure, then verdict meaning, then status, then three appended paragraphs on battery geometry, ML review and Gabor (exploratory material gets as much space as the decision path).
- **findings doc** leads with the problem and data, then the KPI catalogue, then sixteen numbered sections in chronological order (E18 morphology, E24 benchmark, D36 battery hypotheses, E25, D38, E26G, E28J); the decision pipeline itself appears only as §8 "implementation review status". A judge reading top-down meets the exploratory audits before the verdict logic.
- **notebook 02** has the right order (§0 config → §1–§3 reference and comparison → §4–§6 evidence → §7 decision → §8 reports → §9 self-tests → §10 unseen → §11 exploratory) but omits the battery-geometry block that the HTML shows, and its §3b "secondary KPIs" (9 material KPIs) differ from the HTML "Secondary KPIs" appendix (14 trusted KPIs) and from the "Battery geometry candidates" (8 exploratory).
- **HTML report** is the only artefact with all 11 sections in decision order; it prints a different "config hash" from the notebook.

**Proposed single ordering** (apply to README "Status"/top, findings §1 lead-in, notebook §0 preamble; the HTML already follows it):

1. *Primary decision screen*: verdict string, the three outcome columns, decision stability, drivers, usable n per primary KPI, what would move it, thresholds hash + provenance. Both known batches: *consistent within detectable limits*, stability 1.00 (7/7), no driver, Batch 1 one evidence flag (4ih2ggld crack_frac +0.27 MAD, not severe).
2. *What could not have been seen*: the five primary KPIs with MDC (1.75–3.00 MAD at n = 5–7), the reference's own sub-populations (10 ordinary / 4 grey-pore / 3 cracked), the ±5-level band.
3. *Evidence for the decision*: painted voids / outlined particles, local-anomaly table, material-only classifier (0.59 / 0.30, seed range 0.40–0.59 / 0.30–0.71, corroboration on no seed), three acquisition views (+0.28 / −1.39; −0.73 / −3.01, read as "not explained away"), data-derived flags.
4. *Self-tests*: reference-split diagnostics (drift 0.05 vs α 0.05; localized flags 0.92 on 7-v-10 splits because 82 % of draws contain a cracked site; ordinary-only 5-v-5: 0.03 / 0.80 / 0.17 / 0.33), synthetic shifts (pore_frac ×2 and bright_frac ×1.5 fire the drift path; crack_frac ×2 fires the localized path), heterogeneity check (cracked-3 vs ordinary-10 → pending_review at 5.8 / 5.4 / 2.6 MAD).
5. *Exploratory evidence, clearly labelled and never a verdict input*: secondary-KPI appendix (descriptive), battery geometry candidates (8 KPIs, `exploratory`), DINOv2 novelty (tracks acquisition), flag-inclusive classifier (etd_boundary_sharpness carries Batch 1), E26G graph, E27 balanced encoder, E28J Gabor, morphology atlas and annotation pack.
6. *Limits and next actions*: usable n, specimen independence unconfirmed, no tolerances → no "accept", expert mask review pending (A/D), metadata (B), drop procedure (one command, `--review`, `--summary`).

Concretely: move README's three exploratory paragraphs under one heading "Exploratory evidence (not verdict inputs)" after the drop procedure; add a six-line "Where the verdict comes from" box at the top of findings §1 pointing to notebook §7 and the reports; add one sentence to notebook §8 (done) so the battery block is discoverable; keep the HTML as is.

## 4. Follow-up the owner must run (needs raw `Dataset/`)

```bash
/opt/anaconda3/bin/python3 notebooks/_build_02_batch_qc.py
cd notebooks && PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 02_batch_qc.ipynb
```

This regenerates `02_batch_qc.ipynb` (markdown fixes: §2b, §3d, §4, §8) and rewrites `reports/qc_Batch_{1,2}.html` with the corrected integration note, physics-band sentence, classifier placeholder and footer label. Expected: identical verdicts, primary tables, hashes (`99d2bbcae6f3` / `b4f4da2e357c`); only prose changes. Then commit `.ipynb`, generator and reports together (CLAUDE.md Git rule).

Checklist items applied to this audit (decision_log Part B): C04 (code keeps the prose's promises — the ±5-band condition and the band-table source are the two it does not), C07, C14, C16, C17, C21 (lists audited against `KPI_TRUST`), C25 (missing-input defaults re-read in `decide`), C27.
