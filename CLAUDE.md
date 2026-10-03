# CLAUDE.md — Polaron SEM batch-drift QC

Guidance for AI agents working in this repository. `AGENTS.md` points here.

## What this project is

A hackathon entry (started 2026-10-03). Given SEM cross-sections of battery electrode coating from several batches, build an interpretable, uncertainty-aware QC system that compares an incoming batch against an approved baseline and returns **accept / investigate / reject** (the task's framing; our top outcome is "consistent within detectable limits", never "accept" — see Style) with an explanation a materials expert can verify. An unseen batch arrives mid-event. Judged on KPI quality, accuracy on the unseen batch, interpretability, honesty about uncertainty, and usability — not raw accuracy.

Read `docs/problem_and_findings.md` before doing anything substantive. It holds the problem statement, everything we know about the data, the KPI catalogue with trust levels, decisions taken, and open questions.

Provider clarification (2026-10-03, D49P): Batch 3 is the supplier's promised baseline; Batches 1/2 are later variations, not better/worse labels. The challenge target is morphology-based in/out-of-distribution comparison. Battery-harm proof is not a prerequisite for a distribution feature. Preserve the full measurable reference, with disclosed feature-quality exclusions; an ordinary-only subset is sensitivity/context, not the promised distribution. `docs/qc_plan.md` §1.1 describes a planned bounded OOD extension. The five-primary-KPI restrictions below describe the current executed release, not an immutable feature limit; promotion still requires a logged validation/version decision before unseen evaluation. Defect risk, acquisition uncertainty and distribution conformance are separate conclusions.

Evaluation clarification (D53): every held-back sample must receive a batch bet, even if very uncertain, with
confidence and feature explanations. Correct identification measures “different from baseline, and in what
way.” Use the existing E31 combined argmax for the first organiser drop, with uncalibrated model scores and
separate morphology/quality context; fit stability is not probability of correctness. Preserve first predictions
before feedback and do not tune using the drop. A test folder can mix source batches, so do not pool it as one
production batch. `analysis/organiser_drop_01/` holds the protocol, pre-prediction snapshot and outputs.

D52/D54S supersede the first-drop combined-family rule for future judged submissions: the declared v2 primary
is morphology plus acquisition-sensitive ETD/Inlens appearance, with frame height excluded from every family
and explicit session statistics only in comparators. `analysis/submission_v2/` preserves known-only fitted
models and matching reliability evidence. Require the explicit model version, never fall back to combined,
and write new output directories. Texture origin remains unresolved; removing explicit acquisition columns
does not establish material-only evidence. V2 was revised after first-drop inspection and before its truth,
and declared before final evaluation. Save truth separately and evaluate saved bets before any new version.

## Team split

- **Santosh**: data inspection, materials sanity-check of KPIs, narrative and judging story, organiser liaison.
- **Claude**: loader, KPI extraction, statistics, decision logic, notebook skeleton.
Do not silently take over the other side's stage; propose, then do.

## Repository layout

```
Dataset/                 raw TIFFs (git-ignored, 1.6 GB). Batch_{1,2,3}/img_<site>_<BSE|ETD|SE|Inlens>.tif
notebooks/
  01_dataset_analysis.ipynb        executed EDA notebook (the judge-facing artefact for section "data")
  _build_01_dataset_analysis.py    generator: writes the .ipynb from code/markdown cells
  02_batch_qc.ipynb                executed QC notebook (reference characterisation, comparisons, decision, self-tests; §0 asserts FROZEN_HASH)
  _build_02_batch_qc.py            its generator (edit this, never the .ipynb)
polaron_qc/              the QC package: features (FEATURE_VERSION 1.1.0), stats, ml, acquisition, physics, decision (Thresholds, hashed),
                         report (build_result / render_report / CLI), secondary + battery_metrics + void_metrics (exploratory battery geometry).
                         KPI lists and trust levels live in polaron_qc/__init__.py (PRIMARY_KPIS, MATERIAL_KPIS, KPI_TRUST, BATTERY_SECONDARY_KPIS)
tests/                   pytest suite (173 tests): `/opt/anaconda3/bin/python3 -m pytest tests -q`
reports/                 qc_Batch_1.html, qc_Batch_2.html — the per-batch judge-facing reports written by notebook 02 / the CLI
analysis_cache/          per-site / per-image / per-particle feature CSVs produced by notebook 01 (committed; small); analysis_cache/features/
                         is notebook 02's content-hashed parquet cache (ignored by version control); analysis_cache/ml/ holds the exploratory embeddings (committed)
analysis/                parallel label-free audit (scripts, findings.md, report.html, assets). Integrity checks, intensity/texture
                         proxies with bootstrap CIs. Complements notebook 01; do not duplicate its checks, cite them.
                         Sub-folders: battery/ (E25), morphology/ (E18/E24, atlas + annotation pack), ml_options/ (E26G/E27/E28J),
                         qc_review/ (E21 probes), rehearsal_h/ (E28 drop rehearsal), g_reconcile/ (documentation-vs-code audit)
docs/problem_and_findings.md       state of knowledge — keep it current
docs/qc_plan.md                    agreed plan for the QC notebook (components, methods, decision logic, checkpoints)
docs/workflow.md                   Mermaid diagram of the whole pipeline + data contracts between modules; update when an interface changes
docs/experiment_log.md             Part A: ledger of experiments with numbers; Part B: progress scoreboard vs the target.
experiments/registry.csv           one row per reported metric (experiment, metric, reference, batch, kpi, value) — append, never edit
docs/decision_log.md               Part A: every consequential decision with rationale; Part B: pre-presentation review
                                   checklist built from reviewer catches; Part C: open items. Append, never delete.
docs/assumption_register.html      self-contained HTML: every interpretive assumption with an annotated example image and a
                                   review status; rebuild with `python3 docs/_build_assumption_register.py` when assumptions change
                                   (ids A1–A18 are referenced from other docs; append, don't renumber)
```

## How notebooks are built and run

Notebooks are **generated from a Python script**, then executed in place. Edit the `_build_*.py` file, never the `.ipynb` directly, or edits will be lost on the next build.

```bash
python3 notebooks/_build_01_dataset_analysis.py
cd notebooks && PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 01_dataset_analysis.ipynb
```

- Feature cells are cached in `analysis_cache/`. If you change a feature definition, delete the corresponding CSV/NPZ before re-executing or you will get stale numbers. `image_quality.csv` is cheap; `site_features.csv`, `bright_particles.csv`, `fft_spectra.npz`, `bse_histograms.npz` take ~5 min; `etd_inlens_features.csv` ~2 min.
- Full execution from empty cache is ~8 min on the M2 Max. Use a long timeout.
- Notebook 02 is built and executed the same way (`python3 notebooks/_build_02_batch_qc.py`, then nbconvert on `02_batch_qc.ipynb`); it needs the raw `Dataset/` folder (≈ 60–70 s per compared batch with a warm `analysis_cache/features/`, ≈ 6 min for the reference cold) and writes `reports/qc_<batch>.html`. Running the generator alone overwrites the executed `.ipynb` with an unexecuted one, so always execute after generating.
- After executing, check for errors programmatically (loop over cells for `output_type == "error"`) and look at the figures. Numbers quoted in markdown cells must match the executed outputs; re-read the outputs before writing prose.

Environment: anaconda `python3` at `/opt/anaconda3/bin/python3` (3.12), numpy 1.26, scipy, scikit-image 0.23, scikit-learn 1.4, pandas 2.2, matplotlib, statsmodels, tifffile, torch 2.9 with MPS. No OpenCV. Kernel name `python3`.

## Hard-won rules about this data

1. **Never use raw intensity of any channel as a material KPI.** Half the images were contrast-stretched after acquisition; the Inlens channel is charging-dominated. Intensity statistics go in an "instrument flags" bucket.
2. **All lengths are pixels.** Microscope settings did not survive; TIFF resolution tags give a nominal 25.0 nm/px (export metadata, unverified). Say "px", optionally "(≈ x µm nominal)", and only compare relatively.
3. **Segment BSE with per-image histogram-anchored thresholds**, never a fixed value and not plain multi-Otsu (it failed on low-contrast images, inflating the bright fraction 3×). The current scheme: graphite mode → valley to the bright mode when resolved, else mode + 3.5 σ with `bright_low_contrast = True`.
4. **Known sub-populations** that must be handled explicitly, not averaged in:
   - Batch 1 low-contrast sites `4ih2ggld`, `5n1q8atc` → bright-phase KPIs unreliable.
   - Batch 3 grey-pore group `71vgq3fw`, `kbdh4tri`, `tuy3zymq`, `x7u69zsw` → pore KPIs on fallback threshold; likely different preparation/session.
   - Batch 3 cracked sites `hzumfsms`, `0grcilhi`, `ufdvpb81` → the only clear material anomaly (large delamination-like voids).
5. **Inlens intra-particle texture is confounded** with Inlens brightness (~75 % of variance). It may not drive a verdict without local-contrast normalisation and must be labelled as a candidate.
6. **Detector label `SE` means `ETD`.** Merge them. Channels are pixel-aligned; BSE masks can be applied to ETD/Inlens directly.
7. **Trim bright edge bands** (current collector / stitching) before measuring; `features.bright_bands()` does this (copies still exist in `ml.py` and `physics.py`, workflow hazard 6).
8. **Per-image percentile thresholds make densities constant by construction.** Use one absolute threshold chosen across sites (see ETD ridge threshold `T_STAR`).
9. **Batch 3 is the working reference, not a clean baseline** (confirmed by the problem providers: three supplier batches of one product; Batch 3 is one batch with more samples; the task is to differentiate). Use robust statistics, show Batch 3's own sub-populations, report all pairwise comparisons, and keep the reference selectable in one config cell.
10. Frame height is a session fingerprint as much as a thickness proxy; do not present it as thickness without caveat.
11. **Fresh graphite–Si/SiOx electrode is confirmed by Santosh (2026-10-03).** Interpret features as as-manufactured structure and possible susceptibility during formation/later cycling, never evidence of cycling-induced damage, SEI or lithium plating. Exact Si versus SiOx chemistry, recipe, binder identity and pixelwise chemical labels remain unspecified. A bright threshold fragment is not automatically a Si/SiOx particle; residual-solid adjacency is not electrical contact. See `docs/battery_microstructure_review.md` for the hypothesis checks and wording audit.

## Style for judge-facing material

- Every KPI gets a one-line materials interpretation and a trust level.
- Every verdict must list its drivers, its decision stability (not "confidence" or "probability correct"), the acquisition flags, and what would move it. "Consistent with the working reference, within detectable limits" is the top outcome; never write "accept" unless an equivalence test against agreed tolerances has been run. Batch-wide drift and localized defects are separate decision paths.
- Small-sample discipline (docs/qc_plan.md §2.0): five primary KPIs (observed 2-D measurements, not stereological estimates) carry verdicts; usable site counts after flags, not folder counts; site-level permutation tests with all reference-dependent fitting inside the loop; jackknife stability; patches never count as n; exceeding the reference maximum is an evidence flag, not a defect call; provenance wording is "developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived".
- Physics statements are qualitative and relative (direction of change), never performance percentages; 2-D tortuosity is a section index, not a 3-D bound.
- Missing verdict inputs default to the conservative side, never to "no evidence against": no classifier run → no Check A reject; no acquisition view (`attenuation.available` False) → no drift reject; MDC infeasible → "not available", never ∞; image review has three states (unreviewed = pending, confirmed = credible, refuted = closed). Localized rule is tiered (D33): one site between 2 and 3 MAD beyond the ordinary max is routed to review without changing the verdict; 3 MAD, two sites, both promotable KPIs or a confirmed review flip it to investigate. Acquisition flags come from the batch's own images (`acquisition.derive_flags`) and are written into the site tables before any statistic runs; the constant site lists are a cross-check for the reference only.
- Show the evidence on the image (painted voids, particle outlines) when a KPI drives a decision.
- State uncertainty in numbers (bootstrap intervals, n sites), not adjectives.
- If a feature looks too good, test it for an acquisition confound before reporting it.

## Plots

Batch colours are fixed: Batch 1 `#2a78d6`, Batch 2 `#eb6834`, Batch 3 `#1baf7a`, Batch 3 grey-pore group `#eda100`, low-contrast marker `#e34948` ring, any other (unseen) batch `UNSEEN_COLOR` `#6f42c1`. Never index `BATCH_COLORS[b]`; use `.get(b, UNSEEN_COLOR)` (an unknown folder name crashed the notebook in the drop rehearsal, E23). One y-axis per plot, legend whenever ≥ 2 series, bar = median in strip plots.

## Git

- `Dataset/` is ignored. Do not add raw TIFFs or any file over ~20 MB.
- Executed notebooks are committed (judges read them). Keep `.ipynb` and its `_build_*.py` in sync in the same commit.
- Commit only when asked. Commit messages end with the attribution line required by the session.

## Before presenting anything

Run the checklist in `docs/decision_log.md` Part B against the plan, result, figure or verdict you are about to present, and say which items you checked. When you take a consequential decision, append it to Part A with rationale and alternatives. When you run an experiment or produce a number you will quote, add an entry to `docs/experiment_log.md` Part A and rows to `experiments/registry.csv`, and update the Part B scoreboard if the status of a capability changed. When a reviewer catches something the checklist did not cover, add a checklist item in the same change.

## Keeping docs current

When a finding changes, update `docs/problem_and_findings.md` (and this file if a rule changes) in the same change as the code. When an interpretation changes or the organisers answer a question, update the corresponding card in `docs/_build_assumption_register.py` and rebuild the HTML. The findings doc is what a teammate or judge reads first.

Morphology experiments also have a live inventory: `analysis/morphology/metric_register.json`.
When adding a metric or changing its status, update its definition, evidence, review
and history there, then regenerate `docs/morphology_metrics.md` and
`analysis/morphology/output/metric_atlas.html`. Implementation, expert validation
and QC role are separate. Benchmark predictions cannot stand in for human labels;
unreviewed or uncertain regions never count as ground-truth accuracy evidence.


E25 adds eight **experimental secondary** battery geometry KPIs in `polaron_qc/secondary.py` and a separate report section. They are absent from PRIMARY_KPIS/MATERIAL_KPIS and all verdict inputs; never promote automatically. Exact bright/void raster adjacency is constant zero on usable known sites and stays diagnostic. All current pore-mode flags are unresolved, so quality filtering uses the separate data-derived grey-pore flag and reports the method diagnostic. Eight scalar comparisons use site n and missing-value abstention; image-window/component coverage is not additional evidence. Paired threshold sensitivity does not validate phase boundaries or battery mechanisms. `analysis/battery/output/report.html` and A18 collect the new audits; keep source snapshots and explicit warm-cache reuse provenance. Physics readings do not infer transport benefit, lithiation time, total expansion or intact-particle share from these descriptors.
