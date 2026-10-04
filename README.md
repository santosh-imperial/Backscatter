# polaron-sem-qc

**Interpretable, uncertainty-aware batch QC for battery-electrode SEM cross-sections.**

`polaron_qc` takes a folder of scanning-electron-microscope cross-section images of an electrode coating, compares it with an approved reference batch, and returns three separate, auditable answers:

| answer | question | output |
|---|---|---|
| **QC verdict** | Has the material moved beyond what this reference and sample size can detect, and in which direction? | `consistent within detectable limits` · `investigate — batch-wide drift` · `investigate — localized anomaly` · `reject (provisional)`, with drivers, decision stability, acquisition flags, the minimum detectable change, and the next QC action |
| **Baseline distance** | How unusual is this sample's measured morphology relative to the reference? | k-nearest-neighbour robust distance to the reference sites, reported as a leave-one-site-out percentile (descriptive, never a probability) |
| **Batch resemblance** | Which known batch does this sample most resemble? | 3-class model scores from a frozen L1 logistic regression on morphology and texture features, with per-feature contributions (uncalibrated; a fingerprint of the sample as imaged) |

Every number is computed on **sites** (one stitched cross-section = one unit of evidence), every reference-dependent fit is repeated inside the permutation and leave-one-site-out loops, every threshold is frozen and hashed, and every KPI that drives a verdict is painted on the image so a materials expert can check it.

Built in 48 hours for the Polaron hackathon by Santosh Kumar Saravanan with Claude and Codex as engineering teammates. The honest headline: on the organisers' first held-back drop the batch-resemblance model scored 1 of 3, and the repository records why.

---

## Contents

1. [What it does](#what-it-does)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Data layout](#data-layout)
5. [How it works](#how-it-works)
6. [Results on the known data](#results-on-the-known-data)
7. [Limitations, stated plainly](#limitations-stated-plainly)
8. [Repository layout](#repository-layout)
9. [Reproducibility and governance](#reproducibility-and-governance)
10. [Demo interface](#demo-interface)
11. [Project status and experiment history](#project-status-and-experiment-history)
12. [Contributing, citation, licence](#contributing-citation-licence)

---

## What it does

Given per-site BSE, ETD and Inlens images (pixel-aligned, three detectors over the same field), the library:

- **Segments** the BSE image into voids, graphite and the bright higher-atomic-number additive phase with per-image, histogram-anchored thresholds, and flags sites whose contrast cannot support a reliable bright-phase mask.
- **Measures** five primary KPIs (crack-like void fraction, largest void diameter, void fraction, bright-phase fraction, bright-phase D50) plus a catalogue of secondary and exploratory descriptors, each with a trust level. All lengths are in pixels; the 25 nm/px tag in the source TIFFs is nominal and unverified.
- **Derives acquisition flags from the batch's own images** (contrast stretching, raised black level with grey pores, low bright-phase contrast) and writes them into the site table before any statistic runs.
- **Tests batch-wide drift** with site-level permutation tests on the Hodges–Lehmann shift (Holm-corrected over the five primary KPIs) and an energy-distance test, and prints the **minimum detectable change** for the batch's usable sample size next to every "not detected".
- **Checks for localized defects** separately: a single site beyond the ordinary-reference maximum is routed to human image review at 2 MAD and changes the verdict at 3 MAD or on agreement of two sites or two KPIs. Reviews have three states: unreviewed, confirmed, refuted.
- **Guards against imaging artefacts**: the multivariate test is repeated in unadjusted, stratified and covariate-adjusted views; a drift reject is withheld when no view is available or when adjustment attenuates the shift strongly.
- **Reads the physics qualitatively**: direction-only statements per driver KPI, gated by a ±5 grey-level threshold-sensitivity band; stereology and void geometry are shown as context, never as verdict inputs; no performance or cycling claims are made.
- **Estimates decision stability** by re-running the whole pipeline, classifier and acquisition views included, with each batch site left out.
- **Writes one self-contained HTML report per batch** with the verdict first and the exploratory evidence clearly labelled underneath.

## Installation

Python 3.12 is the tested interpreter (the project was developed on an Anaconda install on macOS; no OpenCV is needed).

```bash
git clone https://github.com/santosh-imperial/polaron-sem-qc.git
cd polaron-sem-qc
python3 -m venv .venv && source .venv/bin/activate      # or use your conda environment
pip install -r requirements.txt
python3 -m pytest tests -q                              # 241 tests, ≈ 3 min, no raw images needed
```

The package is used in place (`python3 -m polaron_qc.<module>`); no install step beyond the requirements is required. The exploratory DINOv2 novelty member needs `torch` and network access to the hub weights the first time; nothing in the verdict depends on it.

## Quick start

**1. Compare an incoming batch with the reference and write the report.**

```bash
python3 -m polaron_qc.report Dataset/Batch_3 Dataset/Batch_2 reports/qc_Batch_2.html
```

Runs feature extraction (≈ 1 min per seven-site batch cold, cached afterwards), flag derivation, the site-level tests, the material-only classifier, the three acquisition views, the localized check, the decision and the leave-one-site-out stability, and writes the HTML. About 70 s with a warm cache. Useful flags:

```bash
--summary reports/qc_Batch_2.json      # diff-able JSON snapshot of every verdict input
--review <site>:crack_frac=yes|no      # record a human image review; the verdict is recomputed with that state
--cache-dir DIR                        # separate feature cache (cold timing, rehearsals)
--no-images                            # skip evidence images
```

**2. Score held-back samples against the known batches.**

```bash
python3 -m polaron_qc.categorise Dataset/Batch_3 Dataset/Batch_1 Dataset/Batch_2                        # leave-one-site-out evaluation
python3 -m polaron_qc.categorise Dataset/Batch_3 Dataset/Batch_1 Dataset/Batch_2 --score Dataset/<new>   # fit on the known folders, score every site in <new>
```

Writes a per-site table with model scores per feature family, the baseline-distance percentile, nearest reference sites, the top contributing features with sign against the reference median ± MAD, and the acquisition flags. For judged submissions use the frozen workflow in [`analysis/submission_v2/README.md`](analysis/submission_v2/README.md), which pins the model version and preserves first predictions.

**3. Rebuild the executed notebooks.**

```bash
python3 notebooks/_build_02_batch_qc.py
cd notebooks && PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 02_batch_qc.ipynb
```

Notebooks are generated from `_build_*.py` scripts and executed in place; edit the script, never the `.ipynb`. The QC notebook's first cell asserts a frozen configuration hash.

## Data layout

```
Dataset/
  Batch_1/img_<site>_BSE.tif   img_<site>_ETD.tif   img_<site>_Inlens.tif
  Batch_2/...
  Batch_3/...                  # the reference
```

Each site is one stitched, ion-polished cross-section (≈ 7000 × 1600–2300 px, 8-bit grey stored as RGB) imaged by three detectors. `SE` is accepted as an alias of `ETD`. Raw images are not tracked by git.

**Provenance, as clarified by the organisers on 2026-10-04:** the images are crops from around 15 source electrode images, arranged into artificial visual batches; Batch 3 is the designated reference; Batches 1 and 2 are later variations that are neither better nor worse. Parent-image identifiers were not supplied, so related crops may fall into different cross-validation folds, and no claim of supplier-lot identity or manufacturing validation follows from anything in this repository. The tests here are therefore crop-held-out development diagnostics.

## How it works

```
images (BSE · ETD · Inlens per site)
   └─ features.extract_batch ──► site table (five primary KPIs, secondary descriptors, diagnostics)
   └─ acquisition.derive_flags ─► quality flags from the batch's own images (written into the site table)
         │
         ├─ stats.compare_kpis       site-level permutation tests, Holm, energy distance, MDC by simulation
         ├─ acquisition.three_views  unadjusted / stratified / adjusted; attenuation of the multivariate statistic
         ├─ ml.c2st                  material-only classifier two-sample test (corroboration only)
         ├─ stats.local_exceedance   single-site evidence flags against the ordinary-reference maximum
         ├─ physics                  direction-only readings, threshold-band gate, sanity checks
         └─ decision.decide          Check A (drift) · Check B (localized) · quality abstention → verdict + next action
               └─ report.render_report   one HTML per batch; stability by leave-one-site-out of the whole pipeline

   └─ categorise                      batch resemblance (frozen L1 logistic, v2) and baseline distance (k-NN robust z), kept separate
```

The data contracts between modules, the Mermaid diagram and the known hazards are in [`docs/workflow.md`](docs/workflow.md). The method is explained for a non-specialist reader on the demo's "Our approach" page (see [Demo interface](#demo-interface)).

## Results on the known data

| what | result | where |
|---|---|---|
| Batch 1 and Batch 2 vs the reference, five primary KPIs | both *consistent within detectable limits*; decision stability 1.00 (7 of 7 leave-one-site-out re-verdicts); no primary KPI below Holm α | `reports/qc_Batch_1.html`, `qc_Batch_2.html`, E22 |
| Minimum detectable change at 5–7 usable sites | 1.5–3.0 reference MADs per KPI, printed next to every result | E17, E22 |
| Drift-path false-alarm rate on reference splits | 0.05 at α = 0.05 | E17 |
| Localized path on the three known cracked reference sites | recovered at 5.8, 5.4 and 2.6 MAD beyond the ordinary maximum; clean-batch flip rate ≈ 0.12 under the tiered rule | E17, E19 |
| What differs between the batches | 24 of 119 descriptors at p < 0.05 (13 ETD/Inlens texture, 7 BSE morphology); the texture contrast is not a pure gain/offset artefact but stays brightness-linked | [`docs/batch_signatures.md`](docs/batch_signatures.md), E32 |
| Batch resemblance, v2 primary (morphology + texture, no session statistics) | balanced accuracy 0.61, 21 of 31 sites, permutation p 0.005; morphology alone at chance (0.35); scores not calibrated | E31, E34 |
| Session-memorisation check | leave-one-acquisition-cluster-out 0.60–0.65, the same as leave-one-site-out | E35 |
| **First organiser drop (3 samples)** | **1 of 3 correct**: the low-contrast sample was called Batch 1 and is Batch 2; the mask-dependent texture features that drove the bet were measured inside an unreliable bright mask | E33, [`analysis/feedback_drop_01/report.md`](analysis/feedback_drop_01/report.md) |

Every number above is logged in [`docs/experiment_log.md`](docs/experiment_log.md) and [`experiments/registry.csv`](experiments/registry.csv).

## Limitations, stated plainly

- **Small n.** Seven and seventeen sites per batch, with specimen independence unconfirmed and parent-image identifiers unavailable. Intervals are wide and hold-out tests are weak.
- **2-D sections, pixel units.** No 3-D information; the nominal pixel size is an unverified export tag; no tortuosity value is reported because no site has a connected void path through the section.
- **Chemistry unconfirmed.** The bright phase is "possibly Si or SiOx"; the electrodes are uncycled, so nothing here speaks about cycling damage, SEI, lithium plating or performance.
- **Masks are unreviewed.** Segmentation is algorithmic; independent expert annotation of phase boundaries is still pending, and mask-dependent features on low-contrast sites are unreliable (the cause of the first-drop miss).
- **"Consistent" is not "accept".** No equivalence test against agreed tolerances exists, so the top verdict is bounded by the minimum detectable change and release stays a QC decision.
- **Texture is acquisition-sensitive.** ETD ridge and Inlens texture descriptors separate the batches but their origin (microstructure versus charging and session) is unresolved; they never drive the verdict and are labelled wherever they appear.

## Repository layout

```
polaron_qc/            the library: features, acquisition, stats, ml, physics, decision, report, categorise,
                       secondary / battery_metrics / void_metrics (exploratory battery geometry)
tests/                 pytest suite (241 tests; synthetic data plus guarded real-cache smoke tests)
notebooks/             01_dataset_analysis.ipynb (EDA) and 02_batch_qc.ipynb (QC), each with its generator script
reports/               the per-batch HTML reports written by the pipeline
docs/                  problem_and_findings.md · qc_plan.md · workflow.md · batch_signatures.md · decision_log.md ·
                       experiment_log.md · next_steps.md · assumption_register.html (18 reviewed assumptions with images)
experiments/           registry.csv — one row per reported number
analysis/              audits and pilots, each with its own protocol, outputs and receipts (battery geometry, morphology
                       atlas and benchmark pack, graph / encoder / Gabor / shape pilots, drop rehearsals, feedback review,
                       quality-policy and confidence audits, frozen submission workflow)
analysis_cache/        committed per-site feature tables and ML embeddings; the parquet feature cache is git-ignored
CLAUDE.md / AGENTS.md  rules for human and AI contributors (data hazards, style, governance)
```

## Reproducibility and governance

- **Frozen thresholds and configuration.** `decision.Thresholds` is hashed (`b4f4da2e357c`) and printed on every report; the QC notebook asserts its frozen configuration hash (`99d2bbcae6f3`) before running.
- **Decision log** `docs/decision_log.md`: every consequential choice with rationale and rejected alternatives (D01 onward), a pre-presentation checklist grown from reviewer catches (C01 onward), and open items.
- **Experiment log and registry**: every quoted number has an entry and a CSV row; superseded figures carry pointers rather than edits.
- **Pre-registration.** Exploratory pilots commit their definitions before the first run; predictions for organiser drops are committed before the truth arrives and are never edited afterwards.
- **Rehearsals.** The drop procedure was rehearsed cold on renamed copies of known data, which found three drop-only failures before the real drop.
- **Conservative defaults.** A missing classifier run, a missing acquisition view or an infeasible power simulation can only make a verdict more cautious, never less.

## Demo interface

A FastAPI front end, **Backscatter**, wraps the same frozen commands (upload a lot, run the one-command QC, score samples, review receipts) and includes an "Our approach" page that explains the method in Simplified Technical English. It lives on the branch `claude/model-integration-ui-plan-c0e590`:

```bash
git checkout claude/model-integration-ui-plan-c0e590
pip install fastapi uvicorn
python3 -m app.server --port 8770      # then open http://localhost:8770/approach
```

## Project status and experiment history

The project is a hackathon entry and remains under active review. The sections below summarise the record; the detailed history is in the logs.

- **Organiser feedback on the first drop (2026-10-04).** Batch 1 and Batch 2 bets were swapped; Batch 3 was correct. Both preserved model versions score 1 of 3. All v2 bets survived 31 of 31 training-site deletions, including both mistakes, so stability is not confidence. The low-contrast query is Batch 2, so low contrast is not exclusive to Batch 1. Truth and saved bets are kept apart in `analysis/feedback_drop_01/`.
- **Follow-up audits (E37S–E43S).** A quality policy that removes rejected-value influence from mask-dependent inputs lowers balanced accuracy on the known sites (0.57 vs 0.61) and leaves the three labelled failure bets unchanged; bounded model and feature alternatives (gated L1, random forest, extra geometry) did not establish a gain; a scalar confidence correction improved log loss but worsened Brier on the bet and was not promoted. v2 stays the declared submission model. Reports are under `analysis/quality_policy_audit/`, `analysis/forest_geometry/`, `analysis/confidence_audit/`, `analysis/confidence_calibration/`.
- **Measurement validation next.** A raw-only boundary review (`analysis/boundary_review/`) with development and held-out crops awaits expert masks; a full TIFF header audit recovered no specimen or session identifiers from 102 files; the source-crop overlap audit (`analysis/source_crop_audit/`) could not recover a mapping to the ~15 parent images.
- **Exploratory evidence kept out of the verdict.** Battery geometry audit (`analysis/battery/`), morphology atlas and annotation pack (`analysis/morphology/`), fixed graph geometry (E26G), balanced frozen-encoder novelty (E27), Gabor appearance (E28J), shape and arrangement descriptors (E30K), and the morphology OOD audit (E31L); each has a protocol, a qualification note and a retain-or-defer finding.
- **Open work** is tracked in [`docs/next_steps.md`](docs/next_steps.md) with owners and status.

## Contributing, citation, licence

Contributors, human or AI, should read [`CLAUDE.md`](CLAUDE.md) first: it holds the hard-won rules about this data (never use raw intensity as a material KPI, all lengths in pixels, known sub-populations, conservative defaults) and the governance expected around every change. Run the checklist in `docs/decision_log.md` before presenting a result, and log every quoted number.

If you use this work, please cite the repository: *polaron-sem-qc — interpretable, uncertainty-aware batch QC for battery-electrode SEM cross-sections*, Santosh Kumar Saravanan, 2026, https://github.com/santosh-imperial/polaron-sem-qc.

Licence: to be decided by the author before public release (no `LICENSE` file is present yet). The SEM images belong to the challenge organisers and are not part of this repository.
