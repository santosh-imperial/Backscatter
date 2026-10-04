# Customer QC interface — plan

Status 2026-10-04: agreed design (D56U). First build done: `app/export_lot.py`, `app/build_ui.py`,
`app/ui_template.html`, tests in `tests/test_ui_build.py`. Owner split: Claude engineering, Santosh wording,
materials lines and demo script.

## 1. Who it is for

A cell manufacturer's incoming-material QC engineer, explaining a lot decision to a materials expert. The judges
see the same interface during the demo. The question it answers: has this incoming electrode lot changed
against the supplier's approved baseline (Batch 3), what drives the difference, and what should QC do next?

## 2. Agreed decisions (D56U)

1. **No combined rule.** The recommended action comes only from the existing frozen verdict rules
   (`polaron_qc.decision`, thresholds hash `b4f4da2e357c`). The other lanes are shown beside it, never folded in.
2. **Verdict wording.** Top state is "Consistent within detectable limits", never "Accept". The other states are
   "Investigate — drift", "Investigate — localized", "Reject" and the quality abstention.
3. **Sample resemblance is a secondary view.** The categoriser answers "which known supplier lot does each
   sample most resemble, and in what way". It does not drive the lot decision.
4. **"What this could mean for the cell" panel.** It interprets the lanes and has no verdict of its own. Tier 1
   statements appear only for KPIs that drove the decision. Tiers 2 and 3 sit in a collapsed drawer, with tier 3
   (E25 battery geometry) labelled experimental.
5. **First-drop miss shown openly.** It appears on a Track record screen and, inline, as a warning on any sample
   whose flags match the failure pattern.
6. **Static HTML delivery.** One self-contained `index.html` per build. For the demo, all lots are pre-built. One
   folder is scored live in the terminal, and the page is then rebuilt and opened.

## 3. Screens

Left navigation: approved baseline, incoming lots (Batch_1, Batch_2, unseen), sample sets, track record, method
and provenance.

### 3.1 Lot review (main screen)

Before building, the operator declares each input as one of two kinds:

- **Single lot.** The full lot review below.
- **Sample set (may mix lots).** No lot verdict (D53). Sample resemblance, distance per sample and quality only.

Order on the screen:

1. **Header:** lot ID vs baseline, sites imaged, usable n per KPI after quality flags, quality chips.
2. **Recommended action:** verdict, reason and next QC action, taken verbatim from the decision module.
3. **Four lanes, each with its own conclusion:**

   | Lane | Shows | Never claims |
   |---|---|---|
   | 1 Defect-relevant drift (5 primary KPIs) | verdict state, energy-distance p with permutation count, nearest-to-moving KPI, decision stability "k of n", **MDC next to every "not detected"** (C07) | equivalence, acceptance |
   | 2 Localized anomalies | sites beyond the ordinary-baseline max, the escalation tier, review state (pending / confirmed / refuted) | a defect call from exceeding the max alone |
   | 3 Distance from baseline (morphology) | sites beyond the observed baseline range, median percentile, top driver read against that site's flags (C32) | calibrated membership or OOD p-values (C33R) |
   | 4 Image quality and acquisition | data-derived flags per site, KPIs excluded by those flags, acquisition fingerprint percentile, unadjusted / stratified / adjusted views | that the instrument explains a share (C11) |

4. **What drives it:** robust shift bars (MAD) with bootstrap intervals for the primary KPIs. Clicking a KPI
   opens the evidence.
5. **Evidence viewer:** BSE / ETD / Inlens for the selected site, with overlays for painted crack-like voids,
   bright-phase outlines and patch novelty. A typical baseline site sits alongside for comparison.
6. **What this could mean for the cell:** see §4.
7. **Footer:** model and thresholds versions, hashes, and the provenance wording "developed using exploratory
   analysis of Batches 1–3; frozen before the unseen batch arrived" (C16).

**Revision after the design critique (2026-10-04, score 26/40; Santosh chose story first, image as canvas, all
issues).**

- **Decision band.** The verdict, its reason (shown open) and the next action sit above four collapsible lane rows.
  Each row states whether that lane sets the action, taken from the frozen outcome columns. For example:
  "Does not change the action: below the escalation rule". The distance lane says "Context only". Semantic colour
  belongs to the action; lane rows use neutral icons.
- **Evidence workbench.** The SEM image is the canvas: the lot site above a baseline site of the chosen group
  (typical, grey-pore or cracked), at equal scale. A yellow box on the image marks the object behind a localized
  flag and opens the full-resolution crop. The "What drives it" panel sits beside the image:
  - Selecting a KPI switches the overlay and selects the lot site furthest from the baseline median.
  - The dots on the per-site strip are keyboard-reachable radio buttons.
  - With no driver, the panel opens on the KPI closest to being one.
- **Interaction.** Only the changed region re-renders, so open disclosures stay open. State is kept in a deep link
  `#lot-<lot>~<kpi>~<site>~<view>~<comparison>`. The page title follows the view, focus moves to the heading, and
  there is a skip link and a polite live region.
- **Second critique pass (score 27/40; Santosh chose evidence path first, verdict pinned above the image, all
  issues).**
  - The decision band is cut to the verdict, the next action and stability. The reason and "what would change
    it" sit under Details.
  - Lane rows are a compact legend. A lane that sets the action is in ink at weight 600; the others are muted.
    Each lane has "Show this on the image".
  - The lot and baseline images sit side by side at equal scale, with no label over the image. At 1440×900 the
    first image starts at about y = 794.
  - The workbench opens on whatever bears on the action: a localized flag opens its site and KPI with the crop
    open. Otherwise it opens on the driver, or the KPI closest to being one.
  - A quality abstention is shown as "Investigate: too few usable sites (quality abstention)". The frozen verdict
    string stays in Details and the footer.
- **Third critique pass (score 24/40, after a regression in the opening view).**
  - **Opening view.** The opening view follows the lane that sets the action:
    - pending or credible localized: the flagged site with its crop open
    - drift: the driver KPI
    - abstention: the first excluded site in the bright-phase view
    - otherwise: the KPI closest to being a driver, with no crop

    Flag-only evidence shows its yellow box with the crop closed, captioned "Below the review threshold (2.0 MAD);
    no review needed".
  - **Image review.** Routed or pending flags get Confirm and Refute buttons with a reviewer name. The review is
    stored only in that browser and labelled "Not yet applied". The page shows the exact
    `python -m app.export_lot lot ... --review SITE:KPI=yes|no` command, offers a JSON download, and changes no
    verdict itself.
  - **`--review` on the exporter.** `app.export_lot lot --review` passes reviews to `build_result` and records them
    in `reviews_applied`.
  - **Decision text.** "Why" and "what would change it" are built from structured fields and the frozen
    `decision.Thresholds`. The builder embeds the thresholds and refuses a lot whose thresholds hash differs. The
    raw decision strings are kept under "Frozen decision text" in the footer.
  - **Workbench layout.** The images are stacked in the left column at full width, with the KPI rail on the right
    at 1100 px and wider. A change that would leave the image off-screen scrolls it back into view.
- **Lane 4 wording (approved by Santosh 2026-10-04).** When only the covariate-adjusted view is below 0.05: "Only
  the covariate-adjusted view falls below 0.05. Adjustment on a mixed baseline can create signal, and the frozen
  rules do not use this view to raise drift. Read it as a prompt to check session metadata, not as drift."
- **Layout fixtures.** `app.make_layout_fixtures` writes four labelled copies of a real lot that show drift, localized
  pending review, reject and abstention. `app.build_ui` accepts them only through `--fixture-lot`, shows a banner and a
  separate navigation group, and refuses a fixture passed as `--lot`. They are not results.

### 3.2 Approved baseline

Typical Batch 3 sites, its sub-populations (grey-pore, cracked, ordinary), and the five primary KPI
distributions with trust levels. It says plainly that the baseline is heterogeneous (rule 9).

### 3.3 Sample resemblance (secondary)

One card per sample:

- the predicted batch, with three score bars labelled "uncalibrated" and the margin
- deletion stability "31 of 31" labelled "stability, not correctness" (C17)
- the morphology-only and acquisition-only comparators
- quality flags
- a waterfall of the exact logit contributions for assigned minus runner-up (sums exactly), each feature against
  the Batch 3 median ± MAD
- an inline warning when the sample is low contrast or its masks are unreliable: the first-drop miss pattern
  (E35, E36S)

### 3.4 Track record

- Known-site leave-one-site-out: balanced accuracy 0.608, recall B1 / B2 / B3 0.714 / 0.286 / 0.824, permutation
  p 0.005, confusion matrix and reliability bins (6 / 18 / 7 sites)
- First drop: 1 of 3, with the explanation (a low-contrast Batch 2 site, a combination of quality and batch
  never seen in training)
- Leave-one-acquisition-cluster-out check (E35)
- The lot verdict on the known lots: Batch 1 and Batch 2 are both "consistent within detectable limits" at
  n = 7, with the MDC shown

These are development results, not new-session accuracy.

### 3.5 Method and provenance

How each lane is computed, in one paragraph each, plus the KPI catalogue with trust levels, the frozen hashes and
receipts, and the limitations (pixels only, 2-D, uncycled, Batch 3 heterogeneous).

## 4. What this could mean for the cell

All display text in this section uses ASD-STE100 Simplified Technical English (STE). §4.4 says how the text was
checked. The implementation notes in §4.5 are for engineers and are not STE.

### 4.1 Panel text

> This panel shows possible effects on the cell. It shows text only for the KPIs that cause the verdict. The text
> gives only the direction of the change. It does not give a performance value.

If no KPI causes the verdict, the panel shows this text:

> No primary KPI has a change that is larger than the minimum detectable change. The system does not show
> possible effects for smaller changes.

### 4.2 Tier 1: one text for each primary KPI (approved by Santosh 2026-10-04; deployed text lives in `app/build_ui.py` `STE`)

| KPI | Display text (STE) |
|---|---|
| `crack_frac` crack-like void area fraction | This value is the area of long voids in the section. These voids are similar to delamination. An increase can show a decrease in coating cohesion or adhesion. This condition can be important during calendering and formation. Sample preparation can also cause these voids. Examine the images to find the cause. |
| `pore_max_d` largest void | This value is the diameter of the largest void in the section. One large void can be important. This is also true when the lot average does not change. The local defect rule examines this value. |
| `pore_frac` macro-pore area fraction | This value is the area fraction of the voids that the image shows. This value can change with the calendering density. It can also have an effect on electrolyte access. The image does not show fine pores or binder. Use this value only to compare lots. It is not the total porosity. |
| `bright_frac` bright-phase area fraction | This value is the area of the bright additive phase in the section. Possibly, this additive is Si or SiOx. This system does not identify the chemistry. A change can show a change in formulation or dispersion. Ask the supplier about this change. This value does not measure capacity. |
| `bright_d50` bright-phase D50 | This value is the median diameter of the additive particles in the section. An increase can show a change in the particle size from the supplier. It can also show agglomeration. A change can have an effect on mechanical properties during cell operation. This system does not measure this effect. |

### 4.3 Drawer (collapsed by default)

**Tier 2, title "Context values".** Subtitle: "The verdict does not use these values." Items:

- Sum of the phase fractions. This sum must be 1.
- Volume fraction, calculated from the area fraction (Delesse method).
- Additive mass fraction, if the additive is Si. This is a nominal value.
- Delamination index.
- Length of the longest void, as a fraction of the coating thickness.
- Change of each value when the threshold moves by 5 grey levels.

**Tier 3, title "Experimental battery geometry".** Subtitle: "An expert must examine these values before use.
The verdict does not use these values." Items: the eight E25 battery geometry KPIs, with site-bootstrap
intervals and threshold sensitivity.

**Rules for all panel text:**

- Do not show a performance value or a capacity percentage.
- Do not show transport benefit, lithiation time, total expansion or cycle life.
- Do not show SEI, lithium plating or cycle damage. These specimens are new. They did not operate in a cell.
- Do not show a tortuosity value. No site has a connected void path from the top to the bottom.
- Do not write "Si" alone. Write "possibly Si or SiOx".
- Contact between an additive particle and a void does not show electrical contact. Do not write that it does.

### 4.4 How the STE text was checked

The text was checked by hand against the STE writing rules:

- A descriptive sentence has 25 words or fewer. An instruction has 20 words or fewer.
- Each sentence has one topic, and a paragraph has six sentences or fewer.
- Verbs are in the active voice and in the present, past or future tense.
- There are no -ing forms, except in technical names (calendering, coating).
- Articles are used, and noun clusters have three words or fewer.

Technical names are permitted in STE. This text uses these technical names:

- void, coating, delamination, cohesion, adhesion, calendering, formation, porosity, electrolyte, binder,
  additive, phase, formulation, dispersion, agglomeration, capacity, lithiation, tortuosity, SEI
- diameter, fraction, median, index, KPI

The approved words were not compared with the official ASD-STE100 dictionary, so a person who has the
specification must verify them. Words to verify first: "examine", "identify", "possibly", "similar", "nominal",
"compare".

### 4.5 Implementation notes (not STE)

- Tier 1 text is shown only for verdict drivers. Weights come from `physics.CONSEQUENCE_WEIGHTS` (all five primary
  KPIs have weight 3). The direction comes from `physics.qualitative_statement`, rendered as "<KPI> is higher /
  lower than the baseline (robust shift x MAD)" before the STE text.
- A non-significant shift gets no interpretation. When a lot has no driver, the panel shows the §4.1 no-driver
  text and the MDC.

## 5. Data contract

The builder never refits or scores. It reads saved run outputs only.

| UI element | Source |
|---|---|
| Lanes 1, 2 and 4, recommended action, drivers, MDC, stability | `polaron_qc.report.summary(result)` JSON (`--summary`) |
| Lane 4 acquisition views | `result["acquisition"]`; **not in the summary today** |
| Lane 3 per site | single lot from a categoriser run: `baseline_morphology_percentile`, `exceeds_observed_baseline_max` in `predictions.csv`. Known lots: `freeze/evaluation/ood_batch_summary.csv` (development) |
| Shift bars | `summary.primary` (`shift_mad`, `ci_low`, `ci_high`, `p_holm`) |
| Evidence images | `report.paint_crack_voids`, `outline_bright`, `novelty_overlay`, written as JPEGs; **not exported today** |
| Cell panel tier 1 | direction from `result["physics"]["statements"]`, then the §4.2 STE text after Santosh approves it |
| Cell panel tiers 2 and 3 | `result["physics"]`, secondary geometry columns in `sites_batch`; **not in the summary today** |
| Sample resemblance | `analysis/submission_v2/<run>/submission/predictions.csv`, `driver_contrasts.csv`, `report_evidence.json` |
| Track record | `analysis/submission_v2/freeze/evaluation/*`, `analysis/feedback_drop_01/` (E36S outputs; present in the main checkout, not yet committed) |
| Provenance | `prediction_receipt.json`, `submission_receipt.json`, `summary.meta` |

## 6. Engineering work (Claude)

1. **`app/export_lot.py`.** It runs `build_result` for one lot (frozen thresholds; no change to decision logic)
   and writes a lot bundle: summary JSON, acquisition views, physics, secondary geometry, and evidence JPEGs at
   display resolution. This is an interface addition, so `docs/workflow.md` is updated in the same change.
2. **`app/build_ui.py`.** It reads lot bundles, categoriser runs and track-record files, then writes one
   self-contained `ui/index.html` (vanilla JS views, embedded JPEGs, batch colours from `polaron_qc`, light and
   dark themes). Target under 10 MB.
3. **Fixtures and tests.** A renamed known-data fixture covering both the single-lot and sample-set modes. Tests
   that the builder:
   - fails rather than falling back when an input is missing (C34S)
   - never prints "accept" or "confidence"
   - shows the MDC next to every "not detected"
4. **Demo runbook.** Exact terminal commands for the one live scoring and rebuild, with timings.

## 7. Santosh

- Approve or rewrite the five tier-1 lines in §4.
- One-line materials reading for each lane.
- The demo script order, including whether a cracked Batch 3 site is used as the example where the lanes
  disagree.

## 8. Open items

- Whether the unseen final folder is a single lot or a sample set decides which mode the demo leads with. Confirm
  with the organisers (Santosh).
- A v3 categoriser (quality policy for mask-dependent inputs) would appear as a separate selectable version,
  never a silent replacement.

## 9. Build and demo runbook

Export each lot (about 70 s each with a warm feature cache), the sample set, then build. Bundle directories must be
new. `Dataset/` and the test folder are in the main checkout. The build reads saved files only.

```bash
python -m app.export_lot lot --reference ../../../Dataset/Batch_3 --batch ../../../Dataset/Batch_1 --out ui/bundles/lot_Batch_1 --distance-table analysis/submission_v2/freeze/evaluation/site_table.csv --distance-label "Frozen v2 morphology distance model fitted on Batch 3 only (analysis/submission_v2/freeze). This lot's sites were not in the fit, but were explored during development." --with-baseline
python -m app.export_lot samples --run analysis/submission_v2/first_run --input ../../../Hackathon-Polaron-test --out ui/bundles/samples_drop01 --name "First organiser drop (3 samples)"
python -m app.build_ui --lot ui/bundles/lot_Batch_1 --lot ui/bundles/lot_Batch_2 --samples ui/bundles/samples_drop01 --track-record analysis/submission_v2/freeze/evaluation --feedback ../../../analysis/feedback_drop_01 --out ui/index.html --replace
```

For the unseen folder:

- **Sample set.** Score it with the three frozen v2 commands in `analysis/submission_v2/README.md`, then
  `export_lot samples` on that run and rebuild.
- **Single lot.** Also run `export_lot lot`, using the new run's `scored/site_table.csv` as `--distance-table`.

First build (2026-10-04): 7.94 MB, 81 images. The verdicts and numbers match `reports/qc_Batch_{1,2}.html`:
energy p 0.404 and 0.343, MDC 1.75–3.0 and 1.5–2.5 MAD, stability 7 of 7. The waterfall total for 3e122cbj
(+4.858) equals the E36S decomposition.

Layout-only check of the verdict states the known lots never reach (keep the output out of `ui/`, never demo it):

```bash
python -m app.make_layout_fixtures --from ui/bundles/lot_Batch_1 --out ui/fixtures
python -m app.build_ui --lot ui/bundles/lot_Batch_1 --fixture-lot ui/fixtures/fixture_drift --fixture-lot ui/fixtures/fixture_localized --fixture-lot ui/fixtures/fixture_reject --fixture-lot ui/fixtures/fixture_abstention --track-record analysis/submission_v2/freeze/evaluation --out /tmp/fixtures.html
```

## 10. Inspect a lot: the real-world front door (D57U)

Santosh, 2026-10-04: the demo opens on how the tool is used. A QC engineer receives a new lot, adds it and gets the
decision. The analysis page built above becomes the **How it was built** tab.

**Server.** `python -m app.server` serves http://127.0.0.1:8770, on localhost only. It has two tabs:

- **Inspect a lot.** Built from `app/inspect.html`. The engineer adds a folder by drag and drop or the folder picker,
  or picks one from `inbox/` for large lots. They name the lot and declare it either "One lot" (lot verdict plus
  sample resemblance) or "A sample set that may mix lots" (resemblance only, D53). The browser checks file names and
  detector sets before anything uploads. Progress, the result and the lot history follow.
- **How it was built.** The committed `ui/index.html`.

**Pipeline per inspection.** Each inspection runs in `inspections/<UTC time>_<name>/`, which git ignores:

1. Input check: the scorer's own guards, plus rejection of training images by hash and of training site IDs.
2. Frozen v2 scoring.
3. Composition of the per-sample bets.
4. `export_lot lot`, in one-lot mode only.
5. `export_lot samples`.
6. `build_ui --inspection`.

Each step runs as a subprocess with its own log. One inspection runs at a time. Nothing is refitted, and runs are
never overwritten.

**Rehearsal (2026-10-04).** The first-drop folder is the only non-training input available.

- **Sample set, uploaded:** 67 s for 3 sites (scoring 54 s). The bets and scores are identical to the saved
  `analysis/submission_v2/first_run` predictions.
- **One lot, from the inbox:** 158 s for 3 sites (scoring 54 s, lot comparison 102 s). The verdict is "Investigate:
  too few usable sites (quality abstention)", because 3 sites is below `min_usable_sites` 5.
- **7-site lot:** not yet timed.

The lots from Batches 1–3 cannot be inspected; the input check refuses training images.

**Demo order.**

1. Inspect a lot: drop the unseen folder, or run it beforehand and open it from the lot history.
2. The lot review.
3. Switch to How it was built.

The static page still works on its own if the server fails.

**Known limits.**

- Features are extracted twice per inspection, once by the frozen scorer's snapshot and once by the lot exporter's
  live package. That keeps the frozen scorer isolated, at the cost of time.
- The preview pane cannot read the Documents folder, so the server is started from a terminal:
  `/opt/anaconda3/bin/python3 -m app.server`.

