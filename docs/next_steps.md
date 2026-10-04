# Next steps and task ownership

**Final evaluation, 2026-10-04 (D70S/E44S):** six samples have saved v2 assignments. Labels are pending. Preserve these predictions. Evaluate future labels against the saved files. See the [evaluation handoff](../analysis/final_evaluation_v2/README.md).

**Current task, 2026-10-04 (D69S):** retain the frozen v2 classifier and the existing QC rules. The E43S confidence layer stays experimental. Model experiments are closed. Review the demo page against the current evidence. See the [demo review](merged_qc_review.md#demo-page-review--2026-10-04).

Updated 2026-10-03. This is a shared work plan, not an instruction to start every task. Tasks marked unclaimed remain available; claimed ownership and handoffs are recorded below. Suggested leads reflect the existing team split; Santosh can pick an engineering task too. Record owner, status and the output path here when work starts.

The aim is a QC decision a materials expert can verify: reliable measurements, visible evidence, honest uncertainty and an executable unseen-batch handoff. The provider confirms Batch 3 as a heterogeneous working reference: morphology can distinguish batches even when defect counts do not. Prioritise measurement validity and delivery while testing a compact, explicit morphology expansion.

## What is already done

The baseline extraction, site-level statistics, acquisition checks and decision pipeline work on the known batches. Battery neighbourhood/homogeneity and long-void context have been measured; eight experimental secondary KPIs are in extraction/reports. The morphology atlas and independent annotation pack exist. The ML specialist review supplies two protocols: E26G has completed the fixed-graph audit, and the balanced encoder audit is complete. No new defect model is trained. The cold drop rehearsal (H, E28) passed on five renamed fixtures and fixed three drop-only defects. Known-batch report replays and the 174-test suite pass; the documentation has been reconciled with the code once (G, `analysis/g_reconcile/audit.md`); expert mask accuracy and unseen-batch generalisation remain unvalidated.

Start from [current findings](problem_and_findings.md), [battery report](../analysis/battery/output/report.html), [visual atlas](../analysis/morphology/output/metric_atlas.html) and [ML assessment](../analysis/ml_options/assessment.md). Do not repeat the completed broad EDA or literature review as a prerequisite.

## Pick a task

**Provider clarification / next priority (D49P):** Batch 3 is the supplier's promised baseline; later batches illustrate variation, not better/worse labels. Prioritise L below: a bounded morphology in/out-of-distribution comparator using the full measurable reference. A/C/D remain important for measurement/interpretation evidence, but missing proof of battery harm is not a universal bar to OOD features. The protocol and promotion gates are in `qc_plan.md` §1.1; the standalone E31L audit is complete, with live promotion deferred; Claude’s sample deliverables remain separate.

These letters are local plan labels, not external tickets. Priority P0 protects the core deliverable; P1 adds evidence; P2 is conditional on time/data. Effort is relative scope, not a measured completion estimate.

| Pick | Priority / effort | Work package and concrete output | Suggested lead | Dependency / status |
|---|---|---|---|---|
| **L audit** | P0 / medium | Registered morphology OOD promotion audit: baseline versus bright aspect/alignment/256 px phase association; full measurable B3, shared coverage, perturbation/acquisition/axis controls, joint alerts and added detection. | Codex engineering; Santosh image interpretation | **Completed E31L.1 (D50L/D51L)** in isolated `codex/morphology-l-audit`; copied additively to main. [Report](../analysis/morphology/l_audit/report.html) · [Handoff](../analysis/morphology/l_audit/handoff.md). Threshold/floor and abstract detection pass; automatic input deferred by incomplete radiometric observability, section direction and joint-rate precision. Claude keeps L1–L4 ownership below; no live activation or commit. |
| **A** | P0 / medium | Independently annotate the existing phase-mask review pack; export reviewer JSON with uncertain/unmeasurable cases retained. | Santosh / materials reviewer | Ready; unclaimed |
| **B** | P0 / small | Obtain or explicitly record missing specimen/session, chemistry/process, collector and practical tolerance metadata. Produce an answered/unknown evidence sheet. | Santosh / organiser liaison | Ready; unclaimed |
| **C** | P0 / medium | Review manufacturing interpretations: long voids versus preparation damage, bottom-band/collector candidate and bright rim fragments. Produce per-site notes, marked locations and clear unmeasurable cases. | Santosh / materials reviewer | Ready for visual review; B strengthens interpretation; unclaimed |
| **D** | P0 / medium | Evaluate annotated masks against current segmentation and hysteresis; report pixel, object/KPI errors and abstentions, separately for development and held-out sites. | Engineering | Needs exported A annotations; unclaimed |
| **E** | P1 / medium | Implement the fixed particle-neighbourhood graph audit, Protocol G; deliver visible node/edge overlays, site descriptors, sensitivity and acquisition/redundancy screens. | Engineering | **Codex: completed exploratory audit E26G, integrated into main.** [Report](/Users/santoshkumarsaravanan/Documents/Polaron/analysis/ml_options/e_graph/report.html) · [Handoff](/Users/santoshkumarsaravanan/Documents/Polaron/analysis/ml_options/e_graph/handoff.md). A/C expert review still gates material use. |
| **F** | P1 / medium | Implement the balanced frozen-DINO audit, Protocol F; deliver balanced/unbalanced comparisons, nearest-reference crops and resolution/acquisition sensitivity. | Claude (engineering) | **Done 2026-10-03** (pilot, exploratory; E27, D42). Artifact `analysis/ml_options/f_audit/` (`findings.md`, `run_f_audit.py`, `output/`). Equal site influence leaves the E07 ranking unchanged (ρ 0.97); ranks unstable across 518/224 and under gamma; top sites acquisition-flagged. Retained as retrieval/review aid; material interpretation stopped; excluded from verdict. Santosh materials review of the eight nearest-reference panels still open. |
| **G** | P0 / medium | Reconcile generated notebook, reports, metric statuses and assumptions with the current pipeline; produce one coherent judge-facing result. | Engineering + Santosh editorial review | **Engineering part done 2026-10-03 (Claude).** Audit `analysis/g_reconcile/audit.md` (48 items: 40 fixed, 5 decided in D46, 3 historical kept); report restructured as a decision screen first (grouped contents, status strip, exploratory divider, next QC action per verdict); notebook 02 and both reports regenerated on main with E/F/H/J/K integrated (verdicts, hashes unchanged; 141 tests). **Open: Santosh editorial review** — does every conclusion carry an image, a limitation and a next action (see audit.md §3 proposed ordering). |
| **H** | P0 / medium | Repeat the cold unseen-folder rehearsal with current extraction/quality guards; test new IDs, missing/unavailable evidence and the image-review workflow. Record a delivery receipt and reproducible commands. | Claude (engineering) | **Done 2026-10-03, merged** (E28, D43–D45, C31). Receipt `analysis/rehearsal_h/receipt.md`; fixtures `analysis/rehearsal_h/make_fixtures.py`; runner `run_rehearsal.py`. c2st site-id order dependence resolved (D43); three drop-only defects fixed (`--review` crash, abstention wording, classifier order); abstention label precedence and stability wording decided (D45). Repeat once more after the last core change before the drop (C31). |
| **I** | P0 / small | Build the judging story and manual demo checklist: observed change, trustworthy KPI, image explanation, uncertainty and the next QC action. | Santosh, with engineering support | Start from current reports; finalise after G/H; unclaimed |
| **J** | P2 / small pilot | Optional fixed Gabor/spatial-texture comparator. Retain only if it adds interpretable evidence beyond existing FFT/tensor/geometry descriptors. | Engineering | **Codex: E28J completed and integrated additively into main.** Three appearance descriptors computed; material/QC use deferred. [Report](/Users/santoshkumarsaravanan/Documents/Polaron/analysis/ml_options/j_gabor/report.html) · [Additive G handoff](/Users/santoshkumarsaravanan/Documents/Polaron/analysis/ml_options/j_gabor/handoff.md). |
| **K** | P1 / medium | Extend morphology beyond averages: prespecify shape variability and directional phase-association descriptors, with nonlinear image-depth structure as a conditional extension. Deliver definitions, site tables, image explanations and retain/defer findings. | Engineering + Santosh materials review | **Codex: E30K completed and committed (`bd01279`) on `codex/morphology-k-pilot`, isolated managed worktree.** Seven descriptors and two methods measured; all material/QC use and nonlinear depth extension deferred. [Report](/Users/santoshkumarsaravanan/.codex/worktrees/morphology-k-pilot/Polaron/analysis/morphology/k_pilot/report.html) · [Qualified findings](/Users/santoshkumarsaravanan/.codex/worktrees/morphology-k-pilot/Polaron/analysis/morphology/k_pilot/review.md) · [Additive handoff](/Users/santoshkumarsaravanan/.codex/worktrees/morphology-k-pilot/Polaron/analysis/morphology/k_pilot/handoff.md). Eleven mathematical tests and raw/table/atlas checks pass; A/D/C expert gates remain open. Current main has newer G changes, preserved for additive reconciliation. |
| **L** | P0 / medium | Sample-identification deliverable after D49P: three separate answers per sample — (L1) site categorisation among Batches 1/2/3 with model probabilities, fold-local selection, balanced accuracy / recall / confusion counts, morphology-only vs acquisition-only vs combined; (L2) baseline OOD assessment against Batch 3 built separately from the classifier, with descriptive LOO reference percentiles and the legacy tail-rank 1/18 resolution floor (not calibrated p-values); (L3) E20 per-particle Inlens texture under local-contrast normalisation (robustness, not material proof); (L4) batch-signature explanations (what differs, direction, size, family, images). | Claude (engineering), Santosh review | **L1/L2 done 2026-10-03 (E31, D50, C32): `polaron_qc/categorise.py`, `analysis/categoriser/findings.md`; primary balanced accuracy 0.66 (p 0.005) is a batch fingerprint — morphology-only at chance; model probabilities not calibrated; OOD: no Batch 2 site exceeds the observed reference LOO maximum; this does not establish membership/equivalence. Batch 1 exceedances are driven by unreliable bright segmentation (D52R). L4 done (`docs/batch_signatures.md`). L3 done (E32, D51: no normalised Inlens variant qualifies; contrast not a pure gain/offset artefact but brightness-linked; B1 > B2 part amplitude-carried). Notebook §12 shows the three answers side by side. **E33: first test drop assigned (B1 / B2 / B3, frozen before truth). D52: primary v2 = morphology + texture, no session statistics (frame height removed everywhere); assignments unchanged. E36S outcome 2026-10-04: B1/B2 bets swapped, B3 correct; both primaries 1/3 sites.** Open: Santosh's crop reviews (F panels, E20 particles, signature crops) and the organisers' reading of 'categorise' — resemblance or material state.** The frozen five-KPI QC verdict is unchanged and does not headline this deliverable. |

Current drop priorities after the merge: [organiser clarification, crop reviews, final integrated rehearsal and judging walkthrough](merged_qc_review.md). E20 is completed without a qualifying candidate; no further feature promotion is pending a casual choice.

**First organiser drop:** target clarification is answered (D53): always assign a batch and explain uncertainty.
[E33/E34R delivery](../analysis/organiser_drop_01/README.md) saves three bets, all nine image mappings, known-only
frozen models, deletion sensitivity and illustrated evidence. E36S records organiser feedback separately:
the Batch 1/2 bets are wrong and Batch 3 correct (1/3 sites); preserve the original predictions.
The final-folder procedure uses the declared frozen v2 workflow and validates input/source hashes.

**Suggested division for the first working session (historical):** Santosh picks A and either B or C; engineering prepares D's evaluator, G/H and one of E/F/K. For morphology expansion specifically, start with K's shape variability and E's graph arrangement. E, F and K can be independent branches if capacity permits. Each shared pipeline/register/log change has one coordinating owner to avoid conflicting edits. No assignment or agent dispatch is implied by this table.

## A and D: turn predictions into a measurement benchmark

Use the E38S [raw-only development page](../analysis/boundary_review/output/development.html) and its [handoff instructions](../analysis/boundary_review/README.md). It reuses E24's exact crops with neutral IDs and no batch/method hints in the page payload. Complete the six development examples; keep the separate five-site held-out page with an independent reviewer until methods are fixed. Draw void, bright appearance, residual solid and uncertain/ignore regions. Record previous exposure honestly: neutral IDs cannot erase prior knowledge. These are independent-drawing declarations and visual references, not chemical mapping. The original E24 comparison page remains available for later method inspection, after annotation.

Export the JSON with reviewer name, notes and measurable status; preserve the manifest identity. Browser local storage alone is not a durable handoff. Partial labels support pixel comparisons on labelled regions only. Fully specified masks are needed for object-size/width errors; poor contrast can remain unmeasurable. A reviewer can complete the existing pack, but held-out annotations/results must remain sealed from anyone tuning the method until development choices are fixed. All sites were previously explored, so this is a measurement holdout, not new-batch validation.

Engineering imports the raw-only packet export through the adapter, which validates aliases/source hashes and calls the existing evaluator in a fresh directory:

```sh
/opt/anaconda3/bin/python3 -m analysis.boundary_review.prepare --annotations /absolute/path/polaron_development_boundaries.json --out /absolute/path/new_development_evaluation
```

Done means a saved annotation export, separate development/held-out evaluation, usable coverage and abstention counts, plus an explanation of whether errors change relevant KPIs. Choose any error tolerances from the intended measurement/QC use before looking at held-out results. Algorithm agreement is not accuracy. A segmentation change needs new extraction provenance and known-data verification; it is not silently substituted into the frozen release pipeline.

Use a separate output for the already-labelled first-drop review. Held-out import additionally requires `--release-held-out` after choices are fixed; changed recorded method sources fail rather than silently scoring another method. Partial ROI labels do not establish full-site particle-size/width accuracy.

## B and C: clarify the actual battery/QC question

Use [A18 and the other assumption cards](assumption_register.html#A18) and the [battery application review](battery_microstructure_review.md). Record observations separately from hypotheses and supplied metadata.

Ask for specimen IDs and which sites share a specimen; imaging/preparation sessions and section orientation; confirmation of the collector edge; exact Si/SiOx/recipe/binder and coating/calendering information if available; practical KPI tolerances or acceptable/defective outcomes that may be shared. Fresh graphite–Si/SiOx and Batch3's working-reference status are already confirmed and do not need asking again. Unknown is a valid answer.

Review internal versus clipped long voids, possible preparation effects, ambiguous bright rims and a few clearly recognisable graphite plate sections. The bottom band in `epqdaau9` is a collector candidate; full raw context matters because trimming removes it. Existing graphite tensors are image directions, not plate-instance axes. Save site/crop coordinates, reviewer identity and reasoning. Confirming a collector or a visible void does not establish adhesion loss, electrical isolation or later cycling failure.

Done means an answered/unknown metadata sheet and traceable review notes. Apply confirmed/refuted states only to the exact question/KPI reviewed; morphology annotation does not automatically confirm a manufacturing-defect flag. If practical tolerances remain unavailable, the top outcome remains consistent with the working reference within detectable limits; acceptance/equivalence stays unavailable.

## E and F: bounded additional ML evidence

Use the exact proposed settings and stop/go gates in [Protocols G and F](../analysis/ml_options/assessment.md#two-concrete-experiment-protocols). Register precise graph descriptor keys/definitions before extraction. Keep candidate stage in [the ML register](../analysis/ml_options/candidate_register.json), and measured morphology keys/statuses in [the metric register](../analysis/morphology/metric_register.json).

E starts with fixed graphs and four interpretable summaries, with threshold/object-floor/edge coverage sensitivity, node-only size/loading baselines and acquisition controls. A proximity edge is not electrical contact. F uses the existing frozen representation, equal reference-site influence in PCA/memory fitting, fold-local preprocessing and nearest-image explanations. Cached embeddings do not prove material validity. Neither audit trains a defect classifier or changes primary tests/classifier inputs/verdicts.

Each pilot is done when saved site tables, real-image examples, uncertainty/coverage, provenance and an explicit retain/defer finding exist—even if the result is redundant, unstable or confounded. Use one prespecified pilot and one saved all-known-site run if justified; additional rounds require a new development version. Do not select methods or settings by the most attractive known-folder separation. New numerical results belong in the experiment log/registry. Sites remain n; specimen independence is unresolved.

## K: morphology differences beyond defect counts

This workstream complements E's graph geometry and J's fixed appearance filters. Batch identity, morphological change and manufacturing acceptability are separate questions. Keep Batch 3's full usable distribution and sub-populations visible; the ordinary-only reference is a sensitivity view, not a newly declared clean baseline.

| Morphology family | Already measured | Additional work / status |
|---|---|---|
| Size and shape | Bright/pore size summaries and distribution-width ratios; bright circularity, solidity and aspect ratio; pore elongation | **Computed E30K:** equal-count bright aspect/circularity IQR and solidity Q10, at fixed 50/100 px² floors with clipped components excluded. At least 20 eligible objects; source/math checks are distinct from expert validation. Other phase distributions remain unimplemented. |
| Orientation | Axial histograms and alignment of elongated bright/void components; image-tensor feasibility checks | Graphite plate-instance orientation is **deferred** until independently reviewed plate axes and section/collector direction exist. Image directions are not graphite alignment. |
| Spatial arrangement | Boundary-censored bright-centroid nearest-neighbour spacing; local bright dispersion and bright/pore association | **Computed E26G, exploratory:** fixed graph edge lengths, directional arrangement and neighbouring size association; QC use deferred after sensitivity/uncertainty checks. **Computed E30K:** finite-domain bright auto and symmetric bright/void cross-phase correlation x−y contrasts at 64/256 px, with phase-specific quality gates and acquisition/geometry/loading controls. These describe 2-D arrangement, not physical contact or 3-D transport. |
| Image-depth structure | Ten-bin phase-fraction profiles, signed/absolute linear slopes, void-width profiles and long-void locations | **Deferred K extension:** nonlinear depth structure awaits independent phase/section and collector-direction validation. Rows are not confirmed through-thickness/collector coordinates. |
| Texture scale/direction | BSE correlation length/FFT slope; detector/scale image tensors | **Computed E28J:** three fixed Gabor sampled-appearance summaries; material/QC use deferred after redundancy/acquisition/coverage checks. Existing Inlens per-particle texture remains confounded. |

Start with a small prespecified panel, not an open feature sweep. Before extraction, write exact keys, formulas, units, weights, lags/bins, coverage/abstention rules and sensitivity settings into a versioned definition sheet and the live metric register. Show each descriptor on actual images, including excluded/clipped regions. Generated inventory/atlas views must label proposed versus computed versus expert-reviewed status separately.

Use current data-derived quality flags, site-level intervals and deletion sensitivity, paired threshold perturbations and acquisition/size/loading/redundancy controls. Components, graph edges and image windows never increase statistical n. Independent A/D labels gate measurement-accuracy claims; computed geometry alone is provisional. A descriptor that is constant, unstable, unmeasurable or explained by an existing summary gets a defer finding rather than another QC score.

Done means the definition sheet, reproducible measured outputs for the selected pilot, image evidence, usable-site/coverage tables, uncertainty and a retain/defer recommendation, with register/atlas/findings/log updates. New descriptors stay secondary and outside classifier/verdict inputs. Promotion requires a separate logged validation decision and practical QC interpretation; nothing is selected using the unseen batch. E30K computes the bounded seven-descriptor panel; all material/QC use and additional depth structure remain deferred.

## G, H and I: deliver and preserve the unseen test

G updates notebook **generators**, then rebuilds/executes the notebook and checks outputs against current reports. Audit trust labels, usable counts, threshold sensitivity, unavailable values, expert-review status and battery wording. Some previous notebooks/scoreboard notes predate the secondary additions; reconcile against current code/results. Keep a short primary decision screen, with exploratory work clearly accessible underneath. Santosh reviews whether every conclusion has an image, a limitation and a practical next action.

H exercises the documented [drop procedure](../README.md#unseen-batch--the-drop-procedure-rehearsed-see-experiment-log-e23) using a temporary renamed known-data fixture and a fresh cache. Preserve raw originals and historical caches. The prior rehearsal precedes the latest additions, so verify the current path, including missing-quality abstention, finite/absent secondary values and confirmed/refuted image reviews. Include the known limitation that fold assignment/permutation streams can depend on site IDs; resolve or disclose it on known data before freeze. Compare verdict inputs and primary features, not an assumed requirement that every stochastic classifier p-value be identical. Run checks appropriate to changed code and repeat the delivery rehearsal after the last core change.

Before the real drop, snapshot core config/thresholds, feature/method versions, source hashes and reference inputs. Save the unseen report before any follow-up exploration. Run the existing frozen decision path first; its classifier/acquisition views run inside the pipeline. Optional embedding/graph audits do not become verdict drivers. If the unseen batch arrives while a candidate protocol is unfinished, keep it out of candidate selection/tuning; any use for selection makes it development data. An incoming batch outside supported data/quality conditions should show unavailable evidence or investigation, not forced acceptance.

I prepares a short walkthrough: why acquisition matters; which observed geometries are trustworthy enough to use; what the reference itself contains; the QC verdict and image evidence; what uncertainty/tolerances prevent; what the operator should review or measure next. Show one ordinary case and one reviewed challenging case without claiming a ground-truth defect label that we do not have.

## Keep for later

Trained CNN/GNN defect models, supervised U-Net without independent labels, graphite-alignment automation without instances/direction, binder-network segmentation without resolvable evidence, and 3-D transport/performance predictions need additional data/validation. They are future branches with explicit gates in the ML and morphology registers.

Once someone picks work, update this table with actual ownership, status and artifact path. Completion requires the specified output and limitations, not merely code running. This planning pass launches no new experiment, annotation or implementation task. Pre-presentation checks applied: C04/C05/C06/C13/C14/C16–C18/C21/C27–C30.


**E35S / D54S final-folder preparation:** [Frozen v2 workflow](../analysis/submission_v2/README.md) saves
models, family-specific reliability, first bets and image explanations with explicit version checks and
no comparator fallback. Use new output directories for final evaluation. Save organiser truth separately,
evaluate the frozen E33/v1 and v2 bets, and keep any label-informed model revision separate.

**E36S / D55S feedback received 2026-10-04:** [Saved-bet evaluation](../analysis/feedback_drop_01/report.md)
records the Batch 1/2 swap and correct Batch 3 label: both primaries score 1/3 sites, including a 0.84
wrong v2 bet stable under all 31 deletion fits. The first-drop truth item is closed; final evaluation remains open.

The next priority is a bounded known-data audit of measurement-quality handling for all mask-dependent
classifier inputs, followed by Santosh's particle/pore boundary review. In particular, the labelled
low-contrast query is Batch 2 although both low-contrast training sites are Batch 1. Define a consistent
input-validity policy and compare it with v2 using fitting/selection inside known-site folds; group by specimen
or preparation session when metadata exist. Treat these three labelled queries as failure examples in any
feedback-informed revision, not a fresh test or a target to fix by swapping labels. No new model is trained
or feature promoted by recording this outcome. Morphology-only makes the same swaps, so the feedback does
not isolate texture as the cause; acquisition-only's 2/3 is diagnostic and does not authorise its promotion.

**E37S / D56S quality audit — Codex owns, completed 2026-10-04:** the user authorised the next priority.
`analysis/quality_policy_audit/protocol.md` fixes the matched legacy/direct/dependency/quality-only
comparison after feedback, with fold-local fitting and a dependency catalogue. This is a standalone
candidate; frozen v2 remains the submission model. Santosh's independent boundary review stays open.

[Audit findings](../analysis/quality_policy_audit/findings.md) and
[visual review](../analysis/quality_policy_audit/report.html) show functional rejected-value safety,
known balanced accuracy 0.569 versus 0.608 and unchanged failure bets. Candidate promotion is deferred;
this completes the bounded audit. The next work is independent particle/pore boundary review and
specimen/preparation/session metadata, with any later quality-conditioned model registered separately.

**E38S / D57S review handoff — Codex owns, prepared 2026-10-04:** separate raw-only pages reuse E24's
6 development / 5 held-out crops and add 3 fixed first-drop development crops. Reviewer/exposure,
uncertainty/abstention and export/import controls are ready; no expert annotations were created.
All 102 TIFF headers over 34 sites lack explicit grouping/settings IDs, and Santosh confirms no mapping
is available yet. [Handoff](../analysis/boundary_review/README.md), [mapping sheet](../analysis/boundary_review/output/group_mapping_template.csv)
and [unsent organiser questions](../analysis/boundary_review/organiser_questions.md) are ready.
Human boundary review and actual organiser grouping remain open; no classifier/QC promotion.

## Classification improvement assuming no SME annotations (D58S)

**Source-construction update (D59S, 2026-10-04):** organisers confirm crops from around
15 source images arranged into artificial visual batches. Parent IDs remain unknown;
crop-LOO can share parents across folds. These experiments identify curated visual
groups, not actual supplier lots. No source-held-out accuracy or IID significance is
available. M1/M2 are authorised and run under the fixed [E39S protocol](../analysis/classification_m1_m2/protocol.md),
with a separate label-blind overlap audit. Parent-image mapping, when available, must
group both outer and inner folds; non-overlapping related crops remain undetectable.

**M1/M2 outcome (E39S/D60S): completed, no promotion.** Gated L2 balanced accuracy
0.445; adding the fixed24 gives 0.473, versus ungated v2's matched 0.608 and gated
L1 0.569. Appearance alone is 0.300 and gamma-sensitive. Augmented L2 corrects one
revealed failure, but all models still miss low-contrast B2; that is development
evidence, not external success. [Full comparison / response maps](../analysis/classification_m1_m2/report.html).
E40S recovers no overlaps and cannot rule out shared non-overlapping sources. **M3
is now the next experiment**, not a reason to widen M1's search or silently promote
M2. No 34-site submission fit was performed.

Santosh asks us to plan under the assumption that no human SME will be available.
Batch labels already support supervised site categorisation. Expert phase annotations
would validate mask/physical interpretation; their absence does not block a separately
versioned image-signature classifier. A/D remain measurement-validation gaps, rather
than prerequisites for every classification experiment. Current v2 and the frozen
QC verdict remain unchanged. M3–M5 remain proposed; M1/M2 execution is separate from submission promotion.

**Diagnostic basis:** the existing v2 confusion matrix recognises 2/7 known Batch 2
sites, versus 5/7 B1 and 14/17 B3. It uses 29 inputs and independent one-versus-rest
L1 fits. Class balancing is already enabled. E37S's complete dependency gate prevents
invalid-value influence but does not improve accuracy; a flags-only annotation or
median-imputation change is not sufficient. Existing graph/Gabor and DINO novelty
audits do not establish a classification gain.

| Order | Work | Concrete deliverable and rationale |
|---|---|---|
| **M1 — first, bounded model audit** | Compare current matched L1 with **L2 multinomial logistic regression** (primary new candidate) and **shrinkage LDA** (diagnostic comparator), using the existing features and complete mask-dependency policy. Preserve ungated frozen v2 as a comparator. | Shared outer site holdouts, fold-local preprocessing/regularisation, per-class recalls, confusion, balanced accuracy and log-loss. Joint class fitting and covariance regularisation are hypotheses worth testing on correlated inputs, not promised improvements. No large model search. |
| **M2 — new mask-independent signature** | A compact fixed panel of whole-region/patch-pooled BSE, ETD and Inlens multiscale filter-energy ratios, edge-orientation summaries and spatial heterogeneity. | No phase/particle segmentation needed; explain each descriptor with response maps and example patches. Compare with existing Gabor/FFT descriptors for redundancy. Channel-relative normalisation can reduce gain sensitivity but must be challenged under gamma/quantisation. Interpret as image appearance, not chemistry, plate instances or battery harm. No frame dimensions, raw detector means or quality indicators as predictors. |
| **M3 — frozen representation probe** | Reuse cached DINOv2 BSE embeddings for a supervised site-level classifier, rather than rerunning baseline novelty. Freeze the encoder; pool a fixed/equally weighted tile representation, then a small fold-local PCA and regularised linear head. | Batch identification and nearest training-site patch examples; all tiles from a held-out site excluded from every fit. E27's gamma/resolution sensitivity remains a mandatory challenge, not erased by a new task. This is transfer learning from frozen features; no encoder/GNN training or segmentation labels required. |
| **M4 — acquisition-robust training** | Challenge the candidate with prespecified non-clipping gain/offset and mild gamma/quantisation changes on copies of training images. Augmentation applies to every class; detector omission is a separate missing-channel sensitivity. | Train with and without these changes; measure clean held-out recall/log-loss and held-out perturbation failures. Re-extract dependent features/quality on transformed copies. Originals and all their copies stay in the same fold; total weight per original site is fixed. Do not erase/paint defects, use large blur, or rotate away section orientation. Synthetic changes cannot reproduce all real preparation effects or restore lost information. |
| **M5 — delivery and confidence** | Combine views only if their outer-fold errors supply complementary evidence; use a fixed rule or select weights inside training folds. Assess a single positive temperature from inner out-of-fold scores only, with outer assessment, if coverage permits. | Always emit the required batch bet and all class scores, separate robustness/quality warnings and nearest-image/descriptor explanations. Single-temperature scaling preserves the argmax; it is not an accuracy fix or a correctness guarantee under session shift. No fitting calibration on the final drop. |

**Recommended execution:** start M1 and M2, followed by the cached M3 probe. Fix exact
feature definitions, pooling/PCA sizes, the small model grid, nuisance settings and
promotion criteria before each run. Assign experiment IDs at execution, append numerical
results to the ledger, and keep candidate/model-selection steps inside outer training
folds if reporting the resulting selection procedure. Repeated CV seeds are sensitivity
on the same sites, not more data; do not select a favourable seed. A fixed primary
candidate with labelled comparators limits best-of-many optimism.

**Use the revealed labels correctly:** retain the original 31-site benchmark for a
matched comparison, then optionally train a newly versioned final candidate on all
34 labelled development sites (8/8/18 across B1/B2/B3). This uses the organiser's
first-drop labels, including the new low-contrast B2 example. Preserve original E33/v2
predictions; the three first-drop samples are development evidence in every later
revision and cannot be quoted as external validation. Improvement on their training
predictions is not success evidence. Freeze the final candidate before final labels.

**Success evidence:** better crop-held-out diagnostic balanced accuracy and B2 recall, no concealed
collapse of the other classes, competitive log-loss, and fewer prediction changes under
the fixed nuisance challenges. Selection/feature fitting and augmentation stay inside
training folds. Resampling for independent inference requires valid parent-group exchangeability;
IID crop permutations do not establish significance with unknown shared parents. The existing acquisition-cluster holdouts are proxy
stress checks, not real specimen/session validation. If a class is missing or an inner
fit is infeasible, disclose that and use the prespecified fallback, never label crop fallback as grouped validation. New labelled
source images with confirmed grouping are needed to demonstrate new-source improvement; no performance gain is asserted here.

### Confidence and explanation criteria included (E42S/D64S/D65S)

Assignment accuracy is not the only improvement criterion. E42S's fixed L2+8 has mixed
known-score results and2/3 revealed development, while gated29L2 keeps the three original bets
with lower wrong scores and higher correct score. [Audit](../analysis/confidence_audit/report.html)
reports every fixed candidate, common-label proper losses, own-bet correctness, both target
weightings and same-bet/flattening controls. No exact organiser utility or calibration is inferred.

M3 must retain this confidence/explanation audit in addition to balanced accuracy. If adding
calibration, declare one low-capacity method and fit it strictly inside outer training folds using
training-only internal predictions; compare unchanged and flattening controls on held-out crops,
then genuine new-source labels. Parent IDs govern both outer/inner groups if supplied. Do not choose
calibrator, threshold, family or score weighting from the three revealed drop outcomes. Preserve
uncalibrated class scores and observed/imputed explanations; no31-crop bin certifies calibration.

**Promotion approval (D66S):** Santosh authorises replacement when confidence quality improves;
no additional approval is required for a candidate meeting a declared promotion criterion.
E42S compared an anchor-ordered matched legacy reference, which E43S found is not exact
frozen v2 on one fold. E43S's actual-v2 nested confidence layer fails its two-target and
non-regression gate. Declare the next candidate's criterion and tradeoffs before
fitting, then version/freeze any supported replacement while preserving the original. Always
assign the class with the largest score even if all three are below 0.5; uncertainty changes
the explanation of the bet, not the argmax rule.

**E43S completed, no confidence replacement (D68S):** the scalar fitted on internal
crop-held-out training predictions softens wrong bets but weakens correct bets farther.
Equalclass full-label NLL1.064→1.032 improves, own-bet Brier0.209→0.243 worsens;
all31 known/3 revealed assignments stay unchanged. [Report](../analysis/confidence_calibration/report.html)
and saved465pair fits preserve the failed result. Full suite241passes; originalv2/QC
unchanged. Do not widen calibration/threshold search to chase3revealed labels. Next bounded
work remains M3 frozen-DINO, with confidence/explanations and genuine source limits included.

### Completed cheap comparison: Random Forest × compact geometry (D61S/D62S/E41S/D63S)

Santosh asks whether to add morphology/physics inputs or change the model. Recommend
testing both effects separately, using already-computed measurements without new SME
annotations. **Completed E41S, no promotion:** gated29 L1/RF balanced accuracy0.569/0.465;
+8 geometry L1/RF0.521/0.417. All four main candidates retain1/3 on the revealed development
drop. [Report and image references](../analysis/forest_geometry/report.html);
[findings](../analysis/forest_geometry/findings.md) and all fixed diagnostics are saved.
The following records the original recommendation; no wider forest search follows this result.

| Declared input view | Matched L1 | Fixed shallow Random Forest |
|---|---|---|
| Existing29 with complete dependency gate | baseline | isolate model effect |
| Same29 plus fixed eight geometry descriptors | isolate added-measurement effect | examine interactions with added geometry |

Preserve frozen ungated v2 as an additional reference. Proposed forest:500 trees,
depth3, at least4 training crops per leaf, sqrt features per split, balanced class
weights, bootstrap=True and fixedseed0. These are bounded starting settings, not
validated choices. Imputation stays fold-local and numerical validity does not rely
on filenames or explicit quality flags as predictors. No OOB/training-accuracy claim,
native-missing-value shortcut promotion, seed search or calibration claim.

The fixed eight geometry keys are `bright_aspect_count_iqr`,
`bright_circularity_count_iqr`, `bright_solidity_count_q10`,
`void_local_width_d50_px`, `void_local_width_d90_px`,
`graph_edge_length_iqr_ratio`, `local_bright_std_512px`, and
`bright_pore_crosscorr_xy_contrast_256px`. These are count-shape distributions,
section-width tails, centroid-spacing variability, spatial phase dispersion and
finite-image arrangement. Existing definitions, threshold/floor/coverage gates
and bright/pore/joint quality dependencies apply. They remain unreviewed image
geometry; no chemical, physical contact or transport claim follows. Some may be
redundant or acquisition-sensitive. The unresolved low-contrast B2 case cannot gain
observed bright-particle evidence from extra mask-dependent bright descriptors.

Prioritise these measured quantities over guessed electrochemical outputs. Current
`tortuosity_index` is undefined, `bright_void_boundary_frac` is constant, and conditional
mass-fraction/unfolded3D calculations are assumption-limited. They do not supply new
validated battery measurements. Compare feature families using held-out changes in
recall/log loss and images; training impurity importance is insufficient. Crop results
remain diagnostic until parent grouping is available. M3's frozen-DINO supervised
probe remains a separate representation experiment after this small comparison.

Sources: [Random Forest controls](https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
[training impurity versus held-out importance](https://scikit-learn.org/1.4/auto_examples/inspection/plot_permutation_importance.html),
[source-group validation](https://scikit-learn.org/1.4/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

Method references: [multinomial/L1/L2 logistic regression](https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html),
[shrinkage LDA](https://scikit-learn.org/1.4/modules/lda_qda.html),
[frozen DINOv2 transfer](https://arxiv.org/abs/2304.07193),
[nested evaluation](https://scikit-learn.org/1.4/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
A direct-only gate's higher known score does not make its remaining indirect dependencies safe.
