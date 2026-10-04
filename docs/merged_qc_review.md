# Merged QC review and remaining drop checks

Reviewed against main `c14504e` on 2026-10-03. L1/L2, E20/L3 and L4 are integrated; the separate morphology
promotion audit E31L is delivered in the Codex edits. This review changes presentation and evidence tracking,
not feature definitions, fitted-score calculations, primary verdict inputs or frozen thresholds.

**First-drop update (D53/E33/E34R):** the organiser has answered the target question: always take a batch bet,
with qualified confidence and explanations; identification measures how samples differ from baseline.
The three-sample folder has arrived. [Predictions, images and sensitivity checks](../analysis/organiser_drop_01/README.md)
are saved before truth feedback. The calibration corrections below still apply. Session/specimen metadata and
the crop-review tasks remain open.

**Feedback update (E36S/D55S, 2026-10-04):** the organiser swaps the Batch 1/2 bets and confirms Batch 3.
Both preserved primaries score 1/3 sites, and all v2 bets survive 31/31 training-site deletions, including
both errors. This is stable misclassification. [Separate labels and saved-bet review](../analysis/feedback_drop_01/report.md)
verify class encoding and artifact integrity. Low contrast is not exclusive to Batch 1, and several strong
inputs on the labelled Batch 2 query depend on its unreliable bright mask. Prioritise a known-data
input-quality audit and expert boundary review; morphology-only makes the same swaps, so no isolated
texture-cause finding follows. Comparator promotion or another feedback-informed change requires a new
version and makes this drop development evidence. Final evaluation remains pending.

## What the three answers support

The combined categoriser has balanced accuracy 0.66 and site-label permutation p 0.005 on the known sites,
with fold-local selection. It is useful evidence of a **batch fingerprint**. Morphology-only categorisation
has not demonstrated separation; this does not prove that morphology is identical. Acquisition-only p 0.035
does not pass the pre-registered secondary multiplicity threshold. None establishes performance on a new
acquisition session or specimen. The probabilities remain uncalibrated model scores. The reported
`accuracy_ci95` is a descriptive binomial reference interval, not an exact generalisation interval for
dependent LOO predictions. This qualification follows from the overlapping fits; see the error-correlation
analysis by [Bengio & Grandvalet](https://jmlr.org/papers/v5/grandvalet04a.html).
Local results: [E31 findings](../analysis/categoriser/findings.md).

The morphology distance answers **how unusual the measured site is relative to the observed Batch 3
reference**. No Batch 2 site exceeds the observed reference LOO maximum under the primary score; that does
not establish equivalence or distribution membership. The two Batch 1 exceedances are driven by unreliable
bright segmentation on low-contrast sites. Reference heterogeneity is intentional: a cracked site can resemble
an included cracked reference site, but an unseen cracked site is not guaranteed to be below the maximum.

The legacy `ood_rank_p_*` columns are **descriptive tail ranks**, with resolution floor 1/18. Percentiles run
from 0 to 100. Neither supplies a calibrated p-value or a false-alert guarantee. Reference scores refit on 16
sites while queries use all 17; merely ranking those scores does not establish exchangeability. The previous
IID exceedance-probability interpretation is withdrawn, while its preregistration text and all numerical outputs
are preserved. This conclusion follows from [the implementation](../polaron_qc/categorise.py) and the
calibration/symmetric-scoring requirements explained by
[Angelopoulos & Bates](https://arxiv.org/abs/2107.07511). Multiple queries also share a reference maximum, so
their exceedance events cannot simply be multiplied as independent events.

E20 is complete: none of its four primary normalisations qualifies. The original texture retains its
**confounded** status. The CV variant has offset sensitivity; per-particle affine standardisation loses useful
amplitude information; local-z remains acquisition-linked; LBP is noise-sensitive. These are measured outcomes,
not a pending feature-selection choice. E25 was a different sampled-gradient measurement and remains a
separate method. Source: [E20 findings and particle evidence](../analysis/e20_inlens/findings.md).

The frozen batch verdict remains a separate action recommendation with its existing limits. Failure to reject
a difference is not proof of equivalence, and resemblance to a known batch is not battery-performance evidence.

## Work to finish before the drop

1. **Clarify the target with the organisers (Santosh).** Suggested question, drafted only:
   “Will held-back samples be scored for resemblance to the known Batch 1/2/3 images, or for material conformance
   to Batch 3 across new acquisition sessions? Will the samples come from new specimens/sessions, and can they
   belong to a batch outside those three?” This determines how much weight the fingerprint deserves.
2. **Review the three crop sets (Santosh / materials reviewer).** Use the
   [F audit and nearest-reference panels](../analysis/ml_options/f_audit/findings.md),
   [E20 particle crops](../analysis/e20_inlens/figures/evidence_particles.png), and
   [batch-signature crops](../analysis/signatures/fig_signature_crops.png).
   Record whether contrast, edge relief, preparation or internal material structure plausibly explains each
   example; uncertain reviews remain uncertain. Match images to driver columns and acquisition flags.
3. **Run the final integrated rehearsal (H).** Exercise the frozen batch report and new per-site categoriser on
   renamed known-data fixtures, including low-contrast sites and fewer usable sites. Confirm flags, missing
   measurements, known-class argmax, descriptive ranks and image-review states are understandable in the hand-off.
   Save the configuration, versions, source/reference hashes and outputs before the real drop. A renamed fixture
   checks engineering, not unseen accuracy.
4. **Prepare the judging walkthrough (I).** Show fingerprint, descriptive morphology distance and QC action side
   by side, using an ordinary example and a disagreement such as a cracked Batch 3 site categorised as Batch 1.
   Explain uncertainty and the next measurement/review action. Include the known-site confusion matrix.

Further texture, learned CNN/GNN or reference-conformance work needs a new bounded protocol and independent
validation. E31L's retained geometry evidence remains available; no automatic promotion is implied by this review.

## Validation receipt

The companion [review receipt](../analysis/categoriser/merged_review_receipt.json) records numerical-output,
preregistration, production-source and inventory-history preservation, plus regenerated presentation checks.
Historical E31L delivery receipts describe that earlier delivery and are not rewritten to claim a new base.

## Demo page review · 2026-10-04

The page architecture agrees with the implemented system. It keeps three outputs separate: the batch assignment, the descriptive baseline distance and the QC action. The page needs text corrections before the demo. The proposed text below uses short sentences and active voice, as required by Santosh.

The live page is `http://localhost:8770/approach`. The server uses `.claude/worktrees/model-integration-ui-plan-c0e590`. Its receipt records build commit `f35471a`. The frozen model, QC decision code and report code match main. The page has older evidence and metric counts. The [freeze manifest](../analysis/demo_freeze_20261004/manifest.json) records the file hashes.

### 1. Correct the data description

The lead and section 1 call the data three supplier lots. They also call Batch 3 an approved baseline. The organisers describe artificial batches made from crops of about 15 electrode images. We have no source-image mapping. The manufacturing use case does not establish the source of these challenge batches.

Proposed text:

> The challenge data contain 31 crops from about 15 electrode images. The organisers assign these crops to three artificial batches. Batch 3 is the challenge reference. The batches do not identify confirmed supplier lots. No production acceptance limits are available.

Keep the proposed supplier QC use case separate from this data description. Source: `app/build_approach.py`, lines 266 and 272, in the UI worktree.

### 2. Restrict the validation claims

Section 7 reports the correct frozen v2 result: 21 of 31 crop assignments, with balanced accuracy 0.608. Batch 2 recall is 2 of 7. The confusion matrix agrees with the saved evidence. Keep these values.

The page says the model does not simply memorise sessions. The acquisition clusters do not identify confirmed sessions or source images. Related crops can occur in both fit and test data. The historical permutation result does not establish accuracy on independent source images. The same limit applies to the QC tests that treat sites as separate observations.

Proposed text:

> Each test excludes one crop. Related crops can occur in both fit and test data. We do not know which crops share a source image. The result describes the available crops. It does not establish accuracy on new source images.

Label the permutation results and batch signatures as exploratory evidence. Section 8 compares many descriptors against two batches. Its nominal p-values do not establish material causes. Remove the claim that about six descriptors must separate by chance without a defined comparison count. Source: `app/build_approach.py`, lines 320–328.

### 3. Update completed work and metric status

Section 9 calls the input-quality audit the next task. That audit and the later model tests are complete. None met the criteria for deployment. E43S improved one probability loss but made the score for bet correctness worse. The frozen v2 model stays in use.

Proposed text:

> We completed the input-quality audit and the fixed model comparisons. We also tested a confidence layer. No candidate met the criteria for deployment. We retain the frozen v2 classifier. Its scores are not calibrated probabilities of a correct assignment.

The drawer says 88 descriptors were measured. Main now tracks 112 entries: 108 computed, one implemented and three deferred. All 112 entries have an unreviewed status. Register entries are not the classifier feature count. The classifier still uses 29 features.

Proposed text:

> The metric register tracks 112 entries. We computed 108 entries. One entry is implemented, and three are deferred. All entries still require expert review. The QC rules use five primary KPIs. The batch classifier uses 29 features.

Update the links to the current experiment and decision records. Avoid a fixed range of IDs that excludes later work. Source: `app/build_approach.py`, lines 296 and 333.

### 4. Complete the QC rule description

Section 4 omits some conditions for a reject. The shift test alone cannot cause a reject. The batch path also requires a joint test, site consistency and a separate material classifier check. The acquisition checks must permit the result. This material classifier check is separate from the 29-feature batch classifier.

The statement about fewer than five usable sites is too broad. A shortage for one KPI excludes that KPI from the batch test. It does not always cause a global quality abstention. Global abstention depends on the site count, coverage across the primary KPIs and the quality flags.

Proposed text:

> A significant KPI shift alone cannot cause a reject. The batch path also requires evidence from the joint test and the material classifier check. Enough sites must support the shift. The acquisition checks must permit a reject. The local path checks measurement quality and image-review results.

Use the implemented rules for the detailed diagram. Keep the current QC thresholds. Source: `app/build_approach.py`, lines 299–302; `polaron_qc/decision.py`.

### 5. Put the judged output first

The organisers judge batch assignment, confidence and explanation. Section 7 currently places that answer late in the page. Put the Batch 1/2/3 assignment near the top. Show the score, image evidence and quality flags beside it. Keep the QC action and baseline distance as separate outputs.

Proposed text:

> Each sample receives a Batch 1, 2 or 3 assignment. The system chooses the batch with the highest model score. It shows all three scores and the features that support the assignment. The scores are uncalibrated. A high score can occur for a wrong assignment.

The first drop had one correct assignment out of three. Keep both errors visible. A three-sample folder can contain different batches. It is not necessarily one QC lot. QC abstention does not prevent a required sample assignment.

### 6. Retain the unresolved texture cause

Section 2 calls the microscope the cause of the strongest feature. E20 did not establish that cause. Texture responds to image conditions, but its material contribution remains unresolved. Section 11 also makes an unsupported claim about what a learned model would learn.

Proposed text:

> Texture depends on image conditions. The tests do not establish whether its batch differences come from material, image conditions or both. The classifier uses texture with this limit stated. The QC rules do not use it as a primary KPI.

### Review checks

The review checked the live page, its source and its build receipt. The review also checked the saved model and the QC code against main. No model, threshold or UI file changed. The previous code validation passed all 241 tests. Text changes do not require another model experiment.
