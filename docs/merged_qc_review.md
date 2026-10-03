# Merged QC review and remaining drop checks

Reviewed against main `c14504e` on 2026-10-03. L1/L2, E20/L3 and L4 are integrated; the separate morphology
promotion audit E31L is delivered in the Codex edits. This review changes presentation and evidence tracking,
not feature definitions, fitted-score calculations, primary verdict inputs or frozen thresholds.

**First-drop update (D53/E33/E34R):** the organiser has answered the target question: always take a batch bet,
with qualified confidence and explanations; identification measures how samples differ from baseline.
The three-sample folder has arrived. [Predictions, images and sensitivity checks](../analysis/organiser_drop_01/README.md)
are saved before truth feedback. The calibration corrections below still apply. Session/specimen metadata and
the crop-review tasks remain open.

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
