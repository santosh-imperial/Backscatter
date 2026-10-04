# First-drop feedback: the Batch 1 and Batch 2 bets were wrong

On 2026-10-04 Santosh relayed the organiser's response: “Great features you're heading towards, well done! 1 and 2 are swapped, 3 is correct.” [The evidence record](truth.json) preserves that wording and its interpretation against the original submission order. No separate filename-to-label file was supplied. The resulting batch labels are stored separately in [truth.csv](truth.csv).

| Sample | Frozen bet (v1 and v2) | Organiser label | V2 top score | V2 score on true batch | Outcome |
|---|---|---|---:|---:|---|
| `3e122cbj` | Batch 1 | Batch 2 | 0.8382 | 0.0832 | wrong |
| `fn0mhxef` | Batch 2 | Batch 1 | 0.4956 | 0.3342 | wrong |
| `xrv9xvzb` | Batch 3 | Batch 3 | 0.6218 | 0.6218 | correct |

Both primary versions scored **1/3 sites (33.3%)**. These are three labelled sites, each with three aligned detector images; the images are not nine independent tests. Each true batch has only one query. This is an observed outcome, not a reliable estimate of future accuracy or calibration. Specimen independence and preparation/session metadata remain unresolved.

V1 was frozen before feedback. V2 was revised after inspecting the first drop, before its labels, and declared before final evaluation. Its first-drop comparison is consequently not untouched pre-arrival validation. Original predictions, reports, models and verification receipts retain their pre-feedback “truth pending” wording as historical records.

## What the saved evidence shows

The saved model class names are `Batch_1`, `Batch_2`, `Batch_3`; estimator indices are `0`, `1`, `2`. The encoder maps training folder labels to those same names. Replaying the saved models on saved feature rows reproduces all four families' score columns and argmax labels. There is no demonstrated output-column inversion to fix by globally swapping Batch 1 and Batch 2.

**`3e122cbj`: a strong, stable wrong bet on an unreliable bright mask.** Its bright-component count is 148.12/Mpx, compared with Batch 1/2 known-site medians 16.56/14.00, and `bright_low_contrast` is true. Its Inlens particle-texture P90 is 0.7496 and detected ETD ridge coverage in particle interiors is 0.00846. Both use the unreliable bright mask, so their physical interpretation is compromised. In the saved Batch 1-minus-Batch 2 logit explanation, pore count contributes +1.997, particle-texture P90 +1.716, `ridge_p97` +1.019 and particle-interior ETD ridge coverage +0.947. These explain the model's choice; they do not establish material causes. Two low-contrast training sites belong to Batch 1, while this labelled low-contrast query belongs to Batch 2: low contrast cannot be treated as exclusive Batch 1 evidence. The prominent bright-fragment OOD distance remains a measurement-quality finding.

**`fn0mhxef`: the morphology cues were not batch-exclusive.** Pore count 86.91/Mpx and pore fraction 0.07851 resemble known Batch 2 medians 85.94/Mpx and 0.07544, yet the organiser says Batch 1. Together they contribute +1.223 to the saved Batch 2-minus-Batch 1 logit. Summed morphology contributions are +1.402, appearance contributions −0.734, with intercept +0.173. The appearance family in aggregate therefore opposes this wrong assignment. The acquisition comparator chooses Batch 1, driven chiefly by ETD boundary sharpness, but that is session-sensitive evidence and does not justify promoting the comparator after seeing truth.

**`xrv9xvzb`: the baseline bet was correct.** Pore maximum diameter 419.72 px and pore fraction 0.11363 are leading morphology drivers. BSE contrast stretching is flagged. A correct batch label does not validate every mask, prove distribution equivalence or authorise a manufacturing release.

All three v2 assignments survived 31/31 overlapping known-training-site deletion fits. Thus the two wrong bets are direct examples of stable misclassification. Deletion agreement measures sensitivity to this training set; it cannot be presented as correctness confidence. Scores remain uncalibrated.

## Comparators and practical next steps

The saved v2 morphology-only and combined comparators also score 1/3. Acquisition-only scores 2/3, correcting `fn0mhxef` and still missing `3e122cbj`. These models share the same queries and training sites; their results are correlated diagnostics, not extra test data. Morphology-only makes the same two swaps, so this result does not isolate Inlens texture as the cause. Selecting the acquisition family now would use feedback for method selection.

The next bounded audit should start with how measurement-quality flags affect every mask-dependent classifier input, including particle-interior ETD/Inlens features. Define a quality policy on known data, apply it consistently to fitting and scoring, and assess it with selection/imputation/scaling inside each held-out-site fold. Compare with v2 on the original known sites and use these three queries only as labelled failure examples. Flagging low contrast without changing a numerical input does not make that input reliable.

Review the low-contrast particle masks and the two queries' pore boundaries with Santosh. Batch labels do not supply phase or instance annotations. Request specimen/preparation/session grouping before claiming cross-session generalisation; when available, evaluate entire held-out groups. Any feedback-informed feature, quality-policy or model change requires a new version, and this drop becomes development evidence for that version. Do not globally exchange output labels, silently replace the primary with the better comparator, or remove all texture based on three labels. No model, feature list or QC threshold is changed by this evaluation.

## Reproduction and checks

Run `/opt/anaconda3/bin/python3 -m analysis.feedback_drop_01.evaluate` from the repository root. For a replay, use `--out` with a new directory. The script fits nothing, verifies the original artifact manifests, checks class encoding and saved-score reproduction, reconstructs exact assigned-minus-runner logits, evaluates each saved family, and confirms the inputs remain unchanged. Original single-class top-feature contributions are not substituted for the pairwise explanation.

[All saved-family outcomes](output/evaluation.csv) · [Summary](output/summary.csv) · [Primary outcomes and stability](output/primary_outcomes.csv) · [Confusion counts](output/confusion_v2.csv) · [Pairwise family contributions](output/driver_family_contrasts.csv) · [Integrity receipt](output/receipt.json) · [Pre-feedback visual evidence](../submission_v2/first_run/submission/report.html).

Review applied: C01–C06/C09/C11/C14/C16–C18/C21/C27/C32/C33R/C34S and the new C35S feedback/stability check. No battery-performance, mask-accuracy, calibrated-confidence or release claim follows.
