# First organiser drop: prediction protocol

Recorded after folder arrival and before the first prediction or image-statistic inspection. Existing E31
methods were developed on Batches 1–3 and fixed before this drop; this execution snapshot is not a claim that
the new explanatory wrapper was preregistered before arrival. Ground truth is pending.

The input is `Hackathon-Polaron-test`: image sets grouped by site ID, with BSE, ETD/SE and Inlens channels.
The folder can contain multiple batches, so it is not treated as one production batch for a pooled QC verdict.

* Fit the existing E31 combined categoriser on the 31 known sites only, with seed 0, balanced class weights,
  the existing feature family and fold-local C selection. Save fitted models before extracting test features.
* The primary bet is always the existing combined model's argmax over Batch 1/2/3. Existing tie behaviour is
  retained. Morphology-only and acquisition-only predictions are explanatory sensitivities, not replacement
  models chosen after viewing test results.
* Report all three primary model probabilities and the top-minus-runner-up margin. They are uncalibrated
  model scores, not validated probabilities of correct assignment. Do not introduce a new confidence cutoff.
* Score test sites with the combined model's existing 31 known-training-site deletion fits. Report assignment
  counts and score ranges as training-set sensitivity, not independent votes or probabilities of correctness.
  No test site, test label or pseudo-label enters a fit, scaler, imputer, C choice or model selection.
* Fit the existing morphology OOD model on all 17 Batch 3 sites. Report its descriptive distance, percentile,
  nearest references, quality flags and drivers separately from the batch assignment. No calibrated membership
  or false-alert claim follows from its tail rank.
* Use existing per-feature measurements and linear-score contributions to explain the bet and differences
  from Batch 3. Label acquisition and texture evidence explicitly. Example images use fixed display limits
  and documented regions; they are explanatory, not independent material validation.
* Save site-level and image-level submission CSVs, an illustrated report, source/data/model hashes, and the
  first-prediction receipt before feedback arrives. Preserve the first predictions for tomorrow's evaluation.
* No feature/threshold changes, test-driven tuning, alternative-label override or raw-data commit. Organiser
  messages are recorded as supplied by Santosh; the submission is prepared locally, not sent automatically.

Final folders can be scored with the same frozen models and source checks using the delivery command in the
README. The wrapper requires complete aligned three-channel sets; an unsupported format must be reported
explicitly rather than silently discarded.
