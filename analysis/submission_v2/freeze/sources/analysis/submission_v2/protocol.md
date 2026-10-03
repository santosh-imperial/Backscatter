# D52 v2 execution freeze (E35S / D54S)

Recorded after first-drop images and v1/v2 development predictions were inspected, before truth feedback and
before the final evaluation. This is a versioned execution freeze, not a claim that v2 was preregistered
before the first drop. D52 fixes the primary as morphology plus acquisition-sensitive ETD/Inlens appearance,
29 features. Frame height is excluded from every family. Explicit session statistics remain in comparator
families only. Texture remains confounded; "material" is a feature-family identifier, not proven material origin.

Fit only the 31 known sites (Batch 1/2/3: 7/7/17), using existing fold-local imputation/scaling/C selection,
seed 0, C grid 0.1/0.5/2, 3 inner folds and 200 site-label permutations. Preserve known-site reliability
for this family. Fit baseline morphology distance separately on the full measurable Batch 3 reference.
Save fitted models, all primary deletion fits, known tables, raw training manifests, source copies and hashes.
No incoming site enters fitting, evaluation, normalisation estimates, C selection or probability calibration.

For each complete aligned incoming detector triplet, use the saved material-family argmax. Always take a bet.
Report all three uncalibrated model scores, runner-up margin, exact assigned-minus-runner linear-logit
contributions with known class/baseline medians, quality flags, descriptive baseline ranks, and sensitivity
across overlapping known-site deletion fits. Deletion counts are not independent votes or confidence intervals.
Low contrast can invalidate mask-derived measurements. A high score is not proof of material conformance.

Require explicit model version and primary family through scoring and composition. Verify source/model/input
hashes, required score columns and matching reliability evidence. No fallback to a comparator. Use fresh
output directories, preserve E33/v1 files, and never pool a potentially mixed sample folder for a production
batch verdict. Save truth feedback separately; evaluate these saved bets before any subsequent model revision.
Freeze v2 for final evaluation. A change informed by label feedback creates a separately registered version
and makes this first drop development evidence for that new version. No automatic deletion of texture after
one failed prediction. No organiser message or Git commit is part of this workflow.
