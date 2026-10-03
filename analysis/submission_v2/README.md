# Frozen D52 v2 submission

The declared primary is morphology plus acquisition-sensitive ETD/Inlens appearance (29 features).
Frame height is excluded from every family; explicit session statistics occur only in comparators.
The family identifier `material` does not establish material origin: texture remains confounded.
Dimensions still support image QA and area normalisation.

This version was revised after the first drop was inspected, before its truth, and declared before final
evaluation. [Protocol](protocol.md), [saved-model snapshot](freeze/snapshot.json) and
[matching known-site evaluation](freeze/evaluation/categoriser_summary.csv) preserve that provenance.
The snapshot fits only 31 known sites and retains 31 primary known-site deletion fits. Scores are
uncalibrated; overlapping deletion fits measure training-set sensitivity, not correctness probability.

The [v2 illustrated submission](first_run/submission/report.html), [site predictions](first_run/submission/predictions.csv),
[all nine image assignments](first_run/submission/predictions_images.csv), [submission note](first_run/submission/submission.md)
and [delivery verification](verification.json) belong to this version. Original E33/v1 predictions remain
separate and unchanged. The model's weak Batch 2 recall must stay visible beside its overall development score.

## Final evaluation: three commands from the repository root

Use a new output directory for each arrival, and a fresh Python process for each command.
No known-data refitting or model selection takes place in these commands.

```bash
MPLCONFIGDIR=/private/tmp/polaron-v2-mpl /opt/anaconda3/bin/python3 -m analysis.submission_v2.score_folder --model-version categoriser-v2-D52 --input /absolute/path/to/final_folder --out analysis/final_evaluation_v2/scored --cache-dir _scratch/final_evaluation_v2_features
MPLCONFIGDIR=/private/tmp/polaron-v2-mpl /opt/anaconda3/bin/python3 -m analysis.submission_test.compose_submission --model-version categoriser-v2-D52 --primary-family material --run analysis/final_evaluation_v2/scored --out analysis/final_evaluation_v2/submission
MPLCONFIGDIR=/private/tmp/polaron-v2-mpl /opt/anaconda3/bin/python3 -m analysis.submission_v2.build_report --run analysis/final_evaluation_v2/scored --out analysis/final_evaluation_v2/submission
```

Scoring requires complete matching uint8 BSE/ETD/Inlens sets (`SE` aliases `ETD`), distinct incoming site IDs
and intact frozen source/model/dependency hashes. Renamed exact copies of training TIFFs are rejected as
holdouts. The scorer imports the preserved package snapshot, so later live-package edits cannot change its
model version. Changed runner code requires an explicitly registered revision.

Composition requires the declared version and primary family, matching score columns, consistent sample/image
bets and reliability evidence bound to that model snapshot. Missing material scores are an error; there is no
fallback to combined. Existing score/submission/report files are preserved. The submission note prints actual
family-specific reliability counts, without transplanting v1 confidence bands or claiming calibrated confidence.

Use the exact pairwise driver table to explain the forced bet, and known class/baseline medians to describe
what differs. Mask quality flags qualify the measurements. Descriptive baseline ranks cannot establish
membership, equivalence or defect probability. The input folder may mix batches and receives no pooled
manufacturing-batch verdict.

Save tomorrow's truth separately, evaluate the saved predictions, and append the outcome. A feature change
informed by feedback creates a separately registered version; these three labels cannot validate material
causes or justify automatically deleting texture. No external submission or Git commit is performed here.
