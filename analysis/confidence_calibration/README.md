# E43S: scalar score correction for frozen v2

Declared outcome: the criterion is not met; the original frozen v2 remains the submission.

The deployment-only inverse temperature is **α = 0.527967**
(T = 1.894059); it softens the normalized score vector.
The 31 known evaluation rows use distinct outer-fold scalars, not this full fit.
All forced batch bets and correctness targets are preserved.

Equal-class multiclass log loss changes 1.064 → 1.032, while equal-class own-bet Brier changes 0.209 → 0.243. Mean wrong-bet score falls 0.508 → 0.427; mean right-bet score also falls 0.616 → 0.460. This tradeoff does not establish better correctness calibration.

This layer applies `softmax(alpha * log(p))` to **normalized OvR probabilities**
from the saved D52 v2 material classifier. Alpha 1 reproduces the original scores.
It does not apply softmax to raw classifier logits and creates no new material
features or feature attributions. Scores are a fitted correction; new-source
correctness calibration remains unproved.

## Evidence

- [HTML report](report.html): original/corrected paired scores, losses, promotion
  checks, fitted temperatures and all three revealed development examples.
- [Final protocol](protocol.md) and [initial draft](protocol_initial_threefold.md):
  the reviewer-driven change from three-fold internal scoring to paired internal
  crop-LOO was made before fitting.
- [Known scores](output/known_predictions.csv),
  [development scores](output/feedback_predictions.csv),
  [all summary losses](output/summary.csv) and [temperature fits](output/temperatures.csv).
- [Declared promotion gate](output/promotion.json),
  [known-before-feedback receipt](output/known_receipt.json) and
  [complete receipt](output/receipt.json).
- [Preserved v2 model and limitations](../submission_v2/README.md),
  [feature drivers](../submission_v2/first_run/scored/driver_contrasts.csv) and
  [feature explanations](../submission_v2/first_run/scored/explanations.json).

The practical gate requires at least 1% relative improvement in equal-class
multiclass log loss and own-bet correctness Brier, no worsening beyond 1e-6 in
the other six losses, unchanged bets and complete provenance/replay. Only the
fitted scalar can qualify; fixed softening and uniform scores are controls.
Exact organiser utility and confidence cutoffs are unknown.

The E42S matched comparator differs from saved frozen v2 on `epqdaau9`
(C = 0.5 versus C = 2), despite unchanged bets. Its historical log loss
cannot be reused as the actual-v2 baseline. This study replays the saved v2
deletions and sealed v2 score table; the
[comparator difference](output/historical_comparator_difference.json) is retained.

The artificial visual classes contain crops from around 15 source images.
Parent IDs remain unknown, so related crops can cross folds. Crop-held-out losses
are descriptive development evidence, not source-held-out correctness confidence
or manufacturing-performance validation. The three revealed labels were read
after the known-stage decision and never fit or select alpha.

## Render saved evidence

From the repository root:

```bash
MPLCONFIGDIR=/private/tmp/polaron-calibration-report-mpl /opt/anaconda3/bin/python3 -m analysis.confidence_calibration.build_report
```

This command reads the saved results and writes the report, two scientific plots
and this guide. It performs no classifier or scalar fit. The experiment producer
`analysis.confidence_calibration.run` refuses an existing output directory;
preserve the as-run results rather than overwriting them.

Native browser layout remains unverified because local-file navigation was
previously blocked. Scientific charts, saved score identities and links are
checked separately; no browser workaround is used.
