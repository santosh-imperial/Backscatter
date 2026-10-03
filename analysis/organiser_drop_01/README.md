# First organiser drop: frozen bets and explanations

The organiser requires a batch bet even when uncertain. Correct batch identification measures how a sample
differs from Batch 3. These are predictions before feedback; true labels and accuracy remain unknown.

| Sample | Bet | Model scores B1 / B2 / B3 | Same bet in known-training-site deletion fits |
|---|---|---|---|
| 3e122cbj | **Batch 1** | 0.765 / 0.147 / 0.087 | 31 / 31 |
| fn0mhxef | **Batch 2** | 0.403 / 0.451 / 0.146 | 28 / 31 |
| xrv9xvzb | **Batch 3** | 0.081 / 0.177 / 0.742 | 31 / 31 |

The assignment applies to all three detector files of each site. Scores are uncalibrated model preferences.
Deletion fits overlap and measure training-set sensitivity, not independent votes or correctness probabilities.

Open the [illustrated report](first_run/report.html), [site submission](first_run/submission_sites.csv),
[all nine image assignments](first_run/submission_images.csv), [full driver table](first_run/report_driver_contrasts.csv)
and [verification receipt](verification.json). Original predictions are bound by the
[first-prediction receipt](first_run/prediction_receipt.json); its exact wrapper source is preserved under
first_run/source/. They match the independently generated E33 CLI submission in analysis/submission_test/,
which is preserved.

E34R's first_run means the first Codex execution of this drop. E33's team submission is separately saved and
committed; no claim is made that E34R preceded the other run.

3e122cbj has lower ETD ridge response, higher measured Inlens interior texture and higher void-component
density than the baseline medians. Its low-contrast bright mask is unreliable; mask-derived texture and its
100th-percentile morphology distance cannot establish material departure. Missing contrast separation was
imputed by the fitted known-data imputer. This is a Batch 1 fingerprint bet.

fn0mhxef is the close bet: its Batch 2 score exceeds Batch 1 by only 0.048. Lower void-component density
(86.9 /Mpx versus the Batch 3 median 98.1) favours Batch 2. ETD ridge response and boundary sharpness favour
Batch 1 and largely counter it. Morphology-only predicts Batch 2; acquisition-only predicts Batch 1.

xrv9xvzb favours Batch 3 on all three families. Greater segmented void area and maximum equivalent void
diameter favour Batch 3 relative to its runner-up. Its morphology distance is at the 17.6th reference
percentile. BSE contrast stretching is flagged; this is resemblance, not equivalence or a defect-free label.

The driver contrasts exactly decompose assigned-class minus runner-up linear logits. Positive contributions
favour the bet, negative contributions favour its rival. They explain the model, not a physical cause.
Overlay phase fractions reproduce the original full-site measurements; crops use fixed display limits and
recorded coordinates. Browser local-file preview was blocked by URL policy; HTML structure, image bytes,
links and scientific panels were checked directly.

## Replay the frozen v1 model on another evaluation folder

This preserves the first-drop v1 combined model, including its acquisition-sensitive inputs. The parallel
E34/D52 update adopts the material family as the current primary; use the main repository categoriser
workflow for that version. These commands replay v1 as a versioned comparator, not the updated primary.

From the repository root, using the same frozen known-only models:

    MPLCONFIGDIR=/private/tmp/polaron-eval-mpl /opt/anaconda3/bin/python3 -m analysis.organiser_drop_01.score_folder --input /absolute/path/to/evaluation_folder --out analysis/final_evaluation_first_run --cache-dir _scratch/final_evaluation_features
    MPLCONFIGDIR=/private/tmp/polaron-eval-mpl /opt/anaconda3/bin/python3 -m analysis.organiser_drop_01.build_report --out analysis/final_evaluation_first_run

The scorer imports the preserved source snapshot and checks its model/source/dependency hashes, complete
matching uint8 grayscale/RGB detector sets and distinct site IDs. Concurrent edits to the live package are
ignored by this versioned replay; already imported live modules require a fresh Python process.
It fits nothing on incoming data. A nonempty output directory is rejected to preserve first predictions.
Another input format needs an explicit ingestion adaptation; do not silently discard images or change the model.
A folder can mix source batches; pooling it for a manufacturing-batch verdict needs confirmed common provenance.

The [protocol](protocol.md) and [pre-prediction snapshot](freeze/snapshot.json) document the known-only fit.
Save tomorrow's truth separately and append an evaluation without editing these bets. No external message
is sent by this workflow.
