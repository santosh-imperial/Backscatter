# Final evaluation assignments · E44S

The frozen v2 model scored six samples from `Hackathon-Polaron-eval`. Each sample has three detector images. The assignment applies to all three files. True labels are pending. No model was fitted on the evaluation data.

| Sample | Assigned batch | B1 score | B2 score | B3 score | Runner-up | Margin |
|---|---|---:|---:|---:|---|---:|
| 0eryguqq | Batch 3 | 0.067 | 0.178 | 0.755 | Batch 2 | 0.576 |
| 4hq27w4c | Batch 2 | 0.338 | 0.551 | 0.112 | Batch 1 | 0.213 |
| fhwrjtet | Batch 3 | 0.083 | 0.092 | 0.825 | Batch 2 | 0.732 |
| fspqbkxl | Batch 2 | 0.199 | 0.466 | 0.335 | Batch 3 | 0.131 |
| soo2ax3r | Batch 2 | 0.291 | 0.398 | 0.310 | Batch 3 | 0.088 |
| y59rxmxl | Batch 1 | 0.395 | 0.279 | 0.327 | Batch 3 | 0.068 |

These scores are uncalibrated model scores. They are not probabilities of a correct assignment. The model chooses the highest score, even when it is below 0.5. `soo2ax3r` and `y59rxmxl` have the smallest margins. Their assignments have substantial uncertainty.

The model uses 29 features from morphology and ETD/Inlens appearance. Frame height and explicit session statistics are excluded from the primary assignment. The origin of texture differences remains unresolved. Known crops can share source images. The known-data tests do not establish accuracy on new source images.

## What supports each assignment

Each explanation compares the assigned class with its runner-up. Positive contributions support the assignment. Negative contributions oppose it. The values explain this model, not a physical cause. The [full driver table](submission/driver_contrasts.csv) contains the exact contributions and class medians.

- **0eryguqq → Batch 3:** low Inlens particle texture and high ETD graphite ridge fractions support Batch 3 over Batch 2. Low void area opposes this assignment. All 31 deletion fits retain Batch 3.
- **4hq27w4c → Batch 2:** low void area and fewer resolved voids support Batch 2 over Batch 1. Particle size and high-end Inlens texture oppose the assignment. Thirty of 31 deletion fits retain Batch 2.
- **fhwrjtet → Batch 3:** high ETD graphite ridge fractions and low Inlens particle texture support Batch 3 over Batch 2. Morphology alone assigns Batch 1. All 31 deletion fits retain Batch 3.
- **fspqbkxl → Batch 2:** high Inlens particle texture and gradient energy support Batch 2 over Batch 3. ETD graphite ridge fractions oppose this assignment. Morphology alone assigns Batch 3. Twenty-nine of 31 deletion fits retain Batch 2.
- **soo2ax3r → Batch 2:** high Inlens particle texture and gradient energy support Batch 2 over Batch 3. Void area and ETD ridge measurements oppose it. Both separate comparators assign Batch 3. Twenty-eight of 31 deletion fits retain Batch 2.
- **y59rxmxl → Batch 1:** low ETD graphite ridge fractions and higher Inlens particle texture support Batch 1 over Batch 3. The large maximum void diameter opposes the assignment. Morphology alone assigns Batch 3. Twenty-six of 31 deletion fits retain Batch 1.

Deletion fits share most of their fit data. Their agreement measures model sensitivity to the known crops. It does not measure the probability of a correct assignment. The first drop showed that all deletion fits can agree on a wrong assignment.

## Quality and baseline context

None of the four listed acquisition flags fired. This result does not establish reliable phase masks or equivalent image conditions. Six files contain coloured export borders. The three `4hq27w4c` files have one coloured column at the right edge. The three `y59rxmxl` files have two coloured columns at the left edge. The frozen loader uses plane 0 for all RGB exports. No image or loader rule changed.

No sample exceeds the observed maximum baseline morphology distance. This is a descriptive comparison with 17 reference crops. It does not establish baseline membership, acceptable quality or battery performance. The folder can contain different batches. It receives no pooled QC verdict.

## Saved outputs and checks

- [Full sample scores](submission/predictions.csv)
- [Assignments for all 18 image files](submission/predictions_images.csv)
- [Image evidence and model explanations](submission/report.html)
- [First-prediction receipt](scored/prediction_receipt.json)
- [Independent numerical replay](verification.json)
- [Input checks](input_pixel_checks.json)
- [Procedure fixed before inference](protocol.md)

The checks reproduced every assignment and score from the saved model. The largest score difference was below 3 × 10⁻¹⁵. Each feature contribution reproduced its saved linear term. All 18 input hashes and 1,219 earlier file hashes remained unchanged. The report contains six sample cards and 18 image panels. Three detector and mask panels were checked visually.

The report uses the preserved renderer. This handoff supplies the current data limits and short STE text. Save future labels separately. Evaluate the saved predictions before any new model change. No organiser message or Git commit was made.
