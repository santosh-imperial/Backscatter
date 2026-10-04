# E37S: classifier input-quality audit

The quality gate does not improve known-site balanced accuracy in the matched comparison.

Dependency gating: known-site balanced accuracy 0.568627, versus matched legacy 0.607843; delta -0.039216. Frozen D52 v2 was 0.607843; the matched audit uses different shared inner-fold allocation, so these baselines must remain distinct.

[Protocol](protocol.md) was fixed after first-drop feedback, before these comparisons. The drop is development evidence for this policy. All 31 known sites remain included (7/7/17), including cracked Batch 3 reference sites. No height, explicit session statistic or quality indicator enters any 29-input material/appearance candidate. All appearance inputs retain their acquisition qualifications.

| variant            |   n_sites |   balanced_accuracy |   accuracy |   recall_Batch_1 |   recall_Batch_2 |   recall_Batch_3 |   perm_p |   brier |
|:-------------------|----------:|--------------------:|-----------:|-----------------:|-----------------:|-----------------:|---------:|--------:|
| legacy_matched     |        31 |              0.6078 |     0.6774 |           0.7143 |           0.2857 |           0.8235 |   0.0050 |  0.4625 |
| direct_mask_gate   |        31 |              0.6359 |     0.6774 |           0.7143 |           0.4286 |           0.7647 |   0.0050 |  0.4545 |
| dependency_gate    |        31 |              0.5686 |     0.6129 |           0.5714 |           0.4286 |           0.7059 |   0.0100 |  0.5365 |
| quality_flags_only |        31 |              0.4286 |     0.6129 |           0.2857 |           0.0000 |           1.0000 |   0.1095 |  0.5375 |

Paired OOF descriptive site-resampling delta interval [-0.1736694677871148, 0.09523809523809523]; 1 newly correct and 3 newly wrong, 4 labels changed. Overlapping CV fits and unknown specimen dependence prevent a calibrated generalisation interpretation. Primary permutation p is 0.009950 with 200 full nested site-label permutations; advantage permutation diagnostic 0.771144. Other policy/control p-values are secondary diagnostics, not selectable winners.

Bright-only failure invalidates 19 of 29 candidate inputs; raised-black-level pore failure invalidates 16. Both unknown/failing masks leave only the two raw-BSE appearance summaries. Neither remaining inputs nor their imputed baseline terms are automatically trustworthy. Missingness indicators are absent, but imputation patterns can still carry acquisition information, as the quality-flags-only control tests.

ETD particle ridge coverage depends on the bright interior AND the dominant orientation estimated using joint phase interiors. ETD graphite/ridge summaries and Inlens gradient energy also inherit phase-mask selection. Inlens particle standard-deviation summaries depend on the bright interior. A flag shown beside a score does not disable its numerical influence in v2. This audit marks invalid inputs missing before all fitting/selection/scoring.

## Labelled failure examples — development only

| variant            | site     | true_batch   | predicted_batch   | runner_up   |   top_model_score |   true_batch_model_score |   n_observed_inputs |   deletion_same_assignment |
|:-------------------|:---------|:-------------|:------------------|:------------|------------------:|-------------------------:|--------------------:|---------------------------:|
| legacy_matched     | 3e122cbj | Batch_2      | Batch_1           | Batch_2     |            0.8382 |                   0.0832 |                  29 |                         31 |
| legacy_matched     | fn0mhxef | Batch_1      | Batch_2           | Batch_1     |            0.4956 |                   0.3342 |                  29 |                         31 |
| legacy_matched     | xrv9xvzb | Batch_3      | Batch_3           | Batch_2     |            0.6218 |                   0.6218 |                  29 |                         31 |
| direct_mask_gate   | 3e122cbj | Batch_2      | Batch_1           | Batch_2     |            0.7803 |                   0.1379 |                  17 |                         31 |
| direct_mask_gate   | fn0mhxef | Batch_1      | Batch_2           | Batch_1     |            0.5024 |                   0.3597 |                  29 |                         31 |
| direct_mask_gate   | xrv9xvzb | Batch_3      | Batch_3           | Batch_1     |            0.6848 |                   0.6848 |                  29 |                         31 |
| dependency_gate    | 3e122cbj | Batch_2      | Batch_1           | Batch_3     |            0.5532 |                   0.1272 |                  10 |                         31 |
| dependency_gate    | fn0mhxef | Batch_1      | Batch_2           | Batch_1     |            0.5233 |                   0.3048 |                  29 |                         28 |
| dependency_gate    | xrv9xvzb | Batch_3      | Batch_3           | Batch_2     |            0.7092 |                   0.7092 |                  29 |                         28 |
| quality_flags_only | 3e122cbj | Batch_2      | Batch_1           | Batch_2     |            0.8621 |                   0.0690 |                   2 |                         31 |
| quality_flags_only | fn0mhxef | Batch_1      | Batch_3           | Batch_2     |            0.3918 |                   0.2639 |                   2 |                         31 |
| quality_flags_only | xrv9xvzb | Batch_3      | Batch_3           | Batch_2     |            0.3918 |                   0.3918 |                   2 |                         31 |

These examples were read only after known-only evaluation and candidate fits were saved. Their labels did not select C, transformations or variants. Their correctness is not new held-back accuracy; all three labels were already seen before the audit. All three procedures with 29 declared inputs still bet B1/B2/B3, so each remains 1/3 correct on these development examples. The full gate lowers the low-contrast wrong bet from 0.8382 to 0.5532, with true-class score only 0.1272 and Batch 3 now the runner-up. Remaining pore/image inputs still support the mistake; removed mask-dependent values cannot be the whole explanation. Deletion counts are training-set sensitivity, not independent votes or confidence. Exact logit tables identify imputed baseline terms separately from observed inputs. The quality-only model gives that wrong Batch 1 bet score 0.8621, but its overall known-site permutation p 0.1095 does not demonstrate above-null separation.

## Recommendation

Keep the dependency policy as an experimental safety candidate and preserve frozen v2 as the declared submission. Invalid measurements must not be described as reliable material evidence. No result here establishes cross-session accuracy or warrants automatic deployment; a separate version/decision is needed. Independent boundary annotations and specimen/session grouping remain the next validation gates.

## Evidence and review

[Illustrated report](report.html) · [Dependency catalogue](output/feature_dependencies.csv) · [Known OOF scores](output/known_oof_predictions.csv) · [Coverage](output/feature_availability.csv) · [Quality subgroups](output/quality_subgroups.csv) · [Failure eligibility](output/failure_input_eligibility.csv) · [Exact logit terms](output/failure_logit_contrasts.csv) · [Legacy invalid-input influence](output/legacy_invalid_input_influence.csv) · [Known-stage receipt](output/known_stage_receipt.json) · [Complete receipt](output/receipt.json).

Image selections are unreviewed algorithm masks. Full-site overlay fractions reproduce saved values; central crops have recorded raw coordinates and fixed display. Batch labels do not supply phase/instance truth. Safety regression checks cover rejected-value invariance, missing flags, downstream dependencies, fold-local medians, empty training columns and site/row renaming. Frozen submission artifacts and production QC remain unchanged. Review C01–C06/C09/C11/C14/C16–C18/C21/C27–C30/C32–C35S plus C36S extraction dependencies.
