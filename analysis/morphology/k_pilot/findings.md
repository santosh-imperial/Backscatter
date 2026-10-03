# Task K / E30K — shape variability and directional phase association

Retain the measured geometry and image explanations for review. All seven material/QC uses remain deferred; phase truth, section representativeness and specimen independence are unresolved. No feature is selected or promoted from the strongest known-batch separation.

One fixed run: 31 sites, 252 descriptive comparison rows. 2/21 nominal quality-matched intervals exclude zero; these are correlated exploratory intervals without a multiplicity-adjusted test. A null interval is not equivalence.

## Nominal quality-matched comparisons

| KPI | incoming − reference | usable n (reference/incoming) | median difference | site-bootstrap 95% interval | deletion range |
|---|---|---|---|---|---|
| `bright_aspect_count_iqr` | Batch_1 − Batch_3 | 13/5 | 0.16692 | [-0.0845698, 1.39321] | [0.114666, 0.192077] |
| `bright_aspect_count_iqr` | Batch_2 − Batch_3 | 13/7 | 0.159284 | [-0.0840321, 0.302858] | [0.0947813, 0.182257] |
| `bright_aspect_count_iqr` | Batch_2 − Batch_1 | 5/7 | -0.00763676 | [-1.23393, 0.276711] | [-0.0721392, 0.0446181] |
| `bright_solidity_count_q10` | Batch_1 − Batch_3 | 13/5 | -0.0489531 | [-0.10245, -0.00818542] | [-0.0678035, -0.0401592] |
| `bright_solidity_count_q10` | Batch_2 − Batch_3 | 13/7 | -0.0370136 | [-0.0965403, 0.0253951] | [-0.0597129, -0.0282198] |
| `bright_solidity_count_q10` | Batch_2 − Batch_1 | 5/7 | 0.0119395 | [-0.0607167, 0.0824478] | [-0.0107599, 0.0307899] |
| `bright_circularity_count_iqr` | Batch_1 − Batch_3 | 13/5 | 0.00543522 | [-0.0339338, 0.0187492] | [-0.000974869, 0.00572536] |
| `bright_circularity_count_iqr` | Batch_2 − Batch_3 | 13/7 | 0.00444408 | [-0.031102, 0.040484] | [-0.000522408, 0.0129025] |
| `bright_circularity_count_iqr` | Batch_2 − Batch_1 | 5/7 | -0.000991138 | [-0.0263478, 0.0420398] | [-0.00390699, 0.00746725] |
| `bright_autocorr_xy_contrast_64px` | Batch_1 − Batch_3 | 13/5 | 0.0171755 | [-0.0374596, 0.0424825] | [0.00335744, 0.0222931] |
| `bright_autocorr_xy_contrast_64px` | Batch_2 − Batch_3 | 13/7 | 0.022513 | [-0.0172151, 0.0559931] | [0.0146396, 0.0281458] |
| `bright_autocorr_xy_contrast_64px` | Batch_2 − Batch_1 | 5/7 | 0.00533755 | [-0.0283654, 0.0618365] | [-0.000699908, 0.0191556] |
| `bright_autocorr_xy_contrast_256px` | Batch_1 − Batch_3 | 13/5 | -0.0118176 | [-0.0467592, 0.116071] | [-0.0218368, 0.00304584] |
| `bright_autocorr_xy_contrast_256px` | Batch_2 − Batch_3 | 13/7 | 0.00361714 | [-0.0448069, 0.0442985] | [-0.00660252, 0.0164445] |
| `bright_autocorr_xy_contrast_256px` | Batch_2 − Batch_1 | 5/7 | 0.0154347 | [-0.0867991, 0.0381385] | [0.000571305, 0.0195872] |
| `bright_pore_crosscorr_xy_contrast_64px` | Batch_1 − Batch_3 | 13/5 | -0.000536499 | [-0.00498329, 0.0104926] | [-0.00158979, 0.00239839] |
| `bright_pore_crosscorr_xy_contrast_64px` | Batch_2 − Batch_3 | 13/7 | 0.00116072 | [-0.00542865, 0.00785271] | [-0.000149195, 0.00284373] |
| `bright_pore_crosscorr_xy_contrast_64px` | Batch_2 − Batch_1 | 5/7 | 0.00169722 | [-0.00933185, 0.00691209] | [-0.00123767, 0.00338023] |
| `bright_pore_crosscorr_xy_contrast_256px` | Batch_1 − Batch_3 | 13/5 | 0.0189425 | [0.00547792, 0.0323507] | [0.0166502, 0.0197096] |
| `bright_pore_crosscorr_xy_contrast_256px` | Batch_2 − Batch_3 | 13/7 | 0.00839751 | [-0.010616, 0.0272882] | [0.00579713, 0.00888305] |
| `bright_pore_crosscorr_xy_contrast_256px` | Batch_2 − Batch_1 | 5/7 | -0.010545 | [-0.029154, 0.0119592] | [-0.0131454, -0.00825272] |

## Paired method sensitivity

Shape descriptors vary under ±5 intensity-level mask thresholds and a 100 px² object floor. Raster phase descriptors use the full mask, so their floor100 copies are identical by design; this is not an independent robustness result. No joint floor/threshold interaction was tested.

Pixel-pair marginals are re-estimated in each signed overlapping domain. There is no circular wrap or FFT periodic boundary. Correlations describe finite 2-D arrangement, not contact, chemistry, transport or confirmed collector/through-thickness direction.

## Control qualification

Two fixed ridge controls use fold-local median imputation/scaling and whole-site LOO. Negative R² means these linear controls predict poorly at this n; it does not prove novel material information or acquisition invariance. Count-weighted shape quantiles describe a different population from existing area-weighted means.

| KPI | view | acquisition LOO R² | existing-geometry/loading LOO R² |
|---|---|---|---|
| `bright_aspect_count_iqr` | phase_usable | -0.2513 | -0.4664 |
| `bright_aspect_count_iqr` | quality_matched | -0.3576 | -0.4811 |
| `bright_solidity_count_q10` | phase_usable | 0.1322 | 0.6475 |
| `bright_solidity_count_q10` | quality_matched | -0.03866 | 0.6511 |
| `bright_circularity_count_iqr` | phase_usable | -0.607 | -0.1984 |
| `bright_circularity_count_iqr` | quality_matched | -0.7185 | 0.05135 |
| `bright_autocorr_xy_contrast_64px` | phase_usable | -0.326 | -0.5675 |
| `bright_autocorr_xy_contrast_64px` | quality_matched | -0.6015 | -0.5585 |
| `bright_autocorr_xy_contrast_256px` | phase_usable | -0.4675 | -0.9346 |
| `bright_autocorr_xy_contrast_256px` | quality_matched | -0.6641 | -0.8706 |
| `bright_pore_crosscorr_xy_contrast_64px` | phase_usable | -0.462 | -0.5371 |
| `bright_pore_crosscorr_xy_contrast_64px` | quality_matched | -0.462 | -0.5371 |
| `bright_pore_crosscorr_xy_contrast_256px` | phase_usable | -0.2422 | -0.503 |
| `bright_pore_crosscorr_xy_contrast_256px` | quality_matched | -0.2422 | -0.503 |

## Provenance and validation limits

Shape geometry is a declared warm reuse of E26G verified full-frame component tables at the selected settings; mask phase-pair counts are freshly reconstructed from raw BSE. Original E26G raw/source hashes were checked before reuse. No originals or historical caches were overwritten. Raw replay of prespecified nominal examples checks geometry binding, not chemical identity or expert accuracy.

At least 20 eligible bright components are required for count quantiles. Finite pair domains require at least 10000 pixels and at least 100 pixels of each phase and complement. These are computational coverage gates, not statistical validation. Missing quality flags abstain; grey-pore exclusion is required for cross-phase descriptors.

Nonlinear depth structure is deferred: no confirmed collector direction or independently reviewed phase/section labels. Independent annotation and practical tolerances remain the next gates. Known batches were explored previously; no unseen batch was used.

Review: C01–C07/C09/C11/C13/C14/C16/C17/C21–C23/C28–C30.

## Post-run qualification

Two correlated nominal quality-matched comparisons have intervals excluding zero. Both remain exploratory and outside QC; neither establishes a manufacturing defect, a battery mechanism or acceptance/rejection. Sites may share specimens, and phase masks/image direction are unreviewed.

## Solidity lower tail: sensitive and partly predictable from existing geometry

Batch 1 − Batch 3 is -0.048953, site-bootstrap 95% [-0.102450, -0.008185], at 5 incoming / 13 reference sites. The nominal grey-pore-included interval spans zero. Each of the ±5 threshold and 100 px² floor quality-matched intervals also spans zero. The existing geometry/loading LOO control reaches R² 0.651. This is a count-weighted tail, distinct from the existing area-weighted mean; it is not established incremental material information.

Review whether low-solidity components are physical sections, particle rims, threshold fragments or preparation effects. A lower raster solidity can reflect any of those. Preserve uncertain/unmeasurable cases; independent annotations are needed.

## 256 px bright/void direction contrast: repeatable within this limited mask test

Batch 1 − Batch 3 is 0.018943, site-bootstrap 95% [0.005478, 0.032351], at 5 incoming / 13 reference sites. The nominal, −5 and +5 quality-matched intervals exclude zero with the same direction, as do the corresponding ordinary-reference sensitivity intervals. The floor100 raster duplicate is identical by definition and adds no robustness evidence.

This supports further image/phase/section review of a finite-frame arrangement difference, not an established material or failure mechanism. Only two fixed lags were evaluated. Removing known long-void reference sites changes the reference population; it does not certify a clean baseline. Marginal normalisation and poor linear control predictability do not prove acquisition invariance.

Review bright/void masks at the relevant image-axis spacing, the direction/context of each section and possible preparation/session fingerprints. Confirm specimen independence and repeat sections before a material interpretation. No new tuning, extra lag, trained classifier or unseen-batch selection was run.

Other nominal quality-matched intervals include zero; that is limited evidence at these usable n, not equivalence. These are unadjusted correlated exploratory intervals. All seven material/QC uses stay deferred.
