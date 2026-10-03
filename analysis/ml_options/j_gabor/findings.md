# Task J / E28J: fixed Gabor texture pilot

Known-site development, fixed before new outputs. This is deterministic image filtering, with no neural training, material labels, defect accuracy or unseen-batch validation. Three summaries stay outside primary/ML/verdict inputs.

Four non-overlapping 512px BSE windows are centred at quarter-frame coordinates. The nominal 2×2 area mean precedes per-window mean-zero/unit-SD normalisation. Twelve DC-removed, unit-L2 complex kernels cover raw periods16/32/64px and wavevector angles0/45/90/135°. Sigma is half the period; support is ±3 sigma with fixed square support. A common96px raw border is discarded. Each window contributes equally; four windows are coverage, never n.

Scale/direction refer to appearance of the BSE image, not phase identity, graphite plates, pore transport or material defect labels. Filter energy share is not a particle-size fraction. The full-frame window coordinates and valid-area fractions are saved per site. Spatial representativeness is unvalidated.

All-known view includes acquisition subgroups because no phase mask is used. Quality-matched excludes known low-bright-contrast/grey-pore flags; ordinary-reference additionally excludes known long-void B3 examples without calling that subset clean. Unknown quality does not certify a matched site.

Sampling intervals use4000 percentile site-bootstrap draws, conditional on unknown specimen independence. Whole-site deletion ranges, resolution and gamma/affine sensitivity are separate from sampling uncertainty. No p-values, C2ST or equivalence test.

## Nominal site comparisons

| mode    | view               | kpi                                 | reference   | batch   |   n_reference |   n_batch |   median_reference |   median_batch |   median_difference |     ci_low |    ci_high |   deletion_min |   deletion_max |
|:--------|:-------------------|:------------------------------------|:------------|:--------|--------------:|----------:|-------------------:|---------------:|--------------------:|-----------:|-----------:|---------------:|---------------:|
| nominal | all_known          | gabor_coarse_energy_share           | Batch_3     | Batch_1 |            17 |         7 |            0.82419 |        0.80728 |          -0.016906  | -0.033039  | -0.0059028 |     -0.019219  |    -0.015371   |
| nominal | all_known          | gabor_coarse_energy_share           | Batch_3     | Batch_2 |            17 |         7 |            0.82419 |        0.8208  |          -0.0033888 | -0.02685   |  0.021595  |     -0.007548  |     0.0030001  |
| nominal | all_known          | gabor_coarse_energy_share           | Batch_1     | Batch_2 |             7 |         7 |            0.80728 |        0.8208  |           0.013517  | -0.0056996 |  0.040198  |      0.0093578 |     0.019906   |
| nominal | all_known          | gabor_axial_strength                | Batch_3     | Batch_1 |            17 |         7 |            0.19549 |        0.19806 |           0.0025692 | -0.12209   |  0.047345  |     -0.010228  |     0.0087657  |
| nominal | all_known          | gabor_axial_strength                | Batch_3     | Batch_2 |            17 |         7 |            0.19549 |        0.16035 |          -0.035147  | -0.14367   |  0.015437  |     -0.040113  |    -0.02064    |
| nominal | all_known          | gabor_axial_strength                | Batch_1     | Batch_2 |             7 |         7 |            0.19806 |        0.16035 |          -0.037716  | -0.10294   |  0.033786  |     -0.042446  |    -0.023209   |
| nominal | all_known          | gabor_horizontal_wavevector_balance | Batch_3     | Batch_1 |            17 |         7 |           -0.18505 |       -0.15446 |           0.030589  | -0.063506  |  0.10137   |      0.013545  |     0.031449   |
| nominal | all_known          | gabor_horizontal_wavevector_balance | Batch_3     | Batch_2 |            17 |         7 |           -0.18505 |       -0.15613 |           0.028914  | -0.055486  |  0.099297  |      0.014655  |     0.03254    |
| nominal | all_known          | gabor_horizontal_wavevector_balance | Batch_1     | Batch_2 |             7 |         7 |           -0.15446 |       -0.15613 |          -0.0016757 | -0.079935  |  0.08615   |     -0.015935  |     0.015369   |
| nominal | quality_matched    | gabor_coarse_energy_share           | Batch_3     | Batch_1 |            13 |         5 |            0.82419 |        0.81035 |          -0.013835  | -0.030485  |  0.0072991 |     -0.017493  |    -0.012045   |
| nominal | quality_matched    | gabor_coarse_energy_share           | Batch_3     | Batch_2 |            13 |         7 |            0.82419 |        0.8208  |          -0.0033888 | -0.027021  |  0.021953  |     -0.007548  |     0.0030001  |
| nominal | quality_matched    | gabor_coarse_energy_share           | Batch_1     | Batch_2 |             5 |         7 |            0.81035 |        0.8208  |           0.010447  | -0.017073  |  0.040198  |      0.0062874 |     0.016835   |
| nominal | quality_matched    | gabor_axial_strength                | Batch_3     | Batch_1 |            13 |         5 |            0.20543 |        0.19806 |          -0.0073633 | -0.12209   |  0.05867   |     -0.02016   |     0.0055311  |
| nominal | quality_matched    | gabor_axial_strength                | Batch_3     | Batch_2 |            13 |         7 |            0.20543 |        0.16035 |          -0.04508   | -0.16276   |  0.014847  |     -0.052682  |    -0.030572   |
| nominal | quality_matched    | gabor_axial_strength                | Batch_1     | Batch_2 |             5 |         7 |            0.19806 |        0.16035 |          -0.037716  | -0.11321   |  0.033181  |     -0.050611  |    -0.023209   |
| nominal | quality_matched    | gabor_horizontal_wavevector_balance | Batch_3     | Batch_1 |            13 |         5 |           -0.18677 |       -0.15446 |           0.032309  | -0.057333  |  0.13868   |      0.015264  |     0.033308   |
| nominal | quality_matched    | gabor_horizontal_wavevector_balance | Batch_3     | Batch_2 |            13 |         7 |           -0.18677 |       -0.15613 |           0.030633  | -0.054722  |  0.15222   |      0.016374  |     0.03426    |
| nominal | quality_matched    | gabor_horizontal_wavevector_balance | Batch_1     | Batch_2 |             5 |         7 |           -0.15446 |       -0.15613 |          -0.0016757 | -0.075333  |  0.08615   |     -0.015935  |     0.015369   |
| nominal | ordinary_reference | gabor_coarse_energy_share           | Batch_3     | Batch_1 |            10 |         5 |            0.827   |        0.81035 |          -0.016645  | -0.030372  |  0.010747  |     -0.02115   |    -0.01214    |
| nominal | ordinary_reference | gabor_coarse_energy_share           | Batch_3     | Batch_2 |            10 |         7 |            0.827   |        0.8208  |          -0.0061984 | -0.027021  |  0.030235  |     -0.010704  |     0.00019044 |
| nominal | ordinary_reference | gabor_coarse_energy_share           | Batch_1     | Batch_2 |             5 |         7 |            0.81035 |        0.8208  |           0.010447  | -0.017073  |  0.040198  |      0.0062874 |     0.016835   |
| nominal | ordinary_reference | gabor_axial_strength                | Batch_3     | Batch_1 |            10 |         5 |            0.24999 |        0.19806 |          -0.051931  | -0.14118   |  0.05867   |     -0.096499  |    -0.0073633  |
| nominal | ordinary_reference | gabor_axial_strength                | Batch_3     | Batch_2 |            10 |         7 |            0.24999 |        0.16035 |          -0.089647  | -0.18002   |  0.014832  |     -0.13421   |    -0.04508    |
| nominal | ordinary_reference | gabor_axial_strength                | Batch_1     | Batch_2 |             5 |         7 |            0.19806 |        0.16035 |          -0.037716  | -0.11321   |  0.033181  |     -0.050611  |    -0.023209   |
| nominal | ordinary_reference | gabor_horizontal_wavevector_balance | Batch_3     | Batch_1 |            10 |         5 |           -0.22069 |       -0.15446 |           0.066233  | -0.049811  |  0.15528   |      0.032309  |     0.10016    |
| nominal | ordinary_reference | gabor_horizontal_wavevector_balance | Batch_3     | Batch_2 |            10 |         7 |           -0.22069 |       -0.15613 |           0.064558  | -0.042749  |  0.17684   |      0.030633  |     0.098482   |
| nominal | ordinary_reference | gabor_horizontal_wavevector_balance | Batch_1     | Batch_2 |             5 |         7 |           -0.15446 |       -0.15613 |          -0.0016757 | -0.075333  |  0.08615   |     -0.015935  |     0.015369   |

## Fixed nuisance / redundancy controls

Ridge alpha1 predicts each summary from six acquisition covariates, separately from eleven existing FFT/phase/loading/image-tensor summaries. Median imputation/scaling refit inside every site-LOO fold. Both all-known and quality-matched controls accompany within-batch and ordinary-reference correlations. Prediction is not causal attribution; poor prediction does not prove invariance or independent information.

| view            | kpi                                 | control             |   loo_r2 |   n_sites |
|:----------------|:------------------------------------|:--------------------|---------:|----------:|
| all_known       | gabor_axial_strength                | acquisition         |  -0.3796 |        31 |
| all_known       | gabor_axial_strength                | fft_geometry_tensor |  -0.8241 |        31 |
| all_known       | gabor_coarse_energy_share           | acquisition         |  -0.1684 |        31 |
| all_known       | gabor_coarse_energy_share           | fft_geometry_tensor |  -0.3769 |        31 |
| all_known       | gabor_horizontal_wavevector_balance | acquisition         |  -0.2973 |        31 |
| all_known       | gabor_horizontal_wavevector_balance | fft_geometry_tensor |  -1.016  |        31 |
| quality_matched | gabor_axial_strength                | acquisition         |  -0.7788 |        25 |
| quality_matched | gabor_axial_strength                | fft_geometry_tensor |  -0.4532 |        25 |
| quality_matched | gabor_coarse_energy_share           | acquisition         |  -0.5776 |        25 |
| quality_matched | gabor_coarse_energy_share           | fft_geometry_tensor |  -0.5852 |        25 |
| quality_matched | gabor_horizontal_wavevector_balance | acquisition         |  -0.8213 |        25 |
| quality_matched | gabor_horizontal_wavevector_balance | fft_geometry_tensor |  -0.579  |        25 |

## Fixed sensitivity

| view            | kpi                                 | mode            |   n_sites |   nominal_site_iqr |   median_abs_delta |   max_abs_delta |   median_delta_over_site_iqr |   spearman_nominal_mode |
|:----------------|:------------------------------------|:----------------|----------:|-------------------:|-------------------:|----------------:|-----------------------------:|------------------------:|
| all_known       | gabor_coarse_energy_share           | nominal         |        31 |           0.023256 |         0          |      0          |                   0          |                 1       |
| all_known       | gabor_coarse_energy_share           | full_resolution |        31 |           0.023256 |         0.00091584 |      0.0054642  |                   0.039381   |                 0.99395 |
| all_known       | gabor_coarse_energy_share           | gamma_075       |        31 |           0.023256 |         0.0029317  |      0.015373   |                   0.12606    |                 0.96331 |
| all_known       | gabor_coarse_energy_share           | gamma_125       |        31 |           0.023256 |         0.0023258  |      0.012848   |                   0.10001    |                 0.97137 |
| all_known       | gabor_coarse_energy_share           | affine          |        31 |           0.023256 |         0          |      3.3307e-16 |                   0          |                 1       |
| quality_matched | gabor_coarse_energy_share           | nominal         |        25 |           0.022121 |         0          |      0          |                   0          |                 1       |
| quality_matched | gabor_coarse_energy_share           | full_resolution |        25 |           0.022121 |         0.00091369 |      0.0054642  |                   0.041304   |                 0.99154 |
| quality_matched | gabor_coarse_energy_share           | gamma_075       |        25 |           0.022121 |         0.0033254  |      0.015373   |                   0.15033    |                 0.95615 |
| quality_matched | gabor_coarse_energy_share           | gamma_125       |        25 |           0.022121 |         0.0030579  |      0.012848   |                   0.13824    |                 0.96231 |
| quality_matched | gabor_coarse_energy_share           | affine          |        25 |           0.022121 |         0          |      3.3307e-16 |                   0          |                 1       |
| all_known       | gabor_axial_strength                | nominal         |        31 |           0.09044  |         0          |      0          |                   0          |                 1       |
| all_known       | gabor_axial_strength                | full_resolution |        31 |           0.09044  |         0.0034571  |      0.020902   |                   0.038226   |                 0.99315 |
| all_known       | gabor_axial_strength                | gamma_075       |        31 |           0.09044  |         0.0081584  |      0.03756    |                   0.090208   |                 0.97661 |
| all_known       | gabor_axial_strength                | gamma_125       |        31 |           0.09044  |         0.006416   |      0.031122   |                   0.070942   |                 0.96855 |
| all_known       | gabor_axial_strength                | affine          |        31 |           0.09044  |         2.7756e-17 |      1.1102e-16 |                   3.069e-16  |                 1       |
| quality_matched | gabor_axial_strength                | nominal         |        25 |           0.10375  |         0          |      0          |                   0          |                 1       |
| quality_matched | gabor_axial_strength                | full_resolution |        25 |           0.10375  |         0.0034571  |      0.020902   |                   0.033322   |                 0.98923 |
| quality_matched | gabor_axial_strength                | gamma_075       |        25 |           0.10375  |         0.007464   |      0.03756    |                   0.071942   |                 0.96923 |
| quality_matched | gabor_axial_strength                | gamma_125       |        25 |           0.10375  |         0.0062749  |      0.031122   |                   0.060482   |                 0.96    |
| quality_matched | gabor_axial_strength                | affine          |        25 |           0.10375  |         5.5511e-17 |      1.1102e-16 |                   5.3505e-16 |                 1       |
| all_known       | gabor_horizontal_wavevector_balance | nominal         |        31 |           0.088514 |         0          |      0          |                   0          |                 1       |
| all_known       | gabor_horizontal_wavevector_balance | full_resolution |        31 |           0.088514 |         0.0041366  |      0.01795    |                   0.046734   |                 0.98669 |
| all_known       | gabor_horizontal_wavevector_balance | gamma_075       |        31 |           0.088514 |         0.0079149  |      0.03788    |                   0.08942    |                 0.97823 |
| all_known       | gabor_horizontal_wavevector_balance | gamma_125       |        31 |           0.088514 |         0.0060712  |      0.030613   |                   0.06859    |                 0.98508 |
| all_known       | gabor_horizontal_wavevector_balance | affine          |        31 |           0.088514 |         3.2092e-17 |      1.1449e-16 |                   3.6257e-16 |                 1       |
| quality_matched | gabor_horizontal_wavevector_balance | nominal         |        25 |           0.087649 |         0          |      0          |                   0          |                 1       |
| quality_matched | gabor_horizontal_wavevector_balance | full_resolution |        25 |           0.087649 |         0.0041366  |      0.01795    |                   0.047195   |                 0.97769 |
| quality_matched | gabor_horizontal_wavevector_balance | gamma_075       |        25 |           0.087649 |         0.0077663  |      0.03788    |                   0.088607   |                 0.98077 |
| quality_matched | gabor_horizontal_wavevector_balance | gamma_125       |        25 |           0.087649 |         0.0058984  |      0.030613   |                   0.067296   |                 0.97923 |
| quality_matched | gabor_horizontal_wavevector_balance | affine          |        25 |           0.087649 |         5.5511e-17 |      1.1102e-16 |                   6.3334e-16 |                 1       |

Non-clipping affine copies are a numerical control. Gamma0.75/1.25 changes contrast shape; full resolution changes sampling while retaining raw filter periods. Originals remain read-only. Synthetic controls do not validate real preparation or lost contrast.

## Scope of retain/defer finding

Retain response maps as exploratory appearance references. Defer material/QC use pending expert review, spatial repeatability and independent specimen/process evidence; examine redundancy and acquisition sensitivity before any expanded work. No favourable separation is used to choose a scale, direction, quality subset or a new primary KPI. Numerical and source contracts do not validate a battery mechanism.

Review applied: C01–C07/C09/C11/C13/C14/C16/C17/C21–C23/C28–C30.

## Post-run measured qualification

# Measured qualification · task J / E28J

Retain the response maps for exploratory BSE appearance review. Defer all
three summaries as material/QC drivers. No evidence here establishes useful
information beyond existing geometry or acquisition variables.

The all-known Batch1−Batch3 coarse-energy-share difference is
-0.01691, site-bootstrap95% [-0.03304, -0.00590]
at n7/17 sites. In the quality-matched view it is
-0.01384 [-0.03049, +0.00730], n5/13.
All nine nominal quality-matched intervals include zero. Filtering changes
both the population and sample size; this does not prove the difference was
an acquisition artifact or that the batches are equivalent. These are
correlated exploratory contrasts, with no multiplicity-adjusted test.

Nominal versus full-resolution ranks have Spearman rho
0.987–0.994; median
absolute descriptor movement is 0.038–0.047
of the pooled between-site IQR. Gamma changes move some sites more; complete
paired values and maxima are in `sensitivity_summary.csv`. The non-clipping
affine control agrees to numerical precision, as expected by construction.
This is not evidence that real contrast loss or preparation is corrected.

All twelve acquisition and FFT/geometry/tensor site-LOO ridge R² results are
negative. That means these prespecified linear controls predict poorly at
this n; it does not establish new information or acquisition invariance.
Strong within-batch correlations remain in the correlation table, with
small group counts and different sampled versus full-frame measurands.

Valid convolution interiors cover only 2.53–3.63%
of each trimmed frame. Four windows are coverage, not independent samples;
spatial representativeness and specimen independence remain unvalidated.
Wavevectors are normal to image stripes, not graphite plate-instance axes.
Expert phase validity, battery mechanisms, manufacturing outcomes and
unseen generalisation remain open. Primary/classifier/verdict inputs are
unchanged.

Eight mathematical convention/invariance tests pass. They do not validate
material identity or independent segmentation/defect accuracy.
Review: C01–C07/C09/C11/C13/C14/C16/C17/C21–C23/C28–C30.

