# Fixed battery neighbourhood candidates (E25)

Fresh graphite–Si/SiOx material is user-confirmed; pixelwise identity and these measurement masks remain expert-unvalidated. These are secondary, exploratory measurements, not QC verdict inputs or performance predictions.

Measurement choices were fixed before this run, informed by earlier exploratory analysis of these batches: existing Gaussian/histogram/opening masks; 8-connected bright components >=50 px; image-clipped components excluded; 4-neighbour raster boundary faces; Euclidean 16 px bands outside the segmented component, including threshold holes within its silhouette; complete top-left 512/1024 px windows. Bands classify void, other bright and residual solid separately. Window/components are summarised per site and never add material n.

Quality views exclude low-contrast sites for all quantities and data-derived grey-pore sites for joint quantities. The separate ordinary sensitivity also excludes the three previously identified long-void reference sites. Missing flags are unusable. All31 nominal pore_mode_resolved flags are false: this is the common fallback-threshold method, not evidence that all pore masks are unusable or accurate. Grey-pore exclusion does not validate ordinary masks.

Intervals are 4000-resample percentile site-bootstrap intervals under a working site-independence assumption; specimen independence remains unknown. Threshold envelopes perturb both thresholds together by -5/0/+5 gray levels, keep opening fixed, and are not confidence intervals or an exhaustive independent threshold grid. They do not sample uncertainty in phase chemistry, sample preparation or the mask definition.

Run time: 250.1 s; 31 sites. Raw BSE hashes and input caches checked unchanged. No cache/dataset writes.

## Coverage and definitions

Excluded from between-batch comparisons because constant/non-variable on usable sites: bright_void_boundary_frac.

| KPI | units | usable sites B1/B2/B3 | median B1/B2/B3 |
|---|---|---|---|
| `bright_void_boundary_frac` | fraction | 5/7/13 | 0/0/0 |
| `bright_void_distance_d50_px` | px | 5/7/13 | 6.5557/9.8995/6.7082 |
| `bright_ring_void_frac_16px` | fraction | 5/7/13 | 0.039387/0.021733/0.044495 |
| `local_bright_std_512px` | fraction | 5/7/17 | 0.058566/0.062607/0.059281 |
| `local_bright_pore_spearman_512px` | correlation | 5/7/13 | -0.052591/-0.04414/0.025783 |
| `local_bright_std_1024px` | fraction | 5/7/17 | 0.031222/0.03112/0.029343 |
| `local_bright_pore_spearman_1024px` | correlation | 5/7/13 | 0.13986/-0.23776/-0.35664 |

Component coverage (median site counts by folder; counts are diagnostics):

| batch | retained | edge-clipped | used | full rings | incomplete rings |
|---|---|---|---|---|---|
| Batch_1 | 237 | 17 | 216 | 208 | 8 |
| Batch_2 | 209 | 22 | 184 | 179 | 5 |
| Batch_3 | 191 | 17 | 181 | 174 | 7 |

## Exploratory differences and acquisition sensitivity

Every pair and both quality views are in neighbourhood_comparisons.csv. Below, differences are batch minus reference in raw units; bootstrap intervals and threshold bands describe different uncertainty sources. No acceptance/equivalence or significance claim is made.

| KPI | pair | difference | site-bootstrap 95% interval | threshold endpoint envelope |
|---|---|---|---|---|
| `bright_void_distance_d50_px` | Batch_1 − Batch_3 | -0.1525 | [-0.808, 1.071] | [-2.485, 1.785] |
| `bright_void_distance_d50_px` | Batch_2 − Batch_3 | 3.191 | [-0.07107, 4.777] | [0.459, 5.259] |
| `bright_void_distance_d50_px` | Batch_2 − Batch_1 | 3.344 | [-0.07107, 4.414] | [1.328, 5.09] |
| `bright_ring_void_frac_16px` | Batch_1 − Batch_3 | -0.005108 | [-0.01719, 0.01115] | [-0.03106, 0.02778] |
| `bright_ring_void_frac_16px` | Batch_2 − Batch_3 | -0.02276 | [-0.03721, 0.003943] | [-0.04809, 0.005928] |
| `bright_ring_void_frac_16px` | Batch_2 − Batch_1 | -0.01765 | [-0.03247, 0.007961] | [-0.0382, -0.0006759] |
| `local_bright_std_512px` | Batch_1 − Batch_3 | -0.0007146 | [-0.01472, 0.006339] | [-0.00499, 0.002901] |
| `local_bright_std_512px` | Batch_2 − Batch_3 | 0.003326 | [-0.01095, 0.0106] | [-0.0017, 0.006287] |
| `local_bright_std_512px` | Batch_2 − Batch_1 | 0.00404 | [-0.00405, 0.01214] | [-0.001378, 0.008054] |
| `local_bright_pore_spearman_512px` | Batch_1 − Batch_3 | -0.07837 | [-0.3239, 0.3804] | [-0.1222, -0.06343] |
| `local_bright_pore_spearman_512px` | Batch_2 − Batch_3 | -0.06992 | [-0.2609, 0.1927] | [-0.09596, -0.06177] |
| `local_bright_pore_spearman_512px` | Batch_2 − Batch_1 | 0.008451 | [-0.4354, 0.254] | [-0.02331, 0.05118] |
| `local_bright_std_1024px` | Batch_1 − Batch_3 | 0.001879 | [-0.01066, 0.007606] | [-0.0001931, 0.002935] |
| `local_bright_std_1024px` | Batch_2 − Batch_3 | 0.001777 | [-0.007229, 0.01342] | [-0.0007456, 0.003152] |
| `local_bright_std_1024px` | Batch_2 − Batch_1 | -0.0001019 | [-0.007724, 0.01835] | [-0.002364, 0.002028] |
| `local_bright_pore_spearman_1024px` | Batch_1 − Batch_3 | 0.4965 | [0, 1.3] | [0.3776, 0.5385] |
| `local_bright_pore_spearman_1024px` | Batch_2 − Batch_3 | 0.1189 | [-0.4026, 0.6084] | [0, 0.2168] |
| `local_bright_pore_spearman_1024px` | Batch_2 − Batch_1 | -0.3776 | [-1.181, 0.1323] | [-0.4476, -0.2517] |

Acquisition-only leave-one-site-out ridge predictions (fixed alpha=1, within-fold imputation/scaling; pooled covariates may also encode batch/session):

| KPI | n sites | held-out R² | strongest within-batch |rho| (n≥4) |
|---|---|---|---|
| `bright_void_distance_d50_px` | 25 | -0.161 | Batch_1, H: 0.872 (n=5) |
| `bright_ring_void_frac_16px` | 25 | -0.259 | Batch_1, inlens_p50: -1.000 (n=5) |
| `local_bright_std_512px` | 29 | -0.371 | Batch_2, bse_std: 0.750 (n=7) |
| `local_bright_pore_spearman_512px` | 25 | -0.650 | Batch_3, bright_sep: 0.710 (n=13) |
| `local_bright_std_1024px` | 29 | -0.134 | Batch_1, etd_curtain_frac: 0.600 (n=5) |
| `local_bright_pore_spearman_1024px` | 25 | -0.275 | Batch_1, bse_empty_bin_frac: -0.800 (n=5) |

Correlations/prediction are confound screens, not attribution. Weak or negative held-out prediction does not establish acquisition invariance; strong prediction is grounds to investigate the measurement and spatial sampling. Single-section chance placement, clipped-object selection, ring overlap, mean bright loading and compositional closure can affect these quantities.

A post-inspection redundancy screen against existing mask-derived geometry is in neighbourhood_structural_correlations.csv. Pooled usable-site correlations partly link void proximity/ring fraction to overall pore fraction and bright dispersion to loading/particle sizes; these are not independent mechanisms or covariate-adjusted batch differences.

## Interpretation limits and next review

Zero exact boundary faces can result from the residual-solid gray-level transition between widely separated pore/bright thresholds, especially after Gaussian smoothing/opening. It cannot establish full binder/electronic contact. Bright-to-void distance and rings retain measured geometric variation without treating that transition as a material contact network.

Local bright dispersion describes mask heterogeneity at two chosen pixel scales. A bright–pore correlation is a section association, not causation or three-dimensional co-location. The 16 px band includes threshold holes within a component's silhouette: it is outside segmented pixels, not validated outer space around a physical particle or available expansion volume for Si/SiOx. Use chemical validation, matched section/preparation and reviewed annotations before mechanics claims.

Review phase boundaries and representative neighbourhoods in the four real overlays, then test changed definitions against independent annotations and process/electrochemical outcomes. Keep source masks and excluded objects visible; do not select new scales to maximise folder separation. No morphology accuracy is claimed with zero expert reference masks.

Applied pre-presentation checks: C01/C02 numeric and constant-feature audit; C03 acquisition/within-batch/held-out screen; C04/C05/C13 observable and wording scope; C06/C09/C14 site units and quality counts; C07/C21 no acceptance or primary promotion; C16 exploratory provenance; C22 suppress non-variable KPI comparisons; C28 source coordinates/excluded objects and unreviewed masks.
