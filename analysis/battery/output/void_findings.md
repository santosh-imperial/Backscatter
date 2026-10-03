# E25: B02/B05 long-void spatial context and image-depth widths

Measured from all 31 full/raw BSE sites, with the current band trimming and per-site pore thresholds. Structural context is repeated at threshold −5, nominal and +5. The existing E24 global widths are reused and checked against new nominal medial-axis maps. New width-by-depth profiles cover every site at nominal thresholds; ±5 profiles cover the predeclared 11-site set (all three known long-void sites, all four grey flags, one ordinary control per batch and epqdaau9). Components, centreline pixels and depth bins never become independent statistical n.

## Measurement scope and coverage

- Existing `crack_frac` includes every component ≥30 px² with major axis >500 px, including edge-clipped ones. It is reproduced before any exclusion.
- `long_void_internal_area_frac` measures only components touching no frame edge, divided by the whole trimmed image area. Observed absence is valid zero; it is not acceptance.
- `long_void_y_centroid_norm` measures only those fully observable components, with image top=0 and bottom=1; no internal component gives NaN. Available on 31/31 sites.
- `long_void_edge_clipped_area_share` is an observability diagnostic, not a material KPI. Retained local widths omit these same edge-clipped components; they do not characterise all visible long voids.
- All 31 cached `pore_mode_resolved` entries are False. This invariant cannot select a reliable subset. The data-derived grey-pore flag identifies four preparation/contrast cases, excluded in the quality view but displayed separately. Every segmentation remains an unreviewed prediction.
- Image-row coordinates are not collector depth or calibrated coating thickness. Only epqdaau9 is shown as a candidate visible lower foil/stitch band; its identity and orientation remain unconfirmed. **No collector-interface gap fraction or width is computed.** Raw views are retained for that review.
- Width-profile legend counts are group sizes; usable site counts vary by image-depth bin and are stored in `void_width_depth_summary.csv`. A bin with no retained long-void samples is undefined. Bootstrap resamples with no observed value in a bin remain undefined and are omitted from that bin's percentile interval; finite-resample counts are saved.

## Actual selected-site measurements

| batch   | site     |   crack_frac |   long_void_internal_area_frac |   long_void_y_centroid_norm |   long_void_internal_n_components |   long_void_edge_clipped_area_share |   crack_local_width_d50_px |   band_top |   band_bottom |
|:--------|:---------|-------------:|-------------------------------:|----------------------------:|----------------------------------:|------------------------------------:|---------------------------:|-----------:|--------------:|
| Batch_2 | epqdaau9 |       0.0099 |                         0.0099 |                      0.5615 |                                 4 |                              0.0000 |                    17.0880 |          0 |            15 |
| Batch_3 | 0grcilhi |       0.0624 |                         0.0350 |                      0.4468 |                                10 |                              0.4393 |                    15.6205 |          0 |             0 |
| Batch_3 | 71vgq3fw |       0.0107 |                         0.0096 |                      0.3977 |                                 4 |                              0.1063 |                    18.4391 |          0 |             0 |
| Batch_3 | hzumfsms |       0.0607 |                         0.0316 |                      0.2986 |                                 3 |                              0.4795 |                    16.1245 |          0 |             0 |
| Batch_3 | ufdvpb81 |       0.0478 |                         0.0299 |                      0.5368 |                                11 |                              0.3751 |                    14.4222 |          0 |             0 |
| Batch_3 | xgj4xftb |       0.0191 |                         0.0112 |                      0.1525 |                                 2 |                              0.4132 |                    12.8062 |          0 |             0 |

## Quality-view batch comparisons

All differences are incoming minus reference site medians. Intervals are approximate 95% site-bootstrap intervals (4000 draws). Threshold envelopes are separate measurement-sensitivity ranges, not confidence intervals. No hypothesis test or new alarm threshold is adopted.

| reference   | batch   | kpi                          |   n_ref |   n_batch |   median_delta |   delta_ci_low |   delta_ci_high |   threshold_delta_low |   threshold_delta_high |
|:------------|:--------|:-----------------------------|--------:|----------:|---------------:|---------------:|----------------:|----------------------:|-----------------------:|
| Batch_3     | Batch_1 | long_void_internal_area_frac |      13 |         7 |         0.0004 |        -0.0153 |          0.0133 |               -0.0143 |                 0.0117 |
| Batch_3     | Batch_1 | long_void_y_centroid_norm    |      13 |         7 |         0.0002 |        -0.0657 |          0.1023 |              nan      |               nan      |
| Batch_3     | Batch_1 | crack_local_width_d50_px     |      13 |         7 |         0.3795 |        -1.5643 |          3.6364 |              nan      |               nan      |
| Batch_3     | Batch_2 | long_void_internal_area_frac |      13 |         7 |        -0.0072 |        -0.0156 |         -0.0003 |               -0.0223 |                 0.0068 |
| Batch_3     | Batch_2 | long_void_y_centroid_norm    |      13 |         7 |        -0.0057 |        -0.0839 |          0.1148 |              nan      |               nan      |
| Batch_3     | Batch_2 | crack_local_width_d50_px     |      13 |         7 |         0.0000 |        -3.3509 |          1.5778 |              nan      |               nan      |
| Batch_1     | Batch_2 | long_void_internal_area_frac |       7 |         7 |        -0.0075 |        -0.0188 |          0.0093 |               -0.0172 |                 0.0042 |
| Batch_1     | Batch_2 | long_void_y_centroid_norm    |       7 |         7 |        -0.0059 |        -0.0977 |          0.0930 |              nan      |               nan      |
| Batch_1     | Batch_2 | crack_local_width_d50_px     |       7 |         7 |        -0.3795 |        -4.4458 |          1.4398 |              nan      |               nan      |

## Existing reference subgroups (descriptive)

These sites were selected as known morphological examples during prior exploratory analysis; this comparison is not independent validation of a defect label or a new batch classifier.

| batch            | kpi                          |   n_ref |   n_batch |   median_delta |   delta_ci_low |   delta_ci_high |
|:-----------------|:-----------------------------|--------:|----------:|---------------:|---------------:|----------------:|
| known_long_voids | long_void_internal_area_frac |      10 |         3 |         0.0156 |         0.0109 |          0.0211 |
| known_long_voids | long_void_y_centroid_norm    |      10 |         3 |        -0.0541 |        -0.2370 |          0.0896 |
| known_long_voids | crack_frac                   |      10 |         3 |         0.0405 |         0.0249 |          0.0439 |
| known_long_voids | crack_local_width_d50_px     |      10 |         3 |         0.0047 |        -2.0702 |          1.2976 |
| grey_pore_flags  | long_void_internal_area_frac |      10 |         4 |        -0.0084 |        -0.0139 |         -0.0037 |
| grey_pore_flags  | long_void_y_centroid_norm    |      10 |         4 |         0.0111 |        -0.1005 |          0.2103 |
| grey_pore_flags  | crack_frac                   |      10 |         4 |        -0.0110 |        -0.0160 |         -0.0079 |
| grey_pore_flags  | crack_local_width_d50_px     |      10 |         4 |         0.5087 |        -2.2016 |          2.8233 |

## API and next use

`polaron_qc.void_metrics.long_void_context(pore_mask, *, major_axis_cutoff_px=500, min_area_px=30, n_depth_bins=10, return_details=False)` returns two secondary geometry KPIs plus separate observability diagnostics. `return_details=True` additionally returns component records and masks for audit. The default has no large maps. `KPI_NAMES` and `KPI_DEFINITIONS` define units and interpretation. Width/skeleton profiling remains in this standalone analysis; it is not added to the core extractor.

Use the internal burden and image-row centroid as descriptive context, retaining original crack fraction and clipped-area coverage. Expert phase/void review is still needed; a broad or narrow 2-D section does not establish a 3-D pore throat, wetting rate, adhesion failure or electrical disconnection. Fresh graphite–Si/SiOx is confirmed; no cycling-induced damage is inferred.

## Outputs and verification

See `void_sites.csv`, `void_components.csv.gz`, `void_depth_profiles.csv`, `void_width_depth_profiles.csv`, `void_width_depth_summary.csv`, `void_comparisons.csv`, `void_acquisition_correlations.csv`, `void_acquisition_predictability.csv`, `void_crop_provenance.csv`, `void_summary.json` and `void_manifest.json`. Figures: `void_full_site_overlays.png`, `void_width_examples.png`, `void_width_depth.png`, `void_site_descriptors.png`, `void_collector_candidate.png`.

Input/cache/raw hashes and parameters are recorded for the renderer-recovery snapshot. The initial run saved all numerical tables before a plotting-only failure; its pre/post hashes were not persisted. All 93 geometry rows and six example width-depth profiles were rechecked during recovery. The original measured source is archived in `void_extraction_source.py`. Original crack fractions and E24 global widths agree with the saved values; depth shares sum to one where defined; raw/trimmed offsets and crop bounds are checked. Six core geometry/edge/absence/flip/connectivity tests are supplied. Acquisition correlations and a held-out-site fixed ridge model are exploratory confound screens; poor predictability cannot prove an acquisition-free feature. No QC verdict or primary threshold changes in this analysis.

Checklist applied: C01–C06, C09, C13–C18, C21–C24 and C28. Independent expert annotation, collector-interface review, specimen grouping and physical calibration remain unresolved.
