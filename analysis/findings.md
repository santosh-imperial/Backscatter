# Detailed dataset analysis

This is an exploratory audit of all supplied images, not a manufacturing release decision. No approved baseline, defect labels, material identity, specimen grouping, acquisition settings or verified physical calibration were supplied.

## Inventory

```text
         images  field_ids  width_min  width_max  height_min  height_max
batch                                                                   
Batch_1      21          7       6960       7000        1780        2316
Batch_2      21          7       7000       7000        2048        2272
Batch_3      51         17       6996       7000        1612        2272
```

Field IDs have three detector images each. Treat channels as paired views and patches as subsamples. Field independence and spatial overlap remain unverified. Batch_3 has 17 fields versus seven each in Batch_1 and Batch_2.

## Integrity and calibration

- All 93 TIFFs decoded successfully.
- Exact duplicate pixel groups: 0.
- 39 images contain colored pixels. Maximum original colored-pixel fraction: 0.0392%; maximum after excluding a four-pixel edge: 0.0000%. All measurements use this edge exclusion; original files remain unchanged.
- Nominal TIFF scale range: 24.9992–25.0005 nm/pixel. This is export metadata, not verified calibration.
- Maximum fraction exactly black: 5.48%; exactly white: 6.75%. Endpoint fractions alone cannot establish clipping or image quality.

## Imaging confounders to review

Fields with the greatest endpoint-white fractions (potential detector saturation):

```text
  batch        field channel  white_fraction  p01  p99
Batch_3 img_xgj4xftb  Inlens        0.067535   29  255
Batch_3 img_hawkfj64  Inlens        0.051540   15  255
Batch_3 img_mgxahqnk  Inlens        0.048969   14  255
Batch_1 img_uhdslk0o  Inlens        0.042206    2  255
Batch_3 img_x77cy643  Inlens        0.041079   14  255
Batch_3 img_vc2whyaq  Inlens        0.040424   20  255
```

BSE fields with the highest 1st-percentile intensities (raised dark levels can affect thresholded dark fractions):

```text
  batch        field  p01  p99  std_gray  black_fraction
Batch_3 img_x7u69zsw   25  121 18.008540        0.000000
Batch_3 img_tuy3zymq   24  116 16.515396        0.000000
Batch_3 img_kbdh4tri   24  118 16.892064        0.000000
Batch_3 img_71vgq3fw   23  118 17.084023        0.000000
Batch_3 img_9luzk4jm    6  120 20.585130        0.003347
Batch_3 img_ptg8lmto    6  133 22.209813        0.002729
```

## BSE batch summary

```text
         mean_gray  std_gray  dark_fraction_low  dark_fraction_central  dark_fraction_high  gradient_rms  gradient_anisotropy  tile_mean_sd  dark_run_x_thumb_px  dark_run_y_thumb_px
batch                                                                                                                                                                                
Batch_1    58.4557   23.7388             0.3262                 0.5728              0.7616        0.0554               0.8402        0.0127               7.9105               7.2523
Batch_2    55.1777   23.4534             0.4021                 0.6549              0.8182        0.0524               0.8137        0.0126              13.2040              11.8382
Batch_3    58.8001   21.2351             0.2901                 0.5651              0.8003        0.0478               0.8176        0.0099               8.0265               7.1873
```

Dark fraction is an intensity-threshold proxy, not measured porosity. Threshold ±10 sensitivity is shown above. Dark run lengths use thumbnail pixels and are not segmented particle sizes. Gradient anisotropy is image texture directionality, not a direct particle orientation estimate.

## Sensitivity to raised dark levels

Four Batch_3 BSE fields have raised 1st-percentile intensities (above 10/255). This post-hoc grouping is an imaging-confounder check, not a defect classifier. Compare these fields separately before interpreting a batch-level shift as material change.

```text
                          n_fields  mean_gray  std_gray  gradient_rms  dark_fraction
batch   dark_level_group                                                            
Batch_1 lower (<=10)             7    58.4557   23.7388        0.0554         0.5728
Batch_2 lower (<=10)             7    55.1777   23.4534        0.0524         0.6549
Batch_3 lower (<=10)            13    57.0369   22.4997        0.0501         0.6114
        raised (>10)             4    64.5308   17.1250        0.0405         0.4147
```

## Exploratory pairwise differences

Comparisons use field-level resampling (10,000 draws, seed 42). Intervals assume independent sampled fields and describe these images only. They omit segmentation, acquisition, calibration and between-production-batch uncertainty. Intervals are not adjusted for multiple comparisons; selecting the largest effects is exploratory. Standardized differences use pooled within-group standard deviation, without small-sample correction. Batch_1 is a reference for display only, not assumed approved.

### Batch_2 minus Batch_1, BSE

```text
               metric   delta  delta_ci_low  delta_ci_high  standardized_difference
dark_fraction_central  0.0821       -0.0063         0.1636                   0.9510
            mean_gray -3.2780       -6.6010         0.2929                  -0.9264
  dark_run_x_thumb_px  5.2936       -0.6040        10.8301                   0.8979
         gradient_rms -0.0030       -0.0063         0.0003                  -0.8933
  dark_run_y_thumb_px  4.5859       -0.6957         9.4711                   0.8776
         entropy_bits -0.1485       -0.3344         0.0218                  -0.8044
  gradient_anisotropy -0.0265       -0.0872         0.0306                  -0.4382
             std_gray -0.2855       -1.2112         0.7364                  -0.2812
         tile_mean_sd -0.0001       -0.0029         0.0025                  -0.0399
```

### Batch_3 minus Batch_1, BSE

```text
               metric   delta  delta_ci_low  delta_ci_high  standardized_difference
         gradient_rms -0.0076       -0.0110        -0.0044                  -1.6458
             std_gray -2.5038       -3.9558        -1.1231                  -1.0822
         tile_mean_sd -0.0028       -0.0055        -0.0003                  -0.8584
  gradient_anisotropy -0.0226       -0.0723         0.0227                  -0.5066
         entropy_bits -0.2166       -0.4706        -0.0026                  -0.5004
            mean_gray  0.3444       -1.9468         2.7060                   0.1029
dark_fraction_central -0.0076       -0.0731         0.0513                  -0.0883
  dark_run_x_thumb_px  0.1161       -4.1963         3.5527                   0.0266
  dark_run_y_thumb_px -0.0650       -3.6750         2.9168                  -0.0171
```

### Batch_3 minus Batch_2, BSE

```text
               metric   delta  delta_ci_low  delta_ci_high  standardized_difference
  dark_run_y_thumb_px -4.6509       -8.9369        -0.2244                  -1.0581
  dark_run_x_thumb_px -5.1775      -10.1994        -0.1950                  -1.0393
         gradient_rms -0.0046       -0.0077        -0.0015                  -1.0218
             std_gray -2.2183       -3.6776        -0.8638                  -0.9561
            mean_gray  3.6225        0.1198         7.0424                   0.9354
dark_fraction_central -0.0897       -0.1713        -0.0055                  -0.9342
         tile_mean_sd -0.0026       -0.0048        -0.0006                  -0.8902
         entropy_bits -0.0681       -0.3669         0.2120                  -0.1510
  gradient_anisotropy  0.0039       -0.0344         0.0455                   0.0962
```

## Fields to inspect

The following BSE fields have the highest and lowest image-intensity dark fractions. These are review candidates, not known defects.

```text
  batch        field  dark_fraction_central  mean_gray  gradient_anisotropy
Batch_3 img_x7u69zsw               0.390369  65.828087             0.792769
Batch_3 img_tuy3zymq               0.419880  63.880230             0.798485
Batch_3 img_kbdh4tri               0.421512  64.368909             0.810949
Batch_2 img_epqdaau9               0.706621  53.663001             0.719452
Batch_2 img_3806gxp0               0.756372  51.067417             0.893004
Batch_2 img_avn74qx1               0.779386  49.556579             0.828125
```

## Next measurements

1. Confirm which batch is approved and what each batch label means; obtain acquisition settings, physical calibration and specimen IDs.
2. Review dark regions and particle boundaries with a materials expert. Annotate representative fields from every detector and batch.
3. Validate particle/phase segmentation before reporting particle dimensions, void fraction, crack widths or morphology.
4. Quantify segmentation sensitivity and field/specimen sampling uncertainty separately.
5. Freeze baseline-derived preprocessing and practical tolerance limits before the unseen batch arrives.

## Reproduction

Run `analyze_dataset.py` and then `compare_batches.py` with Python providing numpy, pandas and Pillow. Inputs are read-only. All measurements exclude a four-pixel border. Assets are compressed analysis previews; histograms use full-resolution interiors, texture uses maximum-dimension-1400 thumbnails. See report.html for all field images and charts.
