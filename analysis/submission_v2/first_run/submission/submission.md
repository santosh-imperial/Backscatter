# Batch assignments — categoriser-v2-D52

Declared primary: **material**. Source model SHA256: `0b6d14e030910f9a01bdd4c7cc3105033ba75481354c6a7422d6aaab69e97dba`.
The primary uses morphology and acquisition-sensitive ETD/Inlens appearance. Material origin of texture remains unresolved. Every sample receives a bet. Model scores are uncalibrated; deletion fits measure training-set sensitivity, not correctness probability.

| Sample | Bet | Scores B1 / B2 / B3 | Runner-up | Margin | Same bet in deletion fits | Quality flags |
|---|---|---|---|---|---|---|
| 3e122cbj | **Batch_1** | 0.838 / 0.083 / 0.079 | Batch_2 | 0.755 | 31/31 | bright_low_contrast |
| fn0mhxef | **Batch_2** | 0.334 / 0.496 / 0.170 | Batch_1 | 0.161 | 31/31 | none of listed flags |
| xrv9xvzb | **Batch_3** | 0.151 / 0.228 / 0.622 | Batch_2 | 0.394 | 31/31 | contrast_stretched_bse |

## Matching known-site evaluation

Leave-one-site-out on 31 known sites; balanced accuracy 0.608; recall B1/B2/B3 0.714/0.286/0.824; site-label permutation p 0.004975. These are development results, not new-session accuracy.

Reliability bins below belong to this model family. Small bins and overlapping LOO fits do not establish calibration.

| Score bin | Known sites | Mean top score | Observed correct fraction |
|---|---|---|---|
| [0.33, 0.50) | 6 | 0.412 | 0.333 |
| [0.50, 0.70) | 18 | 0.577 | 0.667 |
| [0.70, 1.00) | 7 | 0.737 | 1.000 |

## Per-sample explanation

The linked driver table exactly decomposes assigned-class minus runner-up linear logits. Baseline/class medians are descriptive and include known quality limitations. Signs explain this model, not material causes.

### 3e122cbj

Primary score drivers: `inlens_particle_texture_p90(+1.72; x=0.7496; ref 0.28±0.114)|ridge_p97(+1.02; x=0.3149; ref 0.4602±0.0628)|pore_count_per_Mpx(+0.65; x=120.8; ref 98.13±5.48)|crack_g_0.2(+0.47; x=0.05435; ref 0.08832±0.0145)|bright_d50(-0.18; x=131.7; ref 145.7±13.5)|crack_count_per_Mpx(-0.08; x=0.2469; ref 0.5526±0.211)|crack_g_0.05(-0.07; x=0.3178; ref 0.3395±0.0373)|inlens_grad_energy(+0.00; x=0.1557; ref 0.07027±0.0166)`. Morphology-only bet: Batch_1; acquisition-only comparator: Batch_1.
Baseline morphology percentile: 100.0. It is a descriptive distance rank, not calibrated membership or a production release verdict.

### fn0mhxef

Primary score drivers: `pore_count_per_Mpx(+0.63; x=86.91; ref 98.13±5.48)|pore_frac(+0.30; x=0.07851; ref 0.09307±0.0183)|inlens_grad_energy(+0.16; x=0.1368; ref 0.07027±0.0166)|etd_crack_density_particles(+0.16; x=0.0003945; ref 0.001503±0.00138)|corr_len_px(+0.04; x=16; ref 21±4.45)|inlens_particle_texture(+0.04; x=0.2637; ref 0.1767±0.0657)`. Morphology-only bet: Batch_2; acquisition-only comparator: Batch_1.
Baseline morphology percentile: 17.6. It is a descriptive distance rank, not calibrated membership or a production release verdict.

### xrv9xvzb

Primary score drivers: `pore_max_d(+0.43; x=419.7; ref 278.2±105)|pore_frac(+0.33; x=0.1136; ref 0.09307±0.0183)|crack_g_0.3(+0.17; x=0.05196; ref 0.05204±0.0112)|inlens_particle_texture(-0.12; x=0.2494; ref 0.1767±0.0657)|crack_g_0.2(+0.11; x=0.08743; ref 0.08832±0.0145)`. Morphology-only bet: Batch_3; acquisition-only comparator: Batch_3.
Baseline morphology percentile: 17.6. It is a descriptive distance rank, not calibrated membership or a production release verdict.

[All image assignments](predictions_images.csv) · [Exact pairwise driver table](driver_contrasts.csv) · [Illustrated report](report.html) · [Prediction provenance](submission_receipt.json)

The input folder may mix batches and is not pooled into a manufacturing-batch QC verdict. Truth is pending; future label feedback must be saved separately without editing these bets.

Version 2 was revised after inspecting the first drop, before its truth, and declared before final evaluation. E33 v1 predictions remain a separate historical record.
