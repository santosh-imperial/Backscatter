# Batch assignments — categoriser-v2-D52

Declared primary: **material**. Source model SHA256: `0b6d14e030910f9a01bdd4c7cc3105033ba75481354c6a7422d6aaab69e97dba`.
The primary uses morphology and acquisition-sensitive ETD/Inlens appearance. Material origin of texture remains unresolved. Every sample receives a bet. Model scores are uncalibrated; deletion fits measure training-set sensitivity, not correctness probability.

| Sample | Bet | Scores B1 / B2 / B3 | Runner-up | Margin | Same bet in deletion fits | Quality flags |
|---|---|---|---|---|---|---|
| 0eryguqq | **Batch_3** | 0.067 / 0.178 / 0.755 | Batch_2 | 0.576 | 31/31 | none of listed flags |
| 4hq27w4c | **Batch_2** | 0.338 / 0.551 / 0.112 | Batch_1 | 0.213 | 30/31 | none of listed flags |
| fhwrjtet | **Batch_3** | 0.083 / 0.092 / 0.825 | Batch_2 | 0.732 | 31/31 | none of listed flags |
| fspqbkxl | **Batch_2** | 0.199 / 0.466 / 0.335 | Batch_3 | 0.131 | 29/31 | none of listed flags |
| soo2ax3r | **Batch_2** | 0.291 / 0.398 / 0.310 | Batch_3 | 0.088 | 28/31 | none of listed flags |
| y59rxmxl | **Batch_1** | 0.395 / 0.279 / 0.327 | Batch_3 | 0.068 | 26/31 | none of listed flags |

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

### 0eryguqq

Primary score drivers: `inlens_particle_texture(+0.82; x=0.1509; ref 0.1767±0.0657)|crack_g_0.3(+0.62; x=0.06605; ref 0.05204±0.0112)|crack_g_0.2(+0.53; x=0.1077; ref 0.08832±0.0145)|pore_frac(-0.13; x=0.07989; ref 0.09307±0.0183)|pore_max_d(-0.00; x=292.2; ref 278.2±105)`. Morphology-only bet: Batch_3; acquisition-only comparator: Batch_3.
Baseline morphology percentile: 17.6. It is a descriptive distance rank, not calibrated membership or a production release verdict.

### 4hq27w4c

Primary score drivers: `pore_frac(+0.82; x=0.05896; ref 0.09307±0.0183)|pore_count_per_Mpx(+0.40; x=90.78; ref 98.13±5.48)|inlens_grad_energy(+0.28; x=0.151; ref 0.07027±0.0166)|etd_crack_density_particles(+0.15; x=0.0004116; ref 0.001503±0.00138)|inlens_particle_texture(+0.12; x=0.3165; ref 0.1767±0.0657)|corr_len_px(+0.01; x=19; ref 21±4.45)`. Morphology-only bet: Batch_2; acquisition-only comparator: Batch_1.
Baseline morphology percentile: 35.3. It is a descriptive distance rank, not calibrated membership or a production release verdict.

### fhwrjtet

Primary score drivers: `crack_g_0.3(+0.91; x=0.07495; ref 0.05204±0.0112)|crack_g_0.2(+0.71; x=0.1166; ref 0.08832±0.0145)|inlens_particle_texture(+0.68; x=0.1658; ref 0.1767±0.0657)|pore_max_d(-0.05; x=279.5; ref 278.2±105)|pore_frac(-0.04; x=0.08708; ref 0.09307±0.0183)`. Morphology-only bet: Batch_1; acquisition-only comparator: Batch_3.
Baseline morphology percentile: 58.8. It is a descriptive distance rank, not calibrated membership or a production release verdict.

### fspqbkxl

Primary score drivers: `inlens_grad_energy(+0.41; x=0.1682; ref 0.07027±0.0166)|inlens_particle_texture(+0.16; x=0.3423; ref 0.1767±0.0657)|pore_frac(+0.13; x=0.08486; ref 0.09307±0.0183)|etd_crack_density_particles(+0.12; x=0.000672; ref 0.001503±0.00138)|pore_count_per_Mpx(+0.10; x=95.84; ref 98.13±5.48)|corr_len_px(-0.04; x=23; ref 21±4.45)`. Morphology-only bet: Batch_3; acquisition-only comparator: Batch_2.
Baseline morphology percentile: 17.6. It is a descriptive distance rank, not calibrated membership or a production release verdict.

### soo2ax3r

Primary score drivers: `inlens_grad_energy(+0.62; x=0.1944; ref 0.07027±0.0166)|etd_crack_density_particles(-0.26; x=0.003462; ref 0.001503±0.00138)|pore_frac(-0.25; x=0.09896; ref 0.09307±0.0183)|inlens_particle_texture(+0.18; x=0.3555; ref 0.1767±0.0657)|pore_count_per_Mpx(+0.10; x=95.94; ref 98.13±5.48)|corr_len_px(+0.04; x=16; ref 21±4.45)`. Morphology-only bet: Batch_3; acquisition-only comparator: Batch_3.
Baseline morphology percentile: 17.6. It is a descriptive distance rank, not calibrated membership or a production release verdict.

### y59rxmxl

Primary score drivers: `ridge_p97(+0.38; x=0.3893; ref 0.4602±0.0628)|crack_g_0.2(+0.32; x=0.06317; ref 0.08832±0.0145)|inlens_particle_texture_p90(+0.11; x=0.4123; ref 0.28±0.114)|bright_d50(+0.09; x=172.8; ref 145.7±13.5)|crack_g_0.05(-0.01; x=0.33; ref 0.3395±0.0373)|pore_count_per_Mpx(+0.00; x=97.8; ref 98.13±5.48)|inlens_grad_energy(+0.00; x=0.1563; ref 0.07027±0.0166)|crack_count_per_Mpx(+0.00; x=0.5319; ref 0.5526±0.211)`. Morphology-only bet: Batch_3; acquisition-only comparator: Batch_1.
Baseline morphology percentile: 76.5. It is a descriptive distance rank, not calibrated membership or a production release verdict.

[All image assignments](predictions_images.csv) · [Exact pairwise driver table](driver_contrasts.csv) · [Illustrated report](report.html) · [Prediction provenance](submission_receipt.json)

The input folder may mix batches and is not pooled into a manufacturing-batch QC verdict. Truth is pending; future label feedback must be saved separately without editing these bets.

Version 2 was revised after inspecting the first drop, before its truth, and declared before final evaluation. E33 v1 predictions remain a separate historical record.
