# Site categorisation, baseline OOD assessment and hand-off

Three separate answers per site; model probabilities are classifier scores, not posteriors or confidence; OOD percentiles are descriptive evidence ranks (0–100), not probabilities of defect or evidence of equivalence. Legacy ood_rank_p columns have tail-rank floor 1/(n_ref+1), not calibrated p-values or false-alert guarantees: reference LOO fits use n_ref−1 sites and query fits use n_ref sites. Texture family: acquisition-sensitive; batch-fingerprint evidence, origin (microstructure vs imaging) not established.

## 1. Site categorisation (leave-one-site-out, fold-local imputer/scaler/C choice, balanced class weights)

| family | primary | n_features | accuracy | accuracy_ci95 | majority_accuracy | balanced_accuracy | chance_balanced | recall_Batch_1 | recall_Batch_2 | recall_Batch_3 | perm_p | n_perm | brier | brier_prior | ece |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| morph | False | 19 | 0.387 | [0.22, 0.58] | 0.548 | 0.347 | 0.333 | 0.571 | 0.000 | 0.471 | 0.159 | 200 | 0.685 | 0.597 | 0.063 |
| acq | False | 9 | 0.613 | [0.42, 0.78] | 0.548 | 0.541 | 0.333 | 0.714 | 0.143 | 0.765 | 0.030 | 200 | 0.528 | 0.597 | 0.099 |
| material | True | 29 | 0.677 | [0.49, 0.83] | 0.548 | 0.608 | 0.333 | 0.714 | 0.286 | 0.824 | 0.005 | 200 | 0.472 | 0.597 | 0.127 |
| combined | False | 38 | 0.677 | [0.49, 0.83] | 0.548 | 0.636 | 0.333 | 0.714 | 0.429 | 0.765 | 0.005 | 200 | 0.456 | 0.597 | 0.101 |

accuracy_ci95 is a binomial reference interval; overlapping LOO fits make it descriptive, not an exact 95% generalisation interval.

### Confusion — morph (morphology-only (trusted BSE geometry KPIs; no intensity statistics))

|  | pred Batch_1 | pred Batch_2 | pred Batch_3 |
|---|---|---|---|
| true Batch_1 | 4 | 1 | 2 |
| true Batch_2 | 5 | 0 | 2 |
| true Batch_3 | 4 | 5 | 8 |

Reliability of the top-class model probability — morph

| bin | n | mean_top_prob | observed_top_accuracy |
|---|---|---|---|
| [0.33, 0.50) | 25 | 0.402 | 0.360 |
| [0.50, 0.70) | 5 | 0.530 | 0.400 |
| [0.70, 1.00) | 1 | 0.756 | 1.000 |

### Confusion — acq (acquisition-only (session / instrument statistics; never material evidence; comparator))

|  | pred Batch_1 | pred Batch_2 | pred Batch_3 |
|---|---|---|---|
| true Batch_1 | 5 | 2 | 0 |
| true Batch_2 | 2 | 1 | 4 |
| true Batch_3 | 1 | 3 | 13 |

Reliability of the top-class model probability — acq

| bin | n | mean_top_prob | observed_top_accuracy |
|---|---|---|---|
| [0.33, 0.50) | 9 | 0.423 | 0.556 |
| [0.50, 0.70) | 9 | 0.565 | 0.556 |
| [0.70, 1.00) | 13 | 0.831 | 0.692 |

### Confusion — material (material = morphology + texture [acquisition-sensitive; batch-fingerprint evidence, origin (microstructure vs imaging) not established]; no session statistics (PRIMARY v2, declared D52 before final evaluation))

|  | pred Batch_1 | pred Batch_2 | pred Batch_3 |
|---|---|---|---|
| true Batch_1 | 5 | 2 | 0 |
| true Batch_2 | 3 | 2 | 2 |
| true Batch_3 | 2 | 1 | 14 |

Reliability of the top-class model probability — material

| bin | n | mean_top_prob | observed_top_accuracy |
|---|---|---|---|
| [0.33, 0.50) | 6 | 0.412 | 0.333 |
| [0.50, 0.70) | 18 | 0.577 | 0.667 |
| [0.70, 1.00) | 7 | 0.737 | 1.000 |

### Confusion — combined (combined = material + acquisition statistics (v1 primary, E31; kept as a labelled batch-fingerprint comparator))

|  | pred Batch_1 | pred Batch_2 | pred Batch_3 |
|---|---|---|---|
| true Batch_1 | 5 | 1 | 1 |
| true Batch_2 | 2 | 3 | 2 |
| true Batch_3 | 1 | 3 | 13 |

Reliability of the top-class model probability — combined

| bin | n | mean_top_prob | observed_top_accuracy |
|---|---|---|---|
| [0.33, 0.50) | 4 | 0.486 | 0.750 |
| [0.50, 0.70) | 11 | 0.568 | 0.455 |
| [0.70, 1.00) | 16 | 0.865 | 0.812 |

## 2. Baseline OOD assessment (reference = all Batch_3 sites; k-NN robust-z distance; percentiles within the reference LOO)

| variant | batch | n_sites | pct_min | pct_median | pct_max | n_exceed_ref_max | frac_exceed_ref_max | n_rank_p_at_floor | matched_pct_median | rms_n_exceed | maha_n_exceed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| morph | Batch_1 | 7 | 11.76 | 29.41 | 100.00 | 2 | 0.29 | 2 | 29.41 | 2 | 2 |
| morph | Batch_2 | 7 | 0.00 | 41.18 | 76.47 | 0 | 0.00 | 0 | 47.06 | 0 | 0 |
| morph_texture | Batch_1 | 7 | 29.41 | 47.06 | 100.00 | 3 | 0.43 | 3 | 64.71 | 3 | 2 |
| morph_texture | Batch_2 | 7 | 0.00 | 41.18 | 88.24 | 0 | 0.00 | 0 | 41.18 | 1 | 0 |
| acq | Batch_1 | 7 | 52.94 | 70.59 | 82.35 | 0 | 0.00 | 0 | 70.59 | 0 | 0 |
| acq | Batch_2 | 7 | 41.18 | 52.94 | 70.59 | 0 | 0.00 | 0 | 52.94 | 0 | 0 |

## 3. Per-site hand-off (abridged; full columns in site_table.csv)

| batch | site | role | acquisition_group | mp_Batch_1__morph | mp_Batch_2__morph | mp_Batch_3__morph | argmax__morph | mp_Batch_1__acq | mp_Batch_2__acq | mp_Batch_3__acq | argmax__acq | mp_Batch_1__material | mp_Batch_2__material | mp_Batch_3__material | argmax__material | mp_Batch_1__combined | mp_Batch_2__combined | mp_Batch_3__combined | argmax__combined | ood_pct_knn__morph | ood_exceeds_knn__morph | ood_pct_knn__morph_texture | ood_exceeds_knn__morph_texture | ood_pct_knn__acq | ood_exceeds_knn__acq | cat_top_features__material | ood_top_features__morph |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Batch_1 | 4ih2ggld | known | low_contrast | 0.54 | 0.06 | 0.40 | Batch_1 | 0.37 | 0.29 | 0.34 | Batch_1 | 0.73 | 0.08 | 0.19 | Batch_1 | 0.68 | 0.09 | 0.23 | Batch_1 | 100.00 | True | 100.00 | True | 52.94 | False | inlens_particle_texture_p90(+0.74; x=0.6377; ref 0.28±0.114)|crack_g_0.2(+0.69; x=0.05857; ref 0.08832±0.0145)|ridge_p97(+0.58; x=0.3185; ref 0.4602±0.0628) | bright_count_per_Mpx(+z=+36.68)|bright_frac(+z=+8.11)|bright_solidity(-z=-7.72) |
| Batch_1 | 5n1q8atc | known | low_contrast | 0.33 | 0.33 | 0.33 | Batch_1 | 0.84 | 0.08 | 0.07 | Batch_1 | 0.73 | 0.23 | 0.04 | Batch_1 | 0.77 | 0.17 | 0.05 | Batch_1 | 100.00 | True | 100.00 | True | 82.35 | False | ridge_p97(+1.16; x=0.2967; ref 0.4602±0.0628)|inlens_particle_texture_p90(+1.09; x=0.6302; ref 0.28±0.114)|pore_count_per_Mpx(+0.59; x=119.5; ref 98.13±5.48) | bright_count_per_Mpx(+z=+39.49)|bright_frac(+z=+13.55)|bright_max_d(+z=+9.85) |
| Batch_1 | f1vzngrs | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.37 | 0.60 | 0.03 | Batch_2 | 0.73 | 0.17 | 0.10 | Batch_1 | 0.56 | 0.36 | 0.07 | Batch_1 | 94.12 | False | 100.00 | True | 82.35 | False | inlens_particle_texture_p90(+0.91; x=0.5583; ref 0.28±0.114)|crack_count_per_Mpx(-0.50; x=0.06651; ref 0.5526±0.211)|ridge_p97(+0.38; x=0.398; ref 0.4602±0.0628) | pore_count_per_Mpx(+z=+6.62)|etd_crack_density_particles(+z=+2.30)|fft_slope(+z=+1.69) |
| Batch_1 | ffwubibz | known | ordinary | 0.10 | 0.52 | 0.38 | Batch_2 | 0.80 | 0.14 | 0.06 | Batch_1 | 0.25 | 0.63 | 0.12 | Batch_2 | 0.30 | 0.55 | 0.16 | Batch_2 | 29.41 | False | 47.06 | False | 76.47 | False | inlens_grad_energy(+0.47; x=0.1549; ref 0.07027±0.0166)|pore_count_per_Mpx(+0.45; x=90.25; ref 98.13±5.48)|etd_crack_density_particles(+0.32; x=0.0002022; ref 0.001503±0.00138) | pore_count_per_Mpx(-z=-1.44)|bright_count_per_Mpx(-z=-0.84)|corr_len_px(-z=-1.12) |
| Batch_1 | fzrt2k6r | known | ordinary | 0.29 | 0.21 | 0.50 | Batch_3 | 0.15 | 0.47 | 0.38 | Batch_2 | 0.29 | 0.38 | 0.33 | Batch_2 | 0.26 | 0.23 | 0.51 | Batch_3 | 17.65 | False | 29.41 | False | 70.59 | False | inlens_grad_energy(+0.83; x=0.1817; ref 0.07027±0.0166)|pore_count_per_Mpx(-0.18; x=100.6; ref 98.13±5.48)|pore_frac(-0.16; x=0.09602; ref 0.09307±0.0183) | bright_d50(+z=+1.42)|bright_max_d(+z=+2.13)|bright_circ(-z=-1.60) |
| Batch_1 | iv6g2oq0 | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.45 | 0.32 | 0.23 | Batch_1 | 0.45 | 0.27 | 0.27 | Batch_1 | 0.49 | 0.21 | 0.31 | Batch_1 | 11.76 | False | 41.18 | False | 52.94 | False | ridge_p97(+0.57; x=0.3696; ref 0.4602±0.0628)|inlens_particle_texture_p90(+0.30; x=0.4662; ref 0.28±0.114)|crack_g_0.2(+0.12; x=0.06466; ref 0.08832±0.0145) | bright_d50(+z=+2.15)|bright_max_d(-z=-1.16)|crack_count_per_Mpx(+z=+1.97) |
| Batch_1 | uhdslk0o | known | ordinary | 0.31 | 0.28 | 0.41 | Batch_3 | 0.54 | 0.28 | 0.18 | Batch_1 | 0.43 | 0.38 | 0.19 | Batch_1 | 0.49 | 0.33 | 0.19 | Batch_1 | 17.65 | False | 41.18 | False | 52.94 | False | ridge_p97(+0.45; x=0.3942; ref 0.4602±0.0628)|inlens_particle_texture_p90(+0.40; x=0.4779; ref 0.28±0.114)|crack_g_0.2(+0.09; x=0.07057; ref 0.08832±0.0145) | crack_count_per_Mpx(+z=+0.98)|bright_count_per_Mpx(+z=+1.13)|bright_d90(+z=+1.51) |
| Batch_2 | 3806gxp0 | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.00 | 0.44 | 0.55 | Batch_3 | 0.18 | 0.53 | 0.28 | Batch_2 | 0.00 | 0.89 | 0.11 | Batch_2 | 29.41 | False | 41.18 | False | 52.94 | False | pore_count_per_Mpx(+0.91; x=80.98; ref 98.13±5.48)|pore_frac(+0.43; x=0.07288; ref 0.09307±0.0183)|etd_crack_density_particles(+0.14; x=8.763e-05; ref 0.001503±0.00138) | pore_count_per_Mpx(-z=-3.13)|corr_len_px(-z=-0.90)|pore_d90(-z=-1.14) |
| Batch_2 | avn74qx1 | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.33 | 0.34 | 0.34 | Batch_3 | 0.08 | 0.55 | 0.37 | Batch_2 | 0.08 | 0.57 | 0.35 | Batch_2 | 76.47 | False | 47.06 | False | 70.59 | False | pore_count_per_Mpx(+1.36; x=70.94; ref 98.13±5.48)|pore_frac(+0.33; x=0.07489; ref 0.09307±0.0183)|etd_crack_density_particles(+0.19; x=0.0003003; ref 0.001503±0.00138) | pore_count_per_Mpx(-z=-4.96)|bright_circ(+z=+1.55)|bright_d90(+z=+1.38) |
| Batch_2 | b3esycq1 | known | ordinary | 0.39 | 0.09 | 0.51 | Batch_3 | 0.76 | 0.04 | 0.20 | Batch_1 | 0.61 | 0.10 | 0.30 | Batch_1 | 0.85 | 0.00 | 0.15 | Batch_1 | 52.94 | False | 41.18 | False | 52.94 | False | ridge_p97(+0.53; x=0.388; ref 0.4602±0.0628)|inlens_grad_energy(+0.50; x=0.1712; ref 0.07027±0.0166)|etd_crack_density_particles(+0.22; x=0.003222; ref 0.001503±0.00138) | bright_frac(+z=+2.93)|bright_count_per_Mpx(+z=+2.66)|bright_d50(+z=+1.97) |
| Batch_2 | epqdaau9 | known | ordinary | 0.49 | 0.07 | 0.44 | Batch_1 | 0.15 | 0.71 | 0.14 | Batch_2 | 0.59 | 0.00 | 0.41 | Batch_1 | 0.56 | 0.16 | 0.29 | Batch_1 | 17.65 | False | 41.18 | False | 64.71 | False | pore_count_per_Mpx(+1.03; x=111.8; ref 98.13±5.48)|crack_count_per_Mpx(-0.53; x=0.2679; ref 0.5526±0.211)|inlens_grad_energy(+0.51; x=0.1501; ref 0.07027±0.0166) | pore_count_per_Mpx(+z=+2.50)|bright_d50(+z=+1.41)|crack_count_per_Mpx(-z=-1.35) |
| Batch_2 | i9jiqjwl | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.40 | 0.36 | 0.24 | Batch_1 | 0.51 | 0.44 | 0.05 | Batch_1 | 0.47 | 0.47 | 0.06 | Batch_2 | 41.18 | False | 88.24 | False | 52.94 | False | inlens_particle_texture_p90(+1.08; x=0.6004; ref 0.28±0.114)|inlens_particle_texture(+0.65; x=0.3987; ref 0.1767±0.0657)|crack_g_0.2(+0.24; x=0.06948; ref 0.08832±0.0145) | bright_max_d(+z=+2.03)|fft_slope(+z=+1.67)|bright_d50(-z=-0.41) |
| Batch_2 | r17byphk | known | ordinary | 0.25 | 0.33 | 0.43 | Batch_3 | 0.18 | 0.29 | 0.53 | Batch_3 | 0.30 | 0.35 | 0.35 | Batch_3 | 0.05 | 0.36 | 0.59 | Batch_3 | 52.94 | False | 41.18 | False | 52.94 | False | crack_g_0.3(-0.20; x=0.04136; ref 0.05204±0.0112)|crack_g_0.05(+0.14; x=0.2788; ref 0.3395±0.0373)|pore_max_d(-0.14; x=253.6; ref 278.2±105) | pore_count_per_Mpx(-z=-2.98)|fft_slope(-z=-0.14)|crack_count_per_Mpx(+z=+1.61) |
| Batch_2 | rxax5ozo | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.12 | 0.13 | 0.75 | Batch_3 | 0.13 | 0.27 | 0.59 | Batch_3 | 0.11 | 0.35 | 0.54 | Batch_3 | 0.00 | False | 0.00 | False | 41.18 | False | inlens_grad_energy(+1.06; x=0.07039; ref 0.07027±0.0166)|pore_frac(-0.12; x=0.08433; ref 0.09307±0.0183)|crack_g_0.2(+0.07; x=0.08384; ref 0.08832±0.0145) | pore_count_per_Mpx(-z=-2.22)|bright_circ(-z=-1.14)|corr_len_px(-z=+0.00) |
| Batch_3 | 0grcilhi | known | cracked | 0.33 | 0.33 | 0.33 | Batch_1 | 0.04 | 0.28 | 0.68 | Batch_3 | 0.41 | 0.22 | 0.37 | Batch_1 | 0.37 | 0.09 | 0.54 | Batch_3 | 94.12 | False | 100.00 | False | 88.24 | False | ridge_p97(+1.04; x=0.3217; ref 0.4602±0.0628)|inlens_particle_texture_p90(-0.75; x=0.173; ref 0.28±0.114)|crack_g_0.2(+0.71; x=0.05649; ref 0.08832±0.0145) | fft_slope(+z=+2.74)|crack_frac(+z=+4.33)|pore_d90(+z=+3.99) |
| Batch_3 | 71vgq3fw | known | grey_pore | 0.23 | 0.35 | 0.42 | Batch_3 | 0.02 | 0.00 | 0.98 | Batch_3 | 0.07 | 0.26 | 0.67 | Batch_3 | 0.02 | 0.00 | 0.98 | Batch_3 | 5.88 | False | 17.65 | False | 29.41 | False | inlens_particle_texture(+0.69; x=0.1643; ref 0.1767±0.0657)|crack_g_0.3(+0.41; x=0.05929; ref 0.05204±0.0112)|crack_g_0.2(+0.33; x=0.0981; ref 0.08832±0.0145) | bright_d50(+z=+0.76)|bright_d90(+z=+0.68)|bright_count_per_Mpx(+z=+0.13) |
| Batch_3 | 9luzk4jm | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.14 | 0.32 | 0.55 | Batch_3 | 0.24 | 0.23 | 0.53 | Batch_3 | 0.00 | 0.00 | 0.99 | Batch_3 | 47.06 | False | 47.06 | False | 82.35 | False | inlens_particle_texture(-0.38; x=0.2741; ref 0.1767±0.0657)|pore_max_d(+0.26; x=340.7; ref 278.2±105)|crack_g_0.2(+0.14; x=0.08832; ref 0.08832±0.0145) | bright_max_d(+z=+1.61)|pore_frac(+z=+2.23)|crack_frac(+z=+1.67) |
| Batch_3 | cfe5vt7s | known | ordinary | 0.28 | 0.47 | 0.24 | Batch_2 | 0.71 | 0.26 | 0.03 | Batch_1 | 0.40 | 0.45 | 0.16 | Batch_2 | 0.36 | 0.62 | 0.01 | Batch_2 | 23.53 | False | 76.47 | False | 58.82 | False | etd_crack_density_particles(+0.26; x=0.0002175; ref 0.001503±0.00138)|pore_count_per_Mpx(+0.24; x=93.61; ref 98.13±5.48)|pore_frac(+0.23; x=0.07986; ref 0.09307±0.0183) | bright_d50(+z=+2.47)|etd_crack_density_particles(-z=-1.12)|bright_count_per_Mpx(+z=+1.38) |
| Batch_3 | hawkfj64 | known | ordinary | 0.34 | 0.23 | 0.43 | Batch_3 | 0.13 | 0.28 | 0.59 | Batch_3 | 0.17 | 0.16 | 0.67 | Batch_3 | 0.10 | 0.02 | 0.89 | Batch_3 | 41.18 | False | 29.41 | False | 35.29 | False | inlens_particle_texture(+0.98; x=0.1209; ref 0.1767±0.0657)|pore_max_d(-0.33; x=235.5; ref 278.2±105)|crack_g_0.2(+0.06; x=0.08517; ref 0.08832±0.0145) | bright_frac(-z=-1.93)|bright_d90(-z=-2.68)|bright_count_per_Mpx(-z=-0.59) |
| Batch_3 | hzumfsms | known | cracked | 0.15 | 0.09 | 0.76 | Batch_3 | 0.01 | 0.25 | 0.74 | Batch_3 | 0.18 | 0.09 | 0.73 | Batch_3 | 0.00 | 0.00 | 1.00 | Batch_3 | 100.00 | False | 94.12 | False | 76.47 | False | pore_max_d(+1.53; x=599.4; ref 278.2±105)|pore_frac(+0.45; x=0.1413; ref 0.09307±0.0183)|crack_g_0.3(+0.10; x=0.0499; ref 0.05204±0.0112) | pore_d90(+z=+7.16)|bright_circ(+z=+2.48)|fft_slope(-z=-2.03) |
| Batch_3 | kbdh4tri | known | grey_pore | 0.32 | 0.34 | 0.33 | Batch_2 | 0.04 | 0.00 | 0.96 | Batch_3 | 0.03 | 0.22 | 0.75 | Batch_3 | 0.01 | 0.00 | 0.99 | Batch_3 | 17.65 | False | 11.76 | False | 5.88 | False | inlens_particle_texture(+1.09; x=0.1227; ref 0.1767±0.0657)|crack_g_0.2(+0.69; x=0.1108; ref 0.08832±0.0145)|crack_g_0.3(+0.62; x=0.06873; ref 0.05204±0.0112) | bright_count_per_Mpx(+z=+3.06)|bright_max_d(-z=-1.29)|pore_count_per_Mpx(+z=+0.50) |
| Batch_3 | mgxahqnk | known | ordinary | 0.49 | 0.29 | 0.22 | Batch_1 | 0.31 | 0.29 | 0.40 | Batch_3 | 0.31 | 0.18 | 0.51 | Batch_3 | 0.29 | 0.18 | 0.52 | Batch_3 | 82.35 | False | 64.71 | False | 52.94 | False | inlens_particle_texture(+0.85; x=0.1391; ref 0.1767±0.0657)|crack_g_0.3(-0.25; x=0.04283; ref 0.05204±0.0112)|pore_max_d(-0.09; x=274.9; ref 278.2±105) | bright_d90(+z=+4.97)|bright_max_d(+z=+3.14)|bright_d50(+z=+4.78) |
| Batch_3 | pl8uabbv | known | ordinary | 0.36 | 0.19 | 0.45 | Batch_3 | 0.10 | 0.84 | 0.06 | Batch_2 | 0.57 | 0.25 | 0.18 | Batch_1 | 0.15 | 0.83 | 0.02 | Batch_2 | 35.29 | False | 82.35 | False | 64.71 | False | inlens_particle_texture_p90(+0.46; x=0.5048; ref 0.28±0.114)|pore_count_per_Mpx(+0.43; x=109; ref 98.13±5.48)|ridge_p97(+0.28; x=0.4102; ref 0.4602±0.0628) | pore_count_per_Mpx(+z=+2.11)|pore_frac(+z=+1.08)|bright_count_per_Mpx(+z=+1.88) |
| Batch_3 | ptg8lmto | known | ordinary | 0.32 | 0.38 | 0.30 | Batch_2 | 0.22 | 0.26 | 0.52 | Batch_3 | 0.13 | 0.28 | 0.58 | Batch_3 | 0.07 | 0.22 | 0.71 | Batch_3 | 70.59 | False | 88.24 | False | 94.12 | False | crack_g_0.2(+0.88; x=0.1046; ref 0.08832±0.0145)|inlens_particle_texture(+0.31; x=0.2055; ref 0.1767±0.0657)|pore_max_d(-0.11; x=265.5; ref 278.2±105) | bright_d90(+z=+2.36)|fft_slope(+z=+3.10)|bright_d50(-z=-0.56) |
| Batch_3 | tuy3zymq | known | grey_pore | 0.27 | 0.48 | 0.26 | Batch_2 | 0.01 | 0.00 | 0.98 | Batch_3 | 0.02 | 0.27 | 0.71 | Batch_3 | 0.04 | 0.15 | 0.82 | Batch_3 | 29.41 | False | 41.18 | False | 11.76 | False | inlens_particle_texture(+1.00; x=0.1324; ref 0.1767±0.0657)|crack_g_0.2(+0.89; x=0.1163; ref 0.08832±0.0145)|crack_g_0.3(+0.69; x=0.07292; ref 0.05204±0.0112) | pore_count_per_Mpx(-z=-1.10)|bright_d90(-z=-0.96)|bright_count_per_Mpx(+z=+2.91) |
| Batch_3 | ufdvpb81 | known | cracked | 0.32 | 0.12 | 0.56 | Batch_3 | 0.23 | 0.45 | 0.32 | Batch_2 | 0.29 | 0.19 | 0.52 | Batch_3 | 0.72 | 0.01 | 0.27 | Batch_1 | 76.47 | False | 58.82 | False | 100.00 | False | pore_max_d(+0.38; x=456.9; ref 278.2±105)|pore_frac(+0.38; x=0.1198; ref 0.09307±0.0183)|inlens_particle_texture(-0.30; x=0.2656; ref 0.1767±0.0657) | pore_d90(+z=+2.62)|fft_slope(-z=-0.99)|crack_frac(+z=+2.86) |
| Batch_3 | utfgcjfa | known | ordinary | 0.33 | 0.22 | 0.44 | Batch_3 | 0.17 | 0.37 | 0.46 | Batch_3 | 0.20 | 0.22 | 0.58 | Batch_3 | 0.02 | 0.08 | 0.90 | Batch_3 | 52.94 | False | 35.29 | False | 17.65 | False | inlens_particle_texture(+0.51; x=0.1834; ref 0.1767±0.0657)|pore_max_d(+0.20; x=418.7; ref 278.2±105)|pore_frac(+0.16; x=0.09758; ref 0.09307±0.0183) | bright_d50(+z=+3.57)|pore_count_per_Mpx(-z=-1.09)|bright_circ(+z=+1.84) |
| Batch_3 | vc2whyaq | known | ordinary | 0.08 | 0.44 | 0.48 | Batch_3 | 0.13 | 0.53 | 0.34 | Batch_2 | 0.06 | 0.41 | 0.52 | Batch_3 | 0.06 | 0.50 | 0.44 | Batch_2 | 58.82 | False | 23.53 | False | 47.06 | False | inlens_particle_texture(+0.52; x=0.1837; ref 0.1767±0.0657)|crack_g_0.3(+0.41; x=0.05519; ref 0.05204±0.0112)|pore_frac(-0.17; x=0.0828; ref 0.09307±0.0183) | pore_count_per_Mpx(-z=-3.45)|bright_d50(+z=+0.71)|pore_d90(+z=+1.22) |
| Batch_3 | x77cy643 | known | ordinary | 0.29 | 0.25 | 0.45 | Batch_3 | 0.15 | 0.39 | 0.46 | Batch_3 | 0.15 | 0.18 | 0.67 | Batch_3 | 0.02 | 0.18 | 0.80 | Batch_3 | 11.76 | False | 5.88 | False | 23.53 | False | inlens_particle_texture(+0.58; x=0.1767; ref 0.1767±0.0657)|crack_g_0.3(+0.30; x=0.05208; ref 0.05204±0.0112)|pore_max_d(-0.12; x=269.2; ref 278.2±105) | bright_max_d(-z=-0.76)|fft_slope(+z=+0.78)|bright_d10(+z=+1.32) |
| Batch_3 | x7u69zsw | known | grey_pore | 0.18 | 0.52 | 0.31 | Batch_2 | 0.14 | 0.06 | 0.79 | Batch_3 | 0.08 | 0.37 | 0.55 | Batch_3 | 0.02 | 0.05 | 0.92 | Batch_3 | 88.24 | False | 70.59 | False | 41.18 | False | inlens_particle_texture(+0.75; x=0.1329; ref 0.1767±0.0657)|pore_frac(-0.69; x=0.06307; ref 0.09307±0.0183)|crack_g_0.2(+0.44; x=0.09851; ref 0.08832±0.0145) | bright_count_per_Mpx(+z=+3.55)|bright_frac(+z=+3.05)|bright_d90(+z=+3.62) |
| Batch_3 | xgj4xftb | known | ordinary | 0.39 | 0.31 | 0.30 | Batch_1 | 0.00 | 0.07 | 0.92 | Batch_3 | 0.04 | 0.17 | 0.79 | Batch_3 | 0.03 | 0.20 | 0.77 | Batch_3 | 64.71 | False | 52.94 | False | 70.59 | False | crack_g_0.2(+1.04; x=0.116; ref 0.08832±0.0145)|inlens_particle_texture(+0.97; x=0.1355; ref 0.1767±0.0657)|crack_g_0.3(+0.61; x=0.07213; ref 0.05204±0.0112) | bright_d50(+z=+3.44)|bright_d90(+z=+3.65)|corr_len_px(-z=-1.48) |

batch-level QC verdict: python3 -m polaron_qc.report <reference_dir> <batch_dir> — separate output, not derived from this table
