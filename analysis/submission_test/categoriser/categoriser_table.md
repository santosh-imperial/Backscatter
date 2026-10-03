# Site categorisation, baseline OOD assessment and hand-off

Three separate answers per site; model probabilities are classifier scores, not posteriors or confidence; OOD percentiles are descriptive evidence ranks (0–100), not probabilities of defect or evidence of equivalence. Legacy ood_rank_p columns have tail-rank floor 1/(n_ref+1), not calibrated p-values or false-alert guarantees: reference LOO fits use n_ref−1 sites and query fits use n_ref sites. Texture family: acquisition-sensitive; batch-fingerprint evidence, origin (microstructure vs imaging) not established.

## 1. Site categorisation (leave-one-site-out, fold-local imputer/scaler/C choice, balanced class weights)

| family | primary | n_features | accuracy | accuracy_ci95 | majority_accuracy | balanced_accuracy | chance_balanced | recall_Batch_1 | recall_Batch_2 | recall_Batch_3 | perm_p | n_perm | brier | brier_prior | ece |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| morph | False | 19 | 0.387 | [0.22, 0.58] | 0.548 | 0.347 | 0.333 | 0.571 | 0.000 | 0.471 | 0.159 | 200 | 0.685 | 0.597 | 0.063 |
| acq | False | 10 | 0.548 | [0.36, 0.73] | 0.548 | 0.473 | 0.333 | 0.571 | 0.143 | 0.706 | 0.080 | 200 | 0.516 | 0.597 | 0.121 |
| combined | True | 39 | 0.677 | [0.49, 0.83] | 0.548 | 0.664 | 0.333 | 0.714 | 0.571 | 0.706 | 0.005 | 200 | 0.457 | 0.597 | 0.110 |

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

### Confusion — acq (acquisition-only (session / instrument statistics; never material evidence))

|  | pred Batch_1 | pred Batch_2 | pred Batch_3 |
|---|---|---|---|
| true Batch_1 | 4 | 3 | 0 |
| true Batch_2 | 2 | 1 | 4 |
| true Batch_3 | 2 | 3 | 12 |

Reliability of the top-class model probability — acq

| bin | n | mean_top_prob | observed_top_accuracy |
|---|---|---|---|
| [0.33, 0.50) | 16 | 0.413 | 0.312 |
| [0.50, 0.70) | 9 | 0.559 | 0.778 |
| [0.70, 1.00) | 6 | 0.806 | 0.833 |

### Confusion — combined (combined = morphology + texture [acquisition-sensitive; batch-fingerprint evidence, origin (microstructure vs imaging) not established] + acquisition (PRIMARY, pre-registered))

|  | pred Batch_1 | pred Batch_2 | pred Batch_3 |
|---|---|---|---|
| true Batch_1 | 5 | 2 | 0 |
| true Batch_2 | 2 | 4 | 1 |
| true Batch_3 | 3 | 2 | 12 |

Reliability of the top-class model probability — combined

| bin | n | mean_top_prob | observed_top_accuracy |
|---|---|---|---|
| [0.33, 0.50) | 10 | 0.437 | 0.400 |
| [0.50, 0.70) | 14 | 0.602 | 0.786 |
| [0.70, 1.00) | 7 | 0.791 | 0.857 |

## 2. Baseline OOD assessment (reference = all Batch_3 sites; k-NN robust-z distance; percentiles within the reference LOO)

| variant | batch | n_sites | pct_min | pct_median | pct_max | n_exceed_ref_max | frac_exceed_ref_max | n_rank_p_at_floor | matched_pct_median | rms_n_exceed | maha_n_exceed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| morph | Batch_1 | 7 | 11.76 | 29.41 | 100.00 | 2 | 0.29 | 2 | 29.41 | 2 | 2 |
| morph | Batch_2 | 7 | 0.00 | 41.18 | 76.47 | 0 | 0.00 | 0 | 47.06 | 0 | 0 |
| morph_texture | Batch_1 | 7 | 29.41 | 47.06 | 100.00 | 3 | 0.43 | 3 | 64.71 | 3 | 2 |
| morph_texture | Batch_2 | 7 | 0.00 | 41.18 | 88.24 | 0 | 0.00 | 0 | 41.18 | 1 | 0 |
| acq | Batch_1 | 7 | 47.06 | 70.59 | 82.35 | 0 | 0.00 | 0 | 70.59 | 0 | 0 |
| acq | Batch_2 | 7 | 23.53 | 47.06 | 70.59 | 0 | 0.00 | 0 | 58.82 | 0 | 0 |

## 3. Per-site hand-off (abridged; full columns in site_table.csv)

| batch | site | role | acquisition_group | mp_Batch_1__morph | mp_Batch_2__morph | mp_Batch_3__morph | argmax__morph | mp_Batch_1__acq | mp_Batch_2__acq | mp_Batch_3__acq | argmax__acq | mp_Batch_1__combined | mp_Batch_2__combined | mp_Batch_3__combined | argmax__combined | ood_pct_knn__morph | ood_exceeds_knn__morph | ood_pct_knn__morph_texture | ood_exceeds_knn__morph_texture | ood_pct_knn__acq | ood_exceeds_knn__acq | cat_top_features__combined | ood_top_features__morph |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Batch_1 | 4ih2ggld | known | low_contrast | 0.54 | 0.06 | 0.40 | Batch_1 | 0.34 | 0.35 | 0.31 | Batch_2 | 0.66 | 0.12 | 0.22 | Batch_1 | 100.00 | True | 100.00 | True | 76.47 | False | ridge_p97(+1.59; x=0.3185; ref 0.4602±0.0628)|etd_boundary_sharpness(-0.55; x=0.6629; ref 0.7495±0.106)|pore_count_per_Mpx(+0.54; x=119.9; ref 98.13±5.48) | bright_count_per_Mpx(+z=+36.68)|bright_frac(+z=+8.11)|bright_solidity(-z=-7.72) |
| Batch_1 | 5n1q8atc | known | low_contrast | 0.33 | 0.33 | 0.33 | Batch_1 | 0.58 | 0.23 | 0.19 | Batch_1 | 0.74 | 0.21 | 0.05 | Batch_1 | 100.00 | True | 100.00 | True | 82.35 | False | ridge_p97(+1.95; x=0.2967; ref 0.4602±0.0628)|inlens_particle_texture_p90(+0.92; x=0.6302; ref 0.28±0.114)|pore_count_per_Mpx(+0.67; x=119.5; ref 98.13±5.48) | bright_count_per_Mpx(+z=+39.49)|bright_frac(+z=+13.55)|bright_max_d(+z=+9.85) |
| Batch_1 | f1vzngrs | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.35 | 0.49 | 0.17 | Batch_2 | 0.56 | 0.36 | 0.07 | Batch_1 | 94.12 | False | 100.00 | True | 70.59 | False | inlens_particle_texture_p90(+0.72; x=0.5583; ref 0.28±0.114)|etd_boundary_sharpness(+0.49; x=1.205; ref 0.7495±0.106)|pore_count_per_Mpx(+0.48; x=134.4; ref 98.13±5.48) | pore_count_per_Mpx(+z=+6.62)|etd_crack_density_particles(+z=+2.30)|fft_slope(+z=+1.69) |
| Batch_1 | ffwubibz | known | ordinary | 0.10 | 0.52 | 0.38 | Batch_2 | 0.57 | 0.21 | 0.22 | Batch_1 | 0.30 | 0.55 | 0.16 | Batch_2 | 29.41 | False | 47.06 | False | 47.06 | False | pore_count_per_Mpx(+0.45; x=90.25; ref 98.13±5.48)|corr_len_px(+0.36; x=16; ref 21±4.45)|pore_max_d(+0.29; x=201; ref 278.2±105) | pore_count_per_Mpx(-z=-1.44)|bright_count_per_Mpx(-z=-0.84)|corr_len_px(-z=-1.12) |
| Batch_1 | fzrt2k6r | known | ordinary | 0.29 | 0.21 | 0.50 | Batch_3 | 0.34 | 0.37 | 0.30 | Batch_2 | 0.33 | 0.33 | 0.33 | Batch_2 | 17.65 | False | 29.41 | False | 58.82 | False | inlens_p50(+1.16; x=148; ref 89±44.5)|bse_p50(-0.45; x=62; ref 57±2.97)|pore_count_per_Mpx(-0.16; x=100.6; ref 98.13±5.48) | bright_d50(+z=+1.42)|bright_max_d(+z=+2.13)|bright_circ(-z=-1.60) |
| Batch_1 | iv6g2oq0 | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.41 | 0.30 | 0.29 | Batch_1 | 0.49 | 0.20 | 0.31 | Batch_1 | 11.76 | False | 41.18 | False | 70.59 | False | ridge_p97(+0.68; x=0.3696; ref 0.4602±0.0628)|inlens_particle_texture_p90(+0.26; x=0.4662; ref 0.28±0.114)|pore_count_per_Mpx(+0.08; x=99.77; ref 98.13±5.48) | bright_d50(+z=+2.15)|bright_max_d(-z=-1.16)|crack_count_per_Mpx(+z=+1.97) |
| Batch_1 | uhdslk0o | known | ordinary | 0.31 | 0.28 | 0.41 | Batch_3 | 0.52 | 0.27 | 0.21 | Batch_1 | 0.49 | 0.33 | 0.19 | Batch_1 | 17.65 | False | 41.18 | False | 70.59 | False | ridge_p97(+0.52; x=0.3942; ref 0.4602±0.0628)|inlens_particle_texture_p90(+0.40; x=0.4779; ref 0.28±0.114)|bse_empty_bin_frac(+0.21; x=0.05224; ref 0.09375±0.119) | crack_count_per_Mpx(+z=+0.98)|bright_count_per_Mpx(+z=+1.13)|bright_d90(+z=+1.51) |
| Batch_2 | 3806gxp0 | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.09 | 0.43 | 0.48 | Batch_3 | 0.10 | 0.63 | 0.27 | Batch_2 | 29.41 | False | 41.18 | False | 41.18 | False | pore_count_per_Mpx(+0.89; x=80.98; ref 98.13±5.48)|bse_p50(+0.62; x=49; ref 57±2.97)|pore_frac(+0.45; x=0.07288; ref 0.09307±0.0183) | pore_count_per_Mpx(-z=-3.13)|corr_len_px(-z=-0.90)|pore_d90(-z=-1.14) |
| Batch_2 | avn74qx1 | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.33 | 0.33 | 0.34 | Batch_3 | 0.08 | 0.57 | 0.35 | Batch_2 | 76.47 | False | 47.06 | False | 47.06 | False | pore_count_per_Mpx(+1.43; x=70.94; ref 98.13±5.48)|bse_p50(+0.65; x=49; ref 57±2.97)|pore_frac(+0.33; x=0.07489; ref 0.09307±0.0183) | pore_count_per_Mpx(-z=-4.96)|bright_circ(+z=+1.55)|bright_d90(+z=+1.38) |
| Batch_2 | b3esycq1 | known | ordinary | 0.39 | 0.09 | 0.51 | Batch_3 | 0.77 | 0.04 | 0.19 | Batch_1 | 0.62 | 0.07 | 0.31 | Batch_1 | 52.94 | False | 41.18 | False | 47.06 | False | ridge_p97(+0.60; x=0.388; ref 0.4602±0.0628)|inlens_grad_energy(+0.36; x=0.1712; ref 0.07027±0.0166)|bse_p50(+0.33; x=62; ref 57±2.97) | bright_frac(+z=+2.93)|bright_count_per_Mpx(+z=+2.66)|bright_d50(+z=+1.97) |
| Batch_2 | epqdaau9 | known | ordinary | 0.49 | 0.07 | 0.44 | Batch_1 | 0.28 | 0.48 | 0.24 | Batch_2 | 0.71 | 0.01 | 0.28 | Batch_1 | 17.65 | False | 41.18 | False | 47.06 | False | pore_count_per_Mpx(+1.43; x=111.8; ref 98.13±5.48)|bse_empty_bin_frac(+0.82; x=0.07463; ref 0.09375±0.119)|etd_boundary_sharpness(+0.79; x=1.06; ref 0.7495±0.106) | pore_count_per_Mpx(+z=+2.50)|bright_d50(+z=+1.41)|crack_count_per_Mpx(-z=-1.35) |
| Batch_2 | i9jiqjwl | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.40 | 0.36 | 0.24 | Batch_1 | 0.47 | 0.47 | 0.06 | Batch_2 | 41.18 | False | 88.24 | False | 70.59 | False | inlens_p50(+0.58; x=141; ref 89±44.5)|pore_frac(+0.30; x=0.07544; ref 0.09307±0.0183)|pore_count_per_Mpx(+0.29; x=92.3; ref 98.13±5.48) | bright_max_d(+z=+2.03)|fft_slope(+z=+1.67)|bright_d50(-z=-0.41) |
| Batch_2 | r17byphk | known | ordinary | 0.25 | 0.33 | 0.43 | Batch_3 | 0.36 | 0.26 | 0.38 | Batch_3 | 0.29 | 0.36 | 0.35 | Batch_2 | 52.94 | False | 41.18 | False | 41.18 | False | pore_count_per_Mpx(+0.77; x=81.8; ref 98.13±5.48)|crack_count_per_Mpx(-0.28; x=0.8929; ref 0.5526±0.211)|inlens_p50(+0.13; x=107; ref 89±44.5) | pore_count_per_Mpx(-z=-2.98)|fft_slope(-z=-0.14)|crack_count_per_Mpx(+z=+1.61) |
| Batch_2 | rxax5ozo | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.22 | 0.33 | 0.46 | Batch_3 | 0.11 | 0.35 | 0.54 | Batch_3 | 0.00 | False | 0.00 | False | 23.53 | False | inlens_grad_energy(+1.04; x=0.07039; ref 0.07027±0.0166)|pore_frac(-0.11; x=0.08433; ref 0.09307±0.0183)|crack_g_0.2(+0.06; x=0.08384; ref 0.08832±0.0145) | pore_count_per_Mpx(-z=-2.22)|bright_circ(-z=-1.14)|corr_len_px(-z=+0.00) |
| Batch_3 | 0grcilhi | known | cracked | 0.33 | 0.33 | 0.33 | Batch_1 | 0.12 | 0.31 | 0.57 | Batch_3 | 0.37 | 0.31 | 0.32 | Batch_1 | 94.12 | False | 100.00 | False | 76.47 | False | ridge_p97(+1.84; x=0.3217; ref 0.4602±0.0628)|inlens_particle_texture_p90(-0.71; x=0.173; ref 0.28±0.114)|etd_boundary_sharpness(-0.57; x=0.3646; ref 0.7495±0.106) | fft_slope(+z=+2.74)|crack_frac(+z=+4.33)|pore_d90(+z=+3.99) |
| Batch_3 | 71vgq3fw | known | grey_pore | 0.23 | 0.35 | 0.42 | Batch_3 | 0.07 | 0.08 | 0.85 | Batch_3 | 0.12 | 0.08 | 0.79 | Batch_3 | 5.88 | False | 17.65 | False | 17.65 | False | inlens_particle_texture(+0.62; x=0.1643; ref 0.1767±0.0657)|crack_g_0.2(+0.51; x=0.0981; ref 0.08832±0.0145)|bse_std(+0.35; x=17.08; ref 22.23±2.02) | bright_d50(+z=+0.76)|bright_d90(+z=+0.68)|bright_count_per_Mpx(+z=+0.13) |
| Batch_3 | 9luzk4jm | known | ordinary | 0.33 | 0.33 | 0.33 | Batch_1 | 0.14 | 0.31 | 0.55 | Batch_3 | 0.11 | 0.19 | 0.70 | Batch_3 | 47.06 | False | 47.06 | False | 64.71 | False | inlens_particle_texture(-0.35; x=0.2741; ref 0.1767±0.0657)|pore_max_d(+0.25; x=340.7; ref 278.2±105)|crack_g_0.2(+0.16; x=0.08832; ref 0.08832±0.0145) | bright_max_d(+z=+1.61)|pore_frac(+z=+2.23)|crack_frac(+z=+1.67) |
| Batch_3 | cfe5vt7s | known | ordinary | 0.28 | 0.47 | 0.24 | Batch_2 | 0.60 | 0.27 | 0.12 | Batch_1 | 0.46 | 0.40 | 0.15 | Batch_1 | 23.53 | False | 76.47 | False | 47.06 | False | inlens_particle_texture_p90(+0.50; x=0.4926; ref 0.28±0.114)|etd_p50(+0.32; x=84; ref 73±4.45)|ridge_p97(+0.31; x=0.407; ref 0.4602±0.0628) | bright_d50(+z=+2.47)|etd_crack_density_particles(-z=-1.12)|bright_count_per_Mpx(+z=+1.38) |
| Batch_3 | hawkfj64 | known | ordinary | 0.34 | 0.23 | 0.43 | Batch_3 | 0.13 | 0.28 | 0.59 | Batch_3 | 0.16 | 0.18 | 0.65 | Batch_3 | 41.18 | False | 29.41 | False | 52.94 | False | inlens_particle_texture(+0.67; x=0.1209; ref 0.1767±0.0657)|pore_max_d(-0.32; x=235.5; ref 278.2±105)|inlens_grad_energy(+0.20; x=0.06806; ref 0.07027±0.0166) | bright_frac(-z=-1.93)|bright_d90(-z=-2.68)|bright_count_per_Mpx(-z=-0.59) |
| Batch_3 | hzumfsms | known | cracked | 0.15 | 0.09 | 0.76 | Batch_3 | 0.33 | 0.33 | 0.33 | Batch_1 | 0.09 | 0.06 | 0.85 | Batch_3 | 100.00 | False | 94.12 | False | 58.82 | False | pore_max_d(+1.49; x=599.4; ref 278.2±105)|pore_frac(+0.48; x=0.1413; ref 0.09307±0.0183)|crack_g_0.2(+0.12; x=0.0857; ref 0.08832±0.0145) | pore_d90(+z=+7.16)|bright_circ(+z=+2.48)|fft_slope(-z=-2.03) |
| Batch_3 | kbdh4tri | known | grey_pore | 0.32 | 0.34 | 0.33 | Batch_2 | 0.10 | 0.07 | 0.83 | Batch_3 | 0.06 | 0.08 | 0.86 | Batch_3 | 17.65 | False | 11.76 | False | 5.88 | False | crack_g_0.2(+0.97; x=0.1108; ref 0.08832±0.0145)|inlens_particle_texture(+0.87; x=0.1227; ref 0.1767±0.0657)|bse_std(+0.50; x=16.91; ref 22.23±2.02) | bright_count_per_Mpx(+z=+3.06)|bright_max_d(-z=-1.29)|pore_count_per_Mpx(+z=+0.50) |
| Batch_3 | mgxahqnk | known | ordinary | 0.49 | 0.29 | 0.22 | Batch_1 | 0.31 | 0.29 | 0.40 | Batch_3 | 0.29 | 0.18 | 0.52 | Batch_3 | 82.35 | False | 64.71 | False | 70.59 | False | inlens_particle_texture(+0.62; x=0.1391; ref 0.1767±0.0657)|crack_g_0.3(-0.15; x=0.04283; ref 0.05204±0.0112)|bse_std(-0.13; x=23.43; ref 22.23±2.02) | bright_d90(+z=+4.97)|bright_max_d(+z=+3.14)|bright_d50(+z=+4.78) |
| Batch_3 | pl8uabbv | known | ordinary | 0.36 | 0.19 | 0.45 | Batch_3 | 0.32 | 0.51 | 0.18 | Batch_2 | 0.40 | 0.47 | 0.13 | Batch_2 | 35.29 | False | 82.35 | False | 88.24 | False | inlens_p50(+0.64; x=133; ref 89±44.5)|bse_p50(+0.62; x=53; ref 57±2.97)|pore_count_per_Mpx(-0.59; x=109; ref 98.13±5.48) | pore_count_per_Mpx(+z=+2.11)|pore_frac(+z=+1.08)|bright_count_per_Mpx(+z=+1.88) |
| Batch_3 | ptg8lmto | known | ordinary | 0.32 | 0.38 | 0.30 | Batch_2 | 0.22 | 0.24 | 0.54 | Batch_3 | 0.11 | 0.25 | 0.64 | Batch_3 | 70.59 | False | 88.24 | False | 100.00 | False | crack_g_0.2(+0.74; x=0.1046; ref 0.08832±0.0145)|inlens_particle_texture(+0.28; x=0.2055; ref 0.1767±0.0657)|pore_max_d(-0.11; x=265.5; ref 278.2±105) | bright_d90(+z=+2.36)|fft_slope(+z=+3.10)|bright_d50(-z=-0.56) |
| Batch_3 | tuy3zymq | known | grey_pore | 0.27 | 0.48 | 0.26 | Batch_2 | 0.07 | 0.07 | 0.86 | Batch_3 | 0.04 | 0.15 | 0.81 | Batch_3 | 29.41 | False | 41.18 | False | 11.76 | False | crack_g_0.2(+1.16; x=0.1163; ref 0.08832±0.0145)|inlens_particle_texture(+0.89; x=0.1324; ref 0.1767±0.0657)|bse_std(+0.57; x=16.53; ref 22.23±2.02) | pore_count_per_Mpx(-z=-1.10)|bright_d90(-z=-0.96)|bright_count_per_Mpx(+z=+2.91) |
| Batch_3 | ufdvpb81 | known | cracked | 0.32 | 0.12 | 0.56 | Batch_3 | 0.23 | 0.45 | 0.32 | Batch_2 | 0.42 | 0.18 | 0.41 | Batch_1 | 76.47 | False | 58.82 | False | 82.35 | False | ridge_p97(-0.67; x=0.4944; ref 0.4602±0.0628)|etd_boundary_sharpness(+0.66; x=1.164; ref 0.7495±0.106)|crack_count_per_Mpx(+0.52; x=1.095; ref 0.5526±0.211) | pore_d90(+z=+2.62)|fft_slope(-z=-0.99)|crack_frac(+z=+2.86) |
| Batch_3 | utfgcjfa | known | ordinary | 0.33 | 0.22 | 0.44 | Batch_3 | 0.17 | 0.38 | 0.45 | Batch_3 | 0.17 | 0.27 | 0.56 | Batch_3 | 52.94 | False | 35.29 | False | 35.29 | False | inlens_particle_texture(+0.46; x=0.1834; ref 0.1767±0.0657)|pore_max_d(+0.18; x=418.7; ref 278.2±105)|pore_frac(+0.16; x=0.09758; ref 0.09307±0.0183) | bright_d50(+z=+3.57)|pore_count_per_Mpx(-z=-1.09)|bright_circ(+z=+1.84) |
| Batch_3 | vc2whyaq | known | ordinary | 0.08 | 0.44 | 0.48 | Batch_3 | 0.24 | 0.39 | 0.37 | Batch_2 | 0.06 | 0.50 | 0.44 | Batch_2 | 58.82 | False | 23.53 | False | 41.18 | False | pore_count_per_Mpx(+1.08; x=81.38; ref 98.13±5.48)|inlens_p50(+0.44; x=119; ref 89±44.5)|pore_frac(+0.23; x=0.0828; ref 0.09307±0.0183) | pore_count_per_Mpx(-z=-3.45)|bright_d50(+z=+0.71)|pore_d90(+z=+1.22) |
| Batch_3 | x77cy643 | known | ordinary | 0.29 | 0.25 | 0.45 | Batch_3 | 0.16 | 0.40 | 0.45 | Batch_3 | 0.14 | 0.28 | 0.58 | Batch_3 | 11.76 | False | 5.88 | False | 29.41 | False | inlens_particle_texture(+0.52; x=0.1767; ref 0.1767±0.0657)|crack_g_0.3(+0.22; x=0.05208; ref 0.05204±0.0112)|bse_std(-0.17; x=23.59; ref 22.23±2.02) | bright_max_d(-z=-0.76)|fft_slope(+z=+0.78)|bright_d10(+z=+1.32) |
| Batch_3 | x7u69zsw | known | grey_pore | 0.18 | 0.52 | 0.31 | Batch_2 | 0.14 | 0.07 | 0.79 | Batch_3 | 0.14 | 0.22 | 0.64 | Batch_3 | 88.24 | False | 70.59 | False | 23.53 | False | pore_frac(-0.68; x=0.06307; ref 0.09307±0.0183)|crack_g_0.2(+0.59; x=0.09851; ref 0.08832±0.0145)|inlens_particle_texture(+0.59; x=0.1329; ref 0.1767±0.0657) | bright_count_per_Mpx(+z=+3.55)|bright_frac(+z=+3.05)|bright_d90(+z=+3.62) |
| Batch_3 | xgj4xftb | known | ordinary | 0.39 | 0.31 | 0.30 | Batch_1 | 0.04 | 0.22 | 0.74 | Batch_3 | 0.03 | 0.19 | 0.77 | Batch_3 | 64.71 | False | 52.94 | False | 94.12 | False | crack_g_0.2(+1.32; x=0.116; ref 0.08832±0.0145)|inlens_particle_texture(+0.77; x=0.1355; ref 0.1767±0.0657)|pore_max_d(+0.14; x=341.2; ref 278.2±105) | bright_d50(+z=+3.44)|bright_d90(+z=+3.65)|corr_len_px(-z=-1.48) |
| Hackathon-Polaron-test | 3e122cbj | scored | low_contrast | 0.63 | 0.05 | 0.32 | Batch_1 | 0.52 | 0.25 | 0.23 | Batch_1 | 0.77 | 0.15 | 0.09 | Batch_1 | 100.00 | True | 100.00 | True | 70.59 | False | ridge_p97(+1.65; x=0.3149; ref 0.4602±0.0628)|inlens_particle_texture_p90(+1.35; x=0.7496; ref 0.28±0.114)|pore_count_per_Mpx(+0.71; x=120.8; ref 98.13±5.48) | bright_count_per_Mpx(+z=+44.28)|bright_solidity(-z=-11.10)|bright_frac(+z=+8.90) |
| Hackathon-Polaron-test | fn0mhxef | scored | ordinary | 0.18 | 0.48 | 0.34 | Batch_2 | 0.57 | 0.33 | 0.10 | Batch_1 | 0.40 | 0.45 | 0.15 | Batch_2 | 17.65 | False | 41.18 | False | 70.59 | False | pore_count_per_Mpx(+0.59; x=86.91; ref 98.13±5.48)|bse_p50(+0.42; x=53; ref 57±2.97)|pore_frac(+0.26; x=0.07851; ref 0.09307±0.0183) | corr_len_px(-z=-1.12)|bright_solidity(+z=+1.23)|bright_d50(+z=+1.13) |
| Hackathon-Polaron-test | xrv9xvzb | scored | ordinary | 0.23 | 0.18 | 0.59 | Batch_3 | 0.04 | 0.30 | 0.66 | Batch_3 | 0.08 | 0.18 | 0.74 | Batch_3 | 17.65 | False | 17.65 | False | 47.06 | False | pore_max_d(+0.43; x=419.7; ref 278.2±105)|pore_frac(+0.32; x=0.1136; ref 0.09307±0.0183)|bse_std(+0.19; x=20.07; ref 22.23±2.02) | bright_d50(+z=+0.49)|bright_max_d(-z=-0.94)|pore_elong(-z=-0.51) |

batch-level QC verdict: python3 -m polaron_qc.report <reference_dir> <batch_dir> — separate output, not derived from this table
