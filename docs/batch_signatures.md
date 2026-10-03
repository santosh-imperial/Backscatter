# Batch signatures against the supplier-promised baseline (Batch 3)

Workstream L4 (next_steps.md), 2026-10-03. Follows the provider's clarification (D49P): Batch 3 is the promised baseline; Batches 1 and 2 arrived later and show the kinds of variation the system must pick up. This page answers *what is different* about each later batch, descriptor by descriptor, with the size and direction of the shift, the family the descriptor belongs to, and what a materials expert should look at. It is **explanatory evidence**: the frozen five-KPI QC verdict (`notebooks/02_batch_qc.ipynb`, `reports/`) and the sample-identification outputs (L1/L2, `polaron_qc/categorise.py`) are separate deliverables.

Source: `analysis/signatures/batch_signatures.csv` (119 descriptors with ≥ 10 usable baseline sites; per descriptor: baseline median and MAD, Batch 1 and Batch 2 median, robust shift in baseline MADs, site-level permutation p on the Hodges–Lehmann shift, 5 000 Monte Carlo relabelings). Sites are n (7 / 7 / 17); low-contrast sites are excluded from bright-phase descriptors, grey-pore sites kept as fallback. Figures: `analysis/signatures/fig_signature_strips.png`, `fig_signature_crops.png`.

## 1. How much separates at all

| family | descriptors scanned | p < 0.05 for Batch 1 or Batch 2 vs baseline | p < 0.01 |
|---|---|---|---|
| texture (ETD ridge density, ETD ridge strength, Inlens particle interior) — acquisition-sensitive | 27 | 13 | 9 |
| morphology (BSE geometry: pores, additive, depth profiles, texture scale) | 50 | 7 | 2 |
| acquisition statistics (black level, contrast, boundary sharpness, channel medians) | 15 | 3 | 0 |
| secondary battery geometry (exploratory) | 27 | 1 | 1 |

Twenty-four of 119 at p < 0.05 against ≈ 6 expected by chance, twelve at p < 0.01 against ≈ 1 (`analysis/signatures/family_counts.csv`). The separation is real but it sits mostly in the ETD/Inlens texture family, not in the void and additive KPIs that carry the QC verdict. That is why both batches read *consistent within detectable limits* on the primary five (largest primary shift: additive D50 +1.4 MAD in Batch 1, Holm-adjusted p above α).

## 2. Batch 1 versus baseline — the signature

| descriptor | family | shift (baseline MADs) | p | reading |
|---|---|---|---|---|
| Inlens particle-interior texture (SD) | texture | **+2.2** | 5e-5 | all seven Batch 1 sites above the baseline median (one above the baseline maximum; the baseline has one high outlier at 0.34) |
| Inlens gradient energy | texture | +5.5 | 8e-5 | same channel, same direction |
| Inlens texture p90 | texture | +2.0 | 5e-4 | |
| ETD ridge density in graphite (every ridge threshold 0.15–0.4) | texture | **−1.3 to −1.6** | 1e-3 to 6e-3 | all seven Batch 1 sites below the baseline median; graphite faces show fewer / weaker fine ridges in ETD |
| ETD ridge strength p97 | texture | −1.4 | 2e-3 | |
| additive fraction in depth bins 8 and 9 of 10 | morphology | **+1.8 / +2.8** | 6e-3 / 7e-3 | more bright additive in the lower part of the strip (orientation unconfirmed, A1) |
| ETD boundary sharpness | acquisition | +2.1 | 0.014 | the flag that alone separated Batch 1 in E06 |
| ETD median level | acquisition | +1.8 | 0.020 | |
| curtaining fraction | texture/acquisition | −0.9 | 0.026 | less ion-milling curtaining |
| additive D50 (primary) | morphology | +1.4 | 0.07 | larger additive particles; not significant after Holm |

**In words:** Batch 1's additive particles look internally more textured in Inlens and sit lower in the strip; its graphite faces are smoother in ETD; the ETD images are sharper and brighter. The first two can be microstructure; the ETD sharpness and level can be session. The two cannot be separated with this data alone (see §5).

## 3. Batch 2 versus baseline — the signature

| descriptor | family | shift (baseline MADs) | p | reading |
|---|---|---|---|---|
| pore count per Mpx | morphology | **−2.2** | 0.013 | fewer resolved pores per area (six of seven Batch 2 sites below the baseline median, two below its minimum) |
| additive-to-void distance D50 (secondary, exploratory) | secondary | **+2.7** | 0.008 | additive particles sit farther from voids (six of seven Batch 2 sites above the baseline median; none above its maximum) |
| additive fraction in depth bin 5 | morphology | −1.4 | 0.015 | |
| pore fraction in depth bins 6–8 | morphology | −0.9 to −1.0 | 0.04–0.10 | |
| Inlens particle-interior texture | texture | +0.9 | 0.013 | between Batch 1 and the baseline; still all seven Batch 2 sites above the baseline median |
| Inlens gradient energy | texture | +3.1 | 0.038 | |
| pore fraction (primary) | morphology | −1.0 | 0.14 | the largest primary shift for Batch 2; not significant |
| additive D50 (primary) | morphology | +1.3 | 0.31 | |

**In words:** Batch 2's change is in the void structure — fewer, and by the primary KPIs slightly less, porosity, with the additive farther from the voids — plus a milder version of Batch 1's Inlens texture shift. Its ETD ridge density is also lower than the baseline but not significantly so.

## 4. The ordering that both batches share

Three texture descriptors order the batches monotonically **Batch 1 > Batch 2 > Batch 3** (Inlens particle texture, Inlens gradient energy) or **Batch 1 < Batch 2 < Batch 3** (ETD graphite ridge density at every threshold). On Inlens particle texture all 14 later-batch sites lie above the baseline median; on ETD ridge density all 14 lie below it. A monotone ordering across two independent later deliveries is harder to produce by session alone than a single shift, but it is not proof: sessions can also be ordered in time, and both descriptors are acquisition-sensitive. E20 (`analysis/e20_inlens/`) tests whether the Inlens ordering survives per-particle local-contrast normalisation; surviving establishes robustness to that transformation, not a material origin, and losing it can mean the normalisation removed useful contrast.

## 5. What an expert should look at, and what this does not claim

- `fig_signature_crops.png`: the same 640 px window in ETD and Inlens for one ordinary site per batch, fixed display limits. The question for the expert is whether the Batch 1 particle interiors show real internal structure (porous composite, fracture) or charging/edge relief, and whether the smoother graphite faces in ETD are a surface property or an imaging difference.
- Known confounds stay on the page: Inlens texture correlates with Inlens brightness (ρ ≈ 0.77, E03); ETD boundary sharpness is an acquisition flag; the two Batch 1 low-contrast sites (rings in the strip plot) sit at the extremes of several ETD descriptors and are excluded from bright-phase ones.
- Nothing here is a defect call or a better/worse judgement — the provider says the later batches are neither explicitly better nor worse. Nothing here enters the frozen verdict thresholds. Descriptor selection for the sample categoriser happens inside its evaluation folds (L1), not from this table.
- 7 sites per later batch give wide intervals; a p of 0.01 on 119 descriptors is expected about once by chance, so the family pattern (13 texture hits) carries more weight than any single row.

## 6. Reproduce

```bash
/opt/anaconda3/bin/python3 analysis/signatures/make_signatures.py
```
