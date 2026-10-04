# Source-crop overlap audit (E40S / D59S)

The organiser's new provenance is useful: the supplied sites are crops from
approximately 15 source electrode images, organised into artificial visual
batches. That does not establish approximately 15 physical specimens. Crop-level
validation may share a parent image between training and testing, so its accuracy
cannot establish generalisation to a new source image or electrode.

**No recoverable pixel-overlap components were found by this bounded audit.**
This does not establish independence. Non-overlapping crops from the same parent
are invisible to pixel matching; the organiser's source-to-crop mapping remains
necessary for true source-held-out validation. No grouping or classifier changed.

| Valid run result | Value |
|---|---:|
| Supplied sites / TIFFs | 34 / 102 |
| All unordered BSE pairs scored | 561 |
| Coarse translation candidates, NCC >= 0.85 | 0 |
| Maximum coarse NCC over any legal pair/translation | 0.588363 |
| Confirmed pixel-overlap edges / nontrivial components | 0 / 0 |
| Targeted synthetic tests | 6 passed |
| Raw TIFF hashes unchanged | 102 / 102 |
| Protected submission files unchanged | 106 / 106 |

The all-pair scores are in `output/all_pair_scores.csv`. NCC here describes pixel
alignment, not classification confidence or similarity to a material batch. With
no coarse candidate, there are no real cross-site full-resolution checks or
verified-overlap panels to present. Empty verification/components files record
that result explicitly.

The fixed method removes four export-border pixels, blurs BSE with sigma 8 px,
samples every 8 px and searches translations whose overlap is at least 512 px
high and 1,024 px wide. Candidate verification would refine within 16 px and
require three separated 256 px patches in **each** of BSE, ETD and Inlens to pass
Pearson/Spearman >= 0.995 and affine residual <= 0.10 target SD. There is no
rotation, reflection, rescaling or nonlinear-registration search. Smaller
overlap, severe photometric changes/resampling, or shared-parent regions with no
overlap can be missed. Technical patches are not independent n.

Both stages of the actual gate pass the synthetic translated/affine control:
coarse NCC 0.963315 and exact `(dy=93, dx=233)` recovery. A separate pre-specified
real-image control chooses the supplied BSE file with the lowest SHA-256, without
visual/class selection. Two constructed crops of that site's three channels,
with a fixed affine transform and uint8 quantisation on the second crop, pass
the candidate gate (NCC 0.993826), recover the exact translation and pass all
nine patch checks. Its shuffled negative control fails the coarse gate
(maximum NCC 0.094032). These are technical detection controls, **not** labelled
organiser source relationships or an estimate of detection recall. The control
PNG is labelled as constructed; it is not an example of overlap between two
supplied sites.

Two preliminary attempts are retained. `failed_pre_score_serialisation/` stopped
before pair scoring due to a NumPy-integer JSON bug. `invalid_control_gate_v1/`
was invalid because its forced fine control bypassed a failed coarse gate.
The corrected control requires the actual gate, and the anti-aliasing blur now
matches the sampling interval. The valid `output/` receipt and source snapshots
supersede both. No classifier accuracy, label or batch-height pattern informed
the repair.

`output/pre_score_receipt.json`, `output/verification.json` and
`output/real_image_controls/receipt.json` preserve source hashes, raw-image
hashes, settings and controls. Source, raw data and frozen submission artifacts
remain unchanged during the valid run. This audit does not provide a source-ID
map, physical specimen count, independent phase annotation or true grouped
generalisation result.
