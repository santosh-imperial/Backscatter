# Conservative source-crop overlap audit

This protocol is saved before scoring the supplied images. It follows the new
organiser statement that the challenge batches were assembled from crops of
approximately 15 source electrode images. Neither source-image IDs nor a mapping
to physical specimens is available.

## Scope and fixed procedure

- Use the 31 known sites plus the three first-drop sites. Discover sites and
  detector aliases from filenames; do not use batch labels, feedback labels,
  acquisition flags, frame dimensions or classifier predictions to select pairs
  or to assign source groups. Dimensions serve only to calculate legal pixel
  intersections. A filename/folder is provenance, not a source-image ID.
- Hash all supplied TIFFs and the audit source before scoring; never alter the
  images or any frozen classifier/QC artifacts.
- Remove four pixels from each outside edge to exclude known export borders.
  Blur BSE with sigma 2 px, sample on an 8 px lattice, and calculate translation
  normalized cross-correlation over every pair. No rotation, reflection,
  rescaling or nonlinear registration is searched.
- Search translations with at least 512 px height and 1,024 px width of overlap.
  Require nonconstant overlapping image regions. Record the maximum score and
  offset for every pair. A coarse NCC of at least 0.85 is only a candidate for
  verification, not evidence of a common source.
- Refine a candidate's translation within 16 px of its coarse offset, using a
  central 512 px BSE patch. Verify three separated, fixed 256 px patches in the
  resulting intersection, avoiding its outer 32 px. All patches must be
  nonconstant. For each detector, report Pearson correlation, Spearman rank
  correlation, pixel equality and affine-fit residual normalized by target
  standard deviation. Patches count as technical checks, never independent n.
- Confirm a pixel-overlap edge only when **all three patches in each of BSE,
  ETD and Inlens** have Pearson >= 0.995, Spearman >= 0.995 and affine residual
  <= 0.10 target SD. Exact pixel identity is additionally reported. Missing
  channels, insufficient coverage or a nonconstant-check failure cannot pass.
- Save coordinates and raw matched-patch PNGs for any confirmed edge. Connected
  components of these confirmed edges are diagnostic pixel-overlap components;
  never call them reconstructed source images or physical specimens. Ask the
  coordinating agent before using these components in classification folds.
- Include synthetic translated-crop controls and unrelated-image controls.
  A translated, affinely transformed crop must be recovered; unrelated and
  constant images must not pass. The synthetic controls do not measure recall
  on the organisers' unknown parent images.

## What this can and cannot establish

A confirmed edge shows repeated spatial content across crops under this limited
registration model. It provides a lower bound on detectable shared content. It
does not identify a physical specimen or a microscopy session. Same-parent crops
with no overlap cannot be recovered by pixel matching, and overlap smaller than
the limits, rotation/rescaling, resampling or changed local contrast may also be
missed. Absence of a whole-frame duplicate or of a detected overlap does **not**
establish independent sites. This audit cannot reconstruct the approximately 15
unknown parents or validate true source-held-out generalisation.

No family selection, classifier fitting, feature promotion, phase annotation or
ground-truth inference is part of this audit. First-drop labels are not read.
