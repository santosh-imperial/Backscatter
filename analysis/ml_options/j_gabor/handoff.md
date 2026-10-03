# Task J integration handoff — E28J / D43J

Owner: Codex. J is integrated additively into the main checkout on top of
`a05860f` (G baseline `a6292be`, E/F integrated, H fixes merged).
These integration edits remain uncommitted. Original pilot measurements were
made in the isolated `gabor-texture-pilot` worktree based on `76cbe53`.
E28J/D43J suffixes preserve the separate H E28/D43 history.

Start with [the measured qualification](review.md), then [the visual report](report.html).
Three descriptors are computed, expert-unreviewed and secondary exploratory:
`gabor_coarse_energy_share`, `gabor_axial_strength`, and
`gabor_horizontal_wavevector_balance`. Retain the fixed bank as an image
appearance reference; all material/QC use is deferred. No incremental material
information, trained CNN, defect accuracy or battery-performance benefit is
demonstrated.

All nine nominal quality-matched intervals include zero. Changing the quality
population also reduces site n, so this does not prove an acquisition artifact
or equivalence. Negative LOO control R² does not establish novelty or invariance.
Valid interiors cover 2.53–3.63% of each frame; spatial representativeness needs
independent review or repeat sections. Wavevectors are normal to stripes, not
graphite plate axes.

## Integration contract

- The full J evidence directory, original measured source snapshots, protocol,
  raw/source hashes, maps, tables and original verification receipt are retained.
  Shared extractor and reused metadata hashes still match; no extraction or
  parameter tuning was repeated.
- Only three metric entries and one method were added to the current register:
  81 metrics and 11 methods. Existing entries, histories and E/F candidate records
  are preserved. The J candidate and explicit sampled-window atlas adapter were
  added; the graph adapter remains intact.
- Exactly 1710 E28J registry rows were appended after the current main history.
  Seven administrative delivery counts are recorded separately under E29J.
  E28J/D43J records and measured qualifications were added without replacing
  E/F/G/H logs. Duplicate stale E/F task rows were reconciled.
- A16 includes both graph and Gabor evidence without confirming expert review.
  Generated inventory, atlas and assumption reports were rebuilt.
- Notebook 02's generator gains an exploratory-evidence Markdown section linking
  E/F/J, the atlas and annotation pack. All code cells and frozen settings remain
  unchanged; the notebook and known-batch reports are re-executed on the H-fixed
  baseline. These are known-data delivery checks, not unseen-material validation.

## Validation evidence

The [integration verifier receipt](integration_verification.json) checks all
155 summaries against 7440 saved window energies, 135 pairwise rows and two
prespecified raw-window replays. It also checks source hashes, five report
images, unchanged production code/historical caches, unchanged notebook code,
preserved register/candidate records and append-only registry history against
`a05860f`. The original [pilot receipt](verification.json) retains the `76cbe53`
scope; its verifier is archived in `provenance/verify_audit_pilot.py`.

The combined atlas verifies numeric/image/source bindings and filter controls.
Twenty-four Gabor/graph/morphology geometry tests pass. See the [notebook
execution receipt](notebook_integration_execution.json) and integration decision
D46J in `docs/decision_log.md` for
the final delivery checks. Independent expert validity, specimen independence,
spatial repeatability and unseen generalisation remain unresolved.

Reproducible verification commands are in [README](README.md).
Review applied: C01–C07/C09/C11/C13/C14/C16/C17/C21–C23/C28–C31;
C27/C31 retain H's rehearsal receipt because this integration changes no
production code or operator steps.
