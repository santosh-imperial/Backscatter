# E38S: phase-boundary review and missing grouping evidence

Start with [development.html](output/development.html). It contains E24's six exact
lossless raw BSE crops, with neutral aliases and no algorithm/batch hints. Inspect
each image, annotate void, bright appearance, residual solid and uncertain regions,
and record measurability, reviewer identity, independent drawing and prior exposure.
Draw boundary neighbourhoods and residual solid as well as obvious phase interiors.
Download JSON; browser-local state alone is not a durable review record.

Keep [held_out_site.html](output/held_out_site.html) with an independent reviewer
until development method choices are fixed. It contains the five existing E24
measurement-holdout sites. Separate files and the release guard enforce a workflow,
not secure isolation on a shared disk. These sites were explored previously.
Previous exposure is retained in the review export, so neutral IDs do not establish
retrospective blinding or specimen independence.

[feedback_development.html](output/feedback_development.html) is a separate three-site
review of the already-labelled organiser drop, using E37S's fixed central 512 px
crops. Treat its results as development/failure evidence. Do not merge this packet
into a held-out accuracy claim or tune the declared v2 submission in place.

The coordinator-only [manifest](output/coordinator_manifest.json) maps aliases back
to exact original crops, raw coordinates, saved methods and file hashes. Send a
reviewer the standalone HTML rather than this folder if you want to hide the mapping.

## Import a downloaded review

Use a fresh output directory for every review. The adapter validates identity,
declarations, polygon bounds and source/prediction hashes, then calls the existing
E24 evaluator without modifying its historical outputs.

```sh
/opt/anaconda3/bin/python3 -m analysis.boundary_review.prepare \
  --annotations /absolute/path/polaron_development_boundaries.json \
  --out /absolute/path/new_development_evaluation
```

The first-drop packet uses the same command with its own exported JSON and a
different output directory. After development decisions are fixed, the held-out
packet needs the additional `--release-held-out` flag. Import fails if recorded
method sources changed; a changed method needs a newly registered/frozen evaluation
protocol rather than bypassing provenance checks. Do not change sources to recover
a more attractive held-out score.

Partial masks score pixel IoU/Dice and phase fractions only on labelled regions.
Whole-crop masks additionally support ROI-local object/width errors, with the
local-width clipping exclusions. The E38S adapter fixes the historical bright-D50
comparison, which included ROI-edge-cut objects despite its notes: it compares
area-weighted D50 of >=50 px² eight-connected, non-ROI-clipped bright objects on
both masks and reports retained/excluded counts. Empty eligible sets are unavailable.
This correction is fixed before annotations and does not rewrite E24 results.
An unmeasurable review is an abstention. Unreviewed
exports score nothing. These are image-appearance references, not chemical ground
truth, full-site KPI accuracy or manufacturing labels. The first-drop classifier
inputs and QC verdict remain unchanged.

## Grouping evidence

[TIFF headers](output/tiff_headers.json) retain actual tags and descriptions, with
source hashes. [Inventory](output/metadata_inventory.csv) summarises possible
grouping/settings keys. All 102 TIFFs over 34 sites contain no such explicit keys;
the descriptions contain export shape, and the software field is `tifffile.py`.
No specimen, preparation-session or imaging-session identifiers were recovered.
Santosh confirms no mapping is available yet. The absence of identifiers does not
establish that sites are independent.

[group_mapping_template.csv](output/group_mapping_template.csv) has one row per
site and blank IDs. Obtain IDs and an evidence source from the organisers. Never
fill them using frame height, acquisition clusters, file timestamps or folder names.
The supplied batch/site identifiers identify observations, not specimens/sessions.
Grouped validation remains unavailable. [Organiser questions](organiser_questions.md)
are a draft for Santosh; nothing has been sent.

## Preparation and checks

```sh
/opt/anaconda3/bin/python3 -m analysis.boundary_review.prepare
/opt/anaconda3/bin/python3 -m pytest tests/test_boundary_review.py -q
/opt/anaconda3/bin/python3 -m analysis.boundary_review.verify_ui
/opt/anaconda3/bin/python3 -m analysis.boundary_review.verify_artifacts
```

Preparation refuses an existing output directory so it cannot reset annotation
state. Source/input/output hashes and protected-artifact checks are in
[receipt.json](output/receipt.json). The current pages contain zero expert labels;
preparation and synthetic tests are not segmentation accuracy evidence.
The final suite passes 191 tests; all 222 protected original artifacts and all
14 exact raw-coordinate crops verify. [validation.json](validation.json) records
the commands and [verification.json](verification.json) the artifact checks.
Native-browser layout is unverified because browser automation blocks local-file
URLs; synthetic control tests and embedded-image checks are separate evidence.
