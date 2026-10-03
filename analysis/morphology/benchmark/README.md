# E24: measurement benchmark and method candidates

Open [review.html](review.html) to annotate eleven real BSE crops. Six sites are
for development; five different sites are held out for measurement checks.
All come from previously explored batches: this is not unseen-batch validation.
Specimen independence is unresolved. Crops are purposive examples centred on
large components, never additional independent material samples.

## Expert review

Draw independent phase polygons on the original **lossless PNG**. Cyan = void,
gold = bright phase, green = residual solid, magenta = uncertain/ignore. Unpainted
pixels are ignored by default. Only declare solid background after inspecting
the whole crop. Later polygons overwrite earlier ones. Algorithm comparison
views are predictions, never ground truth. Mark poor contrast unmeasurable if
necessary; do not force a phase assignment. Add your name and notes, then download
JSON. Local browser state is keyed to the manifest but is not a durable record.

Keep development and held-out-site results separate. Do not tune on held-out labels.

```sh
/opt/anaconda3/bin/python3 -m analysis.morphology.build_benchmark --annotations /absolute/path/polaron_benchmark_annotations.json
```

Outputs: `expert_evaluation.csv` and `expert_evaluation_summary.json`. Partial
manual masks support pixel IoU/Dice and phase-fraction errors on labelled pixels
only. Fully specified masks also support bright D50 and ROI-local width errors,
excluding ROI-clipped components on both sides. Empty classes are undefined;
unmeasurable reviews are abstentions. Evaluating the supplied unreviewed template
produces zero scored rows, never a perfect accuracy score.

## Candidate definitions

- **Local width:** twice distance to solid along a deterministic medial axis of
  ≥30 px² voids, excluding components connected to the full-image edge. Samples
  approximately weight centreline length, not area. No branch pruning; junctions
  and digital boundaries matter. This is not a 3-D pore throat or maximal-disk
  thickness map. Large visible cavities may be excluded. Width PNGs show the
  full-site calculation in a crop; manual evaluation instead recomputes both ROI
  masks with ROI-edge exclusions. These scopes must not be confused.
- **Bright hysteresis:** seeds above the site threshold +5 grow within threshold
  −5 with four-neighbour connectivity, followed by the existing cross opening.
  It can grow rim artefacts and cannot restore missing phase contrast. Algorithm
  agreement is not accuracy; bright low-contrast exclusions still apply.
- **Connectivity:** four- versus eight-connected objects in identical nominal
  masks. Existing primary definitions remain eight-connected.

`local_width_sites.csv` contains nominal and ±5-level measurements;
`local_width_sensitivity.csv` contains method envelopes. `hysteresis_sites.csv`
and `connectivity_sites.csv` quantify method changes. `measurement_diagnostics.csv`
contains descriptive site summaries; these are not confidence intervals or
defect probabilities. `status.json` records the expert-review gap. Lengths are
pixels; calibration is unverified.

## Rebuild and verify

```sh
/opt/anaconda3/bin/python3 -m pytest analysis/morphology/test_benchmark.py analysis/morphology/test_geometry.py -q
/opt/anaconda3/bin/python3 -m analysis.morphology.build_benchmark
/opt/anaconda3/bin/python3 -m analysis.morphology.optimize_review
/opt/anaconda3/bin/python3 -m analysis.morphology.summarize_benchmark
/opt/anaconda3/bin/python3 -m analysis.morphology.metric_register --sync-benchmark
/opt/anaconda3/bin/python3 -m analysis.morphology.build_metric_atlas
/opt/anaconda3/bin/python3 -m analysis.morphology.verify_benchmark
/opt/anaconda3/bin/python3 -m analysis.morphology.verify_review_ui
```

The optimizer compresses only comparison/context previews as JPEG, keeping HTML
below 20 MB; annotation references remain lossless PNG. Raw TIFFs, numerical
arrays and masks are unchanged. UI event/state tests use a synthetic DOM in Node;
they do not verify native browser layout. Inspect image assets for overlay QA.

Maintain `../metric_register.json`, then regenerate the table in
`../../../docs/morphology_metrics.md` and the atlas in `../output/metric_atlas.html`.
Implementation, evidence, expert review and QC role are independent statuses.
No automated operation promotes a new KPI to primary.

Method references: [distance transform and local width](https://bioimagebook.github.io/chapters/2-processing/6-transforms/transforms.html#the-distance-transform)
and [hysteresis / reconstruction](https://bioimagebook.github.io/chapters/2-processing/5-morph/morph.html#hysteresis-thresholding).
The SEM-specific conventions and validation requirements are our adaptations.
The current run retains extraction-source snapshots under `provenance/`: its
experiment ID was corrected to E24 after a parallel workstream used E23. Only
labels changed; verification checks that the measured and rebuild source differ
solely in that experiment label. The original annotation manifest identity stays
stable, and all numerical arrays and parameters are unchanged.
