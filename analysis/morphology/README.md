# E18: morphology differentiation

This independent analysis expands the morphology inspection requested by the
challenge provider. It reads existing caches and BSE images, and writes its own
outputs. It does not change `polaron_qc`, its primary KPI list, or decision rules.

Run from the repository root with the documented Anaconda environment:

```sh
/opt/anaconda3/bin/python3 -m pytest analysis/morphology/test_geometry.py -q
/opt/anaconda3/bin/python3 -m analysis.morphology.run_analysis
/opt/anaconda3/bin/python3 -m analysis.morphology.size_floor_sensitivity
/opt/anaconda3/bin/python3 -m analysis.morphology.make_report
/opt/anaconda3/bin/python3 -m analysis.morphology.verify_outputs
```

Open `output/report.html`. Tables, figures, component coordinates and the input
manifest are alongside it. The manifest uses full SHA-256 hashes of the BSE
inputs and source tables/code. Inputs are checked for changes during the run.

The morphology panel was specified before viewing this run's comparisons:
void elongation and size-distribution width; bright-particle circularity,
solidity, size-distribution width and aspect ratio; axial orientation of voids
and bright particles; bright-particle centroid spacing; image-depth gradient
magnitudes; and BSE texture scale/spectrum slope. The existing material panel
is shown separately, with equal total weights per descriptor family.

Quality-matched comparisons exclude low-contrast and grey-pore sites. Known
cracked reference sites remain in that view, with an additional ordinary-only
reference sensitivity view. Full usable results preserve flagged pore fallback
measurements. All pairwise batch comparisons are reported. Per-site summaries,
not particles or patches, enter tests, bootstrap intervals and sign stability.

Every mask-derived morphology descriptor is recomputed at its histogram-anchored
thresholds and ±5 grey levels. The perturbation envelope is distinct from sampling
uncertainty. Circular orientation is axial; round and edge-clipped components
are excluded from orientation claims. Centroid spacing is normalised against
uniform random point patterns with matching counts, windows and boundary
censoring. This is a geometric comparator, not a validated agglomeration index:
real particles have finite size and cannot overlap.

After inspecting the overlays, `size_floor_sensitivity` audits spacing at
component areas of 50, 500 and 2,000 pixels. Small thresholded fragments near
particle rims may dominate a centroid count. This post-screen measurement audit
does not choose a preferred floor or add new significance tests.

Acquisition checks include pooled and within-batch correlations, quality-matched
and ordinary-site views, and a fixed acquisition-only ridge regression with
preprocessing refitted for each held-out site. These describe sensitivity and
predictability, not causes. BSE texture remains contrast sensitive.

This is an exploratory screen developed on the known batches. Its multiplicity
context does not promote new descriptors into the frozen primary family.
Specimen independence, chemistry, collector orientation and practical release
tolerances remain unresolved. No unseen-batch accuracy is claimed.

## Live metric inventory and measurement review

`metric_register.json` now tracks every morphology metric and candidate method,
including implementation, evidence, expert review, QC role and history. Generated
views are `docs/morphology_metrics.md` and `output/metric_atlas.html`; the latter
illustrates each metric with real SEM evidence. Run
`python -m analysis.morphology.metric_register` and
`python -m analysis.morphology.build_metric_atlas` after updating the register.

The E24 segmentation benchmark, local void-width and hysteresis candidates live
in `benchmark/`. See `benchmark/README.md` for annotation, evaluation and rebuild
commands. Expert validation remains pending; candidate predictions are not labels.


E25 extends the live inventory/atlas with battery geometry measurements and observability. Unlike the original standalone E18 analysis, these eight explicitly experimental secondary scalars are also extracted by `polaron_qc` and displayed in its report. Frozen primary/ML/decision inputs remain unchanged. See `../battery/output/report.html` and `docs/problem_and_findings.md` §12 for limits and reproduction.
