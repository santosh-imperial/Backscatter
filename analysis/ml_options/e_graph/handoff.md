# Task E handoff

Owner: Codex. Isolated worktree: `particle-graph-audit`, based on `f42c663`.
Experiment ID **E26G** and decision **D41G** use a graph suffix to avoid a
parallel H/F experiment-number collision. They are local scientific log IDs.

The completed audit is `analysis/ml_options/e_graph/report.html`; definitions
and frozen settings are in `protocol.json`; full-site/variant/coordinate/control
tables and exact measured sources sit alongside it. Raw TIFFs are accessed via
a locally excluded Dataset symlink to the shared checkout, read only by this workflow.
No TIFFs or historical caches were copied into Git or modified.

Reproduce from this worktree with the Anaconda runtime:

```sh
/opt/anaconda3/bin/python3 -m pytest analysis/ml_options/e_graph/test_graph.py analysis/morphology/test_geometry.py -q
/opt/anaconda3/bin/python3 -m analysis.ml_options.e_graph.run_audit
/opt/anaconda3/bin/python3 -m analysis.morphology.metric_register
/opt/anaconda3/bin/python3 -m analysis.morphology.build_metric_atlas
```

The run hashes all used BSE images, reused metadata, frozen protocol and measured
sources before/after. The runner reconstructs masks from raw BSE and checks
nominal thresholds/brightness-quality flags against reused metadata. Images
illustrate entire-frame graphs cropped for display, not crop-built graphs.

Coverage conventions: the component table starts at the prespecified 50 px²
candidate floor. `graph_components_available` counts those candidates;
`graph_components_below_floor` counts further exclusions at 100 px², rather
than every smaller foreground fragment. No boundary-correction advantage is
claimed; finite-frame neighbour bias remains after clipped-node exclusion.

For G integration, merge the graph directory, atlas adapter and additive graph
register entries. Reconcile candidate/register/findings/log/ownership changes
with H/F/E18 branches; do not replace their newer state with this worktree's
snapshot. The existing QC modules, tests, notebook generators and original
analysis caches are unchanged. Regenerate the atlas and metric Markdown after
reconciling the register. This does not integrate a new primary/classifier or
decision feature. H/F/G remain their existing owner's work.

The fixed graph is retained for exploratory image review. All material/QC use
remains deferred pending expert instances, repeatability and process evidence.
No unseen test, defect accuracy or battery-performance advantage was measured.

Qualification summary: all nominal intervals include zero in the bright-usable
and quality-matched views; 10/12 comparison directions change across settings.
The edge-scale descriptor is partly redundant with node/loading controls.
See the sensitivity table and findings §15 for the scoped interpretation.

Delivery state: task E changes are uncommitted in this isolated worktree:
`/Users/santoshkumarsaravanan/.codex/worktrees/particle-graph-audit/Polaron`.
The main checkout only has the E ownership/handoff row updated. Include the new
graph directory and atlas asset when reviewing/copying the diff; ordinary
`git diff` omits untracked files. No branch merge or push has been performed.

Verification completed: 16 graph/geometry tests passed; all 186 saved graphs
reproduce their four descriptors and all 216 pairwise rows match saved site
values. The 78-entry atlas verifies decoded assets, full-site scalar bindings,
source hashes and filter controls. Source/raw hashes remain current, registry
history bytes are preserved, and primary/classifier/decision/notebook/cache
files are unchanged. See `verification.json`. Assumption HTML was regenerated
without marking expert interpretation as confirmed.

Pre-presentation review: C01–C07/C09/C11/C13/C14/C16/C17/C21–C23/C28–C30.
Coverage and numerical validation are distinct from expert mask accuracy.
