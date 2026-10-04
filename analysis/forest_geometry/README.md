# Fixed forest × geometry experiment

E41S/D62S/D63S: no gain under the fixed model and eight-descriptor comparison;
current submission and QC remain unchanged. [Findings](findings.md) and the
[HTML report with actual-image geometry](report.html) preserve all seven candidates.

The following is the **as-run sequence**, from repository root with
`/opt/anaconda3/bin/python3`. Do not rerun extraction against the saved output:
extraction replaces its receipts and would invalidate this trial. Evaluation needs
a fresh output directory and the explicit staged known receipt before drop extraction:

```sh
python3 -m analysis.forest_geometry.prepare --cohort known
python3 -m analysis.forest_geometry.run --stage known
python3 -m analysis.forest_geometry.prepare --cohort drop --known-stage-receipt analysis/forest_geometry/output/evaluation/known_stage_receipt.json
python3 -m analysis.forest_geometry.run --stage diagnostics
python3 -m analysis.forest_geometry.delivery_report
python3 -m analysis.forest_geometry.verify_delivery
python3 -m pytest tests -q
```

The existing trial cannot be overwritten; use `--out` on the runner for another
evaluation directory. Matching staged extraction uses the exact path to its saved
receipt. The reporting adapter reads the declared canonical evaluation directory;
alternate trial reporting requires a separately declared delivery change. Historical
caches/raw bindings are checked, not silently replaced. Revealed drop labels never
enter fitting, and source groups/SME masks are not inferred. All scores remain
uncalibrated and source dependence is unresolved. No model-selection/promotion follows
the revealed development results.
