# Task K / E30K — fixed morphology audit

[Visual report](report.html) · [Measured qualification](review.md) · [Complete findings](findings.md) · [Frozen definitions](protocol.json) · [Original registration](protocol_registration.json) · [Verification](verification.json) · [Handoff](handoff.md)

Seven descriptors remain secondary and expert-unreviewed. All material/QC use is deferred. Shapes reuse verified E26G component geometry; finite-frame phase pair counts are freshly measured from raw BSE. Pixels, pairs and components are coverage; sites remain n and specimen independence is unknown.

Verify with `/opt/anaconda3/bin/python3`:

```sh
python -m pytest analysis/morphology/k_pilot/test_descriptors.py -q
python -m analysis.morphology.k_pilot.verify_audit
python -m analysis.morphology.metric_register
python -m analysis.morphology.build_metric_atlas --verify-only
```

For reproduction use a separate output checkout so the frozen audit is preserved, then run `python -m analysis.morphology.k_pilot.run_audit` followed once by `python -m analysis.morphology.k_pilot.qualify_report`. The first execution stopped before table/manifest saving on missing control imports; only these imports were repaired before the identical fixed protocol reran. No settings or results were selected/tuned.

`qualify_report.py` reads saved comparison/control/sensitivity values to explain the two nominal findings; its separate presentation manifest and source snapshot preserve the unchanged measured protocol.
