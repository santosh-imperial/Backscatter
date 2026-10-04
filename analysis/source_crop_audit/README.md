# Conservative source-crop audit

Read [findings.md](findings.md) for the valid E40S result and its limits.
The organiser says the dataset consists of crops of approximately 15 original
electrode images; their source-to-crop mapping has not been supplied.

- [Fixed detection protocol](protocol.md)
- [All 561 pair scores](output/all_pair_scores.csv)
- [Valid source/input receipt](output/pre_score_receipt.json)
- [Integrity and summary checks](output/verification.json)
- [Supplementary real-image control protocol](control_real_protocol.md)
- [Real-image control receipt](output/real_image_controls/receipt.json)
- [Constructed positive-control BSE panel](output/real_image_controls/synthetic_affine_quantisation_positive_BSE.png)

No overlap component was recovered; this cannot establish independent sites.
Source-held-out CV was not run and classifier groups remain unchanged.

The audit operates only on pixels and provenance files. It never reads first-drop
feedback labels, classifier predictions or model probabilities. Detector aliases
are normalised, and dimensions are used only for legal intersections. Frame
dimensions cannot become a source-group label or a classification feature.

Run with the project interpreter from the workspace root, into a fresh directory:

```sh
/opt/anaconda3/bin/python3 analysis/source_crop_audit/run.py --out analysis/source_crop_audit/output_recheck
/opt/anaconda3/bin/python3 -m pytest tests/test_source_crop_audit.py -q
```

The delivered supplementary control is bound to `output/file_manifest.csv` and
its code preserves prior receipts by refusing to overwrite its existing output.
Do not remove old receipts to rerun. `failed_pre_score_serialisation/` and
`invalid_control_gate_v1/` document the invalid preliminary attempts and must not
be interpreted as valid findings.
