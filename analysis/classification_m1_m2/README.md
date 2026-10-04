# M1/M2 classification development audit (E39S / D59S)

Read [findings.md](findings.md) or [the standalone report](report.html). Neither fixed
primary candidate improves the current classifier, so there is no submission promotion.
The [protocol](protocol.md) was fixed before this experiment's scores, after first-drop
feedback. Parent-image IDs remain unknown: crop CV is a development diagnostic.

- `appearance.py`: fixed24 mask-independent image-signature API, per-site pooling.
- `extract.py`: aligned raw detectors, exact tile coverage, source/asset hashes and
  fixed-trim gain/offset, gamma and quantisation checks. Existing outputs are retained;
  do not rerun extraction into the recorded directory and overwrite its receipts.
- `models.py`: complete dependency gate, L1/L2/shrinkage-LDA, training-only preprocessing
  and C selection, explicit class mapping, exact linear margin contributions.
- `run.py`: six fixed candidates and shared crop holdouts, then already-revealed failure
  diagnostics. No input to the live QC/submission runtime; fresh evaluation directories.
- `build_report.py`: standalone HTML, scientific charts and actual response maps.
- `update_register.py`: experimental descriptor inventory entries; no trust/QC promotion.

The recorded run used:

```bash
/opt/anaconda3/bin/python3 -m analysis.classification_m1_m2.extract --workers 2
/opt/anaconda3/bin/python3 -m analysis.classification_m1_m2.run
MPLCONFIGDIR=/private/tmp/polaron-e39-mpl /opt/anaconda3/bin/python3 -m analysis.classification_m1_m2.build_report
```

To replay model fitting against the existing hashed extraction, use a **new** output:

```bash
/opt/anaconda3/bin/python3 -m analysis.classification_m1_m2.run --out analysis/classification_m1_m2/output/replay_01
```

The report generator displays the original `output/evaluation/` only. Original
submissions/QC are protected, not rewritten. Full31 fits and31 outer models per
candidate are saved in joblib; these are developmental models, not a newly declared
submission version. No SME annotations, pseudo-labels or real supplier-lot truth.

`output/evaluation/known_stage_receipt.json` precedes reading first-drop features/truth;
the full receipt distinguishes fitting from development failure diagnosis. Tables include
uncalibrated balanced-target model scores, all class recalls, missingness and attributions.
`output/appearance_verification.json` checks all102 raw TIFF hashes and extracted values.
[E40S](../source_crop_audit/findings.md) retains unknown parent dependence after a bounded
overlap search; absence of overlapping pixels cannot establish independent sources.

**As-run definition naming note:** `appearance_definition.json` uses the legacy key
`independent_unit` for “one supplied crop/site vector”. Read this as the **observation
unit**, not established statistical independence. Parent/crop independence is unknown;
no source-independent n is inferred. The numerical definition and hashed receipt are
preserved rather than rewritten after scoring.
