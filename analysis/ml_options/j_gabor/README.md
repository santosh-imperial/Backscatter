# Task J / E28J — fixed Gabor appearance audit

[Visual report](report.html) · [Measured qualification](review.md) · [Frozen protocol](protocol.json) · [Integration handoff](handoff.md) · [Pilot receipt](verification.json) · [Integration receipt](integration_verification.json)

Three computed appearance summaries remain secondary and expert-unreviewed;
all material/QC use is deferred. Start with `review.md` before reading the
unfiltered nominal contrast. `sites.csv` summarises four fixed windows per
site, not the entire frame; `window_energies.csv` has raw/trimmed coordinates.

Verify the integrated saved evidence with `/opt/anaconda3/bin/python3`:

```sh
python -m pytest analysis/ml_options/j_gabor/test_texture.py -q
python -m analysis.ml_options.j_gabor.verify_audit --integration-base a05860f
python -m analysis.morphology.metric_register
python -m analysis.morphology.build_metric_atlas
python docs/_build_assumption_register.py
```

The single measurement run used `python -m analysis.ml_options.j_gabor.run_audit`,
followed by `python -m analysis.ml_options.j_gabor.finalize_report`. For a fresh
reproduction, use a separate checkout/output copy to preserve this frozen audit.
No extraction was rerun or tuned during integration. The original pilot receipt
and verifier source are retained in `verification.json` and
`provenance/verify_audit_pilot.py`; the current verifier checks the additive
integration against the committed H-fixed baseline.

The fixed kernel formulation is grounded in the [scikit-image Gabor API](https://scikit-image.org/docs/0.23.x/api/skimage.filters.html#skimage.filters.gabor_kernel);
our finite square support, DC removal and unit-L2 normalisation are explicit
protocol choices. These sources do not validate the battery interpretation.
