# E31L.1 morphology OOD audit

This isolated development audit implements task **L audit**, supplementing Claude’s L1–L4 categoriser/Inlens/signature work. It evaluates the D49P three-candidate morphology panel without changing production features, classifier inputs or verdicts. The registered protocol and receipt precede new L results; known batches were already development data.

From repository root:

```sh
/opt/anaconda3/bin/python3 -m pytest analysis/morphology/l_audit/test_methods.py -q
/opt/anaconda3/bin/python3 -u -m analysis.morphology.l_audit.run_audit
/opt/anaconda3/bin/python3 -m analysis.morphology.l_audit.verify_audit
```

Read `report.html`, `recommendations.csv`, and `handoff.md`. The report embeds actual-image overlays and scientific plots, so it can be viewed offline. Tables and plots are separate exportable artifacts. Original TIFFs and historical caches remain read-only.

`measurement_receipt.json` allows only identical source/table/raw/code hash reuse of the fresh measurement checkpoint. `simulation_receipt.json` binds the fixed IID/power trial outputs to protocol and execution code. Changes require preserving this run and registering a new development version; do not silently retune existing outputs. Source snapshots record the actual execution definitions. `manifest.json` binds measured tables; `delivery_receipt.json` binds final presentation/verification artifacts.

One execution-only repair changed `radio.transform` to `radio['transform']` after the original run stopped during robustness assembly. The old orchestration source and original measurement/simulation receipts are retained; `execution_repairs.json` records the exact old/new hashes and replacement. Updated checkpoint receipts disclose that lineage. No measurement, statistical definition, protocol threshold or saved simulation value changed.

The isolated verifier checks the strict `87ecee3` protected-file base. On shared main, integration uses its actual source snapshot and explicitly preserves/excludes Claude’s concurrent notebook/report regeneration from the delivery guard. `integration_receipt.json` records those differences without asserting the whole shared repository stayed unchanged. Scientific dependencies, tests and original caches remain checked; `verification_isolated.json` preserves the earlier isolated pass. This is a delivery-scope correction, not a change to the audit’s statistics.

Clarifications fixed in implementation before candidate results: raw axis interchange holds the already-trimmed ROI/thresholds fixed to avoid changing the physical pixels; added-detection gates require both registered candidate shift directions rather than choosing the better sign; missing radiometric paired coverage cannot pass the invariance gate. These are conservative implementations of the registered gates, not after-result threshold changes.

Statistics use sites, not components/pixels. Every permutation refits reference median/MAD, with SD/unit fallbacks disclosed. Exact p-values enumerate all allocations; Monte Carlo uses the plus-one correction. Equal panel weighting is different from the frozen production consequence weights. The combined rule controls two omnibus opportunities via Holm; explanatory univariate tests do not add triggers. The full measurable supplier reference retains its known long voids. Shared-site comparison keeps changes in measurement coverage from masquerading as feature benefit.

IID and shifted summary-space controls check statistical engineering behaviour, not manufacturing sensitivity or external supplier false-alert rates. Exact internal reference splits overlap and are calibrated by construction. Nearest-reference site ranks are descriptive, not calibrated/conformal membership probabilities. Mask accuracy, specimen independence, acquisition/section repeatability and genuine unseen generalisation remain unresolved. Any production integration, release freeze and cold-drop rehearsal are separate actions.
