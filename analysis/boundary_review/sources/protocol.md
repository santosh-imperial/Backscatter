# E38S / D57S — independent boundary-review handoff and grouping-metadata audit

Fixed after first-drop feedback and E37S, before any new annotations. This prepares
measurement evidence; it trains no classifier and promotes no feature or mask policy.

1. Reuse E24's exact six development and five held-out-site BSE crops, split and
   nominal/hysteresis predictions. Do not regenerate, recrop or overwrite E24.
   Reviewer pages contain lossless raw values, neutral aliases and drawing tools;
   they contain no batch/site names, quality flags, thresholds or algorithm overlays.
   The coordinator mapping remains a separate file. This hides available hints;
   it cannot undo a reviewer's previous exposure or guarantee blinding.
2. Prepare a separate three-site first-drop development packet using E37S's fixed
   central 512 px raw coordinates. Its annotations never become unseen validation.
   Save the nominal and E24-defined hysteresis predictions privately, using saved
   first-drop thresholds on the full image before cropping. No threshold selection.
3. Annotators declare their identity, prior algorithm/batch exposure and whether
   they drew boundaries independently. Unpainted/uncertain pixels are ignored;
   unmeasurable cases remain abstentions. Do not invent reviews. Partial masks
   support labelled-pixel errors; whole-crop masks are needed for object/width error.
   Bright appearance is not pixelwise chemical truth. Crops are purposive and not
   extra independent samples. ROI errors cannot be called full-site KPI accuracy.
4. Adapt packet exports to the existing E24 evaluator in a fresh output directory,
   without changing historical annotations/results. Keep each packet separate.
   Held-out evaluation requires explicit release and unchanged recorded method
   sources. If development prompts changes, register/freeze a new method before
   releasing held-out annotations; this command deliberately refuses changed sources.
   Coordinator separation is procedural, not a security boundary on this shared disk.
   E24's bright D50 evaluator includes ROI-clipped objects despite its notes.
   The E38S adapter corrects that comparison: use area-weighted D50 on >=50 px²
   eight-connected objects touching no ROI edge, for both prediction and manual
   whole-crop masks; retain excluded/retained counts, and return unavailable if
   either has no eligible object. Pixel and local-width definitions stay unchanged.
   This correction is fixed before any annotations; historical E24 outputs remain.
5. Inspect all known and first-drop TIFF headers, including ImageDescription and
   nonstandard tag names, for explicit specimen/preparation/imaging IDs and settings.
   Export actual tag inventory, source hashes, missingness and a blank per-site
   mapping sheet. Never infer groups from frame size, intensity, file times, batch
   names or the E35 acquisition-cluster proxy. Santosh confirms no mapping is
   available yet. Record unknowns; prepare an organiser question draft, do not send it.

Completion: reproducible raw-only pages, export/import/evaluation safety checks,
TIFF metadata inventory and unknowns sheet, durable source/output receipts, and
clear expert-review/grouped-validation gaps. Expert accuracy and cross-session
generalisation remain unavailable until the corresponding evidence arrives.
