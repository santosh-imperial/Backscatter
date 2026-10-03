"""polaron_qc — SEM batch-drift QC for electrode cross-sections.

Modules (one owner each during the build; see docs/qc_plan.md):
  features  — batch folder -> per-site / per-image / per-particle / per-patch tables (+ cache)
  stats     — permutation tests, robust shifts, energy distance, MDC, jackknife, reference-split diagnostics
  ml        — grouped-CV classifier two-sample test; exploratory patch-embedding novelty
  physics   — stereology, section tortuosity index, void geometry, consequence weights, qualitative statements
  decision  — verdicts (plan §2.7)            [built after stats review]
  report    — per-batch HTML report           [built after decision review]
All lengths in pixels; NM_PER_PX is nominal (TIFF tag, unverified) and only used for "≈ x µm nominal" strings.
"""
NM_PER_PX = 25.0
BATCH_COLORS = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a", "Batch_3 (grey-pore group)": "#eda100"}
LOW_CONTRAST_SITES = ["4ih2ggld", "5n1q8atc"]
GREY_PORE_SITES = ["71vgq3fw", "kbdh4tri", "tuy3zymq", "x7u69zsw"]
CRACKED_SITES = ["hzumfsms", "0grcilhi", "ufdvpb81"]
PRIMARY_KPIS = ["crack_frac", "pore_max_d", "pore_frac", "bright_frac", "bright_d50"]
