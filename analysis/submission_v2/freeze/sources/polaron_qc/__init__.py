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
UNSEEN_COLOR = "#6f42c1"   # any batch folder not in BATCH_COLORS (the unseen batch); never index BATCH_COLORS[b] directly — use .get(b, UNSEEN_COLOR) (E23)
LOW_CONTRAST_SITES = ["4ih2ggld", "5n1q8atc"]
GREY_PORE_SITES = ["71vgq3fw", "kbdh4tri", "tuy3zymq", "x7u69zsw"]
CRACKED_SITES = ["hzumfsms", "0grcilhi", "ufdvpb81"]
PRIMARY_KPIS = ["crack_frac", "pore_max_d", "pore_frac", "bright_frac", "bright_d50"]

# Trust level per KPI, from docs/problem_and_findings.md §4 (KPI catalogue). Levels:
#   "high"       validated measurement, used in verdicts when usable
#   "medium"     usable, with stated caveats
#   "flag"       acquisition / preparation diagnostic — never a material KPI
#   "confounded" candidate material KPI shown to track acquisition — evidence only
#   "diagnostic" segmentation / threshold bookkeeping
#   "exploratory" measured secondary geometry; expert validation pending, excluded from verdicts/ML
KPI_TRUST = {
    **{k: "high" for k in ["pore_frac", "pore_d50", "pore_elong", "pore_max_d", "pore_d90", "pore_count_per_Mpx",
                           "crack_frac", "crack_count_per_Mpx",
                           "bright_frac", "bright_count_per_Mpx", "bright_d10", "bright_d50", "bright_d90", "bright_circ",
                           "bright_solidity", "bright_max_d", "etd_crack_density_particles"]},
    **{k: "medium" for k in ["fft_slope", "corr_len_px", "graphite_frac"] + [f"profile_pore_{i}" for i in range(10)] + [f"profile_bright_{i}" for i in range(10)]},
    **{k: "flag" for k in ["etd_boundary_sharpness", "etd_curtain_frac", "etd_curtain_anisotropy", "etd_crack_density_graphite",
                           "etd_ridge_dom_angle", "etd_grad_energy", "inlens_grad_energy", "bright_low_contrast", "grey_pore",
                           "pore_mode_resolved", "bright_mode_resolved"]},
    **{k: "confounded" for k in ["inlens_particle_texture", "inlens_particle_texture_p90", "inlens_speckled_particle_frac"]},
    **{k: "diagnostic" for k in ["th_lo", "th_hi", "graphite_mode", "sigma_l", "sigma_r", "bright_mode", "bright_sep", "ridge_p97",
                                 "inlens_particles_measured", "H", "W", "n_patches"]},
}
# KPIs eligible for any "material KPI" list (classifier runs, drift drivers): trust high or medium, never flag/confounded/diagnostic
MATERIAL_KPIS = ["pore_frac", "pore_d50", "pore_elong", "pore_max_d", "crack_frac", "crack_count_per_Mpx",
                 "bright_frac", "bright_count_per_Mpx", "bright_d50", "bright_d90", "bright_circ",
                 "corr_len_px", "fft_slope", "etd_crack_density_particles"]
assert all(KPI_TRUST[k] in ("high", "medium") for k in MATERIAL_KPIS)
assert all(k in MATERIAL_KPIS for k in PRIMARY_KPIS)

# These are extracted and reported in a separate battery-geometry block. Adding
# a measurement here cannot extend the frozen decision/classifier lists.
from .secondary import KPI_NAMES as BATTERY_SECONDARY_KPIS
KPI_TRUST.update({k: "exploratory" for k in BATTERY_SECONDARY_KPIS})
KPI_TRUST["bright_void_boundary_frac"] = "diagnostic"
assert set(BATTERY_SECONDARY_KPIS).isdisjoint(PRIMARY_KPIS + MATERIAL_KPIS)
