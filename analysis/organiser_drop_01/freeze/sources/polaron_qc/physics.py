"""polaron_qc.physics — qualitative, relative physics layer (plan §2.6b).

Turns geometric KPIs measured on 2-D BSE sections into statements a materials engineer can
act on, plus sanity checks on our own measurements. Everything here is **direction only**:
no performance percentages, no absolute conductivity, stress, capacity or diffusion-time numbers.

Every function that produces a statement returns the statement text together with a
``caveats`` list and a ``claims_not_made`` list, so the report prints the scope with the sentence.

Units: all lengths are pixels. ``NM_PER_PX`` in :mod:`polaron_qc` is nominal (TIFF export tag,
unverified) and is only used for "≈ x µm nominal" strings.

Blocks (see docs/qc_plan.md §2.6b; the "what we do NOT claim" column is binding):
  stereology / saltykov_unfold      secondary; Delesse volume fractions, Schwartz–Saltykov unfolding
  section_tortuosity_index          2-D geodesic section index; not a 3-D bound
  void_geometry                     crack-like voids as 2-D geometric observations; not continuity
  additive_size_reading             matched-quantile observed size comparison; no battery-rate forecast
  additive_mechanics_reading        loading/size/ridge descriptors; no inferred binder load or intact share
  CONSEQUENCE_WEIGHTS / _RATIONALE  ordinal weight per KPI (3 high / 2 medium / 1 low)
  sanity_checks / threshold_sensitivity
  qualitative_statement             one-sentence driver reading per KPI from a lookup table

The section at the bottom marked "temporary copies" re-implements image loading, edge-band trimming,
histogram-anchored thresholds and segmentation from notebooks/_build_01_dataset_analysis.py so that
this module can run on raw images now; after integration it is replaced by polaron_qc.features.
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------------------------
# Module constants
# ----------------------------------------------------------------------------------------------

# Densities in g/cm³ — nominal, for recipe comparison only. The chemistry of the bright phase is
# unconfirmed (assumption register A3: "bright higher-Z phase", assumed Si or SiOx).
RHO_GRAPHITE = 2.26  # nominal, for recipe comparison only
RHO_SI = 2.33        # nominal, for recipe comparison only
RHO_SIOX = 2.2       # nominal, for recipe comparison only (SiOx, x ≈ 1, amorphous; literature range ≈ 2.1–2.3)

DOWNSAMPLE = 4                   # factor used for the tortuosity and column-interruption maps
CRACK_LONG_AXIS_PX = 500         # crack-like void cutoff (D09; open item in Part C of the decision log)
SIMILAR_TOL = 0.10               # ±10 % relative change counts as "similar" in direction readings

# Caveats attached to every physics statement (plan §2.6b, last paragraph).
GLOBAL_CAVEATS = [
    "2-D sections; no 3-D information",
    "lengths in px; 25 nm/px is a nominal, unverified export tag",
    "fresh graphite–Si/SiOx material confirmed by Santosh; exact chemistry, recipe and pixelwise phase labels unspecified",
    "uncycled specimens; no electrochemical outcomes, cycling-damage, SEI or plating inference",
]

# ----------------------------------------------------------------------------------------------
# Consequence weights (ordinal: 3 high / 2 medium / 1 low) and rationale, for every KPI column in
# analysis_cache/site_features.csv and analysis_cache/etd_inlens_features.csv, plus derived names.
# ----------------------------------------------------------------------------------------------

KPI_LABELS: Dict[str, str] = {
    "crack_frac": "Crack-like void area fraction",
    "crack_count_per_Mpx": "Crack-like void count per Mpx",
    "pore_max_d": "Largest void (2-D equivalent diameter)",
    "pore_frac": "Macro-pore area fraction",
    "pore_d50": "Macro-pore area-weighted D50",
    "pore_d90": "Macro-pore area-weighted D90",
    "pore_elong": "Macro-pore elongation",
    "pore_count_per_Mpx": "Macro-pore count per Mpx",
    "bright_frac": "Bright-phase (additive) area fraction",
    "bright_count_per_Mpx": "Bright-phase particle count per Mpx",
    "bright_d10": "Bright-phase area-weighted D10",
    "bright_d50": "Bright-phase area-weighted D50",
    "bright_d90": "Bright-phase area-weighted D90",
    "bright_circ": "Bright-phase particle circularity",
    "bright_solidity": "Bright-phase particle solidity",
    "bright_max_d": "Largest bright-phase particle",
    "fft_slope": "Radial power-spectrum slope",
    "corr_len_px": "Texture correlation length",
    "etd_crack_density_particles": "Intra-particle dark-ridge density (ETD, in additive)",
    "graphite_frac": "Graphite-class area fraction",
}
for _k in range(10):
    KPI_LABELS[f"profile_pore_{_k}"] = f"Macro-pore fraction, depth bin {_k}"
    KPI_LABELS[f"profile_bright_{_k}"] = f"Bright-phase fraction, depth bin {_k}"


def _label(kpi: str) -> str:
    return KPI_LABELS.get(kpi, kpi)


CONSEQUENCE_WEIGHTS: Dict[str, int] = {}
CONSEQUENCE_RATIONALE: Dict[str, str] = {}


def _w(kpi: str, weight: int, why: str) -> None:
    CONSEQUENCE_WEIGHTS[kpi] = weight
    CONSEQUENCE_RATIONALE[kpi] = why


# --- high (3): void geometry, additive loading / size, macro-porosity, through-thickness gradients
_w("crack_frac", 3, "area of delamination-like voids; the one clear material anomaly in the data")
_w("crack_count_per_Mpx", 3, "number of crack-like voids per area; localizes delamination-like defects")
_w("pore_max_d", 3, "largest void in the section; a single severe void drives the localized path")
_w("bright_frac", 3, "observed bright-phase area amount; conditional formulation/packing question — primary KPI")
_w("bright_d50", 3, "observed additive section size; local kinetic/mechanical hypothesis requires validation — primary KPI")
_w("bright_count_per_Mpx", 3, "additive number density; agglomeration vs dispersion")
_w("pore_frac", 3, "macro-pore fraction (electrolyte access, calendering density) — primary KPI")
for _k in range(10):
    _w(f"profile_pore_{_k}", 3, "through-thickness macro-pore gradient (binder migration, calendering); orientation to collector unknown")
    _w(f"profile_bright_{_k}", 3, "through-thickness additive gradient (settling, migration); orientation to collector unknown")
# --- medium (2): pore shape / size, texture scale, intra-particle cracking (null on this data)
_w("pore_d50", 2, "equivalent void-section size; transport depends on unresolved network and wetting")
_w("pore_d90", 2, "upper macro-pore size; descriptive")
_w("pore_elong", 2, "2-D void aspect ratio; distinct from directional alignment and transport anisotropy")
_w("pore_count_per_Mpx", 2, "macro-pore number density; descriptive")
_w("bright_circ", 2, "observed additive outline compactness; segmentation-sensitive, not a fracture label")
_w("bright_solidity", 2, "additive shape; descriptive")
_w("bright_d10", 2, "fine tail of the additive size distribution; sensitive to the 50-px area cutoff")
_w("bright_d90", 2, "coarse tail of the additive size distribution; agglomerates")
_w("bright_max_d", 2, "largest additive particle / agglomerate; localized evidence")
_w("corr_len_px", 2, "texture scale, segmentation-free")
_w("fft_slope", 2, "scale-free texture / edge sharpness; partly acquisition-sensitive")
_w("etd_crack_density_particles", 2, "intra-particle cracking of the additive; null on this data")
for _t in ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4"):
    _w(f"crack_p_{_t}", 2, f"intra-particle ridge density at threshold {_t} (grid value; T_STAR is the one used)")
# --- low (1): flags, confounded KPIs, segmentation diagnostics, identifiers
_w("graphite_frac", 1, "derived (1 − pore − bright); carries no independent information")
_w("H", 1, "frame height: session fingerprint as much as thickness proxy")
_w("W", 1, "frame width: acquisition")
_w("th_lo", 1, "segmentation threshold (diagnostic)")
_w("th_hi", 1, "segmentation threshold (diagnostic)")
_w("graphite_mode", 1, "histogram position; raw intensity is never a material KPI")
_w("sigma_l", 1, "histogram width (diagnostic)")
_w("sigma_r", 1, "histogram width (diagnostic)")
_w("bright_mode", 1, "histogram position (diagnostic)")
_w("bright_sep", 1, "bright-phase contrast; acquisition flag")
_w("bright_mode_resolved", 1, "segmentation flag")
_w("bright_low_contrast", 1, "acquisition flag: bright-phase KPIs unreliable when True")
_w("pore_mode_resolved", 1, "segmentation flag (grey-pore group on fallback)")
_w("batch", 1, "identifier")
_w("site", 1, "identifier")
_w("ridge_p97", 1, "per-site ridge-strength percentile used to pick T_STAR; diagnostic")
_w("etd_ridge_dom_angle", 1, "dominant ridge orientation; anisotropy / curtaining diagnostic")
_w("etd_curtain_anisotropy", 1, "preparation flag (ion-mill curtaining)")
_w("etd_boundary_sharpness", 1, "focus / preparation flag")
_w("etd_grad_energy", 1, "channel texture energy; acquisition")
_w("inlens_grad_energy", 1, "Inlens texture energy; charging-dominated")
_w("inlens_particle_texture", 1, "confounded with Inlens brightness (~75 % of variance); candidate only")
_w("inlens_particle_texture_p90", 1, "confounded with Inlens brightness; candidate only")
_w("inlens_speckled_particle_frac", 1, "confounded with Inlens brightness; candidate only")
_w("inlens_particles_measured", 1, "count of particles entering the Inlens texture statistic; diagnostic")
for _t in ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4"):
    _w(f"crack_g_{_t}", 1, "ridge density in graphite: preparation / curtaining flag")
    _w(f"curtain_{_t}", 1, "curtaining fraction: preparation flag")
_w("etd_crack_density_graphite", 1, "ridge density in graphite: preparation flag")
_w("etd_curtain_frac", 1, "curtaining fraction: preparation flag")


def weights_vector(kpi_list: Sequence[str]) -> np.ndarray:
    """Ordinal consequence weights (3/2/1) for a list of KPI names, as a float array.

    Unknown KPI names get weight 1 with a warning, so a new column never silently becomes high-consequence.
    """
    out = []
    for k in kpi_list:
        if k not in CONSEQUENCE_WEIGHTS:
            warnings.warn(f"physics.weights_vector: no consequence weight for '{k}', using 1 (low)")
        out.append(CONSEQUENCE_WEIGHTS.get(k, 1))
    return np.asarray(out, dtype=float)


# Direction-specific consequence text per KPI (weight ≥ 2 only). Qualitative, relative.
CONSEQUENCE_TEXT: Dict[str, Dict[str, str]] = {
    "crack_frac": {"higher": "more delamination-like void area in the section",
                   "lower": "less delamination-like void area in the section"},
    "crack_count_per_Mpx": {"higher": "more long voids per area in the section",
                            "lower": "fewer long voids per area in the section"},
    "pore_max_d": {"higher": "a larger largest void; localized-defect path to be checked",
                   "lower": "a smaller largest void"},
    "pore_frac": {"higher": "more segmented 2-D void area; packing/wetting implications need independent validation",
                  "lower": "less segmented 2-D void area; packing/wetting implications need independent validation"},
    "pore_d50": {"higher": "coarser void sections; connectivity, wetting and ionic conductivity are not established",
                 "lower": "finer void sections; connectivity, wetting and ionic conductivity are not established"},
    "pore_d90": {"higher": "a coarser macro-pore upper tail", "lower": "a finer macro-pore upper tail"},
    "pore_elong": {"higher": "more elongated void sections; this is not transport anisotropy",
                   "lower": "more equiaxed pores"},
    "pore_count_per_Mpx": {"higher": "more, presumably smaller, macro-pores per area",
                           "lower": "fewer macro-pores per area"},
    "bright_frac": {"higher": "more observed additive area; recipe, utilization and mechanical implications need validation",
                    "lower": "less observed additive area; recipe, utilization and mechanical implications need validation"},
    "bright_d50": {"higher": "coarser additive sections; contact, chemistry and electrolyte access also affect later reaction/mechanics",
                   "lower": "finer additive sections; contact, chemistry and electrolyte access also affect later reaction/mechanics"},
    "bright_d90": {"higher": "a coarser additive upper tail: agglomerates or coarse supplier fraction",
                   "lower": "a finer additive upper tail"},
    "bright_d10": {"higher": "a coarser additive fine tail", "lower": "a finer additive fine tail (or more fines resolved)"},
    "bright_count_per_Mpx": {"higher": "more additive particles per area: finer dispersion or more fines",
                             "lower": "fewer additive particles per area: coarser dispersion or agglomeration"},
    "bright_circ": {"higher": "rounder additive particles", "lower": "more irregular additive particle outlines"},
    "bright_solidity": {"higher": "more convex additive particles", "lower": "more re-entrant additive particles"},
    "bright_max_d": {"higher": "a larger largest additive particle / agglomerate", "lower": "a smaller largest additive particle"},
    "corr_len_px": {"higher": "coarser texture scale", "lower": "finer texture scale"},
    "fft_slope": {"higher": "flatter spectrum: sharper edges or more fine texture (partly acquisition-sensitive)",
                  "lower": "steeper spectrum: smoother texture (partly acquisition-sensitive)"},
    "etd_crack_density_particles": {"higher": "more detected dark-ridge pixels inside additive masks; fracture interpretation needs expert review",
                                    "lower": "fewer dark ridges inside additive particles"},
}
for _k in range(10):
    CONSEQUENCE_TEXT[f"profile_pore_{_k}"] = {"higher": f"more macro-porosity in depth bin {_k} (through-thickness gradient; orientation unknown)",
                                              "lower": f"less macro-porosity in depth bin {_k} (through-thickness gradient; orientation unknown)"}
    CONSEQUENCE_TEXT[f"profile_bright_{_k}"] = {"higher": f"more additive in depth bin {_k} (through-thickness gradient; orientation unknown)",
                                                "lower": f"less additive in depth bin {_k} (through-thickness gradient; orientation unknown)"}
for _t in ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4"):
    CONSEQUENCE_TEXT[f"crack_p_{_t}"] = CONSEQUENCE_TEXT["etd_crack_density_particles"]


def qualitative_statement(kpi: str, shift_mad: float, direction: Optional[str], ref_value=None, batch_value=None) -> str:
    """One-sentence physics reading for a KPI shift, direction only.

    Pattern: "<KPI label> is <higher/lower> than the reference (robust shift <x> MAD) → <consequence>, direction only."
    Returns "" for KPIs of weight 1 (flags, confounded or diagnostic columns carry no physics reading).
    ``direction`` may be "higher"/"lower" (also "+"/"-", "up"/"down"); if None it is taken from the sign of
    ``shift_mad`` or from ``batch_value - ref_value``.
    """
    if CONSEQUENCE_WEIGHTS.get(kpi, 1) < 2 or kpi not in CONSEQUENCE_TEXT:
        return ""
    d = _norm_direction(direction, shift_mad, ref_value, batch_value)
    if d is None:
        return ""
    x = abs(float(shift_mad)) if shift_mad is not None and np.isfinite(shift_mad) else float("nan")
    return f"{_label(kpi)} is {d} than the reference (robust shift {x:.1f} MAD) → {CONSEQUENCE_TEXT[kpi][d]}, direction only."


def _norm_direction(direction, shift_mad, ref_value, batch_value):
    m = {"higher": "higher", "lower": "lower", "+": "higher", "-": "lower", "up": "higher", "down": "lower",
         "increase": "higher", "decrease": "lower", "more": "higher", "less": "lower"}
    if direction is not None and str(direction).lower() in m:
        return m[str(direction).lower()]
    if shift_mad is not None and np.isfinite(shift_mad) and shift_mad != 0:
        return "higher" if shift_mad > 0 else "lower"
    if ref_value is not None and batch_value is not None and np.isfinite(ref_value) and np.isfinite(batch_value):
        if batch_value == ref_value:
            return None
        return "higher" if batch_value > ref_value else "lower"
    return None


# ----------------------------------------------------------------------------------------------
# Stereology (secondary)
# ----------------------------------------------------------------------------------------------

STEREOLOGY_COLUMN_NOTES = {
    "vol_frac_pore": "Delesse: volume fraction = area fraction; assumes random sections",
    "vol_frac_bright": "Delesse: volume fraction = area fraction; assumes random sections",
    "vol_frac_graphite": "Conditional area-to-volume estimate: representative spatial sampling and valid phase labels required; plate alignment alone does not invalidate phase-area estimation",
    "nominal_mass_frac_additive_if_Si": "mass fraction of additive within the solid (graphite + additive), if the bright phase is Si (2.33 g/cm³); binder / carbon black unresolved and counted with graphite; nominal, for recipe comparison only",
    "nominal_mass_frac_additive_if_SiOx": "as above with SiOx ≈ 2.2 g/cm³; nominal, for recipe comparison only",
}


def stereology(sites) -> pd.DataFrame:
    """Delesse volume fractions and nominal additive mass fractions per site. **Secondary** (plan §2.0 rule 1).

    Accepts a site row (Series / dict) or a DataFrame with ``pore_frac``, ``bright_frac`` and optionally
    ``graphite_frac`` (else 1 − pore − bright). Returns a DataFrame with ``stereology_secondary=True`` on
    every row, ``caveats`` and ``claims_not_made`` lists in ``df.attrs``, and column notes in
    ``df.attrs["column_notes"]``.
    """
    if isinstance(sites, pd.DataFrame):
        df = sites.copy()
    elif isinstance(sites, pd.Series):
        df = sites.to_frame().T
    else:
        df = pd.DataFrame([dict(sites)])
    keep = [c for c in ("batch", "site", "bright_low_contrast", "pore_mode_resolved") if c in df.columns]
    out = df[keep].copy() if keep else pd.DataFrame(index=df.index)
    pore = pd.to_numeric(df["pore_frac"], errors="coerce").astype(float)
    bright = pd.to_numeric(df["bright_frac"], errors="coerce").astype(float)
    graphite = pd.to_numeric(df["graphite_frac"], errors="coerce").astype(float) if "graphite_frac" in df.columns else 1.0 - pore - bright
    out["vol_frac_pore"] = pore.values
    out["vol_frac_bright"] = bright.values
    out["vol_frac_graphite"] = graphite.values
    solid = bright + graphite
    with np.errstate(invalid="ignore", divide="ignore"):
        out["nominal_mass_frac_additive_if_Si"] = (RHO_SI * bright / (RHO_SI * bright + RHO_GRAPHITE * graphite)).values
        out["nominal_mass_frac_additive_if_SiOx"] = (RHO_SIOX * bright / (RHO_SIOX * bright + RHO_GRAPHITE * graphite)).values
    out["solid_frac"] = solid.values
    out["stereology_secondary"] = True
    out.attrs["column_notes"] = dict(STEREOLOGY_COLUMN_NOTES)
    out.attrs["caveats"] = GLOBAL_CAVEATS + [
        "Delesse assumes random (isotropic uniform) sections; plate-like graphite aligned by calendering violates this",
        "mass fractions are within the segmented solid only; binder / carbon-black network is unsegmented and counted with graphite",
        "secondary: does not carry a verdict",
    ]
    out.attrs["claims_not_made"] = [
        "that the bright phase is Si or SiOx",
        "that the recipe mass fraction equals these numbers (densities nominal; unresolved porosity inflates the solid)",
        "any absolute capacity or energy figure",
    ]
    return out


def saltykov_unfold(diameters_2d_px, n_bins: int = 12) -> pd.DataFrame:
    """Schwartz–Saltykov unfolding of 2-D section diameters into a relative 3-D number density per size class.

    **Spherical assumption**: particles are treated as spheres cut by random planes. Applied to the **bright
    (additive) phase only** — graphite plates are not spheres and must not be unfolded with this method.
    **Secondary** (plan §2.0 rule 1): the output does not carry a verdict.

    Method (Saltykov 1967, equal-width classes of width Δ = max/n_bins): a sphere of diameter D_j = jΔ cut by a
    random plane gives a section diameter in class i ≤ j with probability
    P(i,j) = [√(j²−(i−1)²) − √(j²−i²)] / j, and produces N_V(j)·D_j sections per unit area. So
    N_A(i) = Δ Σ_{j≥i} N_V(j) [√(j²−(i−1)²) − √(j²−i²)], solved by back-substitution from the largest class.
    N_A is taken as the raw counts (unit section area), so ``n_3d_per_volume_relative`` is in counts per
    (unit area × px) and only ratios between batches are meaningful. Negative classes (a known artefact of
    the method with small counts) are clipped to 0 and flagged in ``negative_clipped``.

    Returns a DataFrame with ``bin_center_px``, ``bin_upper_px``, ``n_2d_count``, ``n_3d_per_volume_relative``,
    ``negative_clipped``; NaN-safe (all-NaN densities, no exception) when fewer than 2 finite diameters are given.
    ``df.attrs`` carries ``secondary=True``, ``caveats``, ``claims_not_made`` and ``n_2d_total``.
    """
    d = np.asarray(diameters_2d_px, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    attrs = dict(secondary=True, n_2d_total=int(d.size), spherical_assumption=True, phase="bright (additive) only",
                 caveats=GLOBAL_CAVEATS + ["assumes near-spherical particles cut by random planes",
                                           "equal-width classes; negative classes clipped to 0",
                                           f"small-count instability: {d.size} sections in {n_bins} classes" if d.size < 10 * n_bins else "class counts adequate for the method",
                                           "secondary: does not carry a verdict"],
                 claims_not_made=["that the additive is spherical", "absolute 3-D number densities (unit area unknown)",
                                  "that the unfolded distribution is more reliable than the observed 2-D quantiles"])
    if d.size < 2 or n_bins < 1:
        out = pd.DataFrame({"bin_center_px": np.full(max(n_bins, 1), np.nan), "bin_upper_px": np.nan, "n_2d_count": 0,
                            "n_3d_per_volume_relative": np.nan, "negative_clipped": False})
        out.attrs.update(attrs); out.attrs["caveats"].append("too few sections to unfold (< 2)")
        return out
    dmax = d.max() * (1 + 1e-9)
    delta = dmax / n_bins
    counts, edges = np.histogram(d, bins=n_bins, range=(0.0, dmax))
    j = np.arange(1, n_bins + 1, dtype=float)
    # A[i, j] = Δ [√(j²−(i−1)²) − √(j²−i²)] for j ≥ i (0-based: i = row+1, j = col+1)
    A = np.zeros((n_bins, n_bins))
    for ii in range(n_bins):
        i = ii + 1
        jj = j[ii:]
        A[ii, ii:] = delta * (np.sqrt(jj ** 2 - (i - 1) ** 2) - np.sqrt(np.maximum(jj ** 2 - i ** 2, 0.0)))
    nv = np.zeros(n_bins)
    neg = np.zeros(n_bins, dtype=bool)
    for ii in range(n_bins - 1, -1, -1):
        rest = A[ii, ii + 1:] @ nv[ii + 1:]
        val = (counts[ii] - rest) / A[ii, ii] if A[ii, ii] > 0 else np.nan
        if np.isfinite(val) and val < 0:
            neg[ii] = True; val = 0.0
        nv[ii] = val
    out = pd.DataFrame({"bin_center_px": 0.5 * (edges[:-1] + edges[1:]), "bin_upper_px": edges[1:],
                        "n_2d_count": counts, "n_3d_per_volume_relative": nv, "negative_clipped": neg})
    out.attrs.update(attrs)
    # implied total number of sections from the unfolded distribution (Σ N_V(j) D_j) for a conservation check
    out.attrs["implied_n_2d_from_3d"] = float(np.nansum(nv * edges[1:]))
    return out


def unfolded_quantiles(unfold_df: pd.DataFrame, qs=(0.1, 0.5, 0.9)) -> Dict[str, float]:
    """Number-weighted quantiles (D10/D50/D90 in px) of a saltykov_unfold() result; NaN when the density is empty."""
    nv = unfold_df["n_3d_per_volume_relative"].to_numpy(dtype=float)
    up = unfold_df["bin_upper_px"].to_numpy(dtype=float)
    if not np.isfinite(nv).any() or np.nansum(nv) <= 0:
        return {f"d{int(q*100)}": float("nan") for q in qs}
    nv = np.nan_to_num(nv); cw = np.cumsum(nv) / nv.sum()
    return {f"d{int(q*100)}": float(up[min(int(np.searchsorted(cw, q)), len(up) - 1)]) for q in qs}


# ----------------------------------------------------------------------------------------------
# Section tortuosity index (2-D, descriptive)
# ----------------------------------------------------------------------------------------------

TORTUOSITY_CAVEATS = [
    "2-D section index; not a bound on 3-D tortuosity (paths leave the section; disconnected 2-D masks can be connected in 3-D)",
    "macro-pores only; sub-resolution porosity not represented",
    f"computed on a {DOWNSAMPLE}× block-downsampled mask (majority rule); necks thinner than {DOWNSAMPLE} px can be lost",
]


def _downsample_mask(mask: np.ndarray, f: int = DOWNSAMPLE) -> np.ndarray:
    """Block-downsample a boolean mask by majority rule (block mean ≥ 0.5). Trailing partial blocks are dropped."""
    m = np.asarray(mask, dtype=bool)
    if f <= 1:
        return m
    H, W = (m.shape[0] // f) * f, (m.shape[1] // f) * f
    if H == 0 or W == 0:
        return m
    b = m[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))
    return b >= 0.5


def section_tortuosity_index(pore_mask_bool: np.ndarray, direction: str = "vertical", n_lines: int = 50,
                             downsample: int = DOWNSAMPLE, solid_cost: float = 1e6) -> dict:
    """Geodesic 2-D section tortuosity index of the macro-pore phase, through the strip thickness.

    For up to ``n_lines`` seed columns (evenly spaced among pore pixels on the top edge) the shortest path
    through the pore phase to the bottom edge is found with ``skimage.graph.MCP_Geometric`` on a cost map that is
    1 inside pores and ``solid_cost`` (1e6, effectively impassable) outside. Choice: MCP_Geometric weights
    diagonal steps by √2 so the accumulated cost is a Euclidean geodesic length; ``route_through_array`` is the same
    engine for a single start/end pair, but here each seed must reach *any* bottom pore pixel, which
    ``find_costs(starts, ends)`` handles directly and stops at the first end reached. Seeds whose connected
    component does not touch the bottom edge are counted as not connected (no path exists in 2-D).

    Returns dict with
      index:               median over connected seeds of geodesic length / straight-line height; NaN if none connect
      fraction_connected:  share of seed columns that reach the bottom edge through pores in 2-D
      n_seeds, n_connected, height_px (downsampled) and height_full_px,
      statement:           one sentence (undefined-index case spelled out),
      penalised_index:     one multi-start run with solid cost = ``PENALISED_SOLID_COST`` (20): cheapest top-to-bottom
                           cost / height, where each solid pixel costs 20 pore pixels. Always finite; descriptive only —
                           it ranks sections by how much solid the cheapest 2-D route has to cross,
      caveats, claims_not_made.
    ``direction="horizontal"`` transposes the mask (paths left → right).
    """
    from skimage.graph import MCP_Geometric

    m = np.asarray(pore_mask_bool, dtype=bool)
    if direction == "horizontal":
        m = m.T
    elif direction != "vertical":
        raise ValueError("direction must be 'vertical' or 'horizontal'")
    full_h = m.shape[0]
    md = _downsample_mask(m, downsample)
    H, W = md.shape
    base = dict(index=np.nan, fraction_connected=0.0, n_seeds=0, n_connected=0, height_px=int(H), height_full_px=int(full_h),
                penalised_index=np.nan, downsample=int(downsample), direction=direction,
                caveats=list(TORTUOSITY_CAVEATS),
                claims_not_made=["effective ionic conductivity or any transport coefficient",
                                 "that this index bounds the 3-D tortuosity in either direction",
                                 "anything about porosity below the segmentation resolution"])
    if H < 2 or W < 1 or not md.any():
        base["caveats"].append("empty or degenerate mask"); base["statement"] = "Empty or degenerate mask; no section tortuosity index."
        return base
    # seeds: pore pixels on the top row, evenly spaced
    top_cols = np.where(md[0])[0]
    if top_cols.size == 0:
        base["caveats"].append("no pore pixel on the top edge; no seed available")
        base.update(penalised_index=_penalised_index(md)); base["statement"] = _tortuosity_statement(base)
        return base
    pick = np.unique(np.linspace(0, top_cols.size - 1, min(n_lines, top_cols.size)).round().astype(int))
    seeds = [(0, int(top_cols[p])) for p in pick]
    # connectivity: component touching both top and bottom rows (8-connectivity, same as MCP's neighbourhood)
    lab, _ = ndi.label(md, structure=np.ones((3, 3), int))
    bottom_labels = set(np.unique(lab[-1][md[-1]]).tolist())
    ends = [(H - 1, int(c)) for c in np.where(md[-1])[0]]
    cost = np.where(md, 1.0, float(solid_cost))
    lengths = []
    n_conn = 0
    for s in seeds:
        if lab[s] not in bottom_labels:
            continue
        n_conn += 1
        mcp = MCP_Geometric(cost, fully_connected=True)
        cum, _ = mcp.find_costs(starts=[s], ends=ends, find_all_ends=False)
        c = cum[-1][md[-1]]
        c = c[np.isfinite(c)]
        if c.size:
            lengths.append(float(c.min()) / (H - 1))
    base.update(n_seeds=len(seeds), n_connected=n_conn, fraction_connected=n_conn / len(seeds),
                index=float(np.median(lengths)) if lengths else np.nan,
                penalised_index=_penalised_index(md))
    base["statement"] = _tortuosity_statement(base)
    return base


def _tortuosity_statement(r: dict) -> str:
    if r["n_seeds"] == 0:
        return "No macro-pore pixel on the top edge; no section tortuosity index."
    if r["n_connected"] == 0:
        return (f"No 2-D top-to-bottom path through the macro-pore phase ({r['n_connected']} of {r['n_seeds']} seed columns connect); "
                "the section index is undefined. This is expected for isolated macro-pores at ~10 % area fraction in a single section "
                "and says nothing about 3-D connectivity. "
                f"Penalised index {r['penalised_index']:.1f} (cheapest route with solid costing {PENALISED_SOLID_COST:.0f}× pore; descriptive, relative only).")
    return (f"Section tortuosity index {r['index']:.2f} (median geodesic / straight-line height over {r['n_connected']} of {r['n_seeds']} connected seed columns); "
            f"penalised index {r['penalised_index']:.1f}. 2-D section index, not a 3-D bound.")


PENALISED_SOLID_COST = 20.0


def _penalised_index(md: np.ndarray, solid_cost: float = PENALISED_SOLID_COST) -> float:
    """Cheapest top-to-bottom geodesic cost / height with solid pixels costing ``solid_cost`` pore pixels (descriptive)."""
    from skimage.graph import MCP_Geometric
    H, W = md.shape
    cost = np.where(md, 1.0, float(solid_cost))
    mcp = MCP_Geometric(cost, fully_connected=True)
    starts = [(0, c) for c in range(W)]
    cum, _ = mcp.find_costs(starts=starts)
    return float(np.nanmin(cum[-1]) / (H - 1))


# ----------------------------------------------------------------------------------------------
# Void geometry (2-D geometric observation)
# ----------------------------------------------------------------------------------------------

VOID_CAVEATS = ["2-D geometric observation; not evidence of electronic discontinuity",
                f"crack-like = pore region with major axis > cutoff px (default {CRACK_LONG_AXIS_PX}; judgement call, decision log D09 / Part C)",
                "regions touching a trimmed edge band may be cut by the trim"]


def void_geometry(pore_mask_bool: np.ndarray, crack_long_axis_px: float = CRACK_LONG_AXIS_PX,
                  strip_height_px: Optional[int] = None, min_area_px: int = 30, downsample: int = DOWNSAMPLE) -> dict:
    """Crack-like voids in a 2-D pore mask (full resolution for region measures; ×4 downsampled columns for interruption).

    Returns dict with
      n_crack_like:               number of pore regions (area ≥ min_area_px) with major axis > crack_long_axis_px
      crack_area_frac:            their area / mask area
      longest_void_px:            largest major axis among all pore regions (px, full resolution)
      longest_void_over_thickness: longest_void_px / H (H = strip_height_px or mask height)
      columns_interrupted_frac:   fraction of x-columns (after ×downsample binning) that intersect ≥ 1 crack-like void
      delamination_index:         Σ horizontal extents (bbox widths) of crack-like voids / strip width (can exceed 1 when stacked)
      crack_orientation_deg_median: median |orientation| of crack-like voids from the horizontal (0 = along the coating)
      statements:                 qualitative sentences
      caveats, claims_not_made.
    """
    from skimage.measure import label, regionprops_table

    m = np.asarray(pore_mask_bool, dtype=bool)
    Hm, Wm = m.shape
    H = int(strip_height_px) if strip_height_px else Hm
    out = dict(n_crack_like=0, crack_area_frac=0.0, longest_void_px=0.0, longest_void_over_thickness=0.0,
               columns_interrupted_frac=0.0, delamination_index=0.0, crack_orientation_deg_median=np.nan,
               n_voids=0, crack_long_axis_px=float(crack_long_axis_px), strip_height_px=H,
               statements=[], caveats=list(VOID_CAVEATS),
               claims_not_made=["electronic continuity or discontinuity of the coating",
                                "that a long void in this section is a delamination in 3-D (it may be a sectioned pore)",
                                "mechanical consequence (no stress or adhesion values)"])
    if not m.any():
        out["statements"].append("No pore region in the mask.")
        return out
    lab = label(m, connectivity=2)
    rp = pd.DataFrame(regionprops_table(lab, properties=("label", "area", "major_axis_length", "bbox", "orientation")))
    rp = rp[rp.area >= min_area_px]
    out["n_voids"] = int(len(rp))
    if len(rp) == 0:
        out["statements"].append(f"No pore region of area ≥ {min_area_px} px.")
        return out
    out["longest_void_px"] = float(rp.major_axis_length.max())
    out["longest_void_over_thickness"] = out["longest_void_px"] / H
    big = rp[rp.major_axis_length > crack_long_axis_px]
    out["n_crack_like"] = int(len(big))
    out["crack_area_frac"] = float(big.area.sum() / m.size)
    if len(big):
        widths = (big["bbox-3"] - big["bbox-1"]).to_numpy(dtype=float)
        out["delamination_index"] = float(widths.sum() / Wm)
        # skimage orientation: angle between the major axis and the row (vertical) axis; 90° = along the coating
        out["crack_orientation_deg_median"] = float(np.median(np.abs(90.0 - np.degrees(np.abs(big.orientation.to_numpy())))))
        crack_mask = np.isin(lab, big.label.to_numpy())
        cd = _downsample_mask_any(crack_mask, downsample)
        out["columns_interrupted_frac"] = float(cd.any(axis=0).mean())
    # statements
    s = out["statements"]
    if out["n_crack_like"] == 0:
        s.append(f"No crack-like void (major axis > {crack_long_axis_px:.0f} px) in this section; the longest void spans "
                 f"{out['longest_void_over_thickness']:.2f} of the strip height.")
    else:
        s.append(f"{out['n_crack_like']} crack-like void(s) with major axis > {crack_long_axis_px:.0f} px, covering "
                 f"{100*out['crack_area_frac']:.2f} % of the section area.")
        s.append(f"The longest void spans {out['longest_void_over_thickness']:.2f} of the strip height "
                 f"({out['longest_void_px']:.0f} px, ≈ {out['longest_void_px']*25e-3:.1f} µm nominal).")
        s.append(f"In 2-D, {100*out['columns_interrupted_frac']:.0f} % of through-thickness columns intersect a crack-like void "
                 f"(delamination index {out['delamination_index']:.2f}); this is a geometric observation in one section, "
                 f"not evidence of electronic discontinuity.")
        if np.isfinite(out["crack_orientation_deg_median"]) and out["crack_orientation_deg_median"] < 30:
            s.append("Crack-like voids run mostly along the coating (median tilt < 30° from horizontal), the geometry expected of delamination-like voids.")
    return out


def _downsample_mask_any(mask: np.ndarray, f: int) -> np.ndarray:
    """Block-downsample with an 'any' rule (used for thin crack masks so a 1-block-wide void is not lost)."""
    m = np.asarray(mask, dtype=bool)
    if f <= 1:
        return m
    H, W = (m.shape[0] // f) * f, (m.shape[1] // f) * f
    if H == 0 or W == 0:
        return m
    return m[:H, :W].reshape(H // f, f, W // f, f).any(axis=(1, 3))


# ----------------------------------------------------------------------------------------------
# Additive size and mechanics readings
# ----------------------------------------------------------------------------------------------

def _dir_word(ratio: float, tol: float = SIMILAR_TOL, up: str = "coarser", down: str = "finer", same: str = "similar") -> str:
    if not np.isfinite(ratio):
        return "not measurable"
    if ratio > 1 + tol:
        return up
    if ratio < 1 - tol:
        return down
    return same


def additive_size_reading(ref_quantiles: Dict[str, float], batch_quantiles: Dict[str, float], basis: str = "observed_2d",
                          tol: float = SIMILAR_TOL) -> dict:
    """Matched section-size comparison; no inferred lithiation-time factor.

    Observed quantiles do not determine a diffusion radius, equal section bias,
    diffusivity or contact/access. Unfolded quantiles remain assumption-limited.
    """
    if basis not in ("observed_2d", "unfolded"):
        raise ValueError("basis must be 'observed_2d' or 'unfolded'")
    qs = [q for q in ("d10", "d50", "d90") if q in ref_quantiles and q in batch_quantiles]
    ratios, direction = {}, {}
    for q in qs:
        r, b = float(ref_quantiles[q]), float(batch_quantiles[q])
        ratios[q] = b / r if (np.isfinite(r) and r > 0 and np.isfinite(b)) else float("nan")
        direction[q] = _dir_word(ratios[q], tol)
    parts = [f"{q.upper()} {direction[q]} (×{ratios[q]:.2f})" if np.isfinite(ratios[q]) else f"{q.upper()} not measurable" for q in qs]
    stmt = f"Additive size vs reference, matched quantiles ({'observed 2-D' if basis == 'observed_2d' else 'Saltykov-unfolded, secondary'}): " + "; ".join(parts) + "."
    stmt += " These size descriptors do not determine later lithiation time or total expansion load."
    return dict(basis=basis, ratios=ratios, direction=direction, statement=stmt,
                caveats=GLOBAL_CAVEATS
                        + (["unfolded basis assumes spherical particles; secondary"] if basis == "unfolded" else ["2-D section-size sampling, shape and orientation bias may differ between batches"])
                        + ["contact, chemistry, porosity and electrolyte access remain unresolved",
                           "bright-phase KPIs are unreliable on low-contrast sites; check the usable site count"],
                claims_not_made=["absolute or relative diffusion/lithiation times", "equal section-sampling bias between batches",
                                 "any rate-capability or total expansion forecast"])


def additive_mechanics_reading(ref_row, batch_row, tol: float = SIMILAR_TOL) -> dict:
    """Describe loading/section-size/ridge changes without inferring binder load.

    Ridge coverage counts qualifying interior pixels, not intact particles.
    Neither particle size alone nor 2-D phase amount establishes total expansion.
    """
    def g(row, k):
        try:
            return float(row[k])
        except (KeyError, IndexError, TypeError, ValueError):
            return float("nan")

    def ratio(k):
        r, b = g(ref_row, k), g(batch_row, k)
        return b / r if np.isfinite(r) and r > 0 and np.isfinite(b) else float("nan")

    lr, sr = ratio("bright_frac"), ratio("bright_d50")
    low = bool(_truthy(ref_row, "bright_low_contrast") or _truthy(batch_row, "bright_low_contrast"))
    dl = _dir_word(lr, tol, "more", "less"); ds = _dir_word(sr, tol, "coarser", "finer")
    direction = "unreliable (low-contrast bright phase on at least one side)" if low else "not inferred"
    rr, rb = g(ref_row, "etd_crack_density_particles"), g(batch_row, "etd_crack_density_particles")
    stmt = (f"Observed additive area ×{lr:.2f} ({dl}); section D50 ×{sr:.2f} ({ds}). "
            "Binder stress and total expansion are not inferred from these descriptors.")
    if low:
        stmt += " Bright-phase comparison is unreliable because of low contrast."
    if np.isfinite(rr) and np.isfinite(rb):
        stmt += f" Detected interior ridge-pixel coverage: reference {rr:.4f}, batch {rb:.4f}; this is not an intact-particle share."
    return dict(direction=direction, loading_direction=dl, size_direction=ds, loading_ratio=lr, size_ratio=sr,
                ridge_coverage_ref=rr, ridge_coverage_batch=rb, low_contrast_involved=low, statement=stmt,
                caveats=GLOBAL_CAVEATS + ["area amount and section size are distinct observations",
                                          "confinement, phase utilization, interfaces and binder properties are unresolved",
                                          "ridge coverage is not a validated fracture or electrical-contact measurement"],
                claims_not_made=["stress, strain, adhesion or total expansion values", "binder failure or capacity fade",
                                 "intact-particle fraction", "cycling-induced fracture in fresh specimens"])


def _truthy(row, key) -> bool:
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    try:
        return bool(v) and not (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


# ----------------------------------------------------------------------------------------------
# Sanity checks on our own measurements
# ----------------------------------------------------------------------------------------------

POROSITY_LOW_CUTOFF = 0.20
POROSITY_NOTE = ("consistent with unresolved porosity; segmentation and preparation effects remain alternatives")


def sanity_checks(sites_df: pd.DataFrame, images_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-site physics sanity checks (reported, never hidden).

    Columns: fractions_sum_to_one (|pore+bright+graphite−1| < 1e-6), fraction_sum_residual, porosity_vs_typical
    ("below_typical" when pore_frac < 0.20, with ``porosity_note``), anisotropy_index (= pore_elong − 1; 0 = equiaxed),
    etd_ridge_dom_angle passthrough when present, and the threshold-sensitivity band columns joined from ``images_df``
    (the output of :func:`threshold_sensitivity` rows) when provided. ``df.attrs`` carries caveats / claims_not_made.
    """
    df = sites_df.copy()
    out = df[[c for c in ("batch", "site") if c in df.columns]].copy()
    pore = pd.to_numeric(df.get("pore_frac"), errors="coerce").astype(float)
    bright = pd.to_numeric(df.get("bright_frac"), errors="coerce").astype(float)
    graphite = pd.to_numeric(df["graphite_frac"], errors="coerce").astype(float) if "graphite_frac" in df.columns else 1 - pore - bright
    resid = (pore + bright + graphite - 1.0)
    out["fraction_sum_residual"] = resid.values
    out["fractions_sum_to_one"] = (resid.abs() < 1e-6).values
    out["pore_frac"] = pore.values
    out["porosity_vs_typical"] = np.where(pore.values < POROSITY_LOW_CUTOFF, "below_typical", "within_typical")
    out["porosity_note"] = np.where(pore.values < POROSITY_LOW_CUTOFF, POROSITY_NOTE, "")
    if "pore_elong" in df.columns:
        out["anisotropy_index"] = (pd.to_numeric(df["pore_elong"], errors="coerce") - 1.0).values
    else:
        out["anisotropy_index"] = np.nan
    if "etd_ridge_dom_angle" in df.columns:
        out["etd_ridge_dom_angle"] = df["etd_ridge_dom_angle"].values
    if "pore_mode_resolved" in df.columns:
        out["pore_threshold_fallback"] = ~df["pore_mode_resolved"].astype(bool).values
    if "bright_low_contrast" in df.columns:
        out["bright_low_contrast"] = df["bright_low_contrast"].astype(bool).values
    if images_df is not None and len(images_df):
        keep = [c for c in images_df.columns if c not in ("batch",)]
        out = out.merge(images_df[keep], on="site", how="left")
    out.attrs["caveats"] = GLOBAL_CAVEATS + [
        "typical calendered-anode porosity 25–35 % is a literature range, not a specification for this product",
        "the binder / carbon-black network is unsegmented and falls into the pore or graphite class",
        "anisotropy_index is a 2-D pore-shape statistic; plate alignment in 3-D is not measured",
    ]
    out.attrs["claims_not_made"] = ["that low measured porosity proves unresolved porosity (preparation and segmentation remain alternatives)",
                                    "any transport property from the anisotropy index"]
    return out


def threshold_sensitivity(batch_dir: str, site: str, th_lo: float, th_hi: float, delta: float = 5.0) -> dict:
    """Recompute pore_frac and bright_frac on the real BSE image with th_lo ± delta and th_hi ± delta (slow: loads the TIFF).

    Returns a dict (one row) with the nominal fractions, the values at ±delta, the band half-widths and relative bands
    (band / nominal). Uses the same preprocessing as the notebook (edge-band trim, Gaussian σ=1, one binary opening).
    """
    img = trimmed(load_bse(batch_dir, site))
    sm = _smooth(img)
    res = dict(site=site, th_lo=float(th_lo), th_hi=float(th_hi), delta=float(delta))
    pf = {d: _opened(sm < (th_lo + d)).mean() for d in (-delta, 0.0, +delta)}
    bf = {d: _opened(sm > (th_hi + d)).mean() for d in (-delta, 0.0, +delta)}
    res.update(pore_frac_nominal=pf[0.0], pore_frac_lo_minus=pf[-delta], pore_frac_lo_plus=pf[+delta],
               bright_frac_nominal=bf[0.0], bright_frac_hi_minus=bf[-delta], bright_frac_hi_plus=bf[+delta])
    res["pore_frac_band"] = float(max(pf.values()) - min(pf.values()))
    res["bright_frac_band"] = float(max(bf.values()) - min(bf.values()))
    res["pore_frac_band_rel"] = res["pore_frac_band"] / pf[0.0] if pf[0.0] > 0 else np.nan
    res["bright_frac_band_rel"] = res["bright_frac_band"] / bf[0.0] if bf[0.0] > 0 else np.nan
    return res


# ----------------------------------------------------------------------------------------------
# Per-site driver: run the layer on real images and cache analysis_cache/physics_sites.csv
# ----------------------------------------------------------------------------------------------

def compute_site_physics(batch_dir: str, site: str, th_lo: float, th_hi: float, n_lines: int = 50) -> dict:
    """Load one BSE image, trim edge bands, segment with the given thresholds, run void_geometry and
    section_tortuosity_index. Returns a flat dict (statements joined with ' | ') with runtimes in seconds."""
    t0 = time.perf_counter()
    img = trimmed(load_bse(batch_dir, site))
    pore, bright = segment(img, th_lo, th_hi)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter(); vg = void_geometry(pore, strip_height_px=pore.shape[0]); t_vg = time.perf_counter() - t1
    t2 = time.perf_counter(); tz = section_tortuosity_index(pore, n_lines=n_lines); t_tz = time.perf_counter() - t2
    row = dict(site=site, H_trim=int(pore.shape[0]), W_trim=int(pore.shape[1]), pore_frac_recomputed=float(pore.mean()),
               bright_frac_recomputed=float(bright.mean()))
    for k in ("n_voids", "n_crack_like", "crack_area_frac", "longest_void_px", "longest_void_over_thickness",
              "columns_interrupted_frac", "delamination_index", "crack_orientation_deg_median"):
        row[k] = vg[k]
    row["void_statements"] = " | ".join(vg["statements"])
    for k in ("index", "fraction_connected", "n_seeds", "n_connected", "penalised_index"):
        row[f"tortuosity_{k}"] = tz[k]
    row["tortuosity_statement"] = tz["statement"]
    row.update(t_load_segment_s=t_load, t_void_geometry_s=t_vg, t_tortuosity_s=t_tz)
    return row


def run_all_sites(dataset_root: str, site_features_csv: str, out_csv: Optional[str] = None,
                  sensitivity_sites: Optional[Iterable[str]] = None, etd_csv: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """Run void_geometry + section_tortuosity_index on every site in site_features.csv, stereology on all sites,
    threshold_sensitivity on ``sensitivity_sites``, and write the merged per-site table to ``out_csv``."""
    F = pd.read_csv(site_features_csv)
    rows = []
    for _, r in F.iterrows():
        row = compute_site_physics(os.path.join(dataset_root, r.batch), r.site, r.th_lo, r.th_hi)
        row["batch"] = r.batch
        rows.append(row)
        if verbose:
            print(f"{r.batch} {r.site}: cracks {row['n_crack_like']} cols_int {row['columns_interrupted_frac']:.2f} delam {row['delamination_index']:.2f} "
                  f"| tort idx {row['tortuosity_index']:.3f} conn {row['tortuosity_fraction_connected']:.2f} pen {row['tortuosity_penalised_index']:.2f} "
                  f"| {row['t_void_geometry_s']:.1f}s + {row['t_tortuosity_s']:.1f}s", flush=True)
    P = pd.DataFrame(rows)
    S = stereology(F)
    P = P.merge(S.drop(columns=[c for c in ("bright_low_contrast", "pore_mode_resolved") if c in S.columns]), on=["batch", "site"], how="left")
    if sensitivity_sites:
        srows = []
        for s in sensitivity_sites:
            r = F[F.site == s].iloc[0]
            t0 = time.perf_counter(); d = threshold_sensitivity(os.path.join(dataset_root, r.batch), s, r.th_lo, r.th_hi); d["t_sensitivity_s"] = time.perf_counter() - t0
            srows.append(d)
            if verbose:
                print(f"sensitivity {s}: pore {d['pore_frac_lo_minus']:.3f}/{d['pore_frac_nominal']:.3f}/{d['pore_frac_lo_plus']:.3f} "
                      f"bright {d['bright_frac_hi_minus']:.3f}/{d['bright_frac_nominal']:.3f}/{d['bright_frac_hi_plus']:.3f}", flush=True)
        Sdf = pd.DataFrame(srows).drop(columns=["th_lo", "th_hi"])
        P = P.merge(Sdf, on="site", how="left")
    sites_for_checks = F
    if etd_csv and os.path.exists(etd_csv):
        E = pd.read_csv(etd_csv)
        sites_for_checks = F.merge(E[["batch", "site", "etd_ridge_dom_angle"]], on=["batch", "site"], how="left")
    C = sanity_checks(sites_for_checks)
    P = P.merge(C.drop(columns=[c for c in ("pore_frac", "bright_low_contrast") if c in C.columns]), on=["batch", "site"], how="left")
    cols = ["batch", "site"] + [c for c in P.columns if c not in ("batch", "site")]
    P = P[cols]
    if out_csv:
        P.to_csv(out_csv, index=False)
    return P


# ----------------------------------------------------------------------------------------------
# temporary copies — replaced by polaron_qc.features after integration
# (copied from notebooks/_build_01_dataset_analysis.py: load, bright_bands, trimmed, phase_thresholds, segmentation)
# ----------------------------------------------------------------------------------------------

def load_bse(batch_dir: str, site: str) -> np.ndarray:
    """Grayscale uint8 BSE image of a site (the TIFFs store identical R,G,B planes; channel 0 is used)."""
    import tifffile
    p = os.path.join(batch_dir, f"img_{site}_BSE.tif")
    a = tifffile.imread(p)
    return a[..., 0] if a.ndim == 3 else a


def bright_bands(img, margin=0.2, delta=40, minrows=4):
    """Rows at the top/bottom that are far brighter than the body (current collector / stitching border)."""
    rm = img.mean(1); body = np.median(rm); H = len(rm)
    hot = rm > body + delta
    top = 0
    while top < H * margin and hot[top]: top += 1
    bot = 0
    while bot < H * margin and hot[H - 1 - bot]: bot += 1
    idx = np.where(hot)[0]
    if (idx >= H * (1 - margin)).any(): bot = max(bot, H - idx[idx >= H * (1 - margin)].min())
    if (idx < H * margin).any():     top = max(top, idx[idx < H * margin].max() + 1)
    return (top if top >= minrows else 0), (bot if bot >= minrows else 0)


def trimmed(img):
    t, b = bright_bands(img); return img[t: img.shape[0] - b if b else None]


def phase_thresholds(sm):
    """Per-image pore/graphite/bright thresholds anchored on the graphite mode of a smoothed BSE histogram.
    Bright threshold = valley between graphite mode and a resolved second (bright) mode when one exists,
    otherwise graphite mode + 3.5 sigma_right (flagged). Pore threshold = valley between the dark mode and the
    graphite mode when one exists, otherwise mode - 3.5 sigma_left (flagged)."""
    from scipy.signal import find_peaks
    h = np.bincount(sm[::3, ::3].astype(int).ravel(), minlength=256).astype(float); hs = ndi.gaussian_filter1d(h, 3) + 1e-9
    m = int(np.argmax(hs[10:200])) + 10; half = hs[m] / 2
    l = m
    while l > 0 and hs[l] > half: l -= 1
    r = m
    while r < 255 and hs[r] > half: r += 1
    sl, sr = (m - l) / 1.177, (r - m) / 1.177
    pk, _ = find_peaks(np.log(hs[m + 15:220]), prominence=0.4); pk = pk + m + 15
    if len(pk):
        p2 = int(pk[np.argmax(hs[pk])]); thi = int(m + np.argmin(hs[m:p2])); bright_resolved = True
    else:
        p2 = np.nan; thi = m + 3.5 * sr; bright_resolved = False
    bright_sep = p2 - m if bright_resolved else np.nan
    bright_low_contrast = (not bright_resolved) or (bright_sep < 45)
    pkl, _ = find_peaks(np.log(hs[:max(m - 10, 1)]), prominence=0.4)
    if len(pkl) or hs[0] > hs[max(m - 10, 1):m].min():
        dark_peak = pkl[np.argmax(hs[pkl])] if len(pkl) else 0
        tlo = int(dark_peak + np.argmin(hs[dark_peak:m])); pore_resolved = True
    else:
        tlo = m - 3.5 * sl; pore_resolved = False
    return dict(th_lo=tlo, th_hi=thi, graphite_mode=m, sigma_l=sl, sigma_r=sr, bright_mode=p2, bright_sep=bright_sep,
                bright_mode_resolved=bright_resolved, bright_low_contrast=bright_low_contrast, pore_mode_resolved=pore_resolved, hist=hs / hs.sum())


def _smooth(img):
    from skimage.filters import gaussian
    return gaussian(img, sigma=1.0, preserve_range=True)


def _opened(mask):
    return ndi.binary_opening(mask, iterations=1)


def segment(img, th_lo, th_hi):
    """Pore and bright masks exactly as in notebook 01: Gaussian σ=1, threshold, one binary opening."""
    sm = _smooth(img)
    return _opened(sm < th_lo), _opened(sm > th_hi)


if __name__ == "__main__":  # pragma: no cover
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sens = ["4ih2ggld", "fzrt2k6r", "3806gxp0", "r17byphk", "71vgq3fw", "hzumfsms"]
    t0 = time.perf_counter()
    P = run_all_sites(os.path.join(ROOT, "Dataset"), os.path.join(ROOT, "analysis_cache", "site_features.csv"),
                      out_csv=os.path.join(ROOT, "analysis_cache", "physics_sites.csv"), sensitivity_sites=sens,
                      etd_csv=os.path.join(ROOT, "analysis_cache", "etd_inlens_features.csv"))
    print(f"total {time.perf_counter()-t0:.0f} s; wrote {len(P)} rows")
