"""polaron_qc.report — per-batch QC report (plan output #2) and the evidence-image renderers.

Two public entry points plus four importable evidence helpers::

    from polaron_qc.report import build_result, render_report
    result = build_result("Dataset/Batch_3", "Dataset/Batch_1")          # ~1-2 min with a warm feature cache
    render_report(result, "reports/qc_Batch_1.html")                      # self-contained HTML, embedded JPEGs

``build_result`` is the first end-to-end assembly of the existing modules (features → stats → decision, with the
ml and physics caches read from disk). It exists so the notebook (`notebooks/02_batch_qc.ipynb`) can reuse one
schema; integration problems it exposes are reported in the build log, never patched around silently.

Schema of ``result`` (a plain dict; DataFrames where noted; every key always present, ``None`` when unavailable)
---------------------------------------------------------------------------------------------------------------
meta            {reference, batch, reference_dir, batch_dir, n_sites_ref, n_sites_batch, timestamp,
                 thresholds_hash, provenance, config, config_hash, git_describe, feature_version, runtime_s,
                 primary_kpis, trusted_kpis, notes[list of integration notes produced while building]}
sites_ref       DataFrame, features.extract_batch(...)["sites"] of the reference (one row per site, 87 cols)
sites_batch     DataFrame, same for the incoming batch
images_ref, images_batch   DataFrame, features ``images`` tables (raw per-image statistics)
patches_batch   DataFrame, features ``patches`` table of the batch (y0, x0 in the TRIMMED frame)
flags_ref, flags_batch     DataFrame, PROVISIONAL per-site acquisition flags derived here (the acquisition module
                 will own this): site, contrast_stretched_bse, contrast_stretched_any, bse_p1, raised_black_level,
                 bright_low_contrast, grey_pore, cracked, band_top, band_bottom, acquisition_group
ordinary_ref_sites   list of reference site ids = reference minus GREY_PORE_SITES minus CRACKED_SITES
compare         DataFrame, stats.compare_kpis tidy table (statistic hl_shift) over all trusted KPIs present;
                 Holm on the five primary KPIs; grey-pore sites kept as fallback
energy          dict, stats.energy_distance_test on the primary KPIs (consequence-weighted; low-contrast sites'
                 bright-phase KPIs set to NaN first, so those rows are dropped and counted in n_dropped_batch)
mdc             {kpi: stats.mdc dict} for each primary KPI, n_incoming = that KPI's n_batch_usable, full usable reference
drift           DataFrame, stats.per_site_drift on the primary KPIs (reference sites leave-one-site-out)
local           {kpi: stats.local_exceedance dict (table, ref_max, ref_mad, iid_flag_probability, ...)} against the
                 ORDINARY reference sites; keys = primary KPIs + 'bright_max_d' (descriptive) +
                 'patch_crack_area_frac', 'patch_pore_max_d' (per-site maxima of patch KPIs; descriptive, trimmed frame)
check_a, check_b, abstention, verdict   dicts from polaron_qc.decision (image_reviewed=None → 'pending_review')
stability       {share, n_runs, full_verdict, runs: [{left_out, verdict, outcome_columns}], method}
c2st_material   dict or None (not in the ml cache yet: the cached runs include etd_boundary_sharpness)
c2st_flag_inclusive   dict {auc, null_band, p, n_perm, p_method, cv, n_sites_ref, n_sites_batch, coef: DataFrame,
                 per_site_scores: DataFrame, kpis, variant} or None
c2st_extra      {variant_tag: same dict} for other cached variants of this pair (e.g. without_low_contrast) or {}
novelty         {per_site: DataFrame (this batch), ref_loo: DataFrame, per_patch: DataFrame (this batch; y0/x0 in the
                 ORIGINAL image frame, see ml.patch_embeddings), correlates: DataFrame (all sites), iid_rate,
                 top_site} or None
physics         {sites: DataFrame (physics_sites.csv rows of both batches), statements: {kpi: {statement, gated,
                 band_available, band_abs, delta_abs, note}}, weights: {kpi: int}, rationale: {kpi: str},
                 labels: {kpi: str}, sanity: DataFrame (physics.sanity_checks on the batch), fraction_connected_note}
acquisition     None (placeholder for polaron_qc.acquisition: {unadjusted, stratified, adjusted} views)
limits          {usable_n: {kpi: (n_ref, n_batch)}, specimen_independence: str, reference_heterogeneity: {...},
                 mdc_range_mad: (lo, hi), scale: str, notes: [str]}
KPI_TRUST       the trust dict used (copied in so the report is self-describing)

Coordinate frames (workflow hazard 1): ``patches_batch.y0`` and everything painted by ``paint_crack_voids`` /
``outline_bright`` are in the TRIMMED BSE frame (raw row = y0 + images.band_top of the BSE image). The ml cache's
``novelty_per_patch.y0`` is already in the ORIGINAL frame, so ``novelty_overlay`` is applied to the untrimmed strip.
"""
from __future__ import annotations

import base64
import datetime as _dt
from dataclasses import asdict
import hashlib
import html as _html
import io
import json
import os
import subprocess
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.measure import label, regionprops_table  # noqa: E402

from . import BATCH_COLORS, CRACKED_SITES, GREY_PORE_SITES, LOW_CONTRAST_SITES, NM_PER_PX, PRIMARY_KPIS  # noqa: E402
from . import decision, physics, stats  # noqa: E402

__all__ = ["KPI_TRUST", "TRUSTED_KPIS", "build_result", "render_report", "paint_crack_voids", "outline_bright",
           "novelty_overlay", "crack_void_mask", "fig_to_b64", "array_to_b64", "derive_flags", "DEFAULT_CONFIG"]

# ---------------------------------------------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------------------------------------------
#: Trust level per KPI, transcribed from docs/problem_and_findings.md §4 ("KPI catalogue").
#: TO BE MOVED to polaron_qc/__init__.py once the owner is assigned (workflow.md contract row "(owner to assign)").
KPI_TRUST: dict[str, str] = {
    "pore_frac": "high (grey-pore group: fallback threshold)", "pore_d50": "high", "pore_d90": "high", "pore_elong": "high",
    "pore_max_d": "high", "pore_count_per_Mpx": "high",
    "crack_frac": "high; most defect-relevant", "crack_count_per_Mpx": "high; most defect-relevant",
    "bright_frac": "high when not low-contrast", "bright_count_per_Mpx": "high when not low-contrast",
    "bright_d10": "high when not low-contrast", "bright_d50": "high when not low-contrast", "bright_d90": "high when not low-contrast",
    "bright_circ": "high when not low-contrast", "bright_solidity": "high when not low-contrast", "bright_max_d": "high when not low-contrast",
    "fft_slope": "medium", "corr_len_px": "medium",
    "etd_crack_density_particles": "high; null on this data",
    "etd_crack_density_graphite": "flag", "etd_curtain_frac": "flag", "etd_curtain_anisotropy": "flag", "etd_boundary_sharpness": "flag",
    "inlens_particle_texture": "confounded", "inlens_particle_texture_p90": "confounded", "inlens_speckled_particle_frac": "confounded",
    "bright_sep": "diagnostic", "bright_low_contrast": "diagnostic", "pore_mode_resolved": "diagnostic", "graphite_mode": "diagnostic",
    "th_lo": "diagnostic", "th_hi": "diagnostic", "H": "diagnostic (session fingerprint)", "graphite_frac": "derived",
}
for _k in range(10):
    KPI_TRUST[f"profile_pore_{_k}"] = "medium (orientation unknown)"
    KPI_TRUST[f"profile_bright_{_k}"] = "medium (orientation unknown)"

#: Material KPIs the comparison table reports (trust high / medium; no flags, no confounded, no diagnostics — C21).
TRUSTED_KPIS = ["crack_frac", "pore_max_d", "pore_frac", "bright_frac", "bright_d50",
                "crack_count_per_Mpx", "pore_d50", "pore_d90", "pore_elong", "pore_count_per_Mpx",
                "bright_count_per_Mpx", "bright_d10", "bright_d90", "bright_circ", "bright_solidity", "bright_max_d",
                "corr_len_px", "fft_slope", "etd_crack_density_particles"]

PALETTE = dict(BATCH_COLORS)
PALETTE.update({"grey_pore": "#eda100", "low_contrast": "#e34948", "void": "#e34948", "outline": "#ffd166"})
_VOID_RGB = (227, 73, 72)       # #e34948
_OUTLINE_RGB = (255, 209, 102)  # #ffd166 (annotation colour of the assumption register)

DEFAULT_CONFIG = dict(
    alpha=0.05, power=0.80, seed=0, statistic="hl_shift", n_boot=2000, n_mc=20_000, energy_n_mc=5000,
    mdc_n_sim=400, mdc_n_mc=999, jackknife_n_boot=500, jackknife_energy_n_mc=2000,
    primary=list(PRIMARY_KPIS), thresholds={}, stretch_frac=0.2, black_level_p1=10, patch=512,
    provenance=decision.Thresholds().provenance,
)

_ML_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_cache", "ml")
_PHYSICS_CSV_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_cache", "physics_sites.csv")


# ---------------------------------------------------------------------------------------------------------------
# evidence helpers (importable separately)
# ---------------------------------------------------------------------------------------------------------------
def fig_to_b64(fig, width_px: int = 1100, quality: int = 82) -> str:
    """Matplotlib figure → ``data:image/jpeg;base64,...`` at most ``width_px`` wide (closes the figure)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return array_to_b64(np.asarray(Image.open(buf).convert("RGB")), width_px, quality)


def array_to_b64(rgb: np.ndarray, width_px: int = 1100, quality: int = 82) -> str:
    """uint8 RGB (or grey) array → JPEG data URI, downsampled with LANCZOS to at most ``width_px`` wide."""
    im = Image.fromarray(np.asarray(rgb).astype(np.uint8))
    if im.mode != "RGB":
        im = im.convert("RGB")
    if im.width > width_px:
        im = im.resize((width_px, max(1, int(im.height * width_px / im.width))), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray)
    if g.ndim == 3:
        return g.astype(np.uint8).copy()
    return np.repeat(g.astype(np.uint8)[..., None], 3, axis=2)


def crack_void_mask(pore_mask: np.ndarray, major_axis_px: float = 500, min_area_px: int = 30) -> np.ndarray:
    """Boolean mask of the pore components whose major axis exceeds ``major_axis_px`` (features' crack-like void
    definition: components with area >= 30 px and major_axis_length > 500 px)."""
    lab = label(np.asarray(pore_mask, bool))
    if lab.max() == 0:
        return np.zeros(lab.shape, bool)
    rp = regionprops_table(lab, properties=("label", "area", "major_axis_length"))
    keep = rp["label"][(rp["major_axis_length"] > major_axis_px) & (rp["area"] >= min_area_px)]
    return np.isin(lab, keep)


def paint_crack_voids(bse: np.ndarray, th_lo: float, th_hi: float, major_axis_px: float = 500,
                      color=_VOID_RGB) -> np.ndarray:
    """RGB copy of a (TRIMMED) uint8 BSE image with crack-like voids (pore components, major axis > ``major_axis_px``)
    painted ``color`` (#e34948 by default). Segmentation = features.segment at the site thresholds, so the painted
    pixels are exactly the pixels counted in ``crack_frac``. Coordinates are those of the array passed in: pass the
    trimmed image (features.trimmed) to match the site KPIs, and say so in the caption."""
    from .features import segment
    pore, _, _ = segment(np.asarray(bse, np.uint8), th_lo, th_hi)
    rgb = _gray_to_rgb(bse)
    rgb[crack_void_mask(pore, major_axis_px)] = color
    return rgb


def outline_bright(bse: np.ndarray, th_lo: float, th_hi: float, min_area_px: int = 50, width: int = 2,
                   color=_OUTLINE_RGB) -> np.ndarray:
    """RGB copy of a uint8 BSE image with the outline of every bright-phase particle (area >= ``min_area_px``, the
    features floor) drawn in ``color``. Same frame as the input array."""
    from .features import segment
    _, bright, _ = segment(np.asarray(bse, np.uint8), th_lo, th_hi)
    lab = label(bright)
    if lab.max():
        rp = regionprops_table(lab, properties=("label", "area"))
        bright = np.isin(lab, rp["label"][rp["area"] >= min_area_px])
    edge = bright & ~ndi.binary_erosion(bright, iterations=width)
    rgb = _gray_to_rgb(bse)
    rgb[edge] = color
    return rgb


def novelty_overlay(bse: np.ndarray, per_patch_df: pd.DataFrame, patch: int = 512, alpha: float = 0.45,
                    vmin: float | None = None, vmax: float | None = None, cmap: str = "inferno") -> np.ndarray:
    """Semi-transparent heat overlay of per-patch novelty on a BSE strip.

    ``per_patch_df`` needs columns y0, x0, novelty. The y0/x0 frame is whatever the caller's table uses: the ml cache
    (``analysis_cache/ml/novelty_per_patch.csv``, written by ml.patch_embeddings) stores y0 in the ORIGINAL image
    frame (band_top added back), so pass the untrimmed strip. Patches are ``patch`` px squares; colour = ``cmap``
    scaled between ``vmin``/``vmax`` (default: min/max of the supplied novelty). Returns uint8 RGB.
    """
    rgb = _gray_to_rgb(bse).astype(float)
    df = per_patch_df.dropna(subset=["novelty"])
    if len(df) == 0:
        return rgb.astype(np.uint8)
    lo = float(df.novelty.min()) if vmin is None else float(vmin)
    hi = float(df.novelty.max()) if vmax is None else float(vmax)
    span = (hi - lo) if hi > lo else 1.0
    mapper = plt.get_cmap(cmap)
    H, W = rgb.shape[:2]
    for y0, x0, nv in zip(df.y0.astype(int), df.x0.astype(int), df.novelty.astype(float)):
        if y0 >= H or x0 >= W or y0 < 0 or x0 < 0:
            continue
        col = np.array(mapper((nv - lo) / span)[:3]) * 255.0
        win = (slice(y0, min(y0 + patch, H)), slice(x0, min(x0 + patch, W)))
        rgb[win] = (1 - alpha) * rgb[win] + alpha * col
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _downsample(a: np.ndarray, f: int) -> np.ndarray:
    return a[::f, ::f] if f > 1 else a


def _crop(img: np.ndarray, cy: int, cx: int, h: int, w: int):
    H, W = img.shape[:2]
    h, w = min(h, H), min(w, W)
    y0 = int(np.clip(cy - h // 2, 0, H - h)); x0 = int(np.clip(cx - w // 2, 0, W - w))
    return img[y0:y0 + h, x0:x0 + w], (y0, x0)


# ---------------------------------------------------------------------------------------------------------------
# provisional acquisition flags (the acquisition module will own this)
# ---------------------------------------------------------------------------------------------------------------
def derive_flags(sites: pd.DataFrame, images: pd.DataFrame, stretch_frac: float = 0.2, black_level_p1: float = 10) -> pd.DataFrame:
    """PROVISIONAL per-site acquisition flags from the features tables (workflow.md contract row features → stats:
    contrast_stretched = empty_bin_frac > 0.2, raised_black_level = BSE p1 > 10), plus the site-level flags.
    One row per site. Replace with polaron_qc.acquisition when it lands."""
    bse = images[images.det == "BSE"].set_index("site")
    anyd = images.groupby("site").empty_bin_frac.max()
    rows = []
    for _, r in sites.iterrows():
        s = r.site
        b = bse.loc[s] if s in bse.index else None
        lc = bool(r.get("bright_low_contrast", False)) or s in LOW_CONTRAST_SITES
        gp = bool(r.get("grey_pore", False)) or s in GREY_PORE_SITES
        cr = s in CRACKED_SITES
        rows.append(dict(
            batch=r.batch, site=s,
            contrast_stretched_bse=bool(b is not None and b.empty_bin_frac > stretch_frac),
            contrast_stretched_any=bool(anyd.get(s, 0) > stretch_frac),
            bse_empty_bin_frac=float(b.empty_bin_frac) if b is not None else np.nan,
            bse_p1=float(b.p1) if b is not None else np.nan,
            raised_black_level=bool(b is not None and b.p1 > black_level_p1),
            bright_low_contrast=lc, grey_pore=gp, cracked=cr,
            bright_sep=float(r.get("bright_sep", np.nan)),
            band_top=int(b.band_top) if b is not None else 0, band_bottom=int(b.band_bottom) if b is not None else 0,
            H=int(r.H) if "H" in r else (int(b.H) if b is not None else -1),
            acquisition_group="low_contrast" if lc else ("grey_pore" if gp else ("cracked" if cr else "ordinary")),
        ))
    out = pd.DataFrame(rows)
    out.attrs["status"] = "provisional (derived in polaron_qc.report; polaron_qc.acquisition will own these flags)"
    return out


# ---------------------------------------------------------------------------------------------------------------
# build_result
# ---------------------------------------------------------------------------------------------------------------
def _config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _git_describe(root: str) -> str | None:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=root, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def _mask_unusable(df: pd.DataFrame, kpis, flags=None) -> pd.DataFrame:
    """Copy of ``df`` with each KPI set to NaN on the sites stats.usable_n excludes for it (bright_* on low-contrast
    sites). Grey-pore 'fallback' sites are kept. Used before the multivariate steps so n is the usable n."""
    out = df.copy()
    for k in kpis:
        u = stats.usable_n(df, k, flags)
        if u["excluded_sites"]:
            out.loc[out.site.astype(str).isin(u["excluded_sites"]), k] = np.nan
    return out


def _primary_matrix(df: pd.DataFrame, kpis) -> pd.DataFrame:
    return df.set_index("site")[list(kpis)].astype(float)


def _run_pipeline(ref_sites, batch_sites, ordinary_sites, cfg, th, primary, c2st=None, n_boot=None, energy_n_mc=None,
                  kpis=None, local_extra=True):
    """compare → energy → local → check A/B → abstention → verdict. Used for the full run and inside the jackknife."""
    kpis = list(kpis) if kpis is not None else list(primary)
    compare = stats.compare_kpis(ref_sites, batch_sites, kpis=kpis, primary=primary, seed=cfg["seed"],
                                 n_boot=cfg["n_boot"] if n_boot is None else n_boot, n_mc=cfg["n_mc"], alpha=cfg["alpha"],
                                 statistic=cfg["statistic"])
    w = physics.weights_vector(primary)
    Xr = _primary_matrix(_mask_unusable(ref_sites, primary), primary)
    Xb = _primary_matrix(_mask_unusable(batch_sites, primary), primary)
    energy = stats.energy_distance_test(Xr, Xb, weights=w, n_mc=cfg["energy_n_mc"] if energy_n_mc is None else energy_n_mc, seed=cfg["seed"])
    ord_mask = ref_sites.site.astype(str).isin(ordinary_sites)
    local = {}
    local_kpis = list(primary) + (["bright_max_d"] if local_extra and "bright_max_d" in batch_sites.columns else [])
    for k in local_kpis:
        rv = stats.usable_values(ref_sites[ord_mask.values], k)
        bv = stats.usable_values(batch_sites, k)
        local[k] = stats.local_exceedance(rv, bv, margin_mad=th.severity_margin_mad)
    a = decision.check_a(compare, energy, ref_sites, batch_sites, ord_mask, c2st=c2st, th=th, primary=primary)
    b = decision.check_b({k: v["table"] for k, v in local.items()}, batch_sites, image_reviewed=None, th=th, primary=primary)
    abst = decision.quality_abstention(batch_sites, a, th)
    verdict = decision.decide(a, b, abst, att=None, th=th)
    return dict(compare=compare, energy=energy, local=local, check_a=a, check_b=b, abstention=abst, verdict=verdict,
                X_ref=Xr, X_batch=Xb, weights=w)


def _read_ml(ml_dir, batch, reference, notes):
    """c2st + novelty tables from the ml cache for this pair; None (with a note) when absent."""
    out = dict(c2st_material=None, c2st_flag_inclusive=None, c2st_extra={}, novelty=None)
    p = os.path.join(ml_dir, "c2st_results.json")
    if os.path.exists(p):
        with open(p) as f:
            allres = json.load(f)
        prefix = f"{batch}_vs_{reference}_"
        for key, v in allres.items():
            if not key.startswith(prefix):
                continue
            tag = key[len(prefix):]
            coef = pd.DataFrame(v.get("coef", []))
            cp = os.path.join(ml_dir, f"c2st_{key}_coef.csv")
            if os.path.exists(cp):
                coef = pd.read_csv(cp)
            sp = os.path.join(ml_dir, f"c2st_{key}_sites.csv")
            per_site = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame(v.get("per_site_scores", []))
            feats = coef.feature.tolist() if "feature" in coef else []
            flag_inclusive = any(f in feats for f in ("etd_boundary_sharpness", "etd_curtain_frac", "bright_sep"))
            d = dict(auc=v.get("auc"), null_band=v.get("null_band"), p=v.get("p"), n_perm=v.get("n_perm"),
                     p_method=v.get("p_method"), cv=v.get("cv"), n_sites_ref=v.get("n_sites_ref"), n_sites_batch=v.get("n_sites_batch"),
                     coef=coef, per_site_scores=per_site, kpis=feats, variant=tag, key=key, flag_inclusive=flag_inclusive)
            if tag == "all_sites":
                if flag_inclusive:
                    out["c2st_flag_inclusive"] = d
                else:
                    out["c2st_material"] = d
            elif tag in ("material_only", "material"):
                out["c2st_material"] = d
            else:
                out["c2st_extra"][tag] = d
    # material-only runs (MATERIAL_KPIS, D24) are cached separately in c2st_material.json: {"<batch>_vs_<ref>_material_<variant>": {...}}
    pm = os.path.join(ml_dir, "c2st_material.json")
    if os.path.exists(pm):
        with open(pm) as f:
            mat = json.load(f)
        for variant in ("all_sites", "usable_sites"):
            key = f"{batch}_vs_{reference}_material_{variant}"
            if key in mat:
                v = mat[key]
                d = dict(auc=v.get("auc"), null_band=[v.get("null_lo"), v.get("null_hi")], p=v.get("p"), n_perm=v.get("n_perm"),
                         p_method="Monte Carlo site-label permutation", cv="StratifiedGroupKFold", n_sites_ref=v.get("n_ref"), n_sites_batch=v.get("n_batch"),
                         coef=pd.DataFrame([dict(feature=f_, coef=c_, sign=("+" if c_ > 0 else "−"), selected=True, selection_stability=float("nan"))
                                            for f_, c_ in v.get("selected", [])], columns=["feature", "coef", "sign", "selected", "selection_stability"]),
                         per_site_scores=pd.DataFrame(), kpis=v.get("features", []), variant=f"material_{variant}", key=key, flag_inclusive=False)
                if variant == "all_sites" or out["c2st_material"] is None:
                    out["c2st_material"] = d
                out["c2st_extra"][f"material_{variant}"] = d
    if os.path.exists(p) or os.path.exists(pm):
        if out["c2st_material"] is None:
            notes.append("ml cache: no material-only c2st run for this pair; Check A(iv) ran with c2st=None (reject via Check A is then impossible)")
    else:
        notes.append(f"ml cache: {p} not found; c2st sections empty")
    ps, pp, pc = (os.path.join(ml_dir, f) for f in ("novelty_per_site.csv", "novelty_per_patch.csv", "novelty_correlates.csv"))
    if os.path.exists(ps) and os.path.exists(pp):
        per_site_all = pd.read_csv(ps)
        per_site = per_site_all[per_site_all.batch == batch].copy()
        ref_loo = per_site_all[per_site_all.batch == reference].copy()
        per_patch = pd.read_csv(pp)
        per_patch = per_patch[per_patch.batch == batch].copy()
        corr = pd.read_csv(pc) if os.path.exists(pc) else None
        if len(per_site) == 0:
            notes.append(f"ml cache: novelty tables contain no rows for {batch}; novelty section empty")
            out["novelty"] = None
        else:
            n_ref = int(len(ref_loo)); n_b = int(len(per_site))
            out["novelty"] = dict(per_site=per_site, ref_loo=ref_loo, per_patch=per_patch, correlates=corr,
                                  iid_rate=stats.iid_flag_probability(n_ref, n_b) if n_ref + n_b else np.nan,
                                  top_site=str(per_site.sort_values("novelty_median").site.iloc[-1]),
                                  frame="original image (ml.patch_embeddings adds band_top back to y0)")
    else:
        notes.append("ml cache: novelty tables not found; novelty section empty")
    return out


def _read_physics(physics_csv, sites_ref, sites_batch, compare, drivers, primary, notes):
    """physics_sites.csv rows for both batches + qualitative statements per driver KPI with the band gate."""
    ps = None
    if os.path.exists(physics_csv):
        ps = pd.read_csv(physics_csv)
        ps = ps[ps.batch.isin([sites_ref.batch.iloc[0], sites_batch.batch.iloc[0]])].copy()
        have_band = ps.pore_frac_band.notna().sum() if "pore_frac_band" in ps else 0
        if have_band < len(ps):
            notes.append(f"physics_sites.csv: ±5-level threshold band available for {have_band} of {len(ps)} sites of this pair "
                         "(features.threshold_band(site) for every site is still 'to add' in workflow.md)")
    else:
        notes.append(f"physics cache {physics_csv} not found")
    cmp_ = compare.set_index("kpi")
    statements = {}
    for k in primary:
        if k not in cmp_.index:
            continue
        r = cmp_.loc[k]
        st = physics.qualitative_statement(k, r.shift_mad, r.direction if r.direction != "none" else None)
        band_col = {"pore_frac": "pore_frac_band", "bright_frac": "bright_frac_band"}.get(k)
        band_abs, band_available, gated, note = np.nan, False, True, ""
        delta_abs = float(abs(r.batch_median - r.ref_median)) if np.isfinite(r.batch_median) and np.isfinite(r.ref_median) else np.nan
        if ps is not None and band_col and band_col in ps and ps[band_col].notna().any():
            band_abs = float(np.nanmedian(ps[band_col]))  # band = max-min of the fraction under th ± 5 levels
            band_available = True
            gated = bool(delta_abs > band_abs / 2)        # compare |Δ median| with the band half-width
            note = (f"±5-level band half-width (median of {int(ps[band_col].notna().sum())} sites with a band) = {band_abs / 2:.4f}; "
                    f"|Δ median| = {delta_abs:.4f} → physics reading {'printed' if gated else 'withheld (shift inside the band)'}")
        else:
            note = "±5-level threshold band not available for this KPI/these sites; statement printed ungated"
        statements[k] = dict(statement=st, gated=gated, band_available=band_available, band_abs=band_abs, delta_abs=delta_abs,
                             note=note, is_driver=k in drivers)
    sanity = physics.sanity_checks(sites_batch)
    fc_note = None
    if ps is not None and "tortuosity_fraction_connected" in ps:
        fc = ps.tortuosity_fraction_connected
        fc_note = (f"No 2-D top-to-bottom macro-pore path on any of the {int(fc.notna().sum())} sites of this pair "
                   f"(fraction_connected = {float(fc.max()) if fc.notna().any() else float('nan'):.0f}); the section tortuosity index is undefined and is not reported.")
    return dict(sites=ps, statements=statements, weights={k: int(physics.CONSEQUENCE_WEIGHTS.get(k, 1)) for k in primary},
                rationale={k: physics.CONSEQUENCE_RATIONALE.get(k, "") for k in primary},
                labels={k: physics.KPI_LABELS.get(k, k) for k in TRUSTED_KPIS}, sanity=sanity, fraction_connected_note=fc_note,
                caveats=list(physics.GLOBAL_CAVEATS))


def build_result(reference_batch_dir: str, batch_dir: str, config: dict | None = None,
                 cache_dir: str = "analysis_cache/features", ml_dir: str | None = None, physics_csv: str | None = None,
                 verbose: bool = True) -> dict:
    """Assemble the ``result`` dict documented in the module docstring for one incoming batch against one reference.

    Everything reference-dependent is computed inside stats (permutations, bootstrap, jackknife). Runtime with a warm
    feature cache ≈ 1–2 min (MDC simulation and the 7 leave-one-site-out re-runs dominate); cold extraction of a
    17-site reference adds ~2–3 min.
    """
    from . import features
    t_start = time.time()
    cfg = dict(DEFAULT_CONFIG); cfg.update(config or {})
    th = decision.Thresholds(**cfg.get("thresholds", {}))
    th.alpha = cfg["alpha"]
    primary = list(cfg["primary"])
    notes: list[str] = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ml_dir = ml_dir or _ML_DIR_DEFAULT
    physics_csv = physics_csv or _PHYSICS_CSV_DEFAULT
    log = (lambda *a: print("[report]", *a, flush=True)) if verbose else (lambda *a: None)

    # 1. features ------------------------------------------------------------------------------------------
    ref_t = features.extract_batch(reference_batch_dir, cache_dir=cache_dir, verbose=verbose)
    bat_t = features.extract_batch(batch_dir, cache_dir=cache_dir, verbose=verbose)
    sites_ref, sites_batch = ref_t["sites"].reset_index(drop=True), bat_t["sites"].reset_index(drop=True)
    reference, batch = str(sites_ref.batch.iloc[0]), str(sites_batch.batch.iloc[0])
    for df in (sites_ref, sites_batch):
        df["site"] = df["site"].astype(str)
    flags_ref = derive_flags(sites_ref, ref_t["images"], cfg["stretch_frac"], cfg["black_level_p1"])
    flags_batch = derive_flags(sites_batch, bat_t["images"], cfg["stretch_frac"], cfg["black_level_p1"])
    ordinary = [s for s in sites_ref.site if s not in GREY_PORE_SITES and s not in CRACKED_SITES]
    kpis = [k for k in TRUSTED_KPIS if k in sites_batch.columns and k in sites_ref.columns]
    missing = [k for k in TRUSTED_KPIS if k not in kpis]
    if missing:
        notes.append(f"trusted KPIs missing from the site tables: {missing}")
    log(f"{batch} vs {reference}: {len(sites_batch)} vs {len(sites_ref)} sites; ordinary reference = {len(ordinary)}")

    # 2. compare / energy / local / decision -------------------------------------------------------------
    t0 = time.time()
    full = _run_pipeline(sites_ref, sites_batch, ordinary, cfg, th, primary, c2st=None, kpis=kpis)
    log(f"compare+energy+decision in {time.time() - t0:.0f}s → {full['verdict']['verdict']}")
    compare = full["compare"]
    nan_ci = compare[compare.is_primary & compare.ci_low.isna()].kpi.tolist()
    if nan_ci:
        notes.append(f"bootstrap CI is NaN for primary KPIs {nan_ci}")

    # 3. drift (per-site) -------------------------------------------------------------------------------
    Xr, Xb = full["X_ref"].dropna(), full["X_batch"].dropna()
    drift = stats.per_site_drift(Xr, Xb, weights=full["weights"])
    not_scored = sorted(set(full["X_batch"].index) - set(Xb.index))
    if not_scored:
        notes.append(f"per-site drift score not computed for {not_scored} (bright-phase KPIs unusable: low-contrast sites)")
    drift.attrs["not_scored"] = not_scored

    # 4. MDC per primary KPI -------------------------------------------------------------------------------
    t0 = time.time()
    mdc = {}
    cmp_i = compare.set_index("kpi")
    for k in primary:
        n_in = int(cmp_i.loc[k, "n_batch_usable"])
        xr = stats.usable_values(sites_ref, k).to_numpy()
        mdc[k] = stats.mdc(xr, n_incoming=n_in, alpha=cfg["alpha"], power=cfg["power"], n_sim=cfg["mdc_n_sim"],
                           test=dict(statistic=cfg["statistic"], n_mc=cfg["mdc_n_mc"]), seed=cfg["seed"])
    log(f"MDC ({cfg['mdc_n_sim']} sims × 5 KPIs) in {time.time() - t0:.0f}s: " +
        ", ".join(f"{k} {v['mdc_mad']:.2f}" for k, v in mdc.items()))

    # 5. patch-level local evidence (per-site maxima of patch KPIs; TRIMMED frame) -------------------------
    local = dict(full["local"])
    pr, pb = ref_t["patches"], bat_t["patches"]
    for pk in ("crack_area_frac", "pore_max_d"):
        if pk in pb.columns:
            rmax = pr[pr.site.astype(str).isin(ordinary)].groupby("site")[pk].max()
            bmax = pb.groupby("site")[pk].max()
            local[f"patch_{pk}"] = stats.local_exceedance(rmax, bmax, margin_mad=th.severity_margin_mad)
            # remember the extreme patch per batch site for the evidence crop
            idx = pb.groupby("site")[pk].idxmax().dropna()
            local[f"patch_{pk}"]["extreme_patch"] = pb.loc[idx.values, ["site", "patch_id", "y0", "x0", pk]].reset_index(drop=True)

    # 6. stability by leave-one-site-out --------------------------------------------------------------------
    t0 = time.time()

    def _rerun(kept):
        sub = sites_batch[sites_batch.site.isin(kept)].reset_index(drop=True)
        r = _run_pipeline(sites_ref, sub, ordinary, cfg, th, primary, c2st=None, n_boot=cfg["jackknife_n_boot"],
                          energy_n_mc=cfg["jackknife_energy_n_mc"], kpis=primary, local_extra=False)
        return r["verdict"]

    runs = stats.jackknife(_rerun, sites_batch.site.tolist())
    share = stats.stability_share(runs, full["verdict"], key="verdict")
    stability = dict(share=share, n_runs=len(runs), full_verdict=full["verdict"]["verdict"],
                     runs=[dict(left_out=r["left_out"], verdict=r["result"]["verdict"], outcome_columns=r["result"]["outcome_columns"]) for r in runs],
                     method=f"leave-one-site-out re-run of compare → energy → check A/B → decide ({len(runs)} re-verdicts; "
                            f"n_boot={cfg['jackknife_n_boot']}, energy n_mc={cfg['jackknife_energy_n_mc']}); share of runs returning the full-sample verdict")
    log(f"jackknife {len(runs)} runs in {time.time() - t0:.0f}s → stability share {share:.2f}")

    # 7. ml + physics caches ------------------------------------------------------------------------------
    ml = _read_ml(ml_dir, batch, reference, notes)
    phys = _read_physics(physics_csv, sites_ref, sites_batch, compare, full["verdict"]["drivers"], primary, notes)

    # 8. limits -------------------------------------------------------------------------------------------
    prim = compare[compare.is_primary]
    mdcs = [v["mdc_mad"] for v in mdc.values() if np.isfinite(v["mdc_mad"])]
    limits = dict(
        usable_n={r.kpi: (int(r.n_ref_usable), int(r.n_batch_usable)) for r in prim.itertuples()},
        excluded={r.kpi: (int(r.n_ref_excluded), int(r.n_batch_excluded)) for r in prim.itertuples()},
        fallback={r.kpi: (int(r.n_ref_fallback), int(r.n_batch_fallback)) for r in prim.itertuples()},
        specimen_independence="unconfirmed — whether the sites of a batch come from distinct specimens has been asked of the organisers; n counts sites",
        reference_heterogeneity=dict(grey_pore=[s for s in sites_ref.site if s in GREY_PORE_SITES],
                                     cracked=[s for s in sites_ref.site if s in CRACKED_SITES], ordinary_n=len(ordinary)),
        mdc_range_mad=(float(min(mdcs)), float(max(mdcs))) if mdcs else (np.nan, np.nan),
        scale="2-D sections, 25 nm/px nominal (TIFF export tag, unverified), chemistry of the bright phase unconfirmed",
        seven_v_seven="not applicable" if len(sites_ref) != len(sites_batch) else
                      "7 v 7: smallest attainable two-sided exact p ≈ 0.0006; power is low, most comparisons return 'cannot distinguish'",
        notes=["energy distance and drift score drop batch sites whose bright-phase KPIs are unusable (low-contrast)",
               "patch-level numbers localise evidence only and never enter n"],
    )

    cfg_public = {k: v for k, v in cfg.items()}
    cfg_public["thresholds_resolved"] = asdict(th)
    meta = dict(reference=reference, batch=batch, reference_dir=os.path.abspath(reference_batch_dir), batch_dir=os.path.abspath(batch_dir),
                n_sites_ref=int(len(sites_ref)), n_sites_batch=int(len(sites_batch)),
                timestamp=_dt.datetime.now().isoformat(timespec="seconds"), thresholds_hash=th.hash(), provenance=th.provenance,
                config=cfg_public, config_hash=_config_hash(cfg_public), git_describe=_git_describe(root),
                feature_version=features.FEATURE_VERSION, runtime_s=round(time.time() - t_start, 1),
                primary_kpis=primary, trusted_kpis=kpis, notes=notes)
    log(f"done in {meta['runtime_s']}s; {len(notes)} integration notes")
    return dict(meta=meta, sites_ref=sites_ref, sites_batch=sites_batch, images_ref=ref_t["images"], images_batch=bat_t["images"],
                patches_batch=pb, flags_ref=flags_ref, flags_batch=flags_batch, ordinary_ref_sites=ordinary,
                compare=compare, energy=full["energy"], mdc=mdc, drift=drift, local=local,
                check_a=full["check_a"], check_b=full["check_b"], abstention=full["abstention"], verdict=full["verdict"],
                stability=stability, c2st_material=ml["c2st_material"], c2st_flag_inclusive=ml["c2st_flag_inclusive"],
                c2st_extra=ml["c2st_extra"], novelty=ml["novelty"], physics=phys, acquisition=None, limits=limits,
                KPI_TRUST=dict(KPI_TRUST))


# ---------------------------------------------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------------------------------------------
def _e(x) -> str:
    return _html.escape("" if x is None else str(x))


def _f(x, nd=3, pct=False, signed=False) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)) or (hasattr(x, "__float__") and not np.isfinite(float(x))):
            return "—" if x is None or not np.isfinite(float(x)) else ("∞" if float(x) > 0 else "−∞")
        v = float(x)
    except (TypeError, ValueError):
        return _e(x)
    if not np.isfinite(v):
        return "∞" if v > 0 else "−∞"
    if pct:
        return f"{100 * v:.{nd}f} %"
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace("-", "−")


def _p(p, n_perm=None, method=None) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    s = f"{p:.4f}" if p < 0.001 else f"{p:.3f}"
    return s


def _table(rows: list[list], header: list[str], cls: str = "", caption: str | None = None) -> str:
    h = "".join(f"<th>{c}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="tw"><table class="{cls}">{cap}<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def _kpi_unit(k: str) -> str:
    return "px" if k.endswith(("_d", "_d10", "_d50", "_d90", "_px", "_max_d")) else ("per Mpx" if "per_Mpx" in k else "fraction")


def _pct_change(ref_med, bat_med) -> str:
    if not (np.isfinite(ref_med) and np.isfinite(bat_med)) or ref_med == 0:
        return "—"
    return f"{100 * (bat_med - ref_med) / abs(ref_med):+.0f} %".replace("-", "−")


def _ref_stats_without_anomalous(sites_ref: pd.DataFrame, kpi: str) -> tuple[float, float, int]:
    v = stats.usable_values(sites_ref[~sites_ref.site.isin(GREY_PORE_SITES + CRACKED_SITES)], kpi).to_numpy()
    return (float(np.median(v)), float(stats.mad(v)), int(len(v))) if len(v) else (np.nan, np.nan, 0)


_CSS = """
:root{--bg:#fbfaf7;--card:#ffffff;--ink:#141414;--muted:#5b5a56;--line:#e6e3dc;--accent:#2a78d6;--ok:#1baf7a;--warn:#eda100;--bad:#e34948;--chip:#f1efe9;--b1:#2a78d6;--b2:#eb6834;--b3:#1baf7a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--card:#1f1f1d;--ink:#f2f1ed;--muted:#b9b7af;--line:#333330;--chip:#2a2a27}}
:root[data-theme=dark]{--bg:#161615;--card:#1f1f1d;--ink:#f2f1ed;--muted:#b9b7af;--line:#333330;--chip:#2a2a27}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 80px}h1{font-size:26px;margin:0 0 6px}h2{font-size:19px;margin:0 0 10px}h3{font-size:15px;margin:14px 0 6px;color:var(--muted)}
.lede{color:var(--muted);max-width:960px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:0 0 22px;scroll-margin-top:12px}
.verdict{border-left:6px solid var(--accent)}.verdict.v-consistent{border-left-color:var(--ok)}.verdict.v-investigate{border-left-color:var(--warn)}.verdict.v-reject{border-left-color:var(--bad)}
.vtext{font-size:22px;font-weight:700;margin:0 0 4px}.reason{color:var(--muted);margin:0 0 12px}
.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0}.col{background:var(--chip);border-radius:10px;padding:10px 12px}.col .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.col .v{font-size:18px;font-weight:600}
.chip{display:inline-block;padding:2px 10px;border-radius:999px;background:var(--chip);font-size:13px;margin:0 6px 6px 0}
.chip.ok{background:var(--ok);color:#fff}.chip.warn{background:var(--warn);color:#141414}.chip.bad{background:var(--bad);color:#fff}
.tw{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}th{font-weight:600;color:var(--muted);font-size:12.5px}
td.num,th.num{text-align:right}caption{text-align:left;color:var(--muted);font-size:13px;padding:4px 0}tr.primary td{font-weight:500}
figure{margin:12px 0}figure img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);display:block}figcaption{font-size:13px;color:var(--muted);margin-top:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.note{font-size:13px;color:var(--muted)}.small{font-size:13px}
.drv{padding:8px 12px;border-left:3px solid var(--accent);margin:8px 0;background:var(--chip);border-radius:0 8px 8px 0}.drv .phys{display:block;color:var(--muted);font-size:13.5px;margin-top:4px}
.toc{font-size:14px;margin:0 0 18px;padding-left:18px;columns:2;gap:24px}.toc a{color:var(--ink);text-decoration:none}.toc a:hover{color:var(--accent)}
.ph{border:1px dashed var(--line);border-radius:8px;padding:10px 12px;color:var(--muted)}footer{color:var(--muted);font-size:13px;margin-top:20px;border-top:1px solid var(--line);padding-top:12px}
code{font-size:12.5px;background:var(--chip);padding:1px 5px;border-radius:4px}
@media(max-width:760px){.cols,.grid2{grid-template-columns:1fr}.toc{columns:1}}
@media print{.card{break-inside:avoid}}
"""


# ---------------------------------------------------------------------------------------------------------------
# evidence image builders (need the raw images; each returns (b64, caption) or None)
# ---------------------------------------------------------------------------------------------------------------
def _load_site(batch_dir: str, site: str):
    from .features import bright_bands, load_image
    bse = load_image(batch_dir, site, "BSE")
    t, b = bright_bands(bse)
    return bse, bse[t: bse.shape[0] - b if b else None], t, b


def _site_th(sites: pd.DataFrame, site: str) -> tuple[float, float]:
    r = sites[sites.site == site].iloc[0]
    return float(r.th_lo), float(r.th_hi)


def _strip_figure(panels: list[tuple[np.ndarray, str]], ds: int = 4, h_per: float = 2.4) -> str:
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, h_per * len(panels)))
    for ax, (img, title) in zip(np.atleast_1d(axes).ravel(), panels):
        ax.imshow(_downsample(img, ds), aspect="auto"); ax.set_axis_off(); ax.set_title(title, fontsize=9.5, loc="left")
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig)


def _typical_sites_image(res: dict) -> tuple[str, str] | None:
    d = res["drift"]; d = d[d.group == "batch"].sort_values("distance")
    if len(d) == 0:
        return None
    batch_dir, sites = res["meta"]["batch_dir"], res["sites_batch"]
    panels = []
    for site, tag in ((str(d.site.iloc[0]), "most typical"), (str(d.site.iloc[-1]), "least typical")):
        raw, trim, t, b = _load_site(batch_dir, site)
        th_lo, th_hi = _site_th(sites, site)
        rgb = paint_crack_voids(trim, th_lo, th_hi)
        row = d[d.site == site].iloc[0]
        srow = sites[sites.site == site].iloc[0]
        panels.append((rgb, f"{res['meta']['batch']} / {site} — {tag} by drift score (distance {row.distance:.2f}, top driver {row.top_driver} z = {row.top_driver_z:+.1f}); "
                             f"crack_frac {srow.crack_frac:.4f}, pore_max_d {srow.pore_max_d:.0f} px; trimmed frame (band_top = {t}, band_bottom = {b} rows removed)"))
    cap = ("BSE strips, 4× downsampled, crack-like voids (pore components with major axis > 500 px, i.e. the pixels counted in crack_frac) painted "
           "red (#e34948) with features.segment at the site thresholds. Painted in the TRIMMED frame: raw-image row = shown row + band_top. "
           + (f"Sites not scored (bright-phase KPIs unusable): {', '.join(res['drift'].attrs.get('not_scored', []))}." if res["drift"].attrs.get("not_scored") else ""))
    return _strip_figure(panels), cap


def _anomaly_crops(res: dict, max_crops: int = 4) -> list[tuple[str, str]]:
    """One 1200×600 crop per pending/credible localized anomaly, centred on the offending void / particle."""
    b = res["check_b"]
    recs = (b.get("credible", []) + b.get("credible_pending_review", []))[:max_crops]
    out = []
    batch_dir, sites = res["meta"]["batch_dir"], res["sites_batch"]
    for rec in recs:
        site, k = rec["site"], rec["kpi"]
        raw, trim, t, _ = _load_site(batch_dir, site)
        th_lo, th_hi = _site_th(sites, site)
        from .features import segment
        pore, bright, _ = segment(trim, th_lo, th_hi)
        if k in ("crack_frac", "pore_max_d", "crack_count_per_Mpx"):
            mask = crack_void_mask(pore) if k != "pore_max_d" else pore
            lab = label(mask)
            if lab.max() == 0:
                continue
            rp = regionprops_table(lab, properties=("label", "area", "centroid", "major_axis_length"))
            i = int(np.argmax(rp["area"] if k != "crack_frac" else rp["major_axis_length"]))
            cy, cx = int(rp["centroid-0"][i]), int(rp["centroid-1"][i])
            crop, (y0, x0) = _crop(trim, cy, cx, 600, 1200)
            rgb = paint_crack_voids(crop, th_lo, th_hi) if k != "pore_max_d" else _gray_to_rgb(crop)
            if k == "pore_max_d":
                sub = (lab[y0:y0 + 600, x0:x0 + 1200] == rp["label"][i]); rgb[sub & ~ndi.binary_erosion(sub, iterations=3)] = _VOID_RGB
            what = f"largest {'crack-like void' if k != 'pore_max_d' else 'pore component'} (area {rp['area'][i]:.0f} px², major axis {rp['major_axis_length'][i]:.0f} px)"
        else:
            lab = label(bright)
            if lab.max() == 0:
                continue
            rp = regionprops_table(lab, properties=("label", "area", "centroid"))
            i = int(np.argmax(rp["area"]))
            cy, cx = int(rp["centroid-0"][i]), int(rp["centroid-1"][i])
            crop, (y0, x0) = _crop(trim, cy, cx, 600, 1200)
            rgb = outline_bright(crop, th_lo, th_hi)
            what = f"largest bright-phase particle (area {rp['area'][i]:.0f} px²)"
        fig, ax = plt.subplots(figsize=(11, 5.6)); ax.imshow(rgb); ax.set_axis_off()
        ax.set_title(f"{res['meta']['batch']} / {site} — {k} = {rec['value']:.4g} vs ordinary-reference max {rec['ref_max']:.4g} "
                     f"(margin {rec['margin_in_mad']:.1f} MAD); image review pending", fontsize=10, loc="left")
        out.append((fig_to_b64(fig), f"1200 × 600 px crop at trimmed-frame rows {y0}–{y0 + 600}, cols {x0}–{x0 + 1200} (raw row = +{t}); {what}. "
                                     f"Measurement note: {rec['measurement_note']}. This crop is the evidence the reviewer must confirm before the flag becomes 'credible'."))
    return out


def _novelty_image(res: dict) -> tuple[str, str] | None:
    nv = res.get("novelty")
    if not nv or nv["per_patch"] is None or len(nv["per_patch"]) == 0:
        return None
    site = nv["top_site"]
    pp = nv["per_patch"][nv["per_patch"].site == site]
    if len(pp) == 0:
        return None
    raw, trim, t, _ = _load_site(res["meta"]["batch_dir"], site)
    allpp = nv["per_patch"]
    vmin, vmax = float(allpp.novelty.min()), float(allpp.novelty.max())
    rgb = novelty_overlay(raw, pp, patch=res["meta"]["config"].get("patch", 512), vmin=vmin, vmax=vmax)
    ps = nv["per_site"].set_index("site").loc[site]
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.imshow(_downsample(rgb, 4), aspect="auto"); ax.set_axis_off()
    ax.set_title(f"{res['meta']['batch']} / {site} — highest per-site novelty median in the batch ({ps.novelty_median:.2f}, "
                 f"percentile {ps.pct_median:.0f} of the reference leave-one-site-out distribution; exceeds ref max: {bool(ps.exceeds_ref_loo_max)})", fontsize=9.5, loc="left")
    sm = cm.ScalarMappable(cmap="inferno", norm=matplotlib.colors.Normalize(vmin, vmax)); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01).set_label("patch novelty (kNN distance)", fontsize=8)
    cap = (f"Per-patch novelty from analysis_cache/ml/novelty_per_patch.csv drawn as a semi-transparent heat overlay on the UNTRIMMED BSE strip "
           f"(4× downsampled). Frame: the ml cache stores y0 in the ORIGINAL image frame (ml.patch_embeddings adds band_top = {t} back), "
           f"so no offset was applied. Colour scale spans the batch's patch range {vmin:.1f}–{vmax:.1f}. Exploratory: novelty tracks acquisition "
           f"(see §6), it is evidence of *where* the strip differs, not a material verdict.")
    return fig_to_b64(fig), cap


def _particles_image(res: dict) -> tuple[str, str] | None:
    d = res["drift"]; d = d[d.group == "batch"].sort_values("distance")
    sites = res["sites_batch"]
    cands = [s for s in d.site.astype(str) if not bool(sites.set_index("site").loc[s, "bright_low_contrast"])]
    if not cands:
        cands = d.site.astype(str).tolist()
    if not cands:
        return None
    site = cands[0]
    raw, trim, t, _ = _load_site(res["meta"]["batch_dir"], site)
    th_lo, th_hi = _site_th(sites, site)
    H, W = trim.shape
    # the 1200 x 600 window with the most bright-phase area (box filter on an 8x-downsampled bright mask)
    from .features import segment
    _, bright, _ = segment(trim, th_lo, th_hi)
    dens = ndi.uniform_filter(bright[::8, ::8].astype(float), size=(600 // 8, 1200 // 8), mode="constant")
    cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
    crop, (y0, x0) = _crop(trim, int(cy) * 8, int(cx) * 8, 600, 1200)
    rgb = outline_bright(crop, th_lo, th_hi)
    srow = sites.set_index("site").loc[site]
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.imshow(rgb); ax.set_axis_off()
    ax.set_title(f"{res['meta']['batch']} / {site} — bright-phase particles outlined (th_hi = {th_hi:.0f}, area ≥ 50 px); "
                 f"site bright_frac {srow.bright_frac:.3f}, D50 {srow.bright_d50:.0f} px (≈ {srow.bright_d50 * NM_PER_PX / 1000:.1f} µm nominal)", fontsize=10, loc="left")
    return fig_to_b64(fig), (f"1200 × 600 px centre crop (trimmed-frame rows {y0}–{y0 + 600}, cols {x0}–{x0 + 1200}; raw row = +{t}) of the most typical "
                             f"site without a low-contrast flag, placed on the window with the most bright-phase area. Outline colour #ffd166. The outlined pixels are the pixels counted in bright_frac / bright_d50.")


def _primary_strip_plot(res: dict) -> str:
    """Five panels, one y-axis each: reference sites (grey-pore orange, cracked marked), batch sites, medians as bars."""
    primary = res["meta"]["primary_kpis"]; ref, bat = res["sites_ref"], res["sites_batch"]
    rb, bb = res["meta"]["reference"], res["meta"]["batch"]
    cr, cb = PALETTE.get(rb, "#1baf7a"), PALETTE.get(bb, "#2a78d6")
    fig, axes = plt.subplots(1, len(primary), figsize=(11, 3.3))
    rng = np.random.default_rng(0)
    for ax, k in zip(np.atleast_1d(axes), primary):
        rv = stats.usable_values(ref, k); bv = stats.usable_values(bat, k)
        excl_b = set(stats.usable_n(bat, k)["excluded_sites"])
        for s, v in rv.items():
            c = PALETTE["grey_pore"] if s in GREY_PORE_SITES else cr
            ax.scatter(0 + rng.uniform(-0.12, 0.12), v, s=26, color=c, edgecolor="none", alpha=0.9, zorder=3)
            if s in CRACKED_SITES:
                ax.scatter(0, v, s=70, facecolor="none", edgecolor=cr, linewidth=1.0, zorder=2)
        for s, v in bv.items():
            ax.scatter(1 + rng.uniform(-0.12, 0.12), v, s=26, color=cb, edgecolor="none", alpha=0.9, zorder=3)
        for s in excl_b:
            v = float(bat.set_index("site").loc[s, k])
            ax.scatter(1 + rng.uniform(-0.12, 0.12), v, s=60, facecolor="none", edgecolor=PALETTE["low_contrast"], linewidth=1.4, zorder=2)
        for x, vals, c in ((0, rv.to_numpy(), cr), (1, bv.to_numpy(), cb)):
            if len(vals):
                ax.hlines(np.median(vals), x - 0.22, x + 0.22, color=c, linewidth=2.2, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"{rb}\n(n = {len(rv)})", f"{bb}\n(n = {len(bv)})"], fontsize=8)
        ax.set_title(f"{k} [{_kpi_unit(k)}]", fontsize=9.5); ax.tick_params(axis="y", labelsize=8); ax.set_xlim(-0.5, 1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", color=cr, label=f"{rb} site"),
         plt.Line2D([], [], marker="o", ls="", color=PALETTE["grey_pore"], label="grey-pore group (fallback threshold)"),
         plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", color=cr, label="cracked reference site (ring)"),
         plt.Line2D([], [], marker="o", ls="", color=cb, label=f"{bb} site"),
         plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", color=PALETTE["low_contrast"], label="low-contrast (excluded from bright-phase n)"),
         plt.Line2D([], [], color="#777", linewidth=2.2, label="median")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------------------------------------------
def _verdict_class(v: str) -> str:
    if v.startswith("consistent"):
        return "v-consistent"
    if v.startswith("reject"):
        return "v-reject"
    return "v-investigate"


def _sec(id_: str, title: str, body: str, extra_cls: str = "") -> str:
    return f'<section class="card {extra_cls}" id="{id_}"><h2>{title}</h2>{body}</section>'


def render_report(result: dict, out_path: str, with_images: bool = True, width_px: int = 1100) -> str:
    """Write the self-contained HTML report for one ``result`` (see module docstring for the schema) and return the path.
    Sections follow docs/qc_plan.md §2.8 / §2.0 rule 4 (MDC on the first screen). Evidence images need the raw
    Dataset folder (``meta.batch_dir``); when it is missing or ``with_images`` is False they are replaced by a note."""
    m, v, a, b, abst = result["meta"], result["verdict"], result["check_a"], result["check_b"], result["abstention"]
    st, cmp_, mdc, primary = result["stability"], result["compare"], result["mdc"], list(m["primary_kpis"])
    ref, bat = m["reference"], m["batch"]
    sites_ref, sites_batch = result["sites_ref"], result["sites_batch"]
    trust = result.get("KPI_TRUST", KPI_TRUST)
    phys = result.get("physics") or {}
    weights = (phys.get("weights") or {k: int(physics.CONSEQUENCE_WEIGHTS.get(k, 1)) for k in primary})
    labels = phys.get("labels") or {}
    def lab(k): return labels.get(k) or physics.KPI_LABELS.get(k, k)
    cmp_i = cmp_.set_index("kpi")
    oc = v["outcome_columns"]
    parts = []

    # ---- 1. verdict card ------------------------------------------------------------------------------------
    loc_word = {"none": "none", "flag": "evidence flag only", "pending_review": "pending image review", "credible": "credible"}.get(oc["localized"], oc["localized"])
    cols = (f'<div class="cols"><div class="col"><div class="k">Drift alert</div><div class="v">{"yes" if oc["drift_alert"] else "no"}</div>'
            f'<div class="note">energy distance on {len(primary)} primary KPIs: p = {_p(a["energy_p"])} (Monte Carlo, {result["energy"].get("n_perm", 0)} permutations)</div></div>'
            f'<div class="col"><div class="k">Localized anomaly</div><div class="v">{_e(loc_word)}</div>'
            f'<div class="note">{b["n_sites_flagged"]} site(s) flagged · {b["n_sites_pending"]} pending review · {b["n_sites_credible"]} credible</div></div>'
            f'<div class="col"><div class="k">Quality abstention</div><div class="v">{"yes" if oc["quality_abstention"] else "no"}</div>'
            f'<div class="note">{_e("; ".join(abst["reasons"])) if abst["reasons"] else "no abstention reason"}</div></div></div>')
    drivers = ", ".join(v["drivers"]) if v["drivers"] else "none (no primary KPI meets Holm p &lt; α with |shift| ≥ 1 MAD)"
    jk = "".join(f'<span class="chip {"ok" if r["verdict"] == st["full_verdict"] else "warn"}">− {_e(r["left_out"])}: {_e(r["verdict"].split(" (")[0])}</span>' for r in st["runs"])
    body = (f'<p class="vtext">{_e(v["verdict"])}</p><p class="reason">{_e(v["reason"])}</p>{cols}'
            f'<p><b>Decision stability</b> (leave-one-site-out): {st["share"]:.2f} — {int(round(st["share"] * st["n_runs"]))} of {st["n_runs"]} re-verdicts returned the same verdict. '
            f'<span class="note">This is a stability share, not a probability of being right.</span></p><p>{jk}</p>'
            f'<p><b>Drivers:</b> {drivers}. <b>Usable n:</b> ' + "; ".join(f"{k} {nr}/{nb}" for k, (nr, nb) in result["limits"]["usable_n"].items()) + " (reference/batch sites after quality flags).</p>"
            f'<p><b>What would move it:</b> {_e(v["what_would_move_it"])}.</p>'
            f'<p class="note">Thresholds hash <code>{_e(v["thresholds_hash"])}</code> · {_e(v["provenance"])}. '
            f'Release decisions remain provisional: no equivalence test against agreed tolerances has been run; "consistent" means within what this reference and n can detect (see MDC below).</p>')
    parts.append(_sec("verdict", f"{_e(bat)} vs working reference {_e(ref)} — verdict", body, "verdict " + _verdict_class(v["verdict"])))

    # ---- 2. first-screen table ------------------------------------------------------------------------------
    rows = []
    for k in primary:
        r = cmp_i.loc[k]
        med_o, mad_o, n_o = _ref_stats_without_anomalous(sites_ref, k)
        md = mdc.get(k, {})
        rows.append([f"<b>{_e(k)}</b><br><span class='note'>{_e(lab(k))}</span>",
                     f"{_f(r.ref_median, 4)} ± {_f(r.ref_mad, 4)}<br><span class='note'>w/o anomalous: {_f(med_o, 4)} ± {_f(mad_o, 4)} (n = {n_o})</span>",
                     _f(r.batch_median, 4), f"{_f(r.shift_mad, 2, signed=True)} MAD<br><span class='note'>{_pct_change(r.ref_median, r.batch_median)}</span>",
                     f"{_f(r.ci_low, 2)} to {_f(r.ci_high, 2)}<br><span class='note'>bootstrap over sites, approx.</span>",
                     f"{_p(r.p_perm)}<br><span class='note'>{_e(r.p_method)}, {int(r.n_perm):,} ({_e(m['config']['statistic'])})</span>",
                     f"<b>{_p(r.p_holm)}</b>", f"{int(r.n_ref_usable)} / {int(r.n_batch_usable)}<br><span class='note'>excl. {int(r.n_ref_excluded)}/{int(r.n_batch_excluded)} · fallback {int(r.n_ref_fallback)}/{int(r.n_batch_fallback)}</span>",
                     f"<b>{_f(md.get('mdc_mad'), 2)} MAD</b><br><span class='note'>= {_f(md.get('mdc_abs'), 4)} {_kpi_unit(k)}; n = {md.get('n_incoming', '—')} v remaining</span>",
                     str(weights.get(k, "—")), _e(trust.get(k, "—"))])
    hdr = ["KPI", f"{_e(ref)} median ± MAD", f"{_e(bat)} median", "robust shift", "≈ 95 % CI", "permutation p", "Holm p", "usable n ref / batch",
           "MDC at 80 % power", "weight", "trust"]
    md0 = next(iter(mdc.values()), {})
    body = (f'<p class="note">Robust shift = (batch median − reference median) / reference MAD (MAD × 1.4826). Permutation test statistic: Hodges–Lehmann shift, site labels, '
            f'exact enumeration where affordable. Holm correction across the five primary KPIs only. MDC = smallest shift detected 80 % of the time at α = {m["config"]["alpha"]} by simulation '
            f'({md0.get("n_sim_used", "—")} draws; {_e(md0.get("design", ""))}). A "consistent" verdict never means a shift smaller than the MDC was ruled out.</p>'
            + _table(rows, hdr, "first") + f'<figure><img src="{_primary_strip_plot(result)}" alt="primary KPI strip plots"><figcaption>Per-site values of the five primary KPIs; bar = median; '
            f'reference sub-populations kept visible (grey-pore group in #eda100, cracked sites ringed); low-contrast batch sites ringed in #e34948 and excluded from bright-phase n.</figcaption></figure>')
    parts.append(_sec("kpis", "Primary KPIs — first screen", body))

    # ---- 3. drivers + physics ---------------------------------------------------------------------------------
    stmts = phys.get("statements", {})
    def driver_sentence(k, is_driver):
        r = cmp_i.loc[k]; pk = a["per_kpi"].get(k, {})
        direction = "lower" if r.shift_mad < 0 else "higher"
        pct = _pct_change(r.ref_median, r.batch_median).replace("+", "").replace("−", "")
        cons = f", consistent in {pk.get('n_beyond', 0)} of {pk.get('n_batch_usable', int(r.n_batch_usable))} sites (beyond the ordinary-reference range)" if pk else ""
        s = (f"{bat} has {pct} {direction} {lab(k).lower()} than the reference (robust shift {_f(r.shift_mad, 1, signed=True)} MAD, approx. 95 % CI {_f(r.ci_low, 1)} to {_f(r.ci_high, 1)}; "
             f"permutation p = {_p(r.p_perm)}, Holm p = {_p(r.p_holm)}){cons}")
        ps = stmts.get(k, {})
        if ps.get("statement"):
            if ps.get("gated", True):
                phys_line = f"→ {ps['statement']}"
            else:
                phys_line = "→ physics reading withheld: the shift lies inside the ±5-level threshold band."
            phys_line += f" <i>({ps.get('note', '')})</i>"
        else:
            phys_line = "→ no physics reading for this KPI."
        tag = "" if is_driver else ' <span class="chip warn">not a driver — shown for information</span>'
        return f'<div class="drv">{_e(s)}{tag}<span class="phys">{phys_line}</span></div>'
    if v["drivers"]:
        body = "".join(driver_sentence(k, True) for k in v["drivers"])
    else:
        top = cmp_[cmp_.is_primary].reindex(cmp_[cmp_.is_primary].shift_mad.abs().sort_values(ascending=False).index).kpi.tolist()[:2]
        body = ('<p>No primary KPI meets the driver conditions (Holm p &lt; α and |robust shift| ≥ 1 MAD). The two largest primary shifts are listed for information; '
                'they are not drivers and no verdict rests on them.</p>' + "".join(driver_sentence(k, False) for k in top))
    body += f'<p class="note">Physics statements are qualitative and relative (direction only); caveats: {_e("; ".join(phys.get("caveats", physics.GLOBAL_CAVEATS)))}.</p>'
    parts.append(_sec("drivers", "Drivers and physics reading", body))

    # ---- 4. evidence images -----------------------------------------------------------------------------------
    figs = []
    img_ok = with_images and m.get("batch_dir") and os.path.isdir(m["batch_dir"])
    if img_ok:
        for name, fn in (("typical", _typical_sites_image), ("novelty", _novelty_image), ("particles", _particles_image)):
            try:
                r = fn(result)
                if r:
                    figs.append(r)
            except Exception as ex:  # a missing image must not kill the report; say what failed
                figs.append((None, f"evidence image '{name}' could not be rendered: {ex!r}"))
        try:
            crops = _anomaly_crops(result)
        except Exception as ex:
            crops = [(None, f"anomaly crops could not be rendered: {ex!r}")]
        if crops:
            figs[1:1] = crops
        elif oc["localized"] in ("none", "flag"):
            figs.insert(1, (None, "No pending or credible localized anomaly → no anomaly crop. Flags that did not meet the severity/reliability conditions are listed in §5."))
    else:
        figs.append((None, "Evidence images not rendered: raw Dataset folder not available in this run (meta.batch_dir missing) or with_images=False."))
    body = "".join((f'<figure><img src="{b64}" alt="evidence" loading="lazy"><figcaption>{_e(cap)}</figcaption></figure>' if b64 else f'<p class="ph">{_e(cap)}</p>') for b64, cap in figs)
    parts.append(_sec("evidence", "Evidence images", body))

    # ---- 5. local-anomaly table -------------------------------------------------------------------------------
    rows = []
    for f_ in b["flags"]:
        status = "credible" if f_ in b.get("credible", []) else ("pending review" if f_ in b.get("credible_pending_review", []) else "evidence flag only")
        rows.append([_e(f_["site"]), _e(f_["kpi"]), _f(f_["value"], 4), _f(f_["ref_max"], 4), _f(f_["margin_in_mad"], 2, signed=True),
                     "yes" if f_["severity_ok"] else "no", _e(f_["measurement_note"]),
                     "not reviewed" if f_["image_reviewed"] is None else ("confirmed" if f_["image_reviewed"] else "not confirmed"), status])
    extra_rows = []
    for k, L in result["local"].items():
        if k in primary:
            continue
        t = L["table"]
        for _, r in t[t.exceeds.astype(bool)].iterrows():
            extra_rows.append([_e(r.site), _e(k) + " <span class='note'>(descriptive)</span>", _f(r.value, 4), _f(L["ref_max"], 4), _f(r.margin_in_mad, 2, signed=True),
                               "yes" if bool(r.credible_severity) else "no", "patch-level, trimmed frame" if k.startswith("patch_") else "—", "—", "evidence only (not a primary KPI)"])
    iid = {k: L["iid_flag_probability"] for k, L in result["local"].items() if k in primary}
    body = (f'<p class="note">Every batch site whose value exceeds the maximum of the {len(result["ordinary_ref_sites"])} ordinary reference sites (reference minus grey-pore and cracked). '
            f'Exceeding the maximum is an evidence flag, not a defect call: under an i.i.d. null at least one of {m["n_sites_batch"]} sites exceeds the ordinary-reference maximum with probability '
            + ", ".join(f"{k} {_f(p, 0, pct=True)}" for k, p in iid.items()) + " per KPI (n_batch / (n_ref + n_batch)). A flag becomes credible only with severity margin ≥ "
            f'{m["config"]["thresholds_resolved"]["severity_margin_mad"]} MAD of the ordinary-reference per-site values, a reliable measurement on that site, and a confirmed image review.</p>'
            + (_table(rows + extra_rows, ["site", "KPI", "value", "ordinary-ref max", "margin (MAD)", "severity ok", "measurement", "image review", "status"]) if (rows or extra_rows)
               else "<p>No batch site exceeds the ordinary-reference maximum on any primary KPI.</p>"))
    parts.append(_sec("local", "Local-anomaly evidence flags", body))

    # ---- 6. ML corroboration ----------------------------------------------------------------------------------
    def c2st_block(d, title):
        if d is None:
            return f'<h3>{title}</h3><p class="ph">not available in analysis_cache/ml for this pair (no material-only run has been cached; the cached runs include the acquisition flag etd_boundary_sharpness).</p>'
        coef = d["coef"]
        sel = coef[coef.selected.astype(bool)] if "selected" in coef else coef.iloc[0:0]
        rows = [[_e(r.feature), _e(r.sign), _f(r.coef, 3, signed=True), _f(r.selection_stability, 2), _e(trust.get(r.feature, "—"))] for r in sel.itertuples()]
        band = d.get("null_band") or [np.nan, np.nan]
        return (f'<h3>{title}</h3><p>Cross-validated AUC <b>{_f(d["auc"], 3)}</b>; null 95 % band {_f(band[0], 2)}–{_f(band[1], 2)}; Monte Carlo p = {_p(d["p"])} '
                f'({d.get("n_perm")} site-label permutations; {_e(d.get("cv"))}; {d.get("n_sites_ref")} v {d.get("n_sites_batch")} sites; equal site weight; scaler refit per fold). '
                f'KPIs in the run: {_e(", ".join(d.get("kpis", [])))}.</p>'
                + (_table(rows, ["selected KPI", "sign", "coef", "selection stability", "trust"]) if rows else "<p>No KPI selected.</p>"))
    body = c2st_block(result.get("c2st_material"), "Grouped-CV logistic regression two-sample test — material KPIs only (drives Check A iv)")
    body += c2st_block(result.get("c2st_flag_inclusive"), "Flag-inclusive run — acquisition evidence, not a verdict input")
    for tag, d in (result.get("c2st_extra") or {}).items():
        body += c2st_block(d, f"Variant: {tag} (acquisition evidence)")
    nv = result.get("novelty")
    if nv:
        ps = nv["per_site"].sort_values("novelty_median", ascending=False)
        rows = [[_e(r.site), _f(r.novelty_median, 2), _f(r.novelty_p90, 2), _f(r.pct_median, 0), _f(r.pct_p90, 0), "yes" if bool(r.exceeds_ref_loo_max) else "no", int(r.n_patches)] for r in ps.itertuples()]
        body += (f'<h3>Patch-embedding novelty (exploratory)</h3><p class="note">DINOv2 ViT-S/14 features of per-image-normalised 512-px BSE patches; PCA on the reference (re-fitted per leave-one-site-out fold), kNN(5) distance. '
                 f'Per-site percentiles are within the reference leave-one-site-out distribution ({len(nv["ref_loo"])} sites). "Exceeds reference max" is an evidence flag with an i.i.d. rate of '
                 f'≈ {_f(nv["iid_rate"], 0, pct=True)} for at least one of {len(ps)} sites. Patches never count as n.</p>'
                 + _table(rows, ["site", "novelty median", "novelty p90", "percentile (median)", "percentile (p90)", "exceeds ref LOO max", "patches"]))
        if nv.get("correlates") is not None and len(nv["correlates"]):
            c = nv["correlates"].reindex(nv["correlates"].rho.abs().sort_values(ascending=False).index).head(8)
            rows = [[_e(r.variable), _e(r.kind), _f(r.rho, 2, signed=True), _p(r.p), int(r.n)] for r in c.itertuples()]
            n_acq = int((c.kind == "acquisition").sum()); kpi_sig = c[(c.kind == "kpi") & (c.p < 0.05)].variable.tolist()
            reading = (f"{n_acq} of the top 8 correlates are acquisition statistics and {'no' if not kpi_sig else ', '.join(kpi_sig)} material KPI reaches p &lt; 0.05 → "
                       "novelty is read as acquisition evidence (where the imaging differs), not as a material signal.")
            body += "<h3>What novelty tracks — top 8 Spearman correlates (all 31 sites)</h3>" + _table(rows, ["variable", "kind", "ρ", "p", "n"]) + f"<p>{reading}</p>"
    else:
        body += '<h3>Patch-embedding novelty</h3><p class="ph">not available in analysis_cache/ml for this batch.</p>'
    parts.append(_sec("ml", "ML corroboration (never the sole driver)", body))

    # ---- 7. acquisition ---------------------------------------------------------------------------------------
    fb = result["flags_batch"]
    rows = [[_e(r.site), _e(r.acquisition_group), "yes" if r.contrast_stretched_bse else "no", "yes" if r.contrast_stretched_any else "no",
             f"{_f(r.bse_p1, 0)}{' ▲' if r.raised_black_level else ''}", "yes" if r.bright_low_contrast else "no", _f(r.bright_sep, 0),
             "yes" if r.grey_pore else "no", f"{r.band_top} / {r.band_bottom}", r.H] for r in fb.itertuples()]
    body = (f'<p class="note">{_e(fb.attrs.get("status", "provisional"))}. Rules: contrast stretched = empty histogram bins &gt; {m["config"]["stretch_frac"]:.0%} of the 1–99 % range; '
            f'raised black level = BSE p1 &gt; {m["config"]["black_level_p1"]}; low bright-phase contrast = bright mode unresolved or &lt; 45 levels above graphite; bands = bright edge rows trimmed before measuring.</p>'
            + _table(rows, ["site", "group", "stretched (BSE)", "stretched (any ch.)", "BSE p1", "low contrast", "bright sep (levels)", "grey pore", "bands top / bottom", "H (px)"])
            + '<h3>Sensitivity to acquisition adjustment (unadjusted / stratified / adjusted)</h3>')
    acq = result.get("acquisition")
    if acq:
        rows = [[_e(k), _f(d.get("statistic"), 3), _p(d.get("p")), _e(d.get("n_sites", "—")), _e(d.get("note", ""))] for k, d in acq.items() if isinstance(d, dict)]
        body += _table(rows, ["view", "energy statistic", "p", "sites", "note"]) + '<p class="note">Attenuation under the stratified / adjusted views is a sensitivity analysis, not a causal attribution.</p>'
    else:
        body += '<p class="ph">not yet available — polaron_qc.acquisition is being built; this block renders from result["acquisition"] when present (three views: unadjusted, stratified by acquisition group, residualised on the acquisition variables; no causal percentage is claimed).</p>'
    parts.append(_sec("acq", "Acquisition flags", body))

    # ---- 8. physics sanity ------------------------------------------------------------------------------------
    san = phys.get("sanity")
    body = ""
    if san is not None and len(san):
        n_ok = int(san.fractions_sum_to_one.sum()); body += f"<p>Phase fractions sum to 1 on {n_ok} of {len(san)} batch sites (max |residual| {_f(san.fraction_sum_residual.abs().max(), 2) if san.fraction_sum_residual.notna().any() else '—'}).</p>"
        pv = san.porosity_vs_typical.value_counts().to_dict()
        body += (f"<p>Macro-pore fraction is below the typical calendered-anode range (25–35 %) on {pv.get('below_typical', 0)} of {len(san)} sites — {_e(physics.POROSITY_NOTE)}: "
                 "the binder / carbon-black network is unsegmented and falls into the pore or graphite class, segmentation thresholds and preparation (resin or smearing filling fine pores) remain alternatives; this is a dataset-level observation, not a batch difference.</p>")
    pst = phys.get("sites")
    if pst is not None and len(pst) and "pore_frac_band" in pst:
        hb = pst[pst.pore_frac_band.notna()]
        rows = [[_e(r.batch), _e(r.site), _f(r.pore_frac_nominal, 4), f"{_f(r.pore_frac_lo_minus, 4)} – {_f(r.pore_frac_lo_plus, 4)}", _f(r.pore_frac_band_rel, 0, pct=True),
                 _f(r.bright_frac_nominal, 4), f"{_f(r.bright_frac_hi_plus, 4)} – {_f(r.bright_frac_hi_minus, 4)}", _f(r.bright_frac_band_rel, 0, pct=True)] for r in hb.itertuples()]
        body += (f"<h3>±5-level threshold band (available for {len(hb)} of {len(pst)} sites of this pair)</h3>"
                 + (_table(rows, ["batch", "site", "pore_frac", "band (th_lo ± 5)", "rel. band", "bright_frac", "band (th_hi ± 5)", "rel. band"]) if rows else "")
                 + '<p class="note">The pore-fraction band is as large as the batch differences (±15–25 % relative), which is why physics wording on pore_frac is gated on it; a per-site band for every site is still to be added to features.</p>')
    if phys.get("fraction_connected_note"):
        body += f"<p>{_e(phys['fraction_connected_note'])}</p>"
    if pst is not None and len(pst) and "delamination_index" in pst:
        g = pst.groupby("batch").agg(n=("site", "size"), crack_like=("n_crack_like", "median"), delam=("delamination_index", "median"), longest=("longest_void_over_thickness", "median"))
        rows = [[_e(i), int(r.n), _f(r.crack_like, 0), _f(r.delam, 2), _f(r.longest, 2)] for i, r in g.iterrows()]
        body += "<h3>Void geometry (2-D geometric observations; not continuity)</h3>" + _table(rows, ["batch", "sites", "median crack-like voids / site", "median delamination index", "median longest void / thickness"])
    if pst is not None and len(pst) and "vol_frac_bright" in pst:
        g = pst.groupby("batch").agg(vb=("vol_frac_bright", "median"), vp=("vol_frac_pore", "median"), mSi=("nominal_mass_frac_additive_if_Si", "median"))
        rows = [[_e(i), _f(r.vb, 3), _f(r.vp, 3), _f(r.mSi, 3)] for i, r in g.iterrows()]
        body += ("<h3>Stereology — secondary</h3>" + _table(rows, ["batch", "Delesse vol. frac. bright", "Delesse vol. frac. pore", "nominal mass frac. additive if Si"])
                 + '<p class="note">Delesse assumes random sections (violated by plate alignment); biased by the unsegmented binder; chemistry unconfirmed. Not on the first screen and not a verdict input.</p>')
    parts.append(_sec("physics", "Physics sanity checks", body or "<p class='ph'>physics cache not available.</p>"))

    # ---- 9. secondary KPIs ------------------------------------------------------------------------------------
    sec = cmp_[~cmp_.is_primary]
    rows = [[_e(r.kpi) + f"<br><span class='note'>{_e(lab(r.kpi))}</span>", f"{_f(r.ref_median, 4)} ± {_f(r.ref_mad, 4)}", _f(r.batch_median, 4), f"{_f(r.shift_mad, 2, signed=True)} MAD",
             f"{_f(r.ci_low, 2)} to {_f(r.ci_high, 2)}", _f(r.cliffs_delta, 2, signed=True), _p(r.p_perm), f"{int(r.n_ref_usable)} / {int(r.n_batch_usable)}",
             str(physics.CONSEQUENCE_WEIGHTS.get(r.kpi, "—")), _e(trust.get(r.kpi, "—"))] for r in sec.itertuples()]
    body = ('<p class="note">Descriptive only: uncorrected permutation p (Hodges–Lehmann, site labels), never a verdict driver, not multiplicity-corrected into silence. Lengths in px.</p>'
            + _table(rows, ["KPI", f"{_e(ref)} median ± MAD", f"{_e(bat)} median", "robust shift", "≈ 95 % CI", "Cliff's δ", "p (uncorrected)", "usable n", "weight", "trust"]))
    parts.append(_sec("secondary", "Secondary KPIs — appendix (descriptive)", body))

    # ---- 10. limits + footer ----------------------------------------------------------------------------------
    L = result["limits"]
    lo, hi = L["mdc_range_mad"]
    body = (f"<p>Usable sample sizes per primary KPI (reference / batch): " + "; ".join(f"{k} {nr}/{nb}" for k, (nr, nb) in L["usable_n"].items()) +
            f". Whether sites come from distinct specimens is {_e(L['specimen_independence'])}. The reference is heterogeneous: {len(L['reference_heterogeneity']['grey_pore'])} grey-pore sites "
            f"({_e(', '.join(L['reference_heterogeneity']['grey_pore']))}) rest on a fallback pore threshold and {len(L['reference_heterogeneity']['cracked'])} cracked sites "
            f"({_e(', '.join(L['reference_heterogeneity']['cracked']))}) carry the only clear material anomaly; {L['reference_heterogeneity']['ordinary_n']} ordinary reference sites define the local-anomaly range. "
            f"Minimum detectable change at 80 % power ranges from {_f(lo, 2)} to {_f(hi, 2)} reference MADs across the primary KPIs; smaller shifts cannot be ruled out. "
            f"All measurements are {_e(L['scale'])}. {_e(L['seven_v_seven']) if L['seven_v_seven'] != 'not applicable' else ''} "
            + " ".join(_e(n) + "." for n in L["notes"]) + "</p>")
    if m.get("notes"):
        body += "<h3>Integration notes from this build</h3><ul class='small'>" + "".join(f"<li>{_e(n)}</li>" for n in m["notes"]) + "</ul>"
    parts.append(_sec("limits", "Limits of this dataset", body))
    footer = (f'<footer>Generated {_e(m["timestamp"])} by polaron_qc.report · config hash <code>{_e(m["config_hash"])}</code> · thresholds hash <code>{_e(m["thresholds_hash"])}</code> · '
              f'features v{_e(m.get("feature_version"))} · git {_e(m.get("git_describe") or "n/a")} · build runtime {_e(m.get("runtime_s"))} s · '
              f'statistic {_e(m["config"]["statistic"])}, α = {m["config"]["alpha"]}, power {m["config"]["power"]}, seed {m["config"]["seed"]}.</footer>')

    toc = "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in (("verdict", "Verdict"), ("kpis", "Primary KPIs (MDC)"), ("drivers", "Drivers & physics"), ("evidence", "Evidence images"),
                                                                 ("local", "Local-anomaly flags"), ("ml", "ML corroboration"), ("acq", "Acquisition"), ("physics", "Physics sanity"),
                                                                 ("secondary", "Secondary KPIs"), ("limits", "Limits")))
    head = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>QC report {_e(bat)}</title><style>{_CSS}</style></head><body><div class="wrap">'
            f'<h1>Polaron SEM QC — {_e(bat)} vs working reference {_e(ref)}</h1>'
            f'<p class="lede">{m["n_sites_batch"]} incoming sites against {m["n_sites_ref"]} reference sites (site = unit of evidence; patches never count as n). '
            f'Lengths in px (25 nm/px nominal). Batch-wide drift and localized defects are separate decision paths. Provenance: {_e(m["provenance"])}.</p><ol class="toc">{toc}</ol>')
    html = head + "".join(parts) + footer + "</div></body></html>"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# ---------------------------------------------------------------------------------------------------------------
# command line: python -m polaron_qc.report Dataset/Batch_3 Dataset/Batch_1 reports/qc_Batch_1.html
# ---------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    ref_dir, bat_dir = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join("reports", f"qc_{os.path.basename(os.path.normpath(bat_dir))}.html")
    res = build_result(ref_dir, bat_dir)
    print("wrote", render_report(res, out), f"{os.path.getsize(out) / 1e6:.2f} MB")
