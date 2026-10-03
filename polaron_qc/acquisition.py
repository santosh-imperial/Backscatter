"""polaron_qc.acquisition — sensitivity to acquisition adjustment (docs/qc_plan.md §2.6) and the derived
acquisition flags the rest of the pipeline reads.

What this module is, and is not
-------------------------------
This is a **sensitivity analysis**. It shows how much a batch-vs-reference comparison changes when sites with
atypical acquisition are set aside (stratified view) or when the KPIs are residualised on acquisition
covariates (adjusted view). Nothing here attributes a share of a shift to the microscope: bright-phase
separation, black level and boundary sharpness can change with the material as well as with the session
(checklist C11, decision D19). No function returns anything named "instrument share" and no docstring says
"caused by".

Three views (plan §2.6), one call: :func:`three_views` -> unadjusted / stratified / adjusted +
:func:`polaron_qc.decision.attenuation`.

Flags (contract row ``images`` in docs/workflow.md): :func:`derive_flags` turns the raw per-image statistics
and the per-site segmentation diagnostics into one row per site with boolean flags and an
``acquisition_group``. Everything that can be derived from the data is derived from the data, so an unseen
batch (whose sites are in none of the constant lists in ``polaron_qc.__init__``) gets its flags from its own
images. The one exception is ``cracked_known``, a reference-only label (see the docstring).

Small-sample discipline carried from plan §2.0 / checklist C09: every fit that depends on which sites are
"reference" (the covariate regression, the MAD scaling inside the energy statistic) is **re-fitted inside
every permutation**. ``stats.energy_distance_test`` and ``stats.permutation_test`` have no hook for an
outer fit, so the adjusted view carries its own Monte Carlo permutation loops
(:func:`adjusted_energy_permutation`, :func:`adjusted_kpi_permutation`) with the resample count stated and
the (b + 1)/(n + 1) convention (checklist C15).

Threshold band (hazard 4 in docs/workflow.md): :func:`threshold_band` recomputes pore_frac / bright_frac at
th ± 5 gray levels on the real BSE image with the *features* module's own loader, trimmer and segmenter, for
every site (:func:`threshold_bands`), and the result is cached in ``analysis_cache/acquisition_sites.csv``.

Conventions: one row per site everywhere; keys are (batch, site); lengths in px; seeds are integers.
"""
from __future__ import annotations

import os
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from polaron_qc import CRACKED_SITES, GREY_PORE_SITES, LOW_CONTRAST_SITES, PRIMARY_KPIS
from polaron_qc import stats
from polaron_qc.decision import attenuation as _attenuation
from polaron_qc.features import RIDGE_T, load_image, segment, trimmed

__all__ = [
    "ACQ_COVARIATES", "ACQ_COVARIATES_ADJUST", "ACQUISITION_GROUPS", "STRETCH_EMPTY_BIN_FRAC", "RAISED_BLACK_LEVEL_P1",
    "MIN_SITES_PER_SIDE", "Z_SHIFT",
    "load_site_tables", "derive_flags", "threshold_band", "threshold_bands", "build_acquisition_cache",
    "stratified_comparison", "residualise", "adjusted_energy_permutation", "adjusted_kpi_permutation",
    "adjusted_comparison", "three_views", "three_channel_agreement",
]

# ----------------------------------------------------------------------------------------------------------------
# constants (provenance: developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived)
# ----------------------------------------------------------------------------------------------------------------
#: Contrast stretched after acquisition: comb histogram, share of empty bins between p1 and p99 above this.
STRETCH_EMPTY_BIN_FRAC = 0.2
#: Raised black level: BSE 1st percentile above this gray level (no black pixels; grey-pore signature).
RAISED_BLACK_LEVEL_P1 = 10
#: Fewer usable sites than this on either side and the stratified view refuses to run (plan §2.0; the decision
#: layer's own floor is Thresholds.min_usable_sites = 5 per KPI; 4 is the floor below which even a descriptive
#: energy statistic is meaningless).
MIN_SITES_PER_SIDE = 4
#: Robust-z magnitude that counts as "shifted" in the three-channel pattern label (notebook 01 §6d reads ±2 to ±3).
Z_SHIFT = 2.0
ACQUISITION_GROUPS = ("ordinary", "low_contrast", "grey_pore", "cracked")

#: The full acquisition covariate list named in plan §2.6 (bright-phase separation, black level, stretch /
#: contrast statistics, curtaining, boundary sharpness). Used for correlate tables and available to
#: :func:`adjusted_comparison` with a ridge penalty; **not** the default adjustment set (see below).
ACQ_COVARIATES = ["bright_sep", "graphite_mode", "bse_p1", "bse_std", "bse_empty_bin_frac", "etd_curtain_frac",
                  "etd_boundary_sharpness"]
#: Default adjustment set: three covariates chosen a priori, one per mechanism named in plan §2.6 —
#: bright_sep (bright-phase contrast; drives the low-contrast flag), bse_p1 (black level; drives the grey-pore
#: flag), etd_boundary_sharpness (focus / polish quality; the flag that alone separates Batch 1 from Batch 3 in
#: the classifier, hazard 2). Rationale: the regression is fitted on the reference sites only, and inside the
#: permutation loop the "reference" is 17 of 24 pooled rows; with 7 covariates + intercept that is 8 parameters
#: on 17 rows and the fit absorbs sampling noise (reference R² is inflated by construction), which would make the
#: adjusted view look like it "explains" shifts it has merely over-fitted. Three slopes + intercept on 17 rows
#: keeps ~13 residual degrees of freedom. The choice is exposed as the ``covariates`` argument; the 7-covariate
#: version is available with ``ridge_lambda > 0``.
ACQ_COVARIATES_ADJUST = ["bright_sep", "bse_p1", "etd_boundary_sharpness"]

_INTENSITY_COLS = ["BSE_p50", "ETD_p50", "Inlens_p50"]
_TEXTURE_COLS = ["corr_len_px", "etd_grad_energy", "inlens_grad_energy"]


# ----------------------------------------------------------------------------------------------------------------
# loading the cached tables (convenience for notebooks / tests; the pipeline passes the frames in)
# ----------------------------------------------------------------------------------------------------------------
def load_site_tables(cache_dir: str = "analysis_cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sites, images) from the notebook-01 caches: ``site_features.csv`` merged with ``etd_inlens_features.csv``
    on (batch, site), and ``image_quality.csv``. ``etd_curtain_frac`` is derived from ``curtain_<RIDGE_T>`` when the
    merged table does not carry it (the notebook cache predates that column; ``features.extract_batch`` emits it)."""
    sites = pd.read_csv(os.path.join(cache_dir, "site_features.csv"))
    etd_path = os.path.join(cache_dir, "etd_inlens_features.csv")
    if os.path.exists(etd_path):
        etd = pd.read_csv(etd_path)
        dup = [c for c in etd.columns if c in sites.columns and c not in ("batch", "site")]
        sites = sites.merge(etd.drop(columns=dup), on=["batch", "site"], how="left")
    sites = _ensure_curtain_frac(sites)
    images = pd.read_csv(os.path.join(cache_dir, "image_quality.csv"))
    return sites, images


def _ensure_curtain_frac(sites: pd.DataFrame) -> pd.DataFrame:
    if "etd_curtain_frac" not in sites.columns:
        for key in (f"curtain_{RIDGE_T}", f"curtain_{RIDGE_T!r}", "curtain_0.4"):
            if key in sites.columns:
                sites = sites.copy()
                sites["etd_curtain_frac"] = sites[key]
                break
    return sites


# ----------------------------------------------------------------------------------------------------------------
# derived acquisition flags
# ----------------------------------------------------------------------------------------------------------------
def _norm_det(det: pd.Series) -> pd.Series:
    return det.replace({"SE": "ETD"})


def derive_flags(sites_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """One row per site with the acquisition flags and ``acquisition_group``.

    Inputs are the ``sites`` and ``images`` tables of ``features.extract_batch`` (or the notebook-01 caches via
    :func:`load_site_tables`). Nothing is read from ``sites_df.grey_pore`` (features sets that column from the
    constant list); the flag is re-derived here so an unseen batch is treated identically to the known ones.

    Flags and where they come from
    * ``contrast_stretched_bse``  BSE ``empty_bin_frac > 0.2`` (comb histogram after post-acquisition stretch).
    * ``contrast_stretched_any``  the same on any of the three channels.
    * ``raised_black_level``      BSE ``p1 > 10``: no black pixels in the frame.
    * ``bright_low_contrast``     from ``sites_df`` (features: bright mode unresolved or ``bright_sep < 45``);
      if the column is absent, membership of ``LOW_CONTRAST_SITES`` is used and the source is recorded.
    * ``grey_pore``               ``raised_black_level`` OR membership of ``GREY_PORE_SITES``. The data rule is the
      operative one: on Batches 1–3 it reproduces the constant list exactly (BSE p1 = 23–25 on the four grey-pore
      sites, 0–6 on all 27 others), so the list adds nothing here and on an unseen batch only the data rule acts.
      The list is kept as a cross-check for the reference; ``grey_pore_source`` says which fired ("data",
      "list", "data+list" or "").
    * ``cracked_known``           membership of ``CRACKED_SITES`` — a **reference-only label**. Cracking is the
      material anomaly the primary KPI ``crack_frac`` measures, so it must not be turned into a data-driven
      acquisition flag (that would remove the signal from the comparison). For an unseen batch this column is
      always False; crack-like voids in an unseen batch are found by Check B, not by this flag.
    * ``band_rows``               BSE ``band_top + band_bottom`` (rows trimmed as current collector / stitching band).
    * ``acquisition_group``       ∈ {"ordinary", "low_contrast", "grey_pore", "cracked"} with precedence
      low_contrast > grey_pore > cracked > ordinary (a site is in exactly one group; the booleans keep the overlap).

    Also carried through: ``bse_p1, bse_p50, bse_std, bse_empty_bin_frac, etd_p50, inlens_p50`` (from images) and
    ``bright_sep, etd_curtain_frac, etd_boundary_sharpness, graphite_mode`` (from sites), i.e. every covariate in
    ``ACQ_COVARIATES`` plus the channel medians used by :func:`three_channel_agreement`.
    """
    sites = _ensure_curtain_frac(sites_df.copy())
    sites["site"] = sites["site"].astype(str)
    img = images_df.copy()
    img["site"] = img["site"].astype(str)
    img["det"] = _norm_det(img["det"])
    keys = ["batch", "site"]

    def pick(det, cols, prefix):
        sub = img[img.det == det][keys + cols].drop_duplicates(keys)
        return sub.rename(columns={c: f"{prefix}_{c}" for c in cols})

    out = sites[keys].drop_duplicates().copy()
    out = out.merge(pick("BSE", ["p1", "p50", "std", "empty_bin_frac", "band_top", "band_bottom"], "bse"), on=keys, how="left")
    out = out.merge(pick("ETD", ["p50", "empty_bin_frac"], "etd"), on=keys, how="left")
    out = out.merge(pick("Inlens", ["p50", "empty_bin_frac"], "inlens"), on=keys, how="left")

    out["contrast_stretched_bse"] = (out["bse_empty_bin_frac"] > STRETCH_EMPTY_BIN_FRAC).fillna(False).astype(bool)
    any_stretch = img.groupby(keys)["empty_bin_frac"].max().rename("max_empty_bin_frac").reset_index()
    out = out.merge(any_stretch, on=keys, how="left")
    out["contrast_stretched_any"] = (out["max_empty_bin_frac"] > STRETCH_EMPTY_BIN_FRAC).fillna(False).astype(bool)
    out = out.drop(columns=["max_empty_bin_frac"])
    out["raised_black_level"] = (out["bse_p1"] > RAISED_BLACK_LEVEL_P1).fillna(False).astype(bool)
    out["band_rows"] = (out["bse_band_top"].fillna(0) + out["bse_band_bottom"].fillna(0)).astype(int)
    out = out.drop(columns=["bse_band_top", "bse_band_bottom"])

    # from sites
    carry = [c for c in ("bright_sep", "etd_curtain_frac", "etd_boundary_sharpness", "graphite_mode") if c in sites.columns]
    out = out.merge(sites[keys + carry].drop_duplicates(keys), on=keys, how="left")
    for c in ("bright_sep", "etd_curtain_frac", "etd_boundary_sharpness", "graphite_mode"):
        if c not in out.columns:
            out[c] = np.nan
    if "bright_low_contrast" in sites.columns:
        blc = sites[keys + ["bright_low_contrast"]].drop_duplicates(keys)
        out = out.merge(blc, on=keys, how="left")
        out["bright_low_contrast"] = out["bright_low_contrast"].fillna(False).astype(bool)
        out["bright_low_contrast_source"] = "data"
    else:
        out["bright_low_contrast"] = out["site"].isin(LOW_CONTRAST_SITES)
        out["bright_low_contrast_source"] = "list"

    in_list = out["site"].isin(GREY_PORE_SITES)
    out["grey_pore"] = out["raised_black_level"] | in_list
    out["grey_pore_source"] = np.select([out["raised_black_level"] & in_list, out["raised_black_level"], in_list],
                                        ["data+list", "data", "list"], default="")
    out["cracked_known"] = out["site"].isin(CRACKED_SITES)
    out["acquisition_group"] = np.select([out["bright_low_contrast"], out["grey_pore"], out["cracked_known"]],
                                         ["low_contrast", "grey_pore", "cracked"], default="ordinary")
    cols = keys + ["acquisition_group", "bright_low_contrast", "grey_pore", "cracked_known", "contrast_stretched_bse",
                   "contrast_stretched_any", "raised_black_level", "band_rows", "bse_p1", "bse_p50", "bse_std",
                   "bse_empty_bin_frac", "etd_p50", "inlens_p50", "bright_sep", "etd_curtain_frac",
                   "etd_boundary_sharpness", "graphite_mode", "grey_pore_source", "bright_low_contrast_source",
                   "etd_empty_bin_frac", "inlens_empty_bin_frac"]
    return out[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------
# threshold band on the real image (features' loader / trimmer / segmenter; nothing re-implemented)
# ----------------------------------------------------------------------------------------------------------------
def threshold_band(batch_dir: str, site: str, th_lo: float, th_hi: float, delta: float = 5.0) -> dict:
    """pore_frac / bright_frac of one site at its thresholds and at th ± ``delta`` gray levels.

    Uses ``features.load_image`` (BSE), ``features.trimmed`` (edge bands) and ``features.segment`` (gaussian σ = 1 +
    one opening), so the nominal values reproduce ``site_features.csv`` to float precision. ``segment`` is called
    three times (nominal, both thresholds − δ, both thresholds + δ); the pore mask responds to th_lo and the
    bright mask to th_hi, so three calls give all six fractions. ≈ 1 s per site.

    Returns pore_frac_nominal, pore_frac_lo_minus (th_lo − δ), pore_frac_lo_plus (th_lo + δ), bright_frac_nominal,
    bright_frac_hi_minus (th_hi − δ), bright_frac_hi_plus (th_hi + δ), pore_frac_band_rel and bright_frac_band_rel
    (= (max − min) / nominal over the three values; NaN when the nominal fraction is 0), plus site, th_lo, th_hi,
    delta and elapsed_s. Column names match ``physics.threshold_sensitivity`` so the tables can be joined.
    """
    t0 = time.perf_counter()
    a = trimmed(load_image(batch_dir, site, "BSE"))
    pore0, bright0, _ = segment(a, th_lo, th_hi)
    pore_m, bright_m, _ = segment(a, th_lo - delta, th_hi - delta)
    pore_p, bright_p, _ = segment(a, th_lo + delta, th_hi + delta)
    pf = dict(nominal=float(pore0.mean()), lo_minus=float(pore_m.mean()), lo_plus=float(pore_p.mean()))
    bf = dict(nominal=float(bright0.mean()), hi_minus=float(bright_m.mean()), hi_plus=float(bright_p.mean()))
    out = dict(site=site, th_lo=float(th_lo), th_hi=float(th_hi), delta=float(delta),
               pore_frac_nominal=pf["nominal"], pore_frac_lo_minus=pf["lo_minus"], pore_frac_lo_plus=pf["lo_plus"],
               bright_frac_nominal=bf["nominal"], bright_frac_hi_minus=bf["hi_minus"], bright_frac_hi_plus=bf["hi_plus"])
    out["pore_frac_band_rel"] = (max(pf.values()) - min(pf.values())) / pf["nominal"] if pf["nominal"] > 0 else np.nan
    out["bright_frac_band_rel"] = (max(bf.values()) - min(bf.values())) / bf["nominal"] if bf["nominal"] > 0 else np.nan
    out["elapsed_s"] = time.perf_counter() - t0
    return out


def threshold_bands(batch_dir: str, sites_df: pd.DataFrame, delta: float = 5.0, verbose: bool = False) -> pd.DataFrame:
    """:func:`threshold_band` for every row of ``sites_df`` whose ``batch`` matches ``batch_dir`` (all rows when the
    table has no ``batch`` column). Needs ``site, th_lo, th_hi``. Returns one row per site with ``batch`` first."""
    bname = os.path.basename(os.path.normpath(batch_dir))
    df = sites_df
    if "batch" in df.columns and (df.batch == bname).any():
        df = df[df.batch == bname]
    rows = []
    for r in df.itertuples(index=False):
        rec = threshold_band(batch_dir, str(r.site), float(r.th_lo), float(r.th_hi), delta)
        rec = dict(batch=getattr(r, "batch", bname), **rec)
        rows.append(rec)
        if verbose:
            print(f"{rec['batch']}/{rec['site']}: pore band {rec['pore_frac_band_rel']:.2f}, bright band "
                  f"{rec['bright_frac_band_rel']:.2f} ({rec['elapsed_s']:.1f} s)")
    return pd.DataFrame(rows)


def build_acquisition_cache(dataset_root: str = "Dataset", cache_dir: str = "analysis_cache",
                            out_csv: str | None = None, delta: float = 5.0, verbose: bool = True) -> pd.DataFrame:
    """derive_flags for all cached sites joined with the per-site threshold bands; written to
    ``analysis_cache/acquisition_sites.csv`` (≈ 1 s per site, 31 sites)."""
    sites, images = load_site_tables(cache_dir)
    flags = derive_flags(sites, images)
    bands = pd.concat([threshold_bands(os.path.join(dataset_root, b), sites, delta, verbose) for b in sorted(sites.batch.unique())],
                      ignore_index=True)
    out = flags.merge(bands.drop(columns=["elapsed_s"]), on=["batch", "site"], how="left")
    out_csv = out_csv or os.path.join(cache_dir, "acquisition_sites.csv")
    out.to_csv(out_csv, index=False)
    return out


# ----------------------------------------------------------------------------------------------------------------
# helpers shared by the views
# ----------------------------------------------------------------------------------------------------------------
def _attach_flags(df: pd.DataFrame, flags_df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Left-join ``cols`` of ``flags_df`` onto ``df`` on (batch, site); flag columns win over same-named columns."""
    cols = [c for c in dict.fromkeys(cols) if c in flags_df.columns]
    f = flags_df[["batch", "site"] + cols].copy()
    f["site"] = f["site"].astype(str)
    d = df.copy()
    d["site"] = d["site"].astype(str)
    d = d.drop(columns=[c for c in cols if c in d.columns])
    return d.merge(f, on=["batch", "site"], how="left")


def _usable_mask(df: pd.DataFrame, kpi: str, flags: dict | None) -> np.ndarray:
    return df["site"].astype(str).isin(stats.usable_n(df, kpi, flags)["usable_sites"]).to_numpy()


def _energy_frame(df: pd.DataFrame, kpis: Sequence[str], flags: dict | None) -> pd.DataFrame:
    """KPI matrix for the energy test, indexed by site, with values set to NaN where the KPI is not usable on
    that site (bright_* on low-contrast sites), so ``energy_distance_test`` drops the row and reports it
    (checklist C14: usable n, not folder n)."""
    X = df.set_index(df["site"].astype(str))[list(kpis)].astype(float).copy()
    for k in kpis:
        X.loc[~_usable_mask(df, k, flags), k] = np.nan
    return X


def _energy(ref_df, batch_df, kpis, flags, weights, n_mc, seed) -> dict:
    return stats.energy_distance_test(_energy_frame(ref_df, kpis, flags), _energy_frame(batch_df, kpis, flags),
                                      weights=weights, n_mc=n_mc, seed=seed)


# ----------------------------------------------------------------------------------------------------------------
# view 2: stratified
# ----------------------------------------------------------------------------------------------------------------
def stratified_comparison(ref_df: pd.DataFrame, batch_df: pd.DataFrame, kpis: Sequence[str], flags_df: pd.DataFrame,
                          keep_groups: Sequence[str] = ("ordinary",), energy_kpis: Sequence[str] = PRIMARY_KPIS,
                          weights=None, n_mc_energy: int = 5000, **compare_kwargs) -> dict:
    """The same comparison restricted to sites whose ``acquisition_group`` ∈ ``keep_groups`` on **both** sides.

    Returns dict(compare, energy, n_ref_kept, n_batch_kept, dropped_sites, keep_groups, reason). ``compare`` is
    ``stats.compare_kpis`` on the kept sites (``compare_kwargs`` forwarded: statistic, seed, n_boot, flags, ...);
    ``energy`` is ``stats.energy_distance_test`` on ``energy_kpis`` (default the primary five) with the same
    ``weights`` the unadjusted view used. If fewer than ``MIN_SITES_PER_SIDE`` (4) sites remain on either side the
    view refuses: compare = energy = None and ``reason`` says why (a comparison on 2 sites is not fabricated).
    ``dropped_sites`` lists (batch, site, acquisition_group) for every site set aside.
    """
    keep = set(keep_groups)
    r = _attach_flags(ref_df, flags_df, ["acquisition_group"])
    b = _attach_flags(batch_df, flags_df, ["acquisition_group"])
    for d in (r, b):
        d["acquisition_group"] = d["acquisition_group"].fillna("ordinary")
    rk, bk = r[r.acquisition_group.isin(keep)], b[b.acquisition_group.isin(keep)]
    dropped = pd.concat([r[~r.acquisition_group.isin(keep)], b[~b.acquisition_group.isin(keep)]])
    dropped = [tuple(x) for x in dropped[["batch", "site", "acquisition_group"]].itertuples(index=False)]
    out = dict(compare=None, energy=None, n_ref_kept=int(len(rk)), n_batch_kept=int(len(bk)), dropped_sites=dropped,
               keep_groups=tuple(keep_groups), reason=None)
    if len(rk) < MIN_SITES_PER_SIDE or len(bk) < MIN_SITES_PER_SIDE:
        out["reason"] = (f"stratified view not run: {len(rk)} reference and {len(bk)} batch sites in groups "
                         f"{sorted(keep)} (need >= {MIN_SITES_PER_SIDE} on each side)")
        return out
    flags = compare_kwargs.get("flags")
    seed = int(compare_kwargs.get("seed", 0))
    out["compare"] = stats.compare_kpis(rk, bk, kpis, **compare_kwargs)
    out["energy"] = _energy(rk, bk, energy_kpis, flags, weights, n_mc_energy, seed)
    return out


# ----------------------------------------------------------------------------------------------------------------
# view 3: adjusted (residualised on acquisition covariates, regression fitted on the reference, refit per permutation)
# ----------------------------------------------------------------------------------------------------------------
def _design(C: np.ndarray, fit_rows: slice | np.ndarray, ridge_lambda: float):
    """Standardise covariates by the fit rows (mean / sd; sd 0 -> 1) and prepend an intercept.
    C: (..., N, p). Returns A: (..., N, p + 1) and the penalty matrix (p + 1, p + 1) with 0 on the intercept."""
    Cf = C[..., fit_rows, :]
    mu = Cf.mean(axis=-2, keepdims=True)
    sd = Cf.std(axis=-2, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (C - mu) / sd
    A = np.concatenate([np.ones(Z.shape[:-1] + (1,)), Z], axis=-1)
    P = np.eye(C.shape[-1] + 1) * ridge_lambda
    P[0, 0] = 0.0
    return A, P


def _fit_residuals(Y: np.ndarray, C: np.ndarray, n_fit: int, ridge_lambda: float):
    """OLS (ridge_lambda = 0, minimum-norm via pinv) or ridge regression of Y on [1, C] fitted on the first
    ``n_fit`` rows, residuals for all rows. Batched over leading dimensions.
    Y: (..., N, k)  C: (..., N, p)  ->  residuals (..., N, k), beta (..., p + 1, k)."""
    A, P = _design(C, slice(0, n_fit), ridge_lambda)
    Af, Yf = A[..., :n_fit, :], Y[..., :n_fit, :]
    if ridge_lambda > 0:
        AtA = np.swapaxes(Af, -1, -2) @ Af + P
        beta = np.linalg.solve(AtA, np.swapaxes(Af, -1, -2) @ Yf)
    else:
        beta = np.linalg.pinv(Af) @ Yf
    return Y - A @ beta, beta


def residualise(ref_df: pd.DataFrame, batch_df: pd.DataFrame, kpis: Sequence[str], covariates: Sequence[str],
                flags: dict | None = None, ridge_lambda: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Residualise each KPI on the covariates with a regression **fitted on the usable reference sites only**,
    applied to both sides. Returns (ref_resid, batch_resid, covariate_fit). The residual frames keep every
    non-KPI column (site ids, flag columns) so ``stats.compare_kpis`` can still apply the usable-n gate; the KPI
    columns hold residuals (NaN where the KPI was not usable). ``covariate_fit`` has one row per KPI: r2_ref
    (in-sample R² on the reference fit rows), r2_adj_ref, n_ref_fit, n_params, n_imputed, and the slope per
    covariate in standardised units (``beta_<cov>``).

    Missing covariate values (e.g. ``bright_sep`` is NaN when the bright mode is unresolved) are imputed with the
    pooled (reference + batch) median of that covariate: a label-free constant, so exchangeability under the
    permutation null is preserved. The number of imputed cells is reported.
    """
    covariates = list(covariates)
    pooled = pd.concat([ref_df.assign(_side="ref"), batch_df.assign(_side="batch")], ignore_index=True)
    C = pooled[covariates].astype(float).copy()
    n_imputed = int(C.isna().sum().sum())
    C = C.fillna(C.median())
    Cv = C.to_numpy()
    is_ref = (pooled._side == "ref").to_numpy()
    rr, bb = pooled[is_ref].copy(), pooled[~is_ref].copy()
    fits = []
    for k in kpis:
        y = pooled[k].astype(float).to_numpy()
        usable = np.concatenate([_usable_mask(rr, k, flags), _usable_mask(bb, k, flags)]) & ~np.isnan(y)
        fit_rows = usable & is_ref
        res = np.full(len(y), np.nan)
        rec = dict(kpi=k, r2_ref=np.nan, r2_adj_ref=np.nan, n_ref_fit=int(fit_rows.sum()), n_params=len(covariates) + 1,
                   n_imputed=n_imputed, ridge_lambda=ridge_lambda)
        if fit_rows.sum() >= len(covariates) + 2:
            order = np.concatenate([np.nonzero(fit_rows)[0], np.nonzero(usable & ~fit_rows)[0]])
            R, beta = _fit_residuals(y[order][:, None], Cv[order], int(fit_rows.sum()), ridge_lambda)
            res[order] = R[:, 0]
            yf = y[fit_rows]
            sst = float(((yf - yf.mean()) ** 2).sum())
            sse = float((R[: int(fit_rows.sum()), 0] ** 2).sum())
            n, p = int(fit_rows.sum()), len(covariates)
            rec["r2_ref"] = 1 - sse / sst if sst > 0 else np.nan
            rec["r2_adj_ref"] = 1 - (1 - rec["r2_ref"]) * (n - 1) / (n - p - 1) if sst > 0 and n - p - 1 > 0 else np.nan
            rec.update({f"beta_{c}": float(beta[j + 1, 0]) for j, c in enumerate(covariates)})
        pooled[k] = res
        fits.append(rec)
    ref_res = pooled[is_ref].drop(columns=["_side"]).reset_index(drop=True)
    bat_res = pooled[~is_ref].drop(columns=["_side"]).reset_index(drop=True)
    return ref_res, bat_res, pd.DataFrame(fits)


def adjusted_energy_permutation(Y_ref: np.ndarray, Y_batch: np.ndarray, C_ref: np.ndarray, C_batch: np.ndarray,
                                weights=None, n_mc: int = 5000, seed: int = 0, ridge_lambda: float = 0.0,
                                chunk: int = 500) -> dict:
    """Energy-distance test on covariate-residualised KPIs with the regression **re-fitted on the permuted
    reference rows inside every permutation** (checklist C09), Monte Carlo, p = (b + 1)/(n_mc + 1).

    For the observed allocation and for each of ``n_mc`` random site-label allocations: regress the KPI matrix on
    [1, standardised covariates] using the rows currently labelled reference, take residuals for all rows, then
    ``stats._energy_stat`` (which itself re-standardises by the permuted reference's median / MAD). Rows with any
    NaN in Y or C must be removed by the caller. Returns dict(statistic, p, n_perm, method, n_ref, n_batch, n_cols,
    n_covariates, ridge_lambda, weights).
    """
    Y = np.vstack([np.asarray(Y_ref, float), np.asarray(Y_batch, float)])
    C = np.vstack([np.asarray(C_ref, float), np.asarray(C_batch, float)])
    n_r, n_b = len(Y_ref), len(Y_batch)
    N, k = Y.shape
    w = np.ones(k) if weights is None else np.asarray(weights, float)
    sqrt_w = np.sqrt(w)
    out = dict(statistic=np.nan, p=np.nan, n_perm=0, method="monte_carlo (regression refit per permutation)",
               n_ref=int(n_r), n_batch=int(n_b), n_cols=int(k), n_covariates=int(C.shape[1]), ridge_lambda=ridge_lambda,
               weights=w.tolist())
    if n_r < C.shape[1] + 2 or n_b < 1:
        return out
    R0, _ = _fit_residuals(Y[None], C[None], n_r, ridge_lambda)
    observed = float(stats._energy_stat(R0, n_r, sqrt_w)[0])
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_mc, chunk):
        m = min(chunk, n_mc - start)
        order = np.argsort(rng.random((m, N)), axis=1)
        R, _ = _fit_residuals(Y[order], C[order], n_r, ridge_lambda)
        null = stats._energy_stat(R, n_r, sqrt_w)
        hits += int(stats._hits(null, observed, "greater").sum())
    out.update(statistic=observed, p=(hits + 1) / (n_mc + 1), n_perm=int(n_mc))
    return out


def adjusted_kpi_permutation(y_ref: np.ndarray, y_batch: np.ndarray, C_ref: np.ndarray, C_batch: np.ndarray,
                             statistic="hl_shift", alternative: str = "two-sided", n_mc: int = 5000, seed: int = 0,
                             ridge_lambda: float = 0.0, chunk: int = 2000) -> dict:
    """Univariate analogue of :func:`adjusted_energy_permutation`: site-label Monte Carlo permutation p-value of a
    built-in ``stats`` statistic ('hl_shift', 'median_diff', 'mean_diff', 'cliffs_delta') on residuals whose
    regression is re-fitted on the permuted reference rows in every permutation. Returns dict(p, observed,
    n_perm, method, statistic, n_ref, n_batch)."""
    fn, name, vectorised = stats._resolve_statistic(statistic)
    if not vectorised:
        raise ValueError("adjusted_kpi_permutation needs a built-in vectorised statistic")
    y = np.concatenate([np.asarray(y_ref, float), np.asarray(y_batch, float)])
    C = np.vstack([np.asarray(C_ref, float), np.asarray(C_batch, float)])
    n_r, n_b = len(y_ref), len(y_batch)
    N = len(y)
    out = dict(p=np.nan, observed=np.nan, n_perm=0, method="monte_carlo (regression refit per permutation)",
               statistic=name, alternative=alternative, n_ref=int(n_r), n_batch=int(n_b))
    if n_r < C.shape[1] + 2 or n_b < 1:
        return out
    R0, _ = _fit_residuals(y[None, :, None], C[None], n_r, ridge_lambda)
    observed = float(fn(R0[0, :n_r, 0], R0[0, n_r:, 0]))
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_mc, chunk):
        m = min(chunk, n_mc - start)
        order = np.argsort(rng.random((m, N)), axis=1)
        R, _ = _fit_residuals(y[order][..., None], C[order], n_r, ridge_lambda)
        null = np.asarray(fn(R[:, :n_r, 0], R[:, n_r:, 0]), float)
        hits += int(stats._hits(null, observed, alternative).sum())
    out.update(p=(hits + 1) / (n_mc + 1), observed=observed, n_perm=int(n_mc))
    return out


def adjusted_comparison(ref_df: pd.DataFrame, batch_df: pd.DataFrame, kpis: Sequence[str], flags_df: pd.DataFrame,
                        covariates: Sequence[str] = ACQ_COVARIATES_ADJUST, n_mc: int = 5000, seed: int = 0,
                        ridge_lambda: float = 0.0, energy_kpis: Sequence[str] = PRIMARY_KPIS, weights=None,
                        primary: Sequence[str] = PRIMARY_KPIS, flags: dict | None = None, statistic="hl_shift",
                        alpha: float = 0.05, n_boot: int = 2000) -> dict:
    """Adjusted sensitivity view (plan §2.6 item 3): KPIs residualised on acquisition covariates, comparison re-run.

    Covariates come from ``flags_df`` (:func:`derive_flags`) joined on (batch, site); columns already present in the
    site frames are overridden by the flag table's. The regression (intercept + standardised covariates; OLS when
    ``ridge_lambda == 0``, ridge otherwise, penalty on slopes only) is fitted on the usable **reference** sites and
    applied to both sides.

    Default covariates are the three a-priori ones in ``ACQ_COVARIATES_ADJUST`` (bright_sep, bse_p1,
    etd_boundary_sharpness) — see the constant's comment: 7 covariates + intercept on 17 reference rows is nearly
    saturated and the fit would absorb sampling noise, inflating the apparent attenuation. To use the full
    ``ACQ_COVARIATES`` list pass it with ``ridge_lambda`` of about 1 (units: squared standardised slopes).

    Permutation nulls: the regression is **re-fitted on the permuted reference rows inside every permutation**
    (checklist C09), both for the multivariate energy test (:func:`adjusted_energy_permutation`) and for every
    per-KPI p-value (:func:`adjusted_kpi_permutation`). ``stats.compare_kpis`` is also run on the fixed residuals
    for the effect sizes, bootstrap intervals and Cliff's delta; its own ``p_perm`` (residuals held fixed) is kept in
    ``p_perm_fixed_residuals`` for comparison, and ``p_perm``, ``p_holm`` (primary family), ``p_reported`` and
    ``flag_alpha`` are replaced by the refit-per-permutation values. Both are Monte Carlo with n stated.

    Reading the result (measured on Batches 1–3, 2026-10-03): adjustment is **not** guaranteed to attenuate. With
    Batch 3 as reference the three covariates explain 50–75 % of the reference variance of the pore KPIs, but on
    the 10 ordinary reference sites alone the covariate–KPI |Spearman ρ| is ≤ 0.36: the fit is mostly encoding the
    reference's own sub-populations (bse_p1 identifies the grey-pore group, which has the lowest pore_frac). The
    regression then predicts a *higher* pore_frac for every site with p1 = 0 and the residual reference MAD shrinks,
    so the adjusted energy statistic rose (attenuation −1.4 for Batch 1, −3.0 for Batch 2) while the stratified view
    stayed inside the null. A negative attenuation therefore means "the shift is not explained away by these
    covariates", nothing more; it is not evidence of a larger material shift. ``decision.attenuation`` treats negative
    shares as "no attenuation", which is the intended reading.

    Returns dict(compare_adjusted, energy_adjusted, covariate_fit (per-KPI reference R², adjusted R², n fit rows,
    slopes), n_perm, covariates, ridge_lambda, n_ref_energy, n_batch_energy, residual frames).
    """
    covariates = list(covariates)
    r = _attach_flags(ref_df, flags_df, covariates)
    b = _attach_flags(batch_df, flags_df, covariates)
    ref_res, bat_res, fit = residualise(r, b, kpis, covariates, flags, ridge_lambda)

    cmp_ = stats.compare_kpis(ref_res, bat_res, kpis, primary=primary, flags=flags, seed=seed, n_boot=n_boot,
                              statistic=statistic, alpha=alpha)
    cmp_ = cmp_.rename(columns={"p_perm": "p_perm_fixed_residuals", "p_method": "p_method_fixed_residuals",
                                "n_perm": "n_perm_fixed_residuals"})
    # per-KPI refit-per-permutation p-values on the ORIGINAL values (the residualisation happens inside the loop)
    Cpool = pd.concat([r[covariates], b[covariates]]).astype(float)
    Cpool = Cpool.fillna(Cpool.median())
    Cr, Cb = Cpool.iloc[: len(r)].to_numpy(), Cpool.iloc[len(r):].to_numpy()
    seeds = np.random.SeedSequence(seed).generate_state(max(len(kpis), 1))
    p_refit, n_refit = [], []
    for k, s in zip(kpis, seeds):
        mr = _usable_mask(r, k, flags) & r[k].notna().to_numpy()
        mb = _usable_mask(b, k, flags) & b[k].notna().to_numpy()
        pt = adjusted_kpi_permutation(r.loc[mr, k].to_numpy(float), b.loc[mb, k].to_numpy(float), Cr[mr], Cb[mb],
                                      statistic=statistic, n_mc=n_mc, seed=int(s), ridge_lambda=ridge_lambda)
        p_refit.append(pt["p"])
        n_refit.append(pt["n_perm"])
    cmp_["p_perm"] = p_refit
    cmp_["p_method"] = "monte_carlo (regression refit per permutation)"
    cmp_["n_perm"] = n_refit
    prim = cmp_["is_primary"].to_numpy()
    cmp_["p_holm"] = np.nan
    if prim.any():
        cmp_.loc[prim, "p_holm"] = stats.holm(cmp_.loc[prim, "p_perm"].to_numpy())
    cmp_["p_reported"] = np.where(prim, cmp_["p_holm"], cmp_["p_perm"])
    cmp_["flag_alpha"] = cmp_["p_reported"] <= alpha

    # multivariate: rows usable on every energy KPI (same rule as the unadjusted view), covariates imputed as above
    Xr, Xb = _energy_frame(r, energy_kpis, flags), _energy_frame(b, energy_kpis, flags)
    okr, okb = Xr.notna().all(1).to_numpy(), Xb.notna().all(1).to_numpy()
    energy = adjusted_energy_permutation(Xr[okr].to_numpy(), Xb[okb].to_numpy(), Cr[okr], Cb[okb], weights=weights,
                                         n_mc=n_mc, seed=seed, ridge_lambda=ridge_lambda)
    energy.update(columns=list(energy_kpis), n_dropped_ref=int((~okr).sum()), n_dropped_batch=int((~okb).sum()))
    return dict(compare_adjusted=cmp_, energy_adjusted=energy, covariate_fit=fit, n_perm=int(n_mc),
                covariates=covariates, ridge_lambda=ridge_lambda, n_ref_energy=energy["n_ref"],
                n_batch_energy=energy["n_batch"], ref_residuals=ref_res, batch_residuals=bat_res)


# ----------------------------------------------------------------------------------------------------------------
# the three views in one call
# ----------------------------------------------------------------------------------------------------------------
def three_views(ref_df: pd.DataFrame, batch_df: pd.DataFrame, kpis: Sequence[str], flags_df: pd.DataFrame,
                keep_groups: Sequence[str] = ("ordinary",), covariates: Sequence[str] = ACQ_COVARIATES_ADJUST,
                ridge_lambda: float = 0.0, energy_kpis: Sequence[str] = PRIMARY_KPIS, weights=None,
                n_mc_adjusted: int = 5000, n_mc_energy: int = 5000, **compare_kwargs) -> dict:
    """Unadjusted / stratified / adjusted views of one batch-vs-reference comparison, plus
    ``decision.attenuation`` of the multivariate statistic. This is the call the decision and report layers use.

    ``compare_kwargs`` go to ``stats.compare_kpis`` in every view (statistic='hl_shift' recommended, seed, flags,
    n_boot, alpha, primary). ``weights`` (e.g. ``physics.weights_vector(PRIMARY_KPIS)``) are applied to the energy
    statistic in **all three** views so the attenuation shares compare like with like. The unadjusted energy test
    uses only sites usable on every energy KPI (bright_* excluded on low-contrast sites), reported in
    ``n_dropped_*``.

    Returns dict(unadjusted={compare, energy}, stratified={...stratified_comparison...}, adjusted={...adjusted_comparison...},
    attenuation={stratified, adjusted}, note). Attenuation is descriptive (share by which the energy statistic drops),
    not a causal share; the stratified share is NaN when that view refused to run.
    """
    flags = compare_kwargs.get("flags")
    seed = int(compare_kwargs.get("seed", 0))
    unadj = dict(compare=stats.compare_kpis(ref_df, batch_df, kpis, **compare_kwargs),
                 energy=_energy(ref_df, batch_df, energy_kpis, flags, weights, n_mc_energy, seed))
    strat = stratified_comparison(ref_df, batch_df, kpis, flags_df, keep_groups=keep_groups, energy_kpis=energy_kpis,
                                  weights=weights, n_mc_energy=n_mc_energy, **compare_kwargs)
    adj_kw = {k: compare_kwargs[k] for k in ("primary", "flags", "statistic", "alpha", "n_boot") if k in compare_kwargs}
    adj = adjusted_comparison(ref_df, batch_df, kpis, flags_df, covariates=covariates, n_mc=n_mc_adjusted, seed=seed,
                              ridge_lambda=ridge_lambda, energy_kpis=energy_kpis, weights=weights, **adj_kw)
    att = _attenuation(unadj["energy"], strat["energy"], adj["energy_adjusted"])
    return dict(unadjusted=unadj, stratified=strat, adjusted=adj, attenuation=att,
                note=("sensitivity analysis: attenuation is the share by which the energy statistic drops under the "
                      "stratified / adjusted view; it is not a share of the shift caused by the instrument"))


# ----------------------------------------------------------------------------------------------------------------
# three-channel agreement (notebook 01 §6d)
# ----------------------------------------------------------------------------------------------------------------
def three_channel_agreement(sites_df: pd.DataFrame, images_df: pd.DataFrame, reference_batch: str | None = None,
                            z_shift: float = Z_SHIFT) -> pd.DataFrame:
    """Per-site robust z of the three channel medians (BSE, ETD, Inlens p50) and of three texture measures
    (corr_len_px, etd_grad_energy, inlens_grad_energy), with the across-channel spread of each block, exactly as in
    notebook 01 §6d: z = (x − median) / (1.4826 · MAD + 1e-9), ``intensity_disagreement`` = std of the three
    intensity z (ddof 1), ``texture_disagreement`` = std of the three texture z.

    Centre and scale are taken over **all sites supplied** (the notebook's pooled convention) unless
    ``reference_batch`` is given, in which case they come from that batch's sites and the z-scores read as
    "relative to the reference".

    ``pattern`` (one line per site, fixed thresholds, stated here):
    * intensity shifted  := max |z| over the three channel medians >= ``z_shift`` (2.0);
    * texture shifted    := at least two of the three texture |z| >= ``z_shift`` (a shift that one channel alone
      shows is not cross-channel agreement and does not count);
    * "intensity-only shift"            intensity shifted, texture not  (grey-pore pattern: detector / preparation);
    * "texture shift across channels"   texture shifted, intensity not  (cracked-site pattern: structural);
    * "mixed"                           both;   "none"  neither.
    These labels qualify a verdict's reading (§2.6 "alongside"); they never drive one.
    """
    sites = sites_df.copy()
    sites["site"] = sites["site"].astype(str)
    img = images_df.copy()
    img["site"] = img["site"].astype(str)
    img["det"] = _norm_det(img["det"])
    q = img.pivot_table(index=["batch", "site"], columns="det", values="p50").reset_index()
    q = q.rename(columns={"BSE": "BSE_p50", "ETD": "ETD_p50", "Inlens": "Inlens_p50"})
    keep = ["batch", "site"] + [c for c in _TEXTURE_COLS if c in sites.columns] + \
           [c for c in ("acquisition_group", "bright_low_contrast") if c in sites.columns]
    A = sites[keep].merge(q, on=["batch", "site"], how="inner")
    cols = [c for c in _INTENSITY_COLS + _TEXTURE_COLS if c in A.columns]
    ref = A if reference_batch is None else A[A.batch == reference_batch]
    Z = A.copy()
    for c in cols:
        x = ref[c].astype(float)
        med = np.nanmedian(x)
        s = 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-9
        Z[f"z_{c}"] = (A[c].astype(float) - med) / s
    zi = [f"z_{c}" for c in _INTENSITY_COLS if c in A.columns]
    zt = [f"z_{c}" for c in _TEXTURE_COLS if c in A.columns]
    Z["intensity_disagreement"] = Z[zi].std(axis=1)
    Z["texture_disagreement"] = Z[zt].std(axis=1)
    Z["intensity_channels_shifted"] = (Z[zi].abs() >= z_shift).sum(axis=1).astype(int)
    Z["texture_channels_shifted"] = (Z[zt].abs() >= z_shift).sum(axis=1).astype(int)
    int_shift = Z["intensity_channels_shifted"] >= 1
    tex_shift = Z["texture_channels_shifted"] >= 2
    Z["intensity_shifted"] = int_shift
    Z["texture_shifted"] = tex_shift
    Z["pattern"] = np.select([int_shift & tex_shift, int_shift, tex_shift],
                             ["mixed", "intensity-only shift", "texture shift across channels"], default="none")
    Z.attrs["z_shift"] = z_shift
    Z.attrs["reference_batch"] = reference_batch
    return Z.reset_index(drop=True)
