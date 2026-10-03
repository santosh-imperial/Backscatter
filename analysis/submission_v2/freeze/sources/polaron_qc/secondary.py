"""E25 battery geometry: extraction and descriptive summaries, never verdict inputs.

The original primary/statistical/classifier lists remain unchanged. Threshold
perturbation ranges describe measurement sensitivity; site-bootstrap intervals
describe sampling variability, conditional on unverified specimen independence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import battery_metrics, void_metrics

VERSION = "1.0.0"
DIAGNOSTIC_ONLY = ("bright_void_boundary_frac",)  # E25: constant zero on all usable known sites
EXTRACTED_NAMES = tuple(battery_metrics.KPI_NAMES) + tuple(void_metrics.KPI_NAMES)
KPI_NAMES = tuple(k for k in EXTRACTED_NAMES if k not in DIAGNOSTIC_ONLY)
KPI_DEFINITIONS = {**battery_metrics.KPI_DEFINITIONS, **void_metrics.KPI_DEFINITIONS}
PORE_DEPENDENT = set(void_metrics.KPI_NAMES) | {k for k, d in battery_metrics.KPI_DEFINITIONS.items() if "grey_pore" in d["required_flags"]}
BRIGHT_DEPENDENT = {k for k, d in battery_metrics.KPI_DEFINITIONS.items() if "bright_low_contrast" in d["required_flags"]}


def measure(pore, bright):
    """Nominal secondary values and observability counts from existing masks."""
    return {**battery_metrics.measure_neighbourhoods(pore, bright),
            **void_metrics.long_void_context(pore)}


def extract(smoothed, th_lo, th_hi, *, pore, bright):
    """Reuse nominal masks; perturb both thresholds by -5/+5 grey levels.

    This is a specified coupled perturbation, not a complete independent
    threshold grid or an interval for accuracy. Mask definitions stay unchanged.
    """
    if np.ndim(smoothed) != 2 or np.shape(smoothed) != np.shape(pore) or np.shape(pore) != np.shape(bright):
        raise ValueError("Smoothed image and phase masks must share a 2-D shape")
    if not np.isfinite([th_lo, th_hi]).all() or th_lo >= th_hi:
        raise ValueError("Finite ordered phase thresholds required")
    nominal = measure(pore, bright)
    variants = [nominal]
    for offset in (-5, 5):
        p = ndi.binary_opening(smoothed < th_lo + offset, iterations=1)
        b = ndi.binary_opening(smoothed > th_hi + offset, iterations=1)
        variants.append(measure(p, b))
    for k in EXTRACTED_NAMES:
        values = np.array([v[k] for v in variants], dtype=float)
        finite = values[np.isfinite(values)]
        nominal[k + "_sensitivity_min"] = float(finite.min()) if len(finite) else np.nan
        nominal[k + "_sensitivity_max"] = float(finite.max()) if len(finite) else np.nan
        nominal[k + "_sensitivity_n"] = int(len(finite))
    nominal["battery_secondary_version"] = VERSION
    return nominal


def _bool_column(df, key):
    """Known boolean values only; missing flags cannot imply reliable data."""
    if key not in df:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    s = df[key].astype(str).str.strip().str.lower()
    known = s.isin(["true", "false", "1", "0", "1.0", "0.0"])
    return s.isin(["true", "1", "1.0"]), known


def usable_mask(df, kpi):
    """Quality gating matches data-derived contrast/grey-pore acquisition flags.

    pore_mode_resolved is false on all current sites; it is counted separately
    as a method diagnostic, not silently treated as resolved or used to infer
    accuracy. Phase-boundary expert review remains pending for every site.
    """
    if kpi not in df:
        return pd.Series(False, index=df.index)
    mask = pd.to_numeric(df[kpi], errors="coerce").map(np.isfinite)
    if kpi in BRIGHT_DEPENDENT:
        bad, known = _bool_column(df, "bright_low_contrast")
        mask &= known & ~bad
    if kpi in PORE_DEPENDENT:
        bad, known = _bool_column(df, "grey_pore")
        mask &= known & ~bad
    return mask


def with_observed_quality(raw_sites, derived_sites, flags):
    """Secondary-only quality view; legacy false defaults are not observations.

    Preserve original decision-table flags. Bright quality needs an actual
    extraction boolean; pore quality needs a finite observed BSE black level.
    An acquisition list/default cannot make a missing observation usable.
    """
    out = derived_sites.copy()
    raw = raw_sites.copy()
    raw["site"] = raw.site.astype(str)
    raw = raw.set_index("site")
    assert raw.index.is_unique
    _, bright_known = _bool_column(raw, "bright_low_contrast")
    sid = out.site.astype(str)
    known = sid.map(bright_known).fillna(False).to_numpy()
    out["bright_low_contrast"] = out["bright_low_contrast"].astype(object)
    out.loc[~known, "bright_low_contrast"] = pd.NA
    f = flags.copy()
    f["site"] = f.site.astype(str)
    assert not f.site.duplicated().any()
    black = pd.to_numeric(f.set_index("site")["bse_p1"], errors="coerce") if "bse_p1" in f else pd.Series(dtype=float)
    known = sid.map(black.map(np.isfinite)).fillna(False).to_numpy()
    out["grey_pore"] = out["grey_pore"].astype(object)
    out.loc[~known, "grey_pore"] = pd.NA
    return out


def summarize(reference, batch, *, n_boot=2000, seed=0):
    """Separate raw-unit comparisons; no p-value, score, alert or primary status.

    CIs are percentile bootstrap intervals for difference of site medians,
    unavailable when either side has fewer than two usable sites. Stable
    per-KPI seeds prevent additions from changing existing summaries.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rows = []
    for k in KPI_NAMES:
        rm, bm = usable_mask(reference, k), usable_mask(batch, k)
        r = pd.to_numeric(reference.loc[rm, k], errors="coerce").to_numpy() if k in reference else np.array([])
        b = pd.to_numeric(batch.loc[bm, k], errors="coerce").to_numpy() if k in batch else np.array([])
        row = dict(kpi=k, units=KPI_DEFINITIONS[k]["units"], n_ref_usable=len(r), n_batch_usable=len(b),
                   n_ref_excluded=len(reference)-len(r), n_batch_excluded=len(batch)-len(b),
                   ref_median=float(np.median(r)) if len(r) else np.nan,
                   batch_median=float(np.median(b)) if len(b) else np.nan,
                   median_difference=np.nan, ci_low=np.nan, ci_high=np.nan,
                   interval_available=False, is_primary=False, qc_role="secondary_only_excluded_from_verdict")
        if len(r) and len(b):
            row["median_difference"] = row["batch_median"]-row["ref_median"]
        if len(r) >= 2 and len(b) >= 2:
            # A stable seed uses the metric name rather than its list position.
            import hashlib
            ident = int.from_bytes(hashlib.sha256(k.encode()).digest()[:4], "little")
            rng = np.random.default_rng(np.random.SeedSequence([seed, ident]))
            draws = np.median(rng.choice(b, (n_boot, len(b))), axis=1)-np.median(rng.choice(r, (n_boot, len(r))), axis=1)
            row["ci_low"], row["ci_high"] = map(float, np.quantile(draws, [.025, .975]))
            row["interval_available"] = True
        for df, use, tag in ((reference, rm, "ref"), (batch, bm, "batch")):
            resolved, known = _bool_column(df, "pore_mode_resolved")
            row[f"n_{tag}_pore_threshold_unresolved"] = int((use & known & ~resolved).sum()) if k in PORE_DEPENDENT else 0
            row[f"n_{tag}_pore_threshold_unknown"] = int((use & ~known).sum()) if k in PORE_DEPENDENT else 0
            lo, hi = k + "_sensitivity_min", k + "_sensitivity_max"
            if lo in df and hi in df:
                widths = pd.to_numeric(df.loc[use, hi], errors="coerce")-pd.to_numeric(df.loc[use, lo], errors="coerce")
                row[f"{tag}_median_sensitivity_range"] = float(widths.median()) if widths.notna().any() else np.nan
            else:
                row[f"{tag}_median_sensitivity_range"] = np.nan
            count = k + "_sensitivity_n"
            row[f"n_{tag}_sensitivity_incomplete"] = int((pd.to_numeric(df.loc[use, count], errors="coerce").fillna(0) < 3).sum()) if count in df else int(use.sum())
        rows.append(row)
    return pd.DataFrame(rows)
