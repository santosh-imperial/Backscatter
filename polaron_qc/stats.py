"""polaron_qc.stats — small-sample statistics engine (docs/qc_plan.md §2.0, §2.3, §2.4, §2.7, §2.9).

Scope and discipline
--------------------
* The unit of evidence is the **site**. Every function here takes site-level arrays or
  DataFrames (one row per site). Nothing in this module ever treats patches, particles or
  pixels as n (plan §2.0 rule 6).
* Every quantity that depends on which sites are "reference" (MAD scaling, per-column
  standardisation) is **re-computed inside every bootstrap draw and every permutation**
  (plan §2.0 rule 2, checklist C09).
* p-values come from site-label permutation tests: exact enumeration when the number of
  allocations is affordable, otherwise Monte Carlo with the resample count reported and the
  (b + 1)/(n + 1) convention (checklist C15).
* Bootstrap-over-sites intervals are labelled *approximate* (plan §2.0 rule 3). With 7 sites
  a percentile bootstrap of a median is coarse and tends to under-cover; it is an interval,
  not a probability of being right (checklist C17).
* "Exceeds the reference maximum" is an evidence flag whose i.i.d. false-alarm rate is
  n_batch / (n_ref + n_batch) per KPI; it is exposed, not hidden (checklist C10).
* Usable n after quality flags, not folder counts, enters every test (checklist C14).

Pure numpy / scipy / pandas. No imports from the sibling modules (features, ml, physics),
which are built in parallel.

Conventions
-----------
* Shifts are *batch minus reference*; a positive shift, positive Cliff's delta or
  "higher" direction means the incoming batch is higher than the reference.
* MAD is the median absolute deviation scaled by 1.4826 (consistent with the SD of a
  normal). All "MAD units" in this module use that scaling.
* `seed` is an integer; every function is deterministic for a fixed seed.
"""
from __future__ import annotations

import functools
import itertools
import math
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from polaron_qc import CRACKED_SITES, GREY_PORE_SITES, LOW_CONTRAST_SITES, PRIMARY_KPIS

__all__ = [
    "MAD_SCALE", "mad", "cliffs_delta", "robust_shift", "permutation_test", "holm", "bh",
    "energy_distance_test", "per_site_drift", "mdc", "iid_flag_probability", "local_exceedance",
    "jackknife", "stability_share", "reference_split_diagnostics", "usable_n", "usable_values",
    "flag_rule", "compare_kpis", "DEFAULT_FLAGS",
]

MAD_SCALE = 1.4826

#: Site flags keyed by the quality issue they encode (values are site ids). Mirrors
#: polaron_qc.__init__; `usable_n` also honours a boolean `bright_low_contrast` column
#: when the DataFrame has one, so an unseen batch's own flags are picked up automatically.
DEFAULT_FLAGS = {
    "bright_low_contrast": list(LOW_CONTRAST_SITES),
    "grey_pore": list(GREY_PORE_SITES),
    "cracked": list(CRACKED_SITES),
}


# ----------------------------------------------------------------------------------------
# Basic robust helpers
# ----------------------------------------------------------------------------------------
def _as1d(x) -> np.ndarray:
    """Flatten to float array and drop NaN. Callers are expected to pass usable sites only;
    NaN dropping is a safety net, and reported n values are after dropping."""
    arr = np.asarray(x, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def mad(x, axis=None, scale: float = MAD_SCALE) -> np.ndarray | float:
    """Median absolute deviation times `scale` (1.4826 by default)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x, axis=axis, keepdims=True)
    return scale * np.median(np.abs(x - med), axis=axis)


def cliffs_delta(x_ref, x_batch) -> float:
    """Cliff's delta = P(batch > ref) - P(batch < ref) over all site pairs. Positive means the
    batch tends to be higher than the reference."""
    r, b = _as1d(x_ref), _as1d(x_batch)
    if len(r) == 0 or len(b) == 0:
        return float("nan")
    return float(np.mean(np.sign(b[:, None] - r[None, :])))


def _robust_scale(ref: np.ndarray, axis: int) -> np.ndarray:
    """MAD along `axis` with fallbacks for degenerate columns: MAD == 0 -> SD, SD == 0 -> 1.
    Degenerate columns are rare with continuous KPIs but can occur in small permutations."""
    med = np.median(ref, axis=axis, keepdims=True)
    md = MAD_SCALE * np.median(np.abs(ref - med), axis=axis, keepdims=True)
    sd = ref.std(axis=axis, keepdims=True)
    return med, np.where(md > 0, md, np.where(sd > 0, sd, 1.0))


# ----------------------------------------------------------------------------------------
# Robust standardised shift with approximate bootstrap-over-sites interval
# ----------------------------------------------------------------------------------------
def robust_shift(x_ref, x_batch, n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Robust standardised shift: (median(batch) - median(ref)) / MAD(ref).

    The interval is a percentile bootstrap over **sites** (reference and batch resampled
    independently with replacement); the reference MAD is re-computed on every bootstrap
    draw of the reference, so scaling uncertainty is inside the interval. It is labelled
    approximate: with 7 batch sites the bootstrap distribution of a median is coarse and
    the percentile interval under-covers (see tests/test_stats.py for measured coverage).

    Returns a dict with shift_mad, ci_low, ci_high, ci_method, cliffs_delta, n_ref, n_batch,
    ref_median, batch_median, ref_mad, batch_mad, n_boot, n_boot_valid (draws with a
    non-zero reference MAD). shift_mad is NaN when the reference MAD is zero.
    """
    r, b = _as1d(x_ref), _as1d(x_batch)
    out = dict(shift_mad=np.nan, ci_low=np.nan, ci_high=np.nan,
               ci_method="bootstrap-over-sites percentile (approximate)",
               cliffs_delta=np.nan, n_ref=int(len(r)), n_batch=int(len(b)),
               ref_median=np.nan, batch_median=np.nan, ref_mad=np.nan, batch_mad=np.nan,
               n_boot=int(n_boot), n_boot_valid=0)
    if len(r) < 2 or len(b) < 1:
        return out
    ref_med, bat_med = float(np.median(r)), float(np.median(b))
    ref_mad = float(mad(r))
    out.update(ref_median=ref_med, batch_median=bat_med, ref_mad=ref_mad, batch_mad=float(mad(b)),
               cliffs_delta=cliffs_delta(r, b))
    out["shift_mad"] = (bat_med - ref_med) / ref_mad if ref_mad > 0 else np.nan
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        rb = r[rng.integers(0, len(r), size=(n_boot, len(r)))]
        bb = b[rng.integers(0, len(b), size=(n_boot, len(b)))]
        mad_b = mad(rb, axis=1)
        valid = mad_b > 0
        shifts = np.full(n_boot, np.nan)
        shifts[valid] = (np.median(bb[valid], axis=1) - np.median(rb[valid], axis=1)) / mad_b[valid]
        out["n_boot_valid"] = int(valid.sum())
        if valid.any():
            lo, hi = np.nanpercentile(shifts, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
            out["ci_low"], out["ci_high"] = float(lo), float(hi)
    return out


# ----------------------------------------------------------------------------------------
# Site-label permutation test
# ----------------------------------------------------------------------------------------
def _stat_median_diff(R, B):
    return np.median(B, axis=-1) - np.median(R, axis=-1)


def _stat_mean_diff(R, B):
    return np.mean(B, axis=-1) - np.mean(R, axis=-1)


def _stat_hl_shift(R, B):
    """Hodges-Lehmann shift: median of all pairwise differences b_i - r_j. Robust like the
    median difference but far less discrete under permutation (7 x 17 = 119 pairwise
    differences per allocation instead of two order statistics)."""
    return np.median(B[..., :, None] - R[..., None, :], axis=(-2, -1))


def _stat_cliffs_delta(R, B):
    """Cliff's delta = mean sign(b_i - r_j); rank-based, equivalent to Mann-Whitney U."""
    return np.mean(np.sign(B[..., :, None] - R[..., None, :]), axis=(-2, -1))


_BUILTIN_STATS = {"median_diff": _stat_median_diff, "mean_diff": _stat_mean_diff,
                  "hl_shift": _stat_hl_shift, "cliffs_delta": _stat_cliffs_delta}


def _resolve_statistic(statistic):
    """Return (fn, name, vectorised). Built-in statistics are evaluated on 2-D arrays
    (allocations x sites) at once; a user callable `fn(ref_values, batch_values) -> float`
    is evaluated one allocation at a time."""
    if callable(statistic):
        return statistic, getattr(statistic, "__name__", "callable"), False
    if statistic not in _BUILTIN_STATS:
        raise ValueError(f"unknown statistic {statistic!r}; use 'median_diff', 'mean_diff' or a callable")
    return _BUILTIN_STATS[statistic], statistic, True


@functools.lru_cache(maxsize=8)
def _exact_allocations(n_total: int, n_batch: int):
    """All C(n_total, n_batch) ways of choosing which pooled sites are 'batch'. Returns
    (batch_idx, ref_idx) int16 arrays of shape (n_alloc, n_batch) and (n_alloc, n_ref).
    Cached because the same (24, 7) layout is reused for every KPI and every simulation."""
    n_alloc = math.comb(n_total, n_batch)
    flat = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n_total), n_batch)),
                       dtype=np.int16, count=n_alloc * n_batch)
    batch_idx = flat.reshape(n_alloc, n_batch)
    mask = np.zeros((n_alloc, n_total), dtype=bool)
    np.put_along_axis(mask, batch_idx.astype(np.intp), True, axis=1)
    ref_idx = np.nonzero(~mask)[1].reshape(n_alloc, n_total - n_batch).astype(np.int16)
    return batch_idx, ref_idx


def _eval_stat(fn, vectorised, pooled, ref_idx, batch_idx):
    """Evaluate the statistic for every allocation (rows of ref_idx / batch_idx)."""
    if vectorised:
        return np.asarray(fn(pooled[ref_idx], pooled[batch_idx]), dtype=float)
    return np.array([fn(pooled[ri], pooled[bi]) for ri, bi in zip(ref_idx, batch_idx)], dtype=float)


def _hits(null: np.ndarray, observed: float, alternative: str) -> np.ndarray:
    """Allocations at least as extreme as the observed one, with a floating-point tolerance
    so that the observed allocation itself always counts (ties count as extreme)."""
    tol = 1e-12 * max(1.0, abs(observed))
    if alternative == "two-sided":
        return np.abs(null) >= abs(observed) - tol
    if alternative == "greater":
        return null >= observed - tol
    if alternative == "less":
        return null <= observed + tol
    raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")


def permutation_test(x_ref, x_batch, statistic="median_diff", alternative: str = "two-sided",
                     exact_max: int = 1_000_000, n_mc: int = 20_000, seed: int = 0,
                     chunk: int = 20_000) -> dict:
    """Site-label permutation test of batch vs reference.

    The null is the exchangeability of site labels within the pooled pair, i.e. the null
    that matches the actual comparison and its usable sample sizes (checklist C09).

    * **exact**: when C(n_ref + n_batch, n_batch) <= `exact_max` every allocation is
      enumerated (itertools.combinations over site indices, cached) and
      p = #{allocations with |T| >= |T_obs|} / #allocations. The observed allocation is one
      of them, so p >= 1 / #allocations (7 vs 7: 1/3432, two-sided >= 2/3432 ~ 0.0006).
    * **monte_carlo**: otherwise `n_mc` random allocations are drawn and
      p = (b + 1) / (n_mc + 1), where b counts resamples at least as extreme as observed
      (Phipson & Smyth convention; never returns p = 0).

    Two-sided p uses |T| >= |T_obs| (the scipy convention). For a difference of medians with
    unequal n the null distribution of T need not be exactly symmetric, so this is a
    convention, stated here, not the only possible one.

    `statistic` is 'median_diff' (default, the plan's primary statistic), 'mean_diff',
    'hl_shift' (Hodges-Lehmann median of pairwise differences), 'cliffs_delta' (rank-based)
    or a callable `fn(ref_values, batch_values) -> float` (evaluated per allocation; slow
    for large exact enumerations, use a smaller `exact_max` or Monte Carlo).

    Discreteness warning (measured, see tests): with 7 vs 17 sites the difference of
    medians takes only ~56 distinct |T| values over the 346,104 allocations, so its exact
    p-value is strongly super-uniform (about a fifth of null p-values equal 1.0) and power
    is lost. 'hl_shift' keeps robustness with a nearly continuous null; prefer it when the
    decision layer can accept a different test statistic from the reported shift estimate.

    Returns dict(p, method, n_perm, observed, statistic, alternative, n_ref, n_batch,
    min_p_attainable).
    """
    r, b = _as1d(x_ref), _as1d(x_batch)
    fn, name, vectorised = _resolve_statistic(statistic)
    n_r, n_b = len(r), len(b)
    out = dict(p=np.nan, method=None, n_perm=0, observed=np.nan, statistic=name,
               alternative=alternative, n_ref=int(n_r), n_batch=int(n_b), min_p_attainable=np.nan)
    if n_r < 1 or n_b < 1:
        return out
    pooled = np.concatenate([r, b])
    n_total = n_r + n_b
    observed = float(fn(r, b))
    out["observed"] = observed
    n_alloc = math.comb(n_total, n_b)
    if n_alloc <= exact_max:
        batch_idx, ref_idx = _exact_allocations(n_total, n_b)
        hits = 0
        for start in range(0, n_alloc, chunk):
            sl = slice(start, min(start + chunk, n_alloc))
            null = _eval_stat(fn, vectorised, pooled, ref_idx[sl], batch_idx[sl])
            hits += int(_hits(null, observed, alternative).sum())
        out.update(p=hits / n_alloc, method="exact", n_perm=int(n_alloc), min_p_attainable=1.0 / n_alloc)
        return out
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_mc, chunk):
        m = min(chunk, n_mc - start)
        order = np.argsort(rng.random((m, n_total)), axis=1)
        null = _eval_stat(fn, vectorised, pooled, order[:, n_b:], order[:, :n_b])
        hits += int(_hits(null, observed, alternative).sum())
    out.update(p=(hits + 1) / (n_mc + 1), method="monte_carlo", n_perm=int(n_mc),
               min_p_attainable=1.0 / (n_mc + 1))
    return out


# ----------------------------------------------------------------------------------------
# Multiplicity
# ----------------------------------------------------------------------------------------
def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control, valid under any
    dependence). Used for the **five primary KPIs only** (plan §2.0 rule 1). NaN entries
    are ignored for the family size and returned as NaN."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return adj
    pv = p[ok]
    order = np.argsort(pv)
    stepped = pv[order] * (m - np.arange(m))
    stepped = np.maximum.accumulate(stepped)
    res = np.empty(m)
    res[order] = np.minimum(stepped, 1.0)
    adj[ok] = res
    return adj


def bh(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate). Provided for the
    secondary KPIs as an *optional* view; the plan (§2.4) reports secondary KPIs
    **uncorrected and labelled descriptive**, so `compare_kpis` does not apply this. NaN
    entries are ignored for the family size and returned as NaN."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return adj
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(ranked, 1.0)
    adj[ok] = res
    return adj


# ----------------------------------------------------------------------------------------
# Multivariate: energy distance with reference scaling re-fitted inside each permutation
# ----------------------------------------------------------------------------------------
def _to_matrix(X):
    """Accept DataFrame / 2-D array; return (values float 2-D, index labels, column labels)."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), list(X.index), list(X.columns)
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr, list(range(arr.shape[0])), [f"col{j}" for j in range(arr.shape[1])]


def _energy_stat(Z: np.ndarray, n_ref: int, sqrt_w: np.ndarray) -> np.ndarray:
    """Energy statistic for a stack of permutations.
    Z: (m, N, k) rows already permuted so that the first n_ref rows are the 'reference'.
    Each permutation is standardised by **its own** reference rows (median / MAD per
    column), columns are multiplied by sqrt(weights), and
    E = 2 mean d(X, Y) - mean d(X, X) - mean d(Y, Y) on Euclidean distances (Székely-Rizzo,
    1/n² means including the zero diagonal)."""
    ref = Z[:, :n_ref]
    med, scale = _robust_scale(ref, axis=1)
    Zs = (Z - med) / scale * sqrt_w
    D = np.sqrt(((Zs[:, :, None, :] - Zs[:, None, :, :]) ** 2).sum(-1))
    dxy = D[:, :n_ref, n_ref:].mean(axis=(1, 2))
    dxx = D[:, :n_ref, :n_ref].mean(axis=(1, 2))
    dyy = D[:, n_ref:, n_ref:].mean(axis=(1, 2))
    return 2 * dxy - dxx - dyy


def energy_distance_test(X_ref, X_batch, weights=None, n_mc: int = 5000, seed: int = 0,
                         chunk: int = 1000) -> dict:
    """Energy-distance two-sample test on standardised, optionally consequence-weighted KPIs
    with a site-label permutation p-value.

    Standardisation uses the reference median and MAD per column, **re-computed inside
    every permutation** from the rows the permutation labels as reference (checklist C09).
    `weights` (length k) multiply the squared standardised coordinates, i.e. columns are
    scaled by sqrt(weights) before Euclidean distances. Rows with any NaN are dropped and
    the dropped counts reported. Monte Carlo p = (b + 1) / (n_mc + 1).

    Returns dict(statistic, p, n_perm, method, n_ref, n_batch, n_cols, n_dropped_ref,
    n_dropped_batch, weights).
    """
    R, _, cols = _to_matrix(X_ref)
    B, _, _ = _to_matrix(X_batch)
    if R.shape[1] != B.shape[1]:
        raise ValueError("X_ref and X_batch must have the same columns")
    okR, okB = ~np.isnan(R).any(1), ~np.isnan(B).any(1)
    R, B = R[okR], B[okB]
    n_r, n_b, k = len(R), len(B), R.shape[1]
    w = np.ones(k) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (k,) or (w < 0).any():
        raise ValueError("weights must be a non-negative vector with one entry per column")
    sqrt_w = np.sqrt(w)
    out = dict(statistic=np.nan, p=np.nan, n_perm=0, method="monte_carlo", n_ref=int(n_r), n_batch=int(n_b),
               n_cols=int(k), n_dropped_ref=int((~okR).sum()), n_dropped_batch=int((~okB).sum()),
               weights=w.tolist(), columns=cols)
    if n_r < 2 or n_b < 1:
        return out
    pooled = np.vstack([R, B])
    N = n_r + n_b
    observed = float(_energy_stat(pooled[None], n_r, sqrt_w)[0])
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_mc, chunk):
        m = min(chunk, n_mc - start)
        order = np.argsort(rng.random((m, N)), axis=1)
        null = _energy_stat(pooled[order], n_r, sqrt_w)
        hits += int(_hits(null, observed, "greater").sum())
    out.update(statistic=observed, p=(hits + 1) / (n_mc + 1), n_perm=int(n_mc))
    return out


# ----------------------------------------------------------------------------------------
# Per-site drift score
# ----------------------------------------------------------------------------------------
def per_site_drift(X_ref, X_batch, weights=None) -> pd.DataFrame:
    """Per-site robust distance to the reference centre, for reference and batch sites.

    Distance = sqrt(sum_j w_j * z_j²) with z_j = (x_j - median_ref_j) / MAD_ref_j, i.e. a
    Mahalanobis-like distance with **diagonal** MAD scaling (a full covariance cannot be
    estimated from 10-17 sites). Batch sites are scaled by the full reference. Reference
    sites are scored **leave-one-site-out** (centre and scale from the other reference
    sites) so that their distances are out-of-sample like the batch sites'. The percentile
    of each site is its rank within the leave-one-out reference distances (ties count
    half; a reference site is ranked among the other reference sites).

    Output columns: site, group ('reference' | 'batch'), distance, percentile_in_ref,
    top_driver (column with the largest |z|), top_driver_z, and z_<column> for every column.
    """
    R, idx_r, cols = _to_matrix(X_ref)
    B, idx_b, _ = _to_matrix(X_batch)
    k = R.shape[1]
    w = np.ones(k) if weights is None else np.asarray(weights, dtype=float)

    def score(X, ref):
        med, scale = _robust_scale(ref, axis=0)
        z = (X - med) / scale
        return np.sqrt((w * z ** 2).sum(-1)), z

    d_b, z_b = score(B, R)
    d_r = np.empty(len(R))
    z_r = np.empty_like(R)
    for i in range(len(R)):
        others = np.delete(R, i, axis=0)
        d, z = score(R[i:i + 1], others)
        d_r[i], z_r[i] = d[0], z[0]

    def pct(d, pool):
        return 100.0 * (np.sum(pool < d) + 0.5 * np.sum(pool == d)) / len(pool)

    rows = []
    for i, s in enumerate(idx_r):
        pool = np.delete(d_r, i)
        rows.append(dict(site=s, group="reference", distance=d_r[i],
                         percentile_in_ref=pct(d_r[i], pool) if len(pool) else np.nan,
                         top_driver=cols[int(np.argmax(np.abs(z_r[i])))], top_driver_z=z_r[i][int(np.argmax(np.abs(z_r[i])))],
                         **{f"z_{c}": z_r[i][j] for j, c in enumerate(cols)}))
    for i, s in enumerate(idx_b):
        rows.append(dict(site=s, group="batch", distance=d_b[i], percentile_in_ref=pct(d_b[i], d_r),
                         top_driver=cols[int(np.argmax(np.abs(z_b[i])))], top_driver_z=z_b[i][int(np.argmax(np.abs(z_b[i])))],
                         **{f"z_{c}": z_b[i][j] for j, c in enumerate(cols)}))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Minimum detectable change by simulation
# ----------------------------------------------------------------------------------------
def mdc(x_ref, n_incoming: int, alpha: float = 0.05, power: float = 0.80, n_sim: int = 400,
        test: dict | None = None, shifts=None, seed: int = 0, min_remaining: int = 3,
        replace: bool = False) -> dict:
    """Minimum detectable change (in reference-MAD units) for an incoming batch of
    `n_incoming` sites, by simulation.

    Design (per simulation): draw `n_incoming` sites from `x_ref` **without replacement**
    (default), add shift x MAD(x_ref), and test them against the **remaining** reference
    sites with the Monte Carlo site-label permutation test (default: median difference,
    two-sided, n_mc = 999). Power(shift) = share of simulations with p <= alpha; mdc_mad is
    the smallest grid shift whose empirical power >= `power` (np.inf if none).

    Why without replacement (deviation from the first draft of the contract, which said
    "with replacement"): at zero shift the pooled sample must be exchangeable for the
    permutation test to hold its level. Drawing with replacement puts duplicated values in
    the pseudo-batch only and removes exactly those sites from the remaining reference;
    measured on 17 normal sites this rejected 11 % of the time at shift 0 for alpha = 5 %,
    so a "conservative" MDC cannot be claimed from it. Without replacement the shift-0 case
    is an exact random split of the reference and the level holds. `replace=True` is kept
    for comparison only.

    This is a *n_incoming-vs-remaining* design: with 17 reference sites and 7 draws the
    remaining set has 10 sites, smaller than the real 7-vs-17 comparison, so the MDC
    reported here is **conservative** (the real test has somewhat more power). The draws
    come from the empirical distribution of the reference, with its sub-populations; the
    MDC therefore describes this reference, not an idealised one.

    Common random numbers: the same site draws and the same permutation layouts are reused
    for all shifts, so the power curve is nearly monotone by construction and the grid
    search is stable; the raw curve is returned, together with its running maximum.

    Returns dict(mdc_mad, mdc_abs, power_curve (DataFrame: shift, power, power_monotone,
    n_sim), ref_mad, n_ref, n_incoming, mean_n_remaining, alpha, target_power, test,
    n_sim_used, elapsed_s, design).
    """
    t0 = time.perf_counter()
    r = _as1d(x_ref)
    n_r = len(r)
    m = float(mad(r))
    shifts = np.arange(0, 4.01, 0.25) if shifts is None else np.asarray(shifts, dtype=float)
    kw = dict(statistic="median_diff", alternative="two-sided", n_mc=999)
    kw.update(test or {})
    fn, name, vectorised = _resolve_statistic(kw["statistic"])
    n_mc, alternative = int(kw["n_mc"]), kw["alternative"]
    rng = np.random.default_rng(seed)
    rejections = np.zeros(len(shifts))
    n_used, n_rem_total = 0, 0
    for _ in range(n_sim):
        draw = rng.integers(0, n_r, size=n_incoming) if replace else rng.choice(n_r, size=n_incoming, replace=False)
        remaining = np.setdiff1d(np.arange(n_r), draw)
        n_rem = len(remaining)
        if n_rem < min_remaining or m <= 0:
            continue
        N = n_rem + n_incoming
        pooled = np.concatenate([r[remaining], r[draw]])
        is_batch = np.concatenate([np.zeros(n_rem), np.ones(n_incoming)])
        shifted = pooled[None, :] + is_batch[None, :] * (shifts * m)[:, None]  # (n_shifts, N)
        order = np.argsort(rng.random((n_mc, N)), axis=1)                       # shared across shifts
        if vectorised:
            obs = fn(shifted[:, :n_rem], shifted[:, n_rem:])                      # (n_shifts,)
            perm = shifted[:, order]                                              # (n_shifts, n_mc, N)
            null = fn(perm[..., :n_rem], perm[..., n_rem:])                        # (n_shifts, n_mc)
        else:
            obs = np.array([fn(s[:n_rem], s[n_rem:]) for s in shifted])
            null = np.array([[fn(s[o[:n_rem]], s[o[n_rem:]]) for o in order] for s in shifted])
        hits = np.array([_hits(null[i], obs[i], alternative).sum() for i in range(len(shifts))])
        p = (hits + 1) / (n_mc + 1)
        rejections += p <= alpha
        n_used += 1
        n_rem_total += n_rem
    pw = rejections / n_used if n_used else np.full(len(shifts), np.nan)
    pw_mono = np.maximum.accumulate(pw) if n_used else pw
    reach = np.nonzero(pw >= power)[0]
    mdc_mad = float(shifts[reach[0]]) if len(reach) else float("inf")
    curve = pd.DataFrame(dict(shift=shifts, power=pw, power_monotone=pw_mono, n_sim=n_used))
    return dict(mdc_mad=mdc_mad, mdc_abs=mdc_mad * m, power_curve=curve, ref_mad=m, n_ref=int(n_r),
                n_incoming=int(n_incoming), mean_n_remaining=(n_rem_total / n_used if n_used else np.nan),
                alpha=alpha, target_power=power, test=dict(kw, statistic=name), n_sim_used=int(n_used),
                elapsed_s=time.perf_counter() - t0,
                design=(f"{n_incoming} sites drawn {'with' if replace else 'without'} replacement from {n_r} reference sites vs the remaining "
                        f"(mean {n_rem_total / max(n_used, 1):.1f}) reference sites; Monte Carlo permutation "
                        f"({name}, {alternative}, n_mc={n_mc}); conservative relative to {n_incoming}-vs-{n_r}."))


# ----------------------------------------------------------------------------------------
# Local exceedance (evidence flag path)
# ----------------------------------------------------------------------------------------
def iid_flag_probability(n_ref: int, n_batch: int) -> float:
    """Probability, under an i.i.d. continuous null, that at least one of `n_batch` sites
    exceeds the maximum of `n_ref` reference sites: the overall maximum of the pooled
    n_ref + n_batch values is equally likely to be any of them, so P = n_batch / (n_ref +
    n_batch). For 10 vs 7 that is 7/17 ~ 41 % per KPI (plan §2.0 rule 5, checklist C10)."""
    return n_batch / (n_ref + n_batch)


def local_exceedance(ref_site_values, batch_site_values, margin_mad: float = 1.0) -> dict:
    """Compare each batch site's value with the maximum of the supplied reference sites.

    The caller passes the **ordinary** reference sites (reference minus grey-pore and
    cracked sites) and, for the localized-defect path, per-site maxima of a consequential
    KPI. `exceeds` is an evidence flag only; `credible_severity` additionally requires
    the margin above the reference maximum to be at least `margin_mad` MADs of the
    supplied reference values (plan §2.0 rule 5 condition (a); conditions (b)
    measurement reliability and (c) image review are the decision layer's).

    Returns dict(table, ref_max, ref_mad, n_ref, n_batch, n_exceed, n_credible,
    iid_flag_probability, margin_mad). `table` has one row per batch site: site, value,
    ref_max, exceeds, margin_in_mad, credible_severity.
    """
    ref = pd.Series(ref_site_values, dtype=float)
    bat = pd.Series(batch_site_values, dtype=float)
    ref = ref.dropna()
    ref_max = float(ref.max()) if len(ref) else np.nan
    ref_mad = float(mad(ref.to_numpy())) if len(ref) >= 2 else np.nan
    margin = (bat - ref_max) / ref_mad if (ref_mad and ref_mad > 0) else pd.Series(np.nan, index=bat.index)
    exceeds = bat > ref_max
    table = pd.DataFrame(dict(site=bat.index, value=bat.to_numpy(), ref_max=ref_max, exceeds=exceeds.to_numpy(),
                              margin_in_mad=margin.to_numpy(),
                              credible_severity=(exceeds & (margin >= margin_mad)).fillna(False).to_numpy()))
    n_b = int(bat.notna().sum())
    return dict(table=table, ref_max=ref_max, ref_mad=ref_mad, n_ref=int(len(ref)), n_batch=n_b,
                n_exceed=int(table["exceeds"].sum()), n_credible=int(table["credible_severity"].sum()),
                iid_flag_probability=iid_flag_probability(len(ref), n_b) if len(ref) + n_b else np.nan,
                margin_mad=margin_mad)


# ----------------------------------------------------------------------------------------
# Jackknife driver and decision stability
# ----------------------------------------------------------------------------------------
def jackknife(fn: Callable, batch_site_ids: Iterable, **kwargs) -> list[dict]:
    """Leave-one-site-out driver. Calls `fn(kept_site_ids, **kwargs)` once per site with that
    site removed and returns [{left_out, kept, result}, ...] in input order. `fn` is
    supplied by the decision layer (typically "re-run the verdict on these sites")."""
    ids = list(batch_site_ids)
    runs = []
    for s in ids:
        kept = [t for t in ids if t != s]
        runs.append(dict(left_out=s, kept=kept, result=fn(kept, **kwargs)))
    return runs


def stability_share(outcomes: Sequence, full_outcome, key=None) -> float:
    """Decision stability: share of leave-one-site-out outcomes equal to the full-sample
    outcome. Accepts the list returned by `jackknife` (uses each run's `result`) or a plain
    list of outcomes; `key` selects one field when outcomes are dicts. This is a stability
    share, not a probability that the verdict is correct (checklist C17)."""
    vals = [o["result"] if isinstance(o, dict) and "result" in o and "left_out" in o else o for o in outcomes]
    if key is not None:
        vals = [v[key] for v in vals]
        full_outcome = full_outcome[key]
    if not vals:
        return float("nan")
    return float(np.mean([v == full_outcome for v in vals]))


# ----------------------------------------------------------------------------------------
# Reference-split diagnostics driver
# ----------------------------------------------------------------------------------------
def reference_split_diagnostics(ref_df: pd.DataFrame, kpis: Sequence[str], n_incoming: int,
                                pipeline_fn: Callable[[pd.DataFrame, pd.DataFrame], dict],
                                n_splits: int = 200, seed: int = 0, site_col: str = "site") -> pd.DataFrame:
    """Repeatedly split the reference **by site label** into a pseudo-batch of `n_incoming`
    sites and the rest, call `pipeline_fn(ref_part_df, pseudo_batch_df) -> dict`, and tabulate
    the outputs (one row per split, with the pseudo-batch site ids and part sizes).

    Only whole sites move between parts; rows are never split within a site (`ref_df` must
    have one row per site). Every reference-dependent fit must happen inside `pipeline_fn`
    so it is repeated per split. The pipeline should return its outcomes in separate
    fields (drift alert / local flag / credible anomaly / quality abstention) so the caller
    can count them separately (plan §2.3, §2.9).

    Caveat, stated on purpose: a 7-vs-10 split is a smaller comparison than the real
    7-vs-17 one, so the alert rate measured here is an internal diagnostic with its own
    uncertainty, **not** a validation of the 7-vs-17 procedure's false-alarm rate.
    """
    missing = [k for k in kpis if k not in ref_df.columns]
    if missing:
        raise KeyError(f"kpis missing from ref_df: {missing}")
    sites = ref_df[site_col].to_numpy()
    if len(np.unique(sites)) != len(sites):
        raise ValueError("ref_df must have exactly one row per site")
    if n_incoming >= len(sites):
        raise ValueError("n_incoming must leave at least one reference site")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_splits):
        pseudo = rng.choice(sites, size=n_incoming, replace=False)
        m = ref_df[site_col].isin(pseudo)
        out = pipeline_fn(ref_df.loc[~m].copy(), ref_df.loc[m].copy())
        rows.append(dict(split=i, pseudo_batch_sites=tuple(sorted(map(str, pseudo))),
                         n_ref_part=int((~m).sum()), n_pseudo=int(m.sum()), **(out or {})))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Usable n after quality flags
# ----------------------------------------------------------------------------------------
def flag_rule(kpi: str) -> tuple[str | None, str | None]:
    """Which site flag applies to a KPI and how.
    bright_* -> ('exclude', 'bright_low_contrast'): bright-phase KPIs are unreliable on
    low-contrast sites and those sites are excluded from n.
    pore_* / crack_* -> ('fallback', 'grey_pore'): pore KPIs on grey-pore sites rest on the
    fallback threshold; the sites are **kept** and counted separately.
    Anything else -> (None, None)."""
    if kpi.startswith("bright_"):
        return "exclude", "bright_low_contrast"
    if kpi.startswith(("pore_", "crack_")):
        return "fallback", "grey_pore"
    return None, None


def usable_n(df: pd.DataFrame, kpi: str, flags: dict | None = None, site_col: str = "site") -> dict:
    """Usable site count for one KPI after quality flags (checklist C14).

    A site is usable when its value is not NaN and it is not excluded by the flag relevant
    to the KPI (`flag_rule`). `flags` maps flag name -> list of site ids (default
    DEFAULT_FLAGS). A boolean `bright_low_contrast` column in `df`, if present, is OR-ed
    with the list so an unseen batch's own flags count.

    Returns dict(kpi, n_sites, n_nonnan, n_usable, n_excluded, n_fallback, usable_sites,
    excluded_sites, fallback_sites, excluded_by, fallback_by).
    """
    flags = DEFAULT_FLAGS if flags is None else flags
    sites = df[site_col].astype(str)
    nonnan = df[kpi].notna().to_numpy()
    mode, flag = flag_rule(kpi)
    excluded = np.zeros(len(df), dtype=bool)
    fallback = np.zeros(len(df), dtype=bool)
    if mode == "exclude":
        excluded = sites.isin([str(s) for s in flags.get(flag, [])]).to_numpy()
        if flag in df.columns:
            excluded |= df[flag].fillna(False).astype(bool).to_numpy()
    elif mode == "fallback":
        fallback = sites.isin([str(s) for s in flags.get(flag, [])]).to_numpy()
    usable = nonnan & ~excluded
    return dict(kpi=kpi, n_sites=int(len(df)), n_nonnan=int(nonnan.sum()), n_usable=int(usable.sum()),
                n_excluded=int((excluded & nonnan).sum()), n_fallback=int((fallback & usable).sum()),
                usable_sites=sites[usable].tolist(), excluded_sites=sites[excluded & nonnan].tolist(),
                fallback_sites=sites[fallback & usable].tolist(),
                excluded_by=flag if mode == "exclude" else None, fallback_by=flag if mode == "fallback" else None)


def usable_values(df: pd.DataFrame, kpi: str, flags: dict | None = None, site_col: str = "site") -> pd.Series:
    """The usable site values of one KPI as a Series indexed by site id."""
    u = usable_n(df, kpi, flags, site_col)
    sub = df[df[site_col].astype(str).isin(u["usable_sites"])]
    return pd.Series(sub[kpi].to_numpy(dtype=float), index=sub[site_col].astype(str).to_numpy(), name=kpi)


# ----------------------------------------------------------------------------------------
# Main entry point for the decision layer
# ----------------------------------------------------------------------------------------
def compare_kpis(ref_df: pd.DataFrame, batch_df: pd.DataFrame, kpis: Sequence[str],
                 primary: Sequence[str] = PRIMARY_KPIS, flags: dict | None = None, seed: int = 0,
                 n_boot: int = 2000, exact_max: int = 1_000_000, n_mc: int = 20_000, alpha: float = 0.05,
                 statistic="median_diff", site_col: str = "site") -> pd.DataFrame:
    """Pairwise comparison engine (plan §2.4): one tidy row per KPI.

    For each KPI: usable sites in each frame (`usable_n`), robust standardised shift with
    approximate bootstrap-over-sites interval and Cliff's delta (`robust_shift`), and a
    site-label permutation p-value (`permutation_test`, exact when affordable). Holm
    correction is applied across the **primary KPIs only**; secondary KPIs keep their
    uncorrected p (p_holm = NaN) and are descriptive.

    `direction` is descriptive: 'higher' / 'lower' when the approximate 95 % bootstrap
    interval of the shift excludes zero, else 'none'. It is not the verdict; the decision
    layer combines p_holm, MDC and the localized path.

    Columns: kpi, is_primary, n_ref_usable, n_batch_usable, n_ref_fallback,
    n_batch_fallback, n_ref_excluded, n_batch_excluded, ref_median, batch_median, ref_mad,
    shift_mad, ci_low, ci_high, cliffs_delta, p_perm, p_method, n_perm, p_holm,
    p_reported (p_holm for primary, p_perm for secondary), flag_alpha (p_reported <= alpha),
    direction.
    """
    primary = set(primary)
    seeds = np.random.SeedSequence(seed).generate_state(max(len(kpis), 1))
    rows = []
    for kpi, s in zip(kpis, seeds):
        s = int(s)
        ur, ub = usable_n(ref_df, kpi, flags, site_col), usable_n(batch_df, kpi, flags, site_col)
        xr, xb = usable_values(ref_df, kpi, flags, site_col).to_numpy(), usable_values(batch_df, kpi, flags, site_col).to_numpy()
        row = dict(kpi=kpi, is_primary=kpi in primary, n_ref_usable=ur["n_usable"], n_batch_usable=ub["n_usable"],
                   n_ref_fallback=ur["n_fallback"], n_batch_fallback=ub["n_fallback"],
                   n_ref_excluded=ur["n_excluded"], n_batch_excluded=ub["n_excluded"])
        rs = robust_shift(xr, xb, n_boot=n_boot, seed=s)
        pt = permutation_test(xr, xb, statistic=statistic, exact_max=exact_max, n_mc=n_mc, seed=s)
        row.update(ref_median=rs["ref_median"], batch_median=rs["batch_median"], ref_mad=rs["ref_mad"],
                   shift_mad=rs["shift_mad"], ci_low=rs["ci_low"], ci_high=rs["ci_high"], cliffs_delta=rs["cliffs_delta"],
                   p_perm=pt["p"], p_method=pt["method"], n_perm=pt["n_perm"], p_holm=np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        prim = out["is_primary"].to_numpy()
        if prim.any():
            out.loc[prim, "p_holm"] = holm(out.loc[prim, "p_perm"].to_numpy())
        out["p_reported"] = np.where(prim, out["p_holm"], out["p_perm"])
        out["flag_alpha"] = out["p_reported"] <= alpha
        out["direction"] = np.select(
            [(out["shift_mad"] > 0) & (out["ci_low"] > 0), (out["shift_mad"] < 0) & (out["ci_high"] < 0)],
            ["higher", "lower"], default="none")
    return out
