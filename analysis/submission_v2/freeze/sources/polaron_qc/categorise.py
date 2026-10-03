"""polaron_qc.categorise — per-sample categorisation, baseline-membership assessment and QC hand-off.

Three SEPARATE answers per sample (sample = one site = one BSE/ETD/Inlens image set), never conflated
(pre-registration: analysis/categoriser/findings.md §1, committed before the first run):

1. **Site categorisation** — "does this sample resemble Batch 1, 2 or 3?"  A 3-class L1-logistic model over the
   known batches, evaluated leave-one-site-out with the imputer, scaler and the regularisation choice (C from a
   3-value grid by inner CV) refitted INSIDE every fold and inside every site-label permutation. Three feature
   families are reported side by side: ``morph`` (morphology only), ``acq`` (acquisition only),
   ``material`` (morphology + acquisition-sensitive ETD/Inlens appearance; PRIMARY v2, D52) and
   ``combined`` (appearance + acquisition; comparator). V2 was revised after first-drop inspection,
   before its truth, and declared before final evaluation; the original E31 registration is preserved.
   Outputs are **model probabilities** (normalised one-vs-rest scores) — not posteriors, not confidence; their
   calibration is assessed (:func:`calibration`), not assumed.
2. **Baseline OOD assessment** — "is it outside Batch 3's promised distribution?"  Built independently of (1): a
   3-class model must pick a known batch even for an unfamiliar sample, so a high Batch 3 probability cannot
   establish membership. Robust-z (reference median / MAD) k-nearest-reference distance, with the reference's own
   leave-one-site-out distribution as the yardstick. Percentiles are descriptive evidence ranks (0–100).
   The legacy ``ood_rank_p_*`` columns are tail ranks with a floor of 1/(n_ref + 1), not calibrated p-values:
   reference scores use n_ref−1-site fits whereas queries use an n_ref-site fit. No false-alert guarantee follows.
   They are never a probability of being defective.
3. **Hand-off** — per sample: the top contributing features to (1) and (2) with sign and the Batch 3 median/MAD,
   the data-derived acquisition flags (``report.derive_flags`` / ``apply_derived_flags``) and a pointer to the
   batch-level QC verdict (``polaron_qc.report``), which is a separate output and is not touched here.

The texture family is labelled "acquisition-sensitive; batch-fingerprint evidence, origin (microstructure vs
imaging) not established" wherever it appears. Nothing here changes decision.py, report.py thresholds or the
frozen notebook. One row per site everywhere; patches never enter n; lengths in px.

CLI::

    python3 -m polaron_qc.categorise Dataset/Batch_3 Dataset/Batch_1 Dataset/Batch_2            # LOO evaluation
    python3 -m polaron_qc.categorise Dataset/Batch_3 Dataset/Batch_1 Dataset/Batch_2 --score Dataset/Batch_X
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sps
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import CRACKED_SITES, GREY_PORE_SITES, KPI_TRUST, LOW_CONTRAST_SITES, MATERIAL_KPIS
from .ml import _canonical_order

__all__ = [
    "MORPH_FEATURES", "ACQ_FEATURES", "TEXTURE_FEATURES", "TEXTURE_LABEL", "FAMILIES", "FAMILY_LABELS", "OOD_VARIANTS",
    "PRIMARY_FAMILY", "PRIMARY_OOD_VARIANT", "C_GRID", "N_PERM_DEFAULT",
    "assemble_sites", "load_known", "fit_categoriser", "predict_proba", "loo_evaluate", "calibration", "metrics",
    "fit_ood", "ood_score", "categorise_site", "categorise_sites", "kruskal_context", "run", "main",
]

# ----------------------------------------------------------------------------------------------------------------
# feature families (pre-registered; findings.md §1.2)
# ----------------------------------------------------------------------------------------------------------------
MORPH_EXTRA = ["pore_d90", "pore_count_per_Mpx", "bright_d10", "bright_solidity", "bright_max_d"]
MORPH_FEATURES = list(MATERIAL_KPIS) + MORPH_EXTRA
# v2 (D52): frame height H is excluded from every family — it is a session / stitching fingerprint with no material meaning and
# does not generalise beyond the sessions in the known data. The remaining acquisition statistics stay in a labelled comparator family only.
ACQ_FEATURES = ["bse_p1", "bse_std", "bse_empty_bin_frac", "bright_sep", "etd_boundary_sharpness", "etd_curtain_frac",
                "bse_p50", "etd_p50", "inlens_p50"]
TEXTURE_FEATURES = ["etd_crack_density_graphite", "crack_g_0.05", "crack_g_0.1", "crack_g_0.2", "crack_g_0.3", "ridge_p97",
                    "inlens_particle_texture", "inlens_particle_texture_p90", "inlens_speckled_particle_frac", "inlens_grad_energy"]
TEXTURE_LABEL = "acquisition-sensitive; batch-fingerprint evidence, origin (microstructure vs imaging) not established"
MATERIAL_FEATURES = MORPH_FEATURES + TEXTURE_FEATURES          # v2 primary: no explicit session statistics; texture can still encode acquisition
FAMILIES = {"morph": MORPH_FEATURES, "acq": ACQ_FEATURES, "material": MATERIAL_FEATURES, "combined": MORPH_FEATURES + TEXTURE_FEATURES + ACQ_FEATURES}
FAMILY_LABELS = {"morph": "morphology-only (trusted BSE geometry KPIs; no intensity statistics)",
                 "acq": "acquisition-only (session / instrument statistics; never material evidence; comparator)",
                 "material": f"material = morphology + texture [{TEXTURE_LABEL}]; no session statistics (PRIMARY v2, declared D52 before final evaluation)",
                 "combined": f"combined = material + acquisition statistics (v1 primary, E31; kept as a labelled batch-fingerprint comparator)"}
OOD_VARIANTS = {"morph": MORPH_FEATURES, "morph_texture": MORPH_FEATURES + TEXTURE_FEATURES, "acq": ACQ_FEATURES}
OOD_LABELS = {"morph": "morphology (PRIMARY, pre-registered)",
              "morph_texture": f"texture-inclusive [{TEXTURE_LABEL}]",
              "acq": "acquisition-only (session fingerprint, not material)"}
PRIMARY_FAMILY = "material"   # v2 (D52); v1 (E31) was "combined"
PRIMARY_OOD_VARIANT = "morph"
C_GRID = (0.1, 0.5, 2.0)
N_INNER = 3
K_NN = 3
N_PERM_DEFAULT = 200
CALIB_BIN_EDGES = (1.0 / 3.0, 0.5, 0.7, 1.0 + 1e-9)
FLAG_COLS = ["acquisition_group", "bright_low_contrast", "grey_pore", "raised_black_level", "contrast_stretched_bse", "cracked_known"]
QC_POINTER = "batch-level QC verdict: python3 -m polaron_qc.report <reference_dir> <batch_dir> — separate output, not derived from this table"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "categoriser", "output")
DEFAULT_FEATURE_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_cache", "features")

# C21: the morphology family may only contain catalogue entries of trust high / medium
assert all(KPI_TRUST.get(k) in ("high", "medium") for k in MORPH_FEATURES), "morphology family leaks a non-material column"
assert not set(MORPH_FEATURES) & set(ACQ_FEATURES) and not set(MORPH_FEATURES) & set(TEXTURE_FEATURES)


# ----------------------------------------------------------------------------------------------------------------
# site table assembly
# ----------------------------------------------------------------------------------------------------------------
def assemble_sites(tables_by_batch: dict) -> pd.DataFrame:
    """One row per site across batches: the ``sites`` table of ``features.extract_batch`` plus the acquisition
    covariates and data-derived flags of ``report.derive_flags`` (written back with ``report.apply_derived_flags``).
    ``tables_by_batch`` maps a batch name to the dict returned by ``extract_batch`` (keys ``sites``, ``images``)."""
    from . import report  # local import: report pulls matplotlib / PIL, not needed for the synthetic path
    out = []
    for name, t in tables_by_batch.items():
        sites, images = t["sites"].copy(), t["images"]
        sites["site"] = sites["site"].astype(str)
        if "batch" not in sites.columns:
            sites["batch"] = name
        flags = report.derive_flags(sites, images)
        sites = report.apply_derived_flags(sites, flags)
        carry = [c for c in ("acquisition_group", "cracked_known", "contrast_stretched_bse", "contrast_stretched_any", "raised_black_level",
                             "bse_p1", "bse_p50", "bse_std", "bse_empty_bin_frac", "etd_p50", "inlens_p50", "grey_pore_source")
                 if c in flags.columns]
        f = flags.assign(site=flags.site.astype(str)).drop_duplicates("site").set_index("site")
        for c in carry:
            sites[c] = sites["site"].map(f[c]).to_numpy()
        out.append(sites)
    df = pd.concat(out, ignore_index=True)
    return df.sort_values(["batch", "site"], kind="stable").reset_index(drop=True)


def load_known(batch_dirs: Sequence[str], cache_dir: str | None = DEFAULT_FEATURE_CACHE, verbose: bool = False) -> pd.DataFrame:
    """Extract (or read from cache) every folder in ``batch_dirs`` and assemble the site table."""
    from .features import batch_name, extract_batch
    tabs = {batch_name(d): extract_batch(d, cache_dir=cache_dir, verbose=verbose) for d in batch_dirs}
    return assemble_sites(tabs)


def _matrix(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"site table lacks feature columns {missing}")
    return df[list(features)].astype(float).to_numpy()


# ----------------------------------------------------------------------------------------------------------------
# answer 1 — site categorisation
# ----------------------------------------------------------------------------------------------------------------
def _pipeline(C: float, seed: int) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced", max_iter=2000,
                                               random_state=seed))])


def _proba_full(pipe: Pipeline, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Model probabilities over all ``n_classes`` integer classes (zeros for classes absent from the fit)."""
    P = np.zeros((len(X), n_classes))
    P[:, pipe["lr"].classes_.astype(int)] = pipe.predict_proba(X)
    return P


def _balanced_logloss(y: np.ndarray, P: np.ndarray, n_classes: int, eps: float = 1e-12) -> float:
    """Multiclass log-loss with every class weighted to equal total weight (classes absent from y are ignored)."""
    ll, k = 0.0, 0
    for c in range(n_classes):
        m = y == c
        if m.any():
            ll += -np.mean(np.log(np.clip(P[m, c], eps, 1.0)))
            k += 1
    return ll / max(k, 1)


def _select_C(X: np.ndarray, y: np.ndarray, n_classes: int, C_grid: Sequence[float], seed: int, n_inner: int) -> float:
    """Inner stratified K-fold choice of C by class-balanced log-loss. Ties go to the smallest C (strongest penalty).
    If a class has fewer than 2 members the inner split is impossible and the middle grid value is used."""
    counts = np.bincount(y, minlength=n_classes)
    n_min = int(counts[counts > 0].min())
    if n_min < 2:
        return float(C_grid[len(C_grid) // 2])
    k = min(n_inner, n_min)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    loss = np.zeros(len(C_grid))
    for tr, te in skf.split(X, y):
        for j, C in enumerate(C_grid):
            pipe = _pipeline(C, seed).fit(X[tr], y[tr])
            loss[j] += _balanced_logloss(y[te], _proba_full(pipe, X[te], n_classes), n_classes) * len(te)
    return float(C_grid[int(np.argmin(loss))])      # argmin returns the first (smallest C) on ties


def _loo(X: np.ndarray, y: np.ndarray, n_classes: int, C_grid, seed: int, n_inner: int, keep_models: bool = False):
    """Leave-one-site-out with the full fold-local procedure. Returns (P_oof, C_chosen, models_or_None)."""
    n = len(y)
    P = np.zeros((n, n_classes)); Cs = np.empty(n); models = [] if keep_models else None
    for i in range(n):
        tr = np.r_[0:i, i + 1:n]
        C = _select_C(X[tr], y[tr], n_classes, C_grid, seed, n_inner)
        pipe = _pipeline(C, seed).fit(X[tr], y[tr])
        P[i] = _proba_full(pipe, X[i:i + 1], n_classes)[0]
        Cs[i] = C
        if keep_models:
            models.append(pipe)
    return P, Cs, models


def metrics(y: np.ndarray, P: np.ndarray, classes: Sequence[str]) -> dict:
    """accuracy (+ Clopper–Pearson 95 %), balanced accuracy, per-class recall, confusion counts (rows = true)."""
    y = np.asarray(y); classes = list(classes); K = len(classes)
    pred = P.argmax(axis=1)
    n = len(y); k = int((pred == y).sum())
    lo = sps.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = sps.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    conf = np.zeros((K, K), int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    recall = {classes[c]: (float(conf[c, c] / conf[c].sum()) if conf[c].sum() else float("nan")) for c in range(K)}
    present = [c for c in range(K) if conf[c].sum()]
    bal = float(np.mean([conf[c, c] / conf[c].sum() for c in present]))
    maj = float(np.bincount(y, minlength=K).max() / n)
    return dict(n=n, accuracy=k / n, accuracy_ci95=(float(lo), float(hi)), n_correct=k, balanced_accuracy=bal,
                chance_balanced=1.0 / len(present), majority_accuracy=maj, recall=recall,
                confusion=pd.DataFrame(conf, index=[f"true {c}" for c in classes], columns=[f"pred {c}" for c in classes]))


def calibration(y: np.ndarray, P: np.ndarray, bin_edges: Sequence[float] = CALIB_BIN_EDGES) -> dict:
    """Is the top-class model probability usable as a probability? Multiclass Brier (vs the class-prior Brier),
    reliability of the top-class probability in fixed bins, and the expected calibration error over those bins.
    A statement of what LOO showed on these sites, never a calibration guarantee for an unseen batch."""
    y = np.asarray(y); n, K = P.shape
    onehot = np.eye(K)[y]
    brier = float(np.mean(((P - onehot) ** 2).sum(axis=1)))
    prior = np.bincount(y, minlength=K) / n
    brier_prior = float(np.mean(((prior[None, :] - onehot) ** 2).sum(axis=1)))
    top = P.max(axis=1); correct = (P.argmax(axis=1) == y)
    rows, ece = [], 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (top >= lo) & (top < hi)
        if m.any():
            conf_b, acc_b = float(top[m].mean()), float(correct[m].mean())
            ece += m.sum() / n * abs(acc_b - conf_b)
        else:
            conf_b, acc_b = float("nan"), float("nan")
        rows.append(dict(bin=f"[{lo:.2f}, {min(hi, 1.0):.2f})", n=int(m.sum()), mean_top_prob=conf_b, observed_top_accuracy=acc_b))
    return dict(brier=brier, brier_prior=brier_prior, ece=float(ece), reliability=pd.DataFrame(rows),
                mean_top_prob=float(top.mean()), top_accuracy=float(correct.mean()))


def _encode(df: pd.DataFrame, batch_col: str = "batch"):
    classes = sorted(df[batch_col].astype(str).unique())
    y = df[batch_col].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    return classes, y


def _canonical(X: np.ndarray, y: np.ndarray, sites: np.ndarray):
    order, _, _ = _canonical_order(X, y, np.asarray(sites).astype(str))
    return order


def _perm_stat(X, y_perm, n_classes, C_grid, seed, n_inner):
    P, _, _ = _loo(X, y_perm, n_classes, C_grid, seed, n_inner)
    return metrics(y_perm, P, [str(c) for c in range(n_classes)])["balanced_accuracy"]


def loo_evaluate(df: pd.DataFrame, family: str = PRIMARY_FAMILY, features: Sequence[str] | None = None, n_perm: int = N_PERM_DEFAULT,
                 seed: int = 0, C_grid=C_GRID, n_inner: int = N_INNER, n_jobs: int = -1, batch_col: str = "batch",
                 verbose: bool = False) -> dict:
    """Leave-one-site-out evaluation of one feature family on the known sites (one row per site, column ``batch`` =
    class, column ``site`` = id). Imputer, scaler and the inner C choice are refitted in every fold and inside every
    site-label permutation; statistic = balanced accuracy; p = (b + 1)/(n_perm + 1), one-sided, Monte Carlo.
    Returns dict(family, features, classes, per_site [site, batch, mp_<class>…, argmax, C], metrics, calibration,
    perm_null, p, n_perm, p_method, fold_models, X, y)."""
    features = list(FAMILIES[family] if features is None else features)
    classes, y = _encode(df, batch_col)
    X = _matrix(df, features)
    sites = df["site"].astype(str).to_numpy()
    order = _canonical(np.nan_to_num(X, nan=-1e30), y, sites)      # NaN-safe content order (D43)
    X, y, sites = X[order], y[order], sites[order]
    t0 = time.time()
    P, Cs, models = _loo(X, y, len(classes), C_grid, seed, n_inner, keep_models=True)
    m = metrics(y, P, classes)
    cal = calibration(y, P)
    per_site = pd.DataFrame({"site": sites, "batch": [classes[i] for i in y]})
    for i, c in enumerate(classes):
        per_site[f"mp_{c}"] = P[:, i]
    per_site["argmax"] = [classes[i] for i in P.argmax(axis=1)]
    per_site["C"] = Cs
    null = np.empty(0); p = float("nan")
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        perms = [rng.permutation(y) for _ in range(n_perm)]
        null = np.array(Parallel(n_jobs=n_jobs)(delayed(_perm_stat)(X, yp, len(classes), C_grid, seed, n_inner) for yp in perms))
        p = float((1 + np.sum(null >= m["balanced_accuracy"] - 1e-12)) / (n_perm + 1))
    if verbose:
        print(f"[categorise] {family}: bal.acc {m['balanced_accuracy']:.3f} acc {m['accuracy']:.3f} p {p} ({time.time() - t0:.0f} s)")
    return dict(family=family, label=FAMILY_LABELS.get(family, family), features=features, classes=classes, per_site=per_site,
                metrics=m, calibration=cal, perm_null=null, p=p, n_perm=int(n_perm),
                p_method=f"Monte Carlo site-label permutation of the whole fold-local LOO procedure, {n_perm} resamples, "
                         f"statistic = balanced accuracy, p = (b+1)/(n+1), one-sided",
                fold_models=models, X=X, y=y, sites=sites, seed=seed, C_grid=tuple(C_grid))


@dataclass
class CategoriserModel:
    family: str
    features: list
    classes: list
    pipeline: Pipeline
    C: float
    n_train: int
    ref_batch: str
    ref_median: pd.Series
    ref_mad: pd.Series
    label: str = ""
    note: str = "outputs are model probabilities (normalised one-vs-rest L1-logistic scores), not posteriors or confidence"


def _ref_stats(df: pd.DataFrame, features, ref_batch: str, batch_col="batch"):
    ref = df[df[batch_col].astype(str) == ref_batch] if ref_batch in set(df[batch_col].astype(str)) else df
    med = ref[features].astype(float).median()
    mad = (ref[features].astype(float) - med).abs().median() * 1.4826
    return med, mad


def fit_categoriser(df: pd.DataFrame, family: str = PRIMARY_FAMILY, features: Sequence[str] | None = None, seed: int = 0,
                    C_grid=C_GRID, n_inner: int = N_INNER, ref_batch: str = "Batch_3", batch_col: str = "batch") -> CategoriserModel:
    """Fit the family's model on all known sites (C chosen by the same inner CV) for scoring new samples."""
    features = list(FAMILIES[family] if features is None else features)
    classes, y = _encode(df, batch_col)
    X = _matrix(df, features)
    order = _canonical(np.nan_to_num(X, nan=-1e30), y, df["site"].astype(str).to_numpy())
    X, y = X[order], y[order]
    C = _select_C(X, y, len(classes), C_grid, seed, n_inner)
    pipe = _pipeline(C, seed).fit(X, y)
    med, mad = _ref_stats(df, features, ref_batch, batch_col)
    return CategoriserModel(family, features, classes, pipe, C, len(y), ref_batch, med, mad, FAMILY_LABELS.get(family, family))


def predict_proba(model: CategoriserModel, df: pd.DataFrame) -> pd.DataFrame:
    """Model probabilities (columns ``mp_<class>``) and ``argmax`` for every row of ``df``."""
    P = _proba_full(model.pipeline, _matrix(df, model.features), len(model.classes))
    out = pd.DataFrame({f"mp_{c}": P[:, i] for i, c in enumerate(model.classes)}, index=df.index)
    out["argmax"] = [model.classes[i] for i in P.argmax(axis=1)]
    return out


def categoriser_contributions(pipe: Pipeline, x: np.ndarray, features: Sequence[str], class_idx: int, ref_median=None, ref_mad=None,
                              top: int = 3) -> list[dict]:
    """Top ``top`` features by |coef × standardised value| for the one-vs-rest score of ``class_idx``. Sign '+' means the
    feature pushed the sample TOWARDS that class. Includes the raw value and the reference median / MAD when given."""
    x = np.asarray(x, float).reshape(1, -1)
    z = pipe["scale"].transform(pipe["impute"].transform(x))[0]
    lr = pipe["lr"]
    row = int(np.flatnonzero(lr.classes_.astype(int) == class_idx)[0]) if class_idx in lr.classes_.astype(int) else None
    if row is None:
        return []
    coef = lr.coef_[row] if lr.coef_.shape[0] > 1 else (lr.coef_[0] if lr.classes_[1] == class_idx else -lr.coef_[0])
    contrib = coef * z
    idx = np.argsort(-np.abs(contrib))[:top]
    out = []
    for j in idx:
        if contrib[j] == 0:
            continue
        out.append(dict(feature=features[j], contribution=float(contrib[j]), sign="+" if contrib[j] > 0 else "-", value=float(x[0, j]),
                        ref_median=float(ref_median[features[j]]) if ref_median is not None else float("nan"),
                        ref_mad=float(ref_mad[features[j]]) if ref_mad is not None else float("nan")))
    return out


# ----------------------------------------------------------------------------------------------------------------
# answer 2 — baseline OOD assessment
# ----------------------------------------------------------------------------------------------------------------
def _robust_scale(Xref: np.ndarray):
    """Per-feature (median, scale, keep). scale = 1.4826·MAD, else IQR/1.349, else the feature is dropped."""
    med = np.nanmedian(Xref, axis=0)
    mad = np.nanmedian(np.abs(Xref - med), axis=0) * 1.4826
    q75, q25 = np.nanpercentile(Xref, [75, 25], axis=0)
    iqr = (q75 - q25) / 1.349
    scale = np.where(mad > 0, mad, iqr)
    keep = scale > 0
    scale = np.where(keep, scale, 1.0)
    return med, scale, keep


def _z(X: np.ndarray, med, scale, keep) -> np.ndarray:
    X = np.where(np.isnan(X), med, X)
    return ((X - med) / scale)[:, keep]


def _knn(Zref: np.ndarray, Zq: np.ndarray, k: int):
    """(mean distance to the k nearest reference rows, index array of those rows) for every query row."""
    d = np.sqrt(((Zq[:, None, :] - Zref[None, :, :]) ** 2).sum(axis=2))
    k = min(k, Zref.shape[0])
    nn = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(d, nn, axis=1).mean(axis=1), nn


def _rms(Zq: np.ndarray) -> np.ndarray:
    return np.sqrt((Zq ** 2).mean(axis=1))


def _maha(Zref: np.ndarray, Zq: np.ndarray) -> np.ndarray:
    lw = LedoitWolf().fit(Zref)
    dz = Zq - lw.location_
    return np.sqrt(np.einsum("ij,jk,ik->i", dz, lw.precision_, dz).clip(min=0))


@dataclass
class OODModel:
    variant: str
    features: list
    ref_batch: str
    ref_sites: list
    ref_groups: list
    k: int
    med: np.ndarray
    scale: np.ndarray
    keep: np.ndarray
    Zref: np.ndarray
    ref_loo: pd.DataFrame                      # site, group, score_knn, score_rms, score_maha
    loo_fits: list = field(default_factory=list)   # per held-out ref site: (med, scale, keep, Zref_minus) for the matched-count view
    dropped: list = field(default_factory=list)
    label: str = ""
    note: str = ("percentiles are descriptive evidence ranks (0–100) within the reference's own leave-one-site-out distribution; "
                 "legacy ood_rank_p columns have tail-rank floor 1/(n_ref+1), not calibrated p-values or false-alert guarantees; "
                 "reference and query fits have different sizes; never a probability of defect, out-of-spec or rejection")

    @property
    def n_ref(self) -> int:
        return len(self.ref_sites)


def fit_ood(reference: pd.DataFrame, variant: str = PRIMARY_OOD_VARIANT, features: Sequence[str] | None = None, k: int = K_NN,
            ref_batch: str | None = None, group_col: str = "acquisition_group") -> OODModel:
    """Reference distribution model on the Batch 3 sites: robust scale on all reference sites, and the reference's own
    leave-one-site-out scores (each site scored against the other n−1 with median/MAD and neighbours recomputed)."""
    features = list(OOD_VARIANTS[variant] if features is None else features)
    X = _matrix(reference, features)
    sites = reference["site"].astype(str).tolist()
    groups = reference[group_col].astype(str).tolist() if group_col in reference.columns else ["?"] * len(sites)
    med, scale, keep = _robust_scale(X)
    Zref = _z(X, med, scale, keep)
    rows, fits = [], []
    for i in range(len(sites)):
        m = np.ones(len(sites), bool); m[i] = False
        med_i, scale_i, keep_i = _robust_scale(X[m])
        Zr, Zq = _z(X[m], med_i, scale_i, keep_i), _z(X[i:i + 1], med_i, scale_i, keep_i)
        rows.append(dict(site=sites[i], group=groups[i], score_knn=float(_knn(Zr, Zq, k)[0][0]), score_rms=float(_rms(Zq)[0]),
                         score_maha=float(_maha(Zr, Zq)[0]) if Zr.shape[0] > 2 else float("nan")))
        fits.append((med_i, scale_i, keep_i, Zr))
    rb = ref_batch or str(reference["batch"].iloc[0]) if "batch" in reference.columns else (ref_batch or "reference")
    return OODModel(variant, features, rb, sites, groups, k, med, scale, keep, Zref, pd.DataFrame(rows), fits,
                    [f for f, kp in zip(features, keep) if not kp], OOD_LABELS.get(variant, variant))


def _pct_rank(ref_scores: np.ndarray, s: float):
    pct = float(sps.percentileofscore(ref_scores, s, kind="weak"))
    n_ge = int(np.sum(ref_scores >= s - 1e-12))
    return pct, (n_ge + 1) / (len(ref_scores) + 1), bool(s > ref_scores.max())


def ood_score(model: OODModel, df: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Score every row of ``df`` against the reference: primary k-NN score with descriptive percentile / tail rank / exceedance in the
    reference LOO distribution, the two secondary scores, the matched-count sensitivity (median percentile over the
    n_ref LOO reference sets), the nearest reference sites (with their acquisition group) and the top contributing
    features (largest mean squared z-difference to the k nearest neighbours, sign = sign of the query's robust z)."""
    X = _matrix(df, model.features)
    Zq = _z(X, model.med, model.scale, model.keep)
    s_knn, nn = _knn(model.Zref, Zq, model.k)
    s_rms, s_maha = _rms(Zq), _maha(model.Zref, Zq)
    ref = model.ref_loo
    kept_features = [f for f, kp in zip(model.features, model.keep) if kp]
    rows = []
    for i in range(len(df)):
        r = {}
        for name, s, col in (("knn", s_knn[i], "score_knn"), ("rms", s_rms[i], "score_rms"), ("maha", s_maha[i], "score_maha")):
            pct, rp, ex = _pct_rank(ref[col].to_numpy(), float(s))
            r[f"ood_score_{name}"] = float(s); r[f"ood_pct_{name}"] = pct; r[f"ood_rank_p_{name}"] = rp; r[f"ood_exceeds_{name}"] = ex
        # matched-count sensitivity: query vs each LOO reference set (n_ref − 1 sites), percentile within the LOO scores
        pcts = []
        for j, (med_j, scale_j, keep_j, Zr_j) in enumerate(model.loo_fits):
            zq = _z(X[i:i + 1], med_j, scale_j, keep_j)
            pcts.append(_pct_rank(ref["score_knn"].to_numpy(), float(_knn(Zr_j, zq, model.k)[0][0]))[0])
        r["ood_pct_knn_matched_median"] = float(np.median(pcts)) if pcts else float("nan")
        r["ood_nearest_ref"] = "|".join(f"{model.ref_sites[j]}({model.ref_groups[j]})" for j in nn[i])
        diff2 = ((Zq[i][None, :] - model.Zref[nn[i]]) ** 2).mean(axis=0)
        idx = np.argsort(-diff2)[:top]
        r["ood_top_features"] = "|".join(f"{kept_features[j]}({'+' if Zq[i][j] > 0 else '-'}z={Zq[i][j]:+.2f})" for j in idx)
        rows.append(r)
    out = pd.DataFrame(rows, index=df.index)
    out["ood_variant"] = model.variant
    return out


# ----------------------------------------------------------------------------------------------------------------
# answer 3 — hand-off
# ----------------------------------------------------------------------------------------------------------------
def _fmt_contrib(cs: list[dict]) -> str:
    return "|".join(f"{c['feature']}({c['sign']}{abs(c['contribution']):.2f}; x={c['value']:.4g}; ref {c['ref_median']:.4g}±{c['ref_mad']:.3g})" for c in cs)


def categorise_site(site_row: pd.Series | pd.DataFrame, model: CategoriserModel, ood_models: dict | None = None, top: int = 3) -> dict:
    """One sample through one categoriser (and optional OOD models). ``site_row`` is a Series or one-row frame with the
    feature columns (and flag columns if present). Returns a flat dict: model probabilities, argmax, top categoriser
    contributions, per-variant OOD outputs, flags, and the QC pointer."""
    df = site_row.to_frame().T if isinstance(site_row, pd.Series) else site_row.iloc[:1]
    pr = predict_proba(model, df).iloc[0]
    out = {"site": str(df["site"].iloc[0]) if "site" in df.columns else "?", "family": model.family}
    out.update({k: float(v) for k, v in pr.items() if k.startswith("mp_")})
    out["argmax"] = pr["argmax"]
    ci = model.classes.index(pr["argmax"])
    out["cat_top_features"] = _fmt_contrib(categoriser_contributions(model.pipeline, _matrix(df, model.features)[0], model.features, ci,
                                                                     model.ref_median, model.ref_mad, top))
    for v, om in (ood_models or {}).items():
        o = ood_score(om, df, top).iloc[0]
        out.update({f"{k}__{v}": (o[k].item() if hasattr(o[k], "item") else o[k]) for k in o.index if k != "ood_variant"})
    for c in FLAG_COLS:
        out[c] = df[c].iloc[0] if c in df.columns else None
    out["qc_pointer"] = QC_POINTER
    out["probability_note"] = model.note
    return out


def categorise_sites(df: pd.DataFrame, models: dict, ood_models: dict, role: str = "scored", top: int = 3,
                     loo_results: dict | None = None) -> pd.DataFrame:
    """Per-site hand-off table for every row of ``df``. ``models`` maps family -> CategoriserModel (fit on all known
    sites), ``ood_models`` maps variant -> OODModel. When ``loo_results`` (from :func:`loo_evaluate`, keyed by family)
    is given for KNOWN sites, their model probabilities and contributions come from the leave-one-site-out fold model
    that did not see them, not from the full fit."""
    rows = []
    for i in range(len(df)):
        d = df.iloc[i:i + 1]
        site = str(d["site"].iloc[0])
        r = {"batch": str(d["batch"].iloc[0]) if "batch" in d.columns else "", "site": site, "role": role}
        for c in FLAG_COLS:
            r[c] = d[c].iloc[0] if c in d.columns else None
        for fam, m in models.items():
            loo = loo_results.get(fam) if loo_results else None
            if loo is not None and site in set(loo["sites"]):
                j = int(np.flatnonzero(loo["sites"] == site)[0])
                ps = loo["per_site"].iloc[j]
                probs = {c: float(ps[f"mp_{c}"]) for c in m.classes}
                am = ps["argmax"]; pipe = loo["fold_models"][j]; src = "loo"
            else:
                pr = predict_proba(m, d).iloc[0]
                probs = {c: float(pr[f"mp_{c}"]) for c in m.classes}
                am = pr["argmax"]; pipe = m.pipeline; src = "full_fit"
            for c, p in probs.items():
                r[f"mp_{c}__{fam}"] = p
            r[f"argmax__{fam}"] = am
            r[f"source__{fam}"] = src
            r[f"cat_top_features__{fam}"] = _fmt_contrib(categoriser_contributions(pipe, _matrix(d, m.features)[0], m.features,
                                                                                  m.classes.index(am), m.ref_median, m.ref_mad, top))
        for v, om in ood_models.items():
            if role == "known" and site in set(om.ref_sites):      # reference site: its own LOO score, not a self-match
                j = om.ref_sites.index(site)
                ref = om.ref_loo.iloc[j]
                for name in ("knn", "rms", "maha"):
                    s = float(ref[f"score_{name}"]); others = om.ref_loo[f"score_{name}"].to_numpy()
                    pct, rp, ex = _pct_rank(others, s)
                    r[f"ood_score_{name}__{v}"] = s; r[f"ood_pct_{name}__{v}"] = pct; r[f"ood_rank_p_{name}__{v}"] = rp
                    r[f"ood_exceeds_{name}__{v}"] = False       # a reference site defines the distribution; it cannot exceed it
                r[f"ood_pct_knn_matched_median__{v}"] = float("nan")
                med_j, scale_j, keep_j, Zr_j = om.loo_fits[j]
                zq = _z(_matrix(d, om.features), med_j, scale_j, keep_j)
                _, nn_j = _knn(Zr_j, zq, om.k)
                others_ids = [s for s in om.ref_sites if s != site]; others_grp = [g for s, g in zip(om.ref_sites, om.ref_groups) if s != site]
                r[f"ood_nearest_ref__{v}"] = "|".join(f"{others_ids[t]}({others_grp[t]})" for t in nn_j[0]) + " (leave-one-out)"
                kept = [f for f, kp in zip(om.features, keep_j) if kp]
                diff2 = ((zq[0][None, :] - Zr_j[nn_j[0]]) ** 2).mean(axis=0)
                r[f"ood_top_features__{v}"] = "|".join(f"{kept[t]}({'+' if zq[0][t] > 0 else '-'}z={zq[0][t]:+.2f})" for t in np.argsort(-diff2)[:top])
            else:
                o = ood_score(om, d, top).iloc[0]
                for k in o.index:
                    if k != "ood_variant":
                        r[f"{k}__{v}"] = o[k].item() if hasattr(o[k], "item") else o[k]
        r["qc_pointer"] = QC_POINTER
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# context only
# ----------------------------------------------------------------------------------------------------------------
def kruskal_context(df: pd.DataFrame, features: Sequence[str] | None = None, batch_col: str = "batch") -> pd.DataFrame:
    """Kruskal–Wallis across batches per feature. CONTEXT ONLY: no selection or model choice reads this table."""
    features = list(FAMILIES["combined"] if features is None else features)
    fam = {**{f: "morph" for f in MORPH_FEATURES}, **{f: "texture" for f in TEXTURE_FEATURES}, **{f: "acq" for f in ACQ_FEATURES}}
    rows = []
    for f in features:
        groups = [g[f].dropna().to_numpy(float) for _, g in df.groupby(batch_col)]
        groups = [g for g in groups if len(g)]
        try:
            H, p = sps.kruskal(*groups) if len(groups) > 1 else (float("nan"), float("nan"))
        except ValueError:
            H, p = float("nan"), float("nan")
        meds = df.groupby(batch_col)[f].median()
        rows.append(dict(feature=f, family=fam.get(f, "?"), H=float(H), p=float(p), **{f"median_{b}": float(v) for b, v in meds.items()}))
    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------
# run / outputs
# ----------------------------------------------------------------------------------------------------------------
def _ci_str(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def summary_tables(results: dict) -> pd.DataFrame:
    rows = []
    for fam, r in results.items():
        m, c = r["metrics"], r["calibration"]
        rows.append(dict(family=fam, primary=(fam == PRIMARY_FAMILY), n_features=len(r["features"]), n_sites=m["n"],
                         accuracy=m["accuracy"], accuracy_ci95=_ci_str(m["accuracy_ci95"]), majority_accuracy=m["majority_accuracy"],
                         balanced_accuracy=m["balanced_accuracy"], chance_balanced=m["chance_balanced"],
                         **{f"recall_{k}": v for k, v in m["recall"].items()},
                         perm_p=r["p"], n_perm=r["n_perm"],
                         null_bal_acc_median=float(np.median(r["perm_null"])) if len(r["perm_null"]) else float("nan"),
                         null_bal_acc_p95=float(np.percentile(r["perm_null"], 95)) if len(r["perm_null"]) else float("nan"),
                         brier=c["brier"], brier_prior=c["brier_prior"], ece=c["ece"], mean_top_prob=c["mean_top_prob"],
                         top_accuracy=c["top_accuracy"], C_values=",".join(f"{v:g}" for v in sorted(set(r["per_site"]["C"]))),
                         label=r["label"]))
    return pd.DataFrame(rows)


def ood_batch_summary(site_table: pd.DataFrame, variants=tuple(OOD_VARIANTS), ref_batch: str = "Batch_3") -> pd.DataFrame:
    rows = []
    for v in variants:
        for b, g in site_table.groupby("batch"):
            if b == ref_batch:
                continue
            pct = g[f"ood_pct_knn__{v}"].to_numpy(float)
            rows.append(dict(variant=v, batch=b, n_sites=len(g), pct_min=pct.min(), pct_median=float(np.median(pct)), pct_max=pct.max(),
                             n_exceed_ref_max=int(g[f"ood_exceeds_knn__{v}"].astype(bool).sum()),
                             frac_exceed_ref_max=float(g[f"ood_exceeds_knn__{v}"].astype(bool).mean()),
                             n_rank_p_at_floor=int((g[f"ood_rank_p_knn__{v}"] <= 1 / (17 + 1) + 1e-9).sum()),
                             matched_pct_median=float(np.nanmedian(g[f"ood_pct_knn_matched_median__{v}"])),
                             rms_n_exceed=int(g[f"ood_exceeds_rms__{v}"].astype(bool).sum()),
                             maha_n_exceed=int(g[f"ood_exceeds_maha__{v}"].astype(bool).sum()), label=OOD_LABELS.get(v, v)))
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k not in ("fold_models", "X", "y", "sites")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def run(known_dirs: Sequence[str], score_dir: str | None = None, cache_dir: str | None = DEFAULT_FEATURE_CACHE, out_dir: str = DEFAULT_OUT,
        n_perm: int | None = None, seed: int = 0, n_jobs: int = -1, ref_batch: str = "Batch_3", verbose: bool = True,
        score_cache_dir: str | None = None) -> dict:
    """Fit on the known folders; LOO-evaluate every family (with permutations unless ``--score`` is given and n_perm is
    None); fit the OOD models on the reference; score the known sites (LOO) and, if given, every site of ``score_dir``;
    write CSVs, a markdown and an HTML table to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    known = load_known(known_dirs, cache_dir, verbose=verbose)
    if ref_batch not in set(known["batch"].astype(str)):
        raise ValueError(f"reference batch {ref_batch!r} not among the known folders {sorted(set(known['batch']))}")
    if n_perm is None:
        n_perm = 0 if score_dir else N_PERM_DEFAULT
    if verbose:
        print(f"[categorise] known sites: {known.groupby('batch').size().to_dict()}; n_perm = {n_perm}", flush=True)
    results = {fam: loo_evaluate(known, fam, n_perm=n_perm, seed=seed, n_jobs=n_jobs, verbose=verbose) for fam in FAMILIES}
    models = {fam: fit_categoriser(known, fam, seed=seed, ref_batch=ref_batch) for fam in FAMILIES}
    reference = known[known["batch"].astype(str) == ref_batch]
    ood_models = {v: fit_ood(reference, v) for v in OOD_VARIANTS}
    known_table = categorise_sites(known, models, ood_models, role="known", loo_results=results)
    tables = [known_table]
    scored_table = None
    if score_dir:
        scored = load_known([score_dir], score_cache_dir if score_cache_dir is not None else cache_dir, verbose=verbose)
        if set(scored["site"]) & set(known["site"]):
            raise ValueError("scored folder shares site ids with the known folders")
        scored_table = categorise_sites(scored, models, ood_models, role="scored")
        tables.append(scored_table)
    site_table = pd.concat(tables, ignore_index=True)
    summ = summary_tables(results)
    oods = ood_batch_summary(known_table, ref_batch=ref_batch)
    kw = kruskal_context(known)
    # outputs
    site_table.to_csv(os.path.join(out_dir, "site_table.csv"), index=False)
    summ.to_csv(os.path.join(out_dir, "categoriser_summary.csv"), index=False)
    oods.to_csv(os.path.join(out_dir, "ood_batch_summary.csv"), index=False)
    kw.to_csv(os.path.join(out_dir, "kruskal_context.csv"), index=False)
    for fam, r in results.items():
        r["metrics"]["confusion"].to_csv(os.path.join(out_dir, f"confusion_{fam}.csv"))
        r["calibration"]["reliability"].to_csv(os.path.join(out_dir, f"reliability_{fam}.csv"), index=False)
        pd.DataFrame({"perm_null_balanced_accuracy": r["perm_null"]}).to_csv(os.path.join(out_dir, f"perm_null_{fam}.csv"), index=False)
    for v, om in ood_models.items():
        om.ref_loo.to_csv(os.path.join(out_dir, f"ood_ref_loo_{v}.csv"), index=False)
    meta = dict(known_dirs=list(known_dirs), score_dir=score_dir, ref_batch=ref_batch, n_perm=n_perm, seed=seed, C_grid=list(C_GRID),
                n_inner=N_INNER, k_nn=K_NN, families={k: v for k, v in FAMILIES.items()}, ood_variants=dict(OOD_VARIANTS),
                primary_family=PRIMARY_FAMILY, primary_ood_variant=PRIMARY_OOD_VARIANT, texture_label=TEXTURE_LABEL,
                chosen_C_full_fit={fam: m.C for fam, m in models.items()}, ood_dropped={v: om.dropped for v, om in ood_models.items()},
                wall_s=time.time() - t0, results={fam: _jsonable({k: v for k, v in r.items() if k in ("metrics", "calibration", "p", "n_perm", "p_method", "features")})
                                                   for fam, r in results.items()})
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fh:
        json.dump(_jsonable(meta), fh, indent=1, default=str)
    md = render_markdown(results, summ, oods, site_table, ref_batch)
    with open(os.path.join(out_dir, "categoriser_table.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(out_dir, "categoriser_table.html"), "w") as fh:
        fh.write(render_html(results, summ, oods, site_table, ref_batch))
    if verbose:
        print(md)
        print(f"[categorise] outputs in {out_dir} ({time.time() - t0:.0f} s)")
    return dict(known=known, results=results, models=models, ood_models=ood_models, site_table=site_table, summary=summ,
                ood_summary=oods, kruskal=kw, scored_table=scored_table)


def _site_view(site_table: pd.DataFrame) -> pd.DataFrame:
    cols = ["batch", "site", "role", "acquisition_group"]
    for fam in FAMILIES:
        cols += [c for c in site_table.columns if c.startswith("mp_") and c.endswith(f"__{fam}")] + [f"argmax__{fam}"]
    for v in OOD_VARIANTS:
        cols += [f"ood_pct_knn__{v}", f"ood_exceeds_knn__{v}"]
    cols += [f"cat_top_features__{PRIMARY_FAMILY}", f"ood_top_features__{PRIMARY_OOD_VARIANT}"]
    return site_table[[c for c in cols if c in site_table.columns]]


def render_markdown(results, summ, oods, site_table, ref_batch) -> str:
    out = ["# Site categorisation, baseline OOD assessment and hand-off", "",
           "Three separate answers per site; model probabilities are classifier scores, not posteriors or confidence; "
           "OOD percentiles are descriptive evidence ranks (0–100), not probabilities of defect or evidence of equivalence. "
           "Legacy ood_rank_p columns have tail-rank floor 1/(n_ref+1), not calibrated p-values or false-alert guarantees: "
           "reference LOO fits use n_ref−1 sites and query fits use n_ref sites. "
           f"Texture family: {TEXTURE_LABEL}.", "",
           "## 1. Site categorisation (leave-one-site-out, fold-local imputer/scaler/C choice, balanced class weights)", "",
           _md_table(summ[["family", "primary", "n_features", "accuracy", "accuracy_ci95", "majority_accuracy", "balanced_accuracy",
                           "chance_balanced"] + [c for c in summ.columns if c.startswith("recall_")] + ["perm_p", "n_perm", "brier", "brier_prior", "ece"]]), "",
           "accuracy_ci95 is a binomial reference interval; overlapping LOO fits make it descriptive, not an exact 95% generalisation interval.", ""]
    for fam, r in results.items():
        out += [f"### Confusion — {fam} ({r['label']})", "", _md_table(r["metrics"]["confusion"].reset_index().rename(columns={"index": ""}), "{:.0f}"), "",
                f"Reliability of the top-class model probability — {fam}", "", _md_table(r["calibration"]["reliability"]), ""]
    out += ["## 2. Baseline OOD assessment (reference = all %s sites; k-NN robust-z distance; percentiles within the reference LOO)" % ref_batch, "",
            _md_table(oods[["variant", "batch", "n_sites", "pct_min", "pct_median", "pct_max", "n_exceed_ref_max", "frac_exceed_ref_max", "n_rank_p_at_floor",
                            "matched_pct_median", "rms_n_exceed", "maha_n_exceed"]], "{:.2f}"), "",
            "## 3. Per-site hand-off (abridged; full columns in site_table.csv)", "", _md_table(_site_view(site_table), "{:.2f}"), "",
            f"{QC_POINTER}", ""]
    return "\n".join(out)


def render_html(results, summ, oods, site_table, ref_batch) -> str:
    css = ("body{font:14px/1.45 system-ui;max-width:1400px;margin:20px auto;padding:0 12px;color:#173044}"
           "table{border-collapse:collapse;margin:8px 0 18px;font-size:12.5px}td,th{border:1px solid #cfd8e3;padding:3px 7px;text-align:right}"
           "th{background:#eef3f8}td:first-child,th:first-child{text-align:left}.note{background:#fff1d5;border:1px solid #c8ae78;padding:8px 12px}")
    def t(df, fmt="{:.3f}"):
        return df.to_html(index=False, float_format=lambda v: fmt.format(v), border=0, na_rep="nan")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Site categoriser</title><style>{css}</style></head><body>",
             "<h1>Site categorisation, baseline OOD assessment and hand-off</h1>",
             f"<p class='note'>Three separate answers per site. <b>Model probabilities</b> are normalised classifier scores, not posteriors "
             f"or confidence (calibration assessed in the findings). <b>OOD percentiles</b> are evidence ranks within the reference's own "
             f"leave-one-site-out distribution (0–100), not probabilities of defect or evidence of equivalence. Legacy ood_rank_p columns have "
             f"tail-rank floor 1/(n_ref+1), not calibrated p-values or false-alert guarantees: reference LOO fits use n_ref−1 sites, queries n_ref sites. "
             f"Texture family: {TEXTURE_LABEL}. {QC_POINTER}.</p>",
             "<h2>1. Site categorisation</h2>", t(summ.drop(columns=["label"])),
             "<p>accuracy_ci95 is a binomial reference interval; overlapping LOO fits make it descriptive, not an exact 95% generalisation interval.</p>"]
    for fam, r in results.items():
        parts += [f"<h3>Confusion — {fam}: {r['label']}</h3>", r["metrics"]["confusion"].to_html(border=0),
                  f"<p>Reliability of the top-class model probability ({fam})</p>", t(r["calibration"]["reliability"])]
    parts += [f"<h2>2. Baseline OOD assessment (reference = {ref_batch})</h2>", t(oods, "{:.2f}"),
              "<h2>3. Per-site hand-off (abridged)</h2>", t(_site_view(site_table), "{:.2f}"), "</body></html>"]
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-site categorisation (model probabilities), baseline OOD percentiles and hand-off.")
    ap.add_argument("known", nargs="+", help="known batch folders; the reference (default Batch_3) must be among them")
    ap.add_argument("--score", default=None, help="folder whose sites are scored with models fitted on the known folders")
    ap.add_argument("--reference", default="Batch_3")
    ap.add_argument("--cache-dir", default=DEFAULT_FEATURE_CACHE, help="feature cache for the known folders")
    ap.add_argument("--score-cache-dir", default=None, help="feature cache for the --score folder (default: --cache-dir); use a scratch dir to leave the shared cache untouched")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-perm", type=int, default=None, help=f"site-label permutations (default {N_PERM_DEFAULT}; 0 with --score)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args(argv)
    run(a.known, a.score, a.cache_dir, a.out, a.n_perm, a.seed, a.n_jobs, a.reference, score_cache_dir=a.score_cache_dir)


if __name__ == "__main__":
    main()
