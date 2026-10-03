"""polaron_qc.ml — ML corroboration (plan §2.5): two members, never the sole driver.

1. ``c2st``   grouped classifier two-sample test. L1-logistic regression reference-vs-batch with the SITE as the
              independent unit: StandardScaler + LogisticRegression refitted inside every fold, grouped CV (sites
              never straddle folds), equal total weight per site regardless of patch count, out-of-fold
              probabilities averaged to site means, AUC at the site level, Monte Carlo null from SITE-level label
              permutations with the whole CV procedure repeated inside each permutation (plan §2.0 rules 2, 6).
2. ``patch_embeddings`` + ``novelty`` + ``novelty_correlates``   exploratory patch-embedding novelty: per-image-
              normalised 512-px BSE patches → pretrained DINOv2 ViT-S/14 CLS embedding (fallback: torchvision
              ResNet-50) → PCA fitted on the reference only → mean distance to the k nearest reference patches.
              Calibrated against the reference's own leave-one-site-out novelty with PCA re-fitted in every fold.
              Correlations with KPIs and acquisition statistics are reported so a reader sees what it tracks.

Both members are corroborators: a classifier that separates beyond the site-permutation null corroborates a KPI
drift and names the KPIs; a novelty signal that fires when no trusted KPI moves forces *investigate* with the
novelty map as evidence (plan §2.5). Neither drives a verdict alone. Patches never enter n.

Inputs are plain arrays / DataFrames so the functions do not depend on the patch table that ``features`` produces.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import CRACKED_SITES, GREY_PORE_SITES, LOW_CONTRAST_SITES

# KPIs the classifier is allowed to see (site-level, trusted or "flags-aware"; see docs/problem_and_findings.md §4).
TRUSTED_KPIS = ["pore_frac", "pore_d50", "pore_elong", "pore_max_d", "crack_frac", "bright_frac",
                "bright_count_per_Mpx", "bright_d50", "bright_circ", "corr_len_px",
                "etd_crack_density_particles", "etd_boundary_sharpness"]
# Acquisition / session statistics the novelty score is correlated against (C03). Never material KPIs.
ACQUISITION_COLS = ["graphite_mode", "bright_sep", "bright_low_contrast", "pore_mode_resolved", "H",
                    "etd_boundary_sharpness", "etd_curtain_frac", "etd_curtain_anisotropy",
                    "bse_p1", "bse_p50", "bse_p99", "bse_std", "bse_empty_bin_frac", "bse_gray_levels",
                    "inlens_p50", "etd_p50", "flag_low_contrast", "flag_grey_pore", "flag_cracked"]
RIDGE_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]          # same grid as notebook 01 (T* rule replicated)
NORM_VERSION = "n1"                                         # bump when the per-image normalisation changes
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_cache", "ml")


# ----------------------------------------------------------------------------------------------------------------
# Part 1 — grouped classifier two-sample test
# ----------------------------------------------------------------------------------------------------------------
def _site_weights(groups: np.ndarray, y: np.ndarray | None = None, class_balance: bool = True) -> np.ndarray:
    """Row weights such that every SITE has total weight 1 whatever its patch count; with ``class_balance`` each
    class (reference / batch) additionally gets equal total weight, counted in sites not rows. Mean weight is 1."""
    groups = np.asarray(groups)
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    if class_balance and y is not None:
        y = np.asarray(y)
        for c in np.unique(y):
            m = y == c
            n_sites_c = len(np.unique(groups[m]))
            w[m] = w[m] / n_sites_c
    return w * len(w) / w.sum()


def _site_table(y: np.ndarray, groups: np.ndarray):
    """Unique sites, their label, and row->site index. Raises if a site carries two labels."""
    sites, idx = np.unique(groups, return_inverse=True)
    site_y = np.full(len(sites), -1)
    for i, s in enumerate(sites):
        ys = np.unique(y[idx == i])
        if len(ys) != 1:
            raise ValueError(f"site {s!r} appears in both reference and batch; site ids must be disjoint")
        site_y[i] = ys[0]
    return sites, site_y, idx


def _canonical_order(X: np.ndarray, y: np.ndarray, groups: np.ndarray, decimals: int = 10):
    """Content-based canonical row order and integer site codes, independent of site ids and of input row order.

    Rows are sorted inside each site by their rounded feature vector (ties: the full vector). Each site then gets the key
    (sha1 of its rounded, row-sorted block, the block's exact bytes, its label) and sites are sorted by that key. Returns
    ``(order, codes, names)``: ``X[order]`` is the canonical row order, ``codes`` the canonical integer site id
    (0 … n_sites−1 in canonical order) of each row *of the reordered array*, ``names[i]`` the caller's id of canonical
    site i. Two calls with identical content give identical outputs whatever the site ids or row order, so grouped CV
    folds and the site-label permutation stream cannot change when sites are relabelled (E23 / D34 open item). Sites with
    identical content and label are exchangeable, so their relative order cannot affect any result. Raises if a site
    carries two labels (site ids must be disjoint between reference and batch)."""
    names, inv = np.unique(np.asarray(groups), return_inverse=True)
    X = np.asarray(X, float); y = np.asarray(y)
    Xr = np.round(X, decimals) + 0.0                       # +0.0 maps -0.0 to 0.0 so the bytes are canonical
    d = X.shape[1]
    within = np.empty(len(X), int)
    keys = []
    for i, name in enumerate(names):
        rows = np.flatnonzero(inv == i)
        labels = np.unique(y[rows])
        if len(labels) != 1:
            raise ValueError(f"site {name!r} appears in both reference and batch; site ids must be disjoint")
        blk, blk_r = X[rows], Xr[rows]
        cols = [blk_r[:, j] for j in range(d)] + [blk[:, j] for j in range(d)]
        o = np.lexsort(cols[::-1]) if d else np.arange(len(rows))   # primary key = rounded column 0, …, then exact values
        within[rows[o]] = np.arange(len(rows))
        keys.append((hashlib.sha1(np.ascontiguousarray(blk_r[o]).tobytes()).hexdigest(),
                     np.ascontiguousarray(blk[o]).tobytes(), int(labels[0])))
    site_rank = np.empty(len(names), int)
    site_rank[sorted(range(len(names)), key=keys.__getitem__)] = np.arange(len(names))
    codes_all = site_rank[inv]
    order = np.lexsort((within, codes_all))
    return order, codes_all[order], names[np.argsort(site_rank)]


def _make_cv(site_y: np.ndarray, cv, seed: int):
    """'auto': StratifiedGroupKFold with min(5, smallest class site count) folds when both classes have >= 3
    sites, else LeaveOneGroupOut. An int forces that many stratified group folds; 'logo' forces leave-one-site-out."""
    n_min = int(min(np.bincount(site_y)))
    if cv == "logo" or (cv == "auto" and n_min < 3):
        return LeaveOneGroupOut(), "LeaveOneGroupOut"
    k = min(5, n_min) if cv == "auto" else int(cv)
    k = max(2, min(k, n_min))
    return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed), f"StratifiedGroupKFold({k})"


def _lr_pipeline(C: float, seed: int = 0) -> Pipeline:
    # liblinear shuffles its active set with its own RNG; without random_state sklearn seeds it from the global RNG and the
    # coefficients differ run to run at the 1e-5 level — seeded so the result depends on content and seed only
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000, random_state=seed))])


def _grouped_cv(X, y, groups, C, seed, cv="auto", class_balance=True):
    """One full grouped-CV pass. Returns site ids, site labels, site-mean out-of-fold probability, site-level AUC,
    and a (n_folds, n_features) boolean matrix of which features each fold selected."""
    sites, site_y, idx = _site_table(y, groups)
    splitter, cv_name = _make_cv(site_y, cv, seed)
    oof = np.full(len(y), np.nan)
    selected = []
    for tr, te in splitter.split(X, y, groups):
        ytr = y[tr]
        if len(np.unique(ytr)) < 2:                      # degenerate fold: no information
            oof[te] = 0.5
            selected.append(np.zeros(X.shape[1], bool))
            continue
        w = _site_weights(groups[tr], ytr, class_balance)
        pipe = _lr_pipeline(C, seed)
        pipe.fit(X[tr], ytr, lr__sample_weight=w)
        oof[te] = pipe.predict_proba(X[te])[:, 1]
        selected.append(pipe["lr"].coef_[0] != 0)
    site_prob = np.array([oof[idx == i].mean() for i in range(len(sites))])
    auc = roc_auc_score(site_y, site_prob)
    return dict(sites=sites, site_y=site_y, site_prob=site_prob, auc=float(auc),
                selected=np.array(selected), cv=cv_name, site_idx=idx)


def c2st(X_ref, X_batch, groups_ref, groups_batch, feature_names: Sequence[str], n_perm: int = 200,
         C: float = 0.5, seed: int = 0, cv="auto", class_weight="balanced", verbose: bool = False) -> dict:
    """Grouped classifier two-sample test, reference (label 0) vs batch (label 1).

    Rows may be sites or patches; ``groups_*`` are site ids (must be disjoint between the two sets).
    Preprocessing (StandardScaler) and the L1-logistic model are fitted inside every fold and inside every
    permutation. Each site contributes equal total weight; ``class_weight="balanced"`` balances the two classes by
    SITE count (sklearn's row-level "balanced" would count patches, which breaks plan §2.0 rule 6). Out-of-fold
    probabilities are averaged per site and the AUC is computed over sites. The null permutes labels at the SITE
    level and repeats the whole CV procedure; p = (b + 1) / (n_perm + 1), one-sided (AUC >= observed), Monte Carlo.
    Before anything runs, rows and sites are put in a canonical content-based order (:func:`_canonical_order`), so the
    result is bit-identical for identical content whatever the site ids or input row order; ``per_site_scores`` reports
    the caller's site ids.

    Returns dict(auc, auc_null, null_band, p, n_perm, p_method, coef, per_site_scores, cv, n_sites_ref,
    n_sites_batch, n_rows). ``coef`` is a refit on all data for reporting only (coef, sign, selected,
    selection_stability = fraction of CV folds selecting the feature, mean_abs_shap = exact linear SHAP, i.e.
    mean |coef_j * (z_ij - mean z_j)| on the standardised features)."""
    X = np.vstack([np.asarray(X_ref, float), np.asarray(X_batch, float)])
    y = np.r_[np.zeros(len(X_ref), int), np.ones(len(X_batch), int)]
    groups = np.r_[np.asarray(groups_ref).astype(str), np.asarray(groups_batch).astype(str)]
    feature_names = list(feature_names)
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match X columns")
    if np.isnan(X).any():
        raise ValueError("X contains NaN; impute or drop before calling c2st")
    cb = class_weight == "balanced"
    rng = np.random.default_rng(seed)

    # canonical content-based order: folds, the permutation stream and the solver see the same arrays whatever the
    # site ids or row order of the input (E23 found AUC 0.51 → 0.41 on identical images under new ids)
    order, groups, site_names = _canonical_order(X, y, groups)
    X, y = X[order], y[order]

    obs = _grouped_cv(X, y, groups, C, seed, cv, cb)
    sites, site_y, idx = obs["sites"], obs["site_y"], obs["site_idx"]

    auc_null = np.empty(n_perm)
    t0 = time.time()
    for b in range(n_perm):
        site_y_perm = rng.permutation(site_y)             # shuffle which SITES are "reference"
        y_perm = site_y_perm[idx]
        auc_null[b] = _grouped_cv(X, y_perm, groups, C, seed, cv, cb)["auc"]
        if verbose and (b + 1) % 50 == 0:
            print(f"  permutation {b + 1}/{n_perm} ({time.time() - t0:.0f} s)")
    p = (1 + np.sum(auc_null >= obs["auc"] - 1e-12)) / (n_perm + 1)

    # refit on everything, reporting only
    w_all = _site_weights(groups, y, cb)
    pipe = _lr_pipeline(C, seed).fit(X, y, lr__sample_weight=w_all)
    coef = pipe["lr"].coef_[0]
    Z = pipe["scale"].transform(X)
    zc = Z - np.average(Z, axis=0, weights=w_all)
    mean_abs_shap = np.average(np.abs(zc * coef), axis=0, weights=w_all)
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coef,
                            "sign": np.where(coef > 0, "+", np.where(coef < 0, "-", "0")),
                            "selected": coef != 0,
                            "selection_stability": obs["selected"].mean(axis=0),
                            "mean_abs_shap": mean_abs_shap})
    coef_df = coef_df.reindex(coef_df.mean_abs_shap.abs().sort_values(ascending=False).index).reset_index(drop=True)
    n_rows = np.bincount(idx)
    per_site = pd.DataFrame({"site": site_names[sites], "label": np.where(site_y == 1, "batch", "reference"),
                             "n_rows": n_rows, "mean_prob": obs["site_prob"]}).sort_values("site", kind="stable").reset_index(drop=True)
    return dict(auc=obs["auc"], auc_null=auc_null,
                null_band=tuple(np.percentile(auc_null, [2.5, 97.5])), p=float(p), n_perm=int(n_perm),
                p_method=f"Monte Carlo site-label permutation, {n_perm} resamples, p = (b+1)/(n+1), one-sided",
                coef=coef_df, per_site_scores=per_site, cv=obs["cv"],
                n_sites_ref=int((site_y == 0).sum()), n_sites_batch=int((site_y == 1).sum()), n_rows=int(len(y)),
                C=C, class_weight="balanced (site-level)" if cb else "per-site equal only")


# ----------------------------------------------------------------------------------------------------------------
# Part 2 — exploratory patch-embedding novelty
# ----------------------------------------------------------------------------------------------------------------
def bright_bands(img: np.ndarray, margin: float = 0.2, delta: float = 40, minrows: int = 4):
    """Rows at the top/bottom far brighter than the body (current collector / stitching border). Replicates
    ``bright_bands`` in notebooks/_build_01_dataset_analysis.py."""
    rm = img.mean(1)
    body = np.median(rm)
    H = len(rm)
    hot = rm > body + delta
    top = 0
    while top < H * margin and hot[top]:
        top += 1
    bot = 0
    while bot < H * margin and hot[H - 1 - bot]:
        bot += 1
    idx = np.where(hot)[0]
    if (idx >= H * (1 - margin)).any():
        bot = max(bot, H - idx[idx >= H * (1 - margin)].min())
    if (idx < H * margin).any():
        top = max(top, idx[idx < H * margin].max() + 1)
    return (top if top >= minrows else 0), (bot if bot >= minrows else 0)


def load_channel0(path: str) -> np.ndarray:
    """Read a TIFF; the images are grayscale stored as RGB (channel 0 == channel 2; channel 1 differs in
    ~0.01 % of pixels). Returns uint8 2-D."""
    import tifffile
    im = tifffile.imread(path)
    if im.ndim == 3:
        im = im[..., 0]
    return im


def normalise_image(img: np.ndarray, clip: float = 3.0) -> tuple[np.ndarray, dict]:
    """Per-IMAGE robust normalisation: (img - median) / IQR, clipped to [-clip, clip], mapped to [0, 1].
    Removes brightness/contrast stretching differences between acquisitions as far as a monotone affine map can.
    Note that the bright phase (+50–70 levels above graphite) and open pores typically saturate at the clip, so the
    embedding sees phase *shapes* and graphite texture, not intra-phase intensity. Returns (float32 image, meta)."""
    x = img.astype(np.float32)
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(q75 - q25)
    if iqr <= 0:
        iqr = float(x.std()) or 1.0
    z = np.clip((x - med) / iqr, -clip, clip)
    sat = float(np.mean(np.abs(z) >= clip))
    return ((z + clip) / (2 * clip)).astype(np.float32), dict(median=med, iqr=iqr, clip=clip, saturated_frac=sat)


def tile_coords(H: int, W: int, patch: int = 512, stride: int = 512) -> list[tuple[int, int, int]]:
    """(patch_id, y0, x0) for every complete patch, row-major."""
    out = []
    pid = 0
    for y0 in range(0, H - patch + 1, stride):
        for x0 in range(0, W - patch + 1, stride):
            out.append((pid, y0, x0))
            pid += 1
    return out


def _device(device=None):
    import torch
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def load_embedding_model(model: str = "dinov2_vits14", device=None):
    """Try ``torch.hub.load('facebookresearch/dinov2', model)``; on failure fall back to torchvision ResNet-50
    (IMAGENET1K_V2) penultimate features. Returns (module in eval mode, name actually used, embedding dim)."""
    import torch
    dev = _device(device)
    if model.startswith("dinov2"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")           # xFormers-not-available chatter
                m = torch.hub.load("facebookresearch/dinov2", model, verbose=False)
            m.eval().to(dev)
            return m, model, int(m.embed_dim)
        except Exception as e:                            # noqa: BLE001 — any hub/network failure → fallback
            print(f"[ml] torch.hub DINOv2 load failed ({type(e).__name__}: {e}); falling back to resnet50")
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    m.eval().to(dev)
    return m, "resnet50_imagenet1k_v2", 2048


def embed_patches(model, patches: np.ndarray, res: int = 224, device=None, batch_size: int = 32) -> np.ndarray:
    """patches: (n, h, w) float32 in [0, 1]. Resize to res×res (bilinear, antialiased), replicate to 3 channels,
    ImageNet-normalise, forward. DINOv2 returns the CLS token (after norm); ResNet-50 the pooled penultimate layer."""
    import torch
    import torch.nn.functional as F
    try:                                                  # inputs must live where the model lives
        dev = next(model.parameters()).device
    except (StopIteration, AttributeError):
        dev = _device(device)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(patches[i:i + batch_size])).to(dev).unsqueeze(1)
            if x.shape[-1] != res:
                x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
            x = (x.repeat(1, 3, 1, 1) - mean) / std
            out.append(model(x).float().cpu().numpy())
    return np.vstack(out)


def _find_image(batch_dir: str, site: str, det: str) -> str:
    names = [det] + (["SE"] if det == "ETD" else []) + (["ETD"] if det == "SE" else [])
    for d in names:
        p = os.path.join(batch_dir, f"img_{site}_{d}.tif")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no {det} image for site {site} in {batch_dir}")


def list_sites(batch_dir: str, det: str = "BSE") -> list[str]:
    dets = {det, "SE", "ETD"} if det in ("SE", "ETD") else {det}
    sites = set()
    for f in glob.glob(os.path.join(batch_dir, "img_*_*.tif")):
        m = re.match(r"img_(.+)_([A-Za-z]+)\.tif$", os.path.basename(f))
        if m and m.group(2) in dets:
            sites.add(m.group(1))
    return sorted(sites)


def patch_embeddings(batch_dir: str, sites: Iterable[str] | None = None, det: str = "BSE", patch: int = 512,
                     stride: int = 512, model: str = "dinov2_vits14", res: int = 518, cache_dir: str = DEFAULT_CACHE,
                     device=None, clip: float = 3.0, batch_size: int = 32, verbose: bool = True,
                     _model_obj=None) -> pd.DataFrame:
    """Per-site patch embeddings of per-image-normalised, edge-trimmed patches.

    Returns DataFrame(batch, site, patch_id, y0, x0, emb_0..emb_{k-1}); ``df.attrs`` carries model_used, res,
    normalisation and per-site meta (bands, median, IQR, saturated fraction). y0 is in the coordinates of the
    ORIGINAL image (band offset added back) so patches can be painted on the strip. ``res`` = model input size:
    measured on the M2 Max (MPS, ViT-S/14, 52 patches): 0.2 s at 224, 2.1 s at 518, so 518 (DINOv2's
    high-resolution adaptation size, 14-px tokens ≈ 14 image px) is the default; 224 is kept as a sensitivity. Cached per
    (batch, site, model, patch, stride, res, normalisation version) as .npz in ``cache_dir``; one site at a time."""
    batch = os.path.basename(os.path.normpath(batch_dir))
    sites = list(sites) if sites is not None else list_sites(batch_dir, det)
    os.makedirs(cache_dir, exist_ok=True)
    norm_tag = f"{NORM_VERSION}c{clip:g}"
    mdl, used, dim = (None, model, None)
    frames, meta = [], {}
    t0 = time.time()
    for site in sites:
        # cache key uses the requested model name; the file records which model was actually used
        cf = os.path.join(cache_dir, f"emb_{batch}_{site}_{det}_{model}_p{patch}_s{stride}_r{res}_{norm_tag}.npz")
        if os.path.exists(cf):
            z = np.load(cf, allow_pickle=False)
            emb, y0, x0, pid = z["emb"], z["y0"], z["x0"], z["patch_id"]
            m = json.loads(str(z["meta"]))
            used = m.get("model_used", used)
        else:
            if mdl is None and _model_obj is None:
                mdl, used, dim = load_embedding_model(model, device)
            elif _model_obj is not None:
                mdl, used = _model_obj, f"{model}(injected)"
            img = load_channel0(_find_image(batch_dir, site, det))
            top, bot = bright_bands(img)
            body = img[top: img.shape[0] - bot if bot else None]
            x, m = normalise_image(body, clip)
            coords = tile_coords(*x.shape, patch, stride)
            if not coords:
                raise ValueError(f"{site}: image {x.shape} smaller than one patch")
            P = np.stack([x[a:a + patch, b:b + patch] for _, a, b in coords])
            emb = embed_patches(mdl, P, res, device, batch_size)
            pid = np.array([c[0] for c in coords]); y0 = np.array([c[1] for c in coords]) + top
            x0 = np.array([c[2] for c in coords])
            m.update(band_top=int(top), band_bottom=int(bot), model_used=used, res=res, H=int(img.shape[0]),
                     W=int(img.shape[1]), n_patches=len(coords))
            np.savez_compressed(cf, emb=emb.astype(np.float32), y0=y0, x0=x0, patch_id=pid, meta=json.dumps(m))
            if verbose:
                print(f"[ml] {batch}/{site}: {len(coords)} patches, bands ({top},{bot}), "
                      f"sat {m['saturated_frac']:.2f}, {time.time() - t0:.0f} s")
        meta[site] = m
        df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
        df.insert(0, "x0", x0); df.insert(0, "y0", y0); df.insert(0, "patch_id", pid)
        df.insert(0, "site", site); df.insert(0, "batch", batch)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.attrs.update(model_used=used, res=res, patch=patch, stride=stride, det=det,
                     normalisation=f"per-image (x-median)/IQR clip ±{clip:g} -> [0,1] ({norm_tag})", site_meta=meta)
    return out


def emb_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("emb_")]
    return df[cols].to_numpy(np.float32)


def _fit_pca(X: np.ndarray, n_components: int, seed: int) -> PCA:
    """Separate so tests can spy on how often PCA is refitted."""
    return PCA(n_components=n_components, random_state=seed).fit(X)


def _knn_novelty(Z_ref: np.ndarray, Z_q: np.ndarray, k: int, return_index: bool = False):
    k = min(k, len(Z_ref))
    d, i = NearestNeighbors(n_neighbors=k).fit(Z_ref).kneighbors(Z_q)
    return (d.mean(axis=1), i) if return_index else d.mean(axis=1)


def _per_site(site: np.ndarray, nov: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"site": site, "novelty": nov})
    g = d.groupby("site").novelty
    return pd.DataFrame({"site": g.median().index, "novelty_median": g.median().values,
                         "novelty_p90": g.quantile(0.9).values, "n_patches": g.size().values})


def novelty(emb_ref, emb_query, groups_ref, groups_query, n_components: int = 32, k: int = 5, seed: int = 0,
            patch_ids_ref=None, patch_ids_query=None) -> dict:
    """EXPLORATORY patch-embedding novelty. Novelty of a patch = mean Euclidean distance to its k nearest
    reference patches in a PCA space fitted on the reference patches only.

    * queries: PCA and kNN index on the full reference;
    * ``ref_loo``: every reference site held out in turn, PCA RE-FITTED on the remaining reference sites, kNN
      against those sites only — the reference's own novelty distribution, which is what a query site's novelty
      is compared with (``query_percentile``, 'weak' percentile of the query site's median / p90 within the
      ref_loo per-site distribution; resolution is 1 / n_ref_sites).
    Caveat: ref_loo uses a reference of n_ref − 1 sites while queries use all n_ref, so query novelty is very
    slightly *lower* than a same-sized reference would give (conservative). Nothing here is a verdict driver;
    the signal must be read next to ``novelty_correlates`` to see what it is tracking (plan §2.5, §5)."""
    emb_ref = np.asarray(emb_ref, np.float32); emb_query = np.asarray(emb_query, np.float32)
    groups_ref = np.asarray(groups_ref).astype(str); groups_query = np.asarray(groups_query).astype(str)
    if patch_ids_ref is None:
        patch_ids_ref = pd.Series(groups_ref).groupby(groups_ref).cumcount().to_numpy()
    if patch_ids_query is None:
        patch_ids_query = pd.Series(groups_query).groupby(groups_query).cumcount().to_numpy()
    nc = int(min(n_components, emb_ref.shape[1], len(emb_ref) - 1))

    pca = _fit_pca(emb_ref, nc, seed)
    Zr, Zq = pca.transform(emb_ref), pca.transform(emb_query)
    nov_q, nn_idx = _knn_novelty(Zr, Zq, k, return_index=True)
    per_patch = pd.DataFrame({"site": groups_query, "patch_id": patch_ids_query, "novelty": nov_q})
    # which reference patches each query patch is closest to (evidence: "nearest reference patches", plan §2.5)
    nearest_ref = pd.DataFrame({"site": np.repeat(groups_query, nn_idx.shape[1]),
                                "patch_id": np.repeat(patch_ids_query, nn_idx.shape[1]),
                                "rank": np.tile(np.arange(nn_idx.shape[1]), len(groups_query)),
                                "ref_site": groups_ref[nn_idx.ravel()], "ref_patch_id": np.asarray(patch_ids_ref)[nn_idx.ravel()]})
    per_site = _per_site(groups_query, nov_q)

    loo_rows, loo_patch, comps = [], [], []
    for s in np.unique(groups_ref):
        m = groups_ref == s
        nc_s = int(min(nc, m.size - m.sum() - 1))
        pca_s = _fit_pca(emb_ref[~m], nc_s, seed)
        nov_s = _knn_novelty(pca_s.transform(emb_ref[~m]), pca_s.transform(emb_ref[m]), k)
        loo_patch.append(pd.DataFrame({"site": s, "patch_id": patch_ids_ref[m], "novelty": nov_s}))
        loo_rows.append(dict(site=s, novelty_median=float(np.median(nov_s)),
                             novelty_p90=float(np.percentile(nov_s, 90)), n_patches=int(m.sum())))
        comps.append(pca_s.components_[0])
    ref_loo = pd.DataFrame(loo_rows)
    ref_loo_per_patch = pd.concat(loo_patch, ignore_index=True)

    qp = per_site.copy()
    qp["pct_median"] = [sps.percentileofscore(ref_loo.novelty_median, v, kind="weak") for v in qp.novelty_median]
    qp["pct_p90"] = [sps.percentileofscore(ref_loo.novelty_p90, v, kind="weak") for v in qp.novelty_p90]
    qp["exceeds_ref_loo_max"] = qp.novelty_median > ref_loo.novelty_median.max()
    return dict(per_patch=per_patch, per_site=per_site, nearest_ref=nearest_ref, ref_loo=ref_loo, ref_loo_per_patch=ref_loo_per_patch,
                query_percentile=qp[["site", "pct_median", "pct_p90", "exceeds_ref_loo_max"]],
                ref_loo_pca_first_component=np.array(comps), n_components=nc, k=k,
                pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
                n_ref_sites=int(len(np.unique(groups_ref))),
                note="exploratory: pretrained-embedding novelty; not a verdict driver; read with novelty_correlates")


def novelty_correlates(per_site_novelty: pd.DataFrame, site_features_df: pd.DataFrame,
                       acquisition_cols: Sequence[str] = ACQUISITION_COLS, kpi_cols: Sequence[str] | None = None,
                       value_col: str = "novelty_median") -> pd.DataFrame:
    """Spearman rho / p of site-level novelty against each KPI and each acquisition flag / statistic.
    ``per_site_novelty`` needs columns site + value_col (concatenate query per_site and ref_loo to use all sites).
    Returns DataFrame(variable, kind, rho, p, n) sorted by |rho|. This is how the report says what the novelty is
    tracking (C03); a strong acquisition correlate means the member cannot be read as a material signal."""
    kpi_cols = list(kpi_cols) if kpi_cols is not None else [c for c in TRUSTED_KPIS if c in site_features_df]
    df = per_site_novelty[["site", value_col]].merge(site_features_df, on="site", how="inner")
    rows = []
    for kind, cols in (("kpi", kpi_cols), ("acquisition", acquisition_cols)):
        for c in cols:
            if c not in df:
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            m = v.notna() & df[value_col].notna()
            if m.sum() < 4 or v[m].nunique() < 2:
                continue
            rho, p = sps.spearmanr(df.loc[m, value_col], v[m])
            rows.append(dict(variable=c, kind=kind, rho=float(rho), p=float(p), n=int(m.sum())))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.reindex(out.rho.abs().sort_values(ascending=False).index).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Helpers for the real data (read notebook-01 caches; features.py will later provide equivalent tables)
# ----------------------------------------------------------------------------------------------------------------
def assemble_site_table(cache_dir: str = os.path.dirname(DEFAULT_CACHE)) -> pd.DataFrame:
    """site_features + etd_inlens_features (T* rule replicated) + BSE/ETD/Inlens image statistics + group flags."""
    S = pd.read_csv(os.path.join(cache_dir, "site_features.csv"))
    E = pd.read_csv(os.path.join(cache_dir, "etd_inlens_features.csv"))
    t_star = min(RIDGE_GRID, key=lambda t: abs(t - E.ridge_p97.median()))
    E = E.assign(etd_crack_density_particles=E[f"crack_p_{t_star}"], etd_crack_density_graphite=E[f"crack_g_{t_star}"],
                 etd_curtain_frac=E[f"curtain_{t_star}"])
    keep = ["batch", "site", "etd_crack_density_particles", "etd_crack_density_graphite", "etd_curtain_frac",
            "etd_curtain_anisotropy", "etd_boundary_sharpness", "etd_grad_energy", "inlens_grad_energy",
            "inlens_particle_texture", "inlens_speckled_particle_frac"]
    df = S.merge(E[keep], on=["batch", "site"], how="left")
    Q = pd.read_csv(os.path.join(cache_dir, "image_quality.csv"))
    Q["det"] = Q.det.replace({"SE": "ETD"})
    for det, cols in (("BSE", ["p1", "p50", "p99", "std", "empty_bin_frac", "gray_levels"]), ("ETD", ["p50"]), ("Inlens", ["p50"])):
        q = Q[Q.det == det][["site"] + cols].rename(columns={c: f"{det.lower()}_{c}" for c in cols})
        df = df.merge(q, on="site", how="left")
    df["flag_low_contrast"] = df.site.isin(LOW_CONTRAST_SITES).astype(int)
    df["flag_grey_pore"] = df.site.isin(GREY_PORE_SITES).astype(int)
    df["flag_cracked"] = df.site.isin(CRACKED_SITES).astype(int)
    df["bright_low_contrast"] = df.bright_low_contrast.astype(int)
    df["pore_mode_resolved"] = df.pore_mode_resolved.astype(int)
    df.attrs["t_star"] = t_star
    return df


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.to_dict(orient="records")
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, tuple):
            out[k] = [float(x) for x in v]
        else:
            out[k] = v
    return out


def run_real_data(dataset_dir: str | None = None, out_dir: str = DEFAULT_CACHE, reference: str = "Batch_3",
                  compare: Sequence[str] = ("Batch_1", "Batch_2"), n_perm: int = 200, res: int = 518,
                  kpis: Sequence[str] = TRUSTED_KPIS, seed: int = 0) -> dict:
    """Driver used for the first real run; writes CSV/JSON into ``out_dir``. Not part of the module API."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = dataset_dir or os.path.join(root, "Dataset")
    os.makedirs(out_dir, exist_ok=True)
    site_df = assemble_site_table()
    summary = {"t_star": site_df.attrs["t_star"], "kpis": list(kpis)}

    # --- c2st at site level ---------------------------------------------------------------------------------
    ref = site_df[site_df.batch == reference]
    c2 = {}
    for b in compare:
        q = site_df[site_df.batch == b]
        variants = {"all_sites": q}
        if q.site.isin(LOW_CONTRAST_SITES).any():
            variants["without_low_contrast"] = q[~q.site.isin(LOW_CONTRAST_SITES)]
        for tag, qq in variants.items():
            t0 = time.time()
            r = c2st(ref[kpis].to_numpy(), qq[kpis].to_numpy(), ref.site, qq.site, kpis, n_perm=n_perm, seed=seed)
            r["runtime_s"] = round(time.time() - t0, 1)
            key = f"{b}_vs_{reference}_{tag}"
            c2[key] = r
            print(f"c2st {key}: AUC {r['auc']:.3f}, null 95% band {r['null_band'][0]:.2f}–{r['null_band'][1]:.2f}, "
                  f"p = {r['p']:.3f} ({r['n_perm']} perms, {r['cv']}), selected: "
                  f"{', '.join(r['coef'][r['coef'].selected].feature)}  [{r['runtime_s']} s]")
            r["coef"].to_csv(os.path.join(out_dir, f"c2st_{key}_coef.csv"), index=False)
            r["per_site_scores"].to_csv(os.path.join(out_dir, f"c2st_{key}_sites.csv"), index=False)
    with open(os.path.join(out_dir, "c2st_results.json"), "w") as f:
        json.dump({k: _jsonable({kk: vv for kk, vv in v.items()}) for k, v in c2.items()}, f, indent=1)
    summary["c2st"] = {k: dict(auc=v["auc"], null_band=[float(x) for x in v["null_band"]], p=v["p"], n_perm=v["n_perm"],
                               cv=v["cv"], n_sites_ref=v["n_sites_ref"], n_sites_batch=v["n_sites_batch"],
                               selected=list(v["coef"][v["coef"].selected].feature), runtime_s=v["runtime_s"])
                       for k, v in c2.items()}

    # --- embeddings -----------------------------------------------------------------------------------------
    t0 = time.time()
    embs = {}
    for b in [reference, *compare]:
        embs[b] = patch_embeddings(os.path.join(dataset_dir, b), res=res, cache_dir=out_dir)
    emb_time = time.time() - t0
    all_emb = pd.concat(embs.values(), ignore_index=True)
    model_used = embs[reference].attrs["model_used"]
    summary["embeddings"] = dict(model_used=model_used, res=res, n_patches=int(len(all_emb)),
                                 n_sites=int(all_emb.site.nunique()), dim=int(emb_matrix(all_emb).shape[1]),
                                 runtime_s=round(emb_time, 1), normalisation=embs[reference].attrs["normalisation"],
                                 site_meta={b: e.attrs["site_meta"] for b, e in embs.items()})
    print(f"embeddings: {model_used} @ {res}px, {len(all_emb)} patches / {all_emb.site.nunique()} sites in {emb_time:.0f} s")

    # --- novelty --------------------------------------------------------------------------------------------
    R = embs[reference]
    Qd = pd.concat([embs[b] for b in compare], ignore_index=True)
    nv = novelty(emb_matrix(R), emb_matrix(Qd), R.site, Qd.site, patch_ids_ref=R.patch_id, patch_ids_query=Qd.patch_id)
    per_site = nv["per_site"].merge(Qd[["site", "batch"]].drop_duplicates(), on="site").merge(nv["query_percentile"], on="site")
    ref_loo = nv["ref_loo"].assign(batch=reference, pct_median=np.nan, pct_p90=np.nan, exceeds_ref_loo_max=False)
    all_sites = pd.concat([per_site, ref_loo], ignore_index=True)
    all_sites["role"] = np.where(all_sites.batch == reference, "reference (leave-one-site-out)", "query")
    all_sites.to_csv(os.path.join(out_dir, "novelty_per_site.csv"), index=False)
    pd.concat([nv["per_patch"].merge(Qd[["site", "patch_id", "y0", "x0", "batch"]], on=["site", "patch_id"]),
               nv["ref_loo_per_patch"].merge(R[["site", "patch_id", "y0", "x0", "batch"]], on=["site", "patch_id"])],
              ignore_index=True).to_csv(os.path.join(out_dir, "novelty_per_patch.csv"), index=False)
    print(all_sites.sort_values(["batch", "novelty_median"]).to_string(index=False))

    nv["nearest_ref"].to_csv(os.path.join(out_dir, "novelty_nearest_ref.csv"), index=False)
    corr = novelty_correlates(all_sites, site_df)
    corr.to_csv(os.path.join(out_dir, "novelty_correlates.csv"), index=False)
    corr_ord = novelty_correlates(all_sites[~all_sites.site.isin(GREY_PORE_SITES + LOW_CONTRAST_SITES + CRACKED_SITES)], site_df)
    corr_ord.to_csv(os.path.join(out_dir, "novelty_correlates_ordinary_sites.csv"), index=False)
    print(corr.head(15).to_string(index=False))
    summary["novelty"] = dict(n_components=nv["n_components"], k=nv["k"], pca_explained_variance=nv["pca_explained_variance"],
                              query_sites=per_site.to_dict(orient="records"), ref_loo=nv["ref_loo"].to_dict(orient="records"),
                              top_correlates=corr.head(12).to_dict(orient="records"),
                              top_correlates_ordinary_sites=corr_ord.head(8).to_dict(orient="records"), note=nv["note"])
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return dict(c2st=c2, embeddings=embs, novelty=nv, correlates=corr, correlates_ordinary=corr_ord, site_df=site_df)


if __name__ == "__main__":
    run_real_data()
