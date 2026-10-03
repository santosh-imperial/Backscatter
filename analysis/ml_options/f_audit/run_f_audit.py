#!/opt/anaconda3/bin/python3
"""Protocol F: balanced frozen-encoder novelty audit (analysis/ml_options/assessment.md).

Standalone analysis. Imports from ``polaron_qc`` (read-only); writes only into ``output/`` next to this file.
Definitions are registered in ``findings.md`` §1 before the first run; this script implements them verbatim.

    python3 run_f_audit.py              # core audit (items 1–6), cached embeddings only, ~1 min CPU
    python3 run_f_audit.py --challenge  # additionally the small acquisition challenge (recomputes embeddings
                                        # for perturbed copies of 5 prespecified sites with the cached DINOv2)

Nothing here is a verdict input. Sites are n; tiles never are. Percentiles within the reference LOO
distribution are evidence flags, not probabilities of defect.
"""
from __future__ import annotations

import os

# OpenBLAS on this Mac spins for ~15 s on a 544x384 SVD when multi-threaded (0.03 s single-threaded); pin threads
# before NumPy is imported. Purely an execution setting; no numerical definition depends on it.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
import time
import zlib

import numpy as np
import pandas as pd
import scipy.stats as sps
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from polaron_qc import acquisition, features, ml  # noqa: E402

# ---- registered settings (findings.md §1) -------------------------------------------------------------------
F_SEED = 20261003
TILES_PER_SITE = 32
N_COMP = 32
K = 5
RESOLUTIONS = (518, 224)            # primary first
REFERENCE = "Batch_3"
COMPARED = ("Batch_1", "Batch_2")
ACQ_COVARIATES = ["bright_sep", "bse_p1", "bse_std", "bse_empty_bin_frac", "etd_boundary_sharpness", "H"]
TOP_TILES_PER_BATCH = 4
CHALLENGE_SITES = {"Batch_1": ["f1vzngrs", "4ih2ggld"], "Batch_2": ["3806gxp0"], "Batch_3": ["9luzk4jm", "71vgq3fw"]}
CHALLENGE_PERTURBATIONS = {"gamma_0.75": ("gamma", 0.75), "gamma_1.25": ("gamma", 1.25), "affine_0.8x+20": ("affine", (0.8, 20.0))}
BATCH_COLOURS = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a"}
GREY_PORE_COLOUR, LOW_CONTRAST_RING = "#eda100", "#e34948"

DATASET = os.path.join(ROOT, "Dataset")
EMB_CACHE = os.path.join(ROOT, "analysis_cache", "ml")
FEAT_CACHE = os.path.join(ROOT, "analysis_cache", "features")
OUT = os.path.join(HERE, "output")
CROPS = os.path.join(OUT, "crops")
FIGS = os.path.join(OUT, "figures")
for d in (OUT, CROPS, FIGS):
    os.makedirs(d, exist_ok=True)


def log(msg: str):
    print(f"[f_audit] {msg}", flush=True)


# ---- inputs ---------------------------------------------------------------------------------------------------
def load_embeddings(res: int) -> pd.DataFrame:
    frames = []
    for b in (REFERENCE, *COMPARED):
        frames.append(ml.patch_embeddings(os.path.join(DATASET, b), res=res, cache_dir=EMB_CACHE, verbose=False))
    df = pd.concat(frames, ignore_index=True)
    df["site"] = df["site"].astype(str)
    return df


def load_site_metadata() -> tuple[pd.DataFrame, dict]:
    """derive_flags on the features sites/images tables (+ frame height H); returns (flags, sites_by_batch)."""
    flags, sites_tabs = [], {}
    for b in (REFERENCE, *COMPARED):
        t = features.extract_batch(os.path.join(DATASET, b), cache_dir=FEAT_CACHE, verbose=False)
        f = acquisition.derive_flags(t["sites"], t["images"])
        f = f.merge(t["sites"][["batch", "site", "H", "W", "th_lo", "th_hi"]], on=["batch", "site"], how="left")
        flags.append(f)
        sites_tabs[b] = t["sites"]
    out = pd.concat(flags, ignore_index=True)
    out["site"] = out["site"].astype(str)
    return out, sites_tabs


def eligibility(emb: pd.DataFrame) -> pd.DataFrame:
    X = ml.emb_matrix(emb)
    fin = pd.Series(np.isfinite(X).all(axis=1), index=emb.index)
    g = emb.assign(_fin=fin).groupby(["batch", "site"], sort=False)
    e = pd.DataFrame({"n_tiles_complete": g.size(), "all_finite": g["_fin"].all()}).reset_index()
    e["eligible"] = e.all_finite & (e.n_tiles_complete >= TILES_PER_SITE)
    return e


def select_tiles(site_rows: pd.DataFrame, site: str) -> np.ndarray:
    """Registered rule: row-major order by (y0, x0); default_rng([F_SEED, crc32(site)]) draws 32 without replacement.
    Returns the selected patch_ids (sorted)."""
    ordered = site_rows.sort_values(["y0", "x0"], kind="stable")
    rng = np.random.default_rng([F_SEED, zlib.crc32(site.encode())])
    idx = np.sort(rng.choice(len(ordered), TILES_PER_SITE, replace=False))
    return ordered.patch_id.to_numpy()[idx]


# ---- memory / scoring -----------------------------------------------------------------------------------------
class Memory:
    """PCA (min(32, rank) components) + k-NN index on the rows handed in. Rows carry (site, patch_id, y0, x0)."""

    def __init__(self, rows: pd.DataFrame):
        X = ml.emb_matrix(rows).astype(np.float64)
        rank = int(np.linalg.matrix_rank(X - X.mean(axis=0)))
        self.n_components = int(min(N_COMP, rank))
        self.pca = PCA(n_components=self.n_components, random_state=0).fit(X)
        self.Z = self.pca.transform(X)
        self.nn = NearestNeighbors(n_neighbors=min(K, len(self.Z))).fit(self.Z)
        self.rows = rows[["site", "patch_id", "y0", "x0"]].reset_index(drop=True)
        self.explained = float(self.pca.explained_variance_ratio_.sum())
        # kept separately so the balance/holdout verification counts the PCA-fit matrix and the memory independently
        self.pca_fit_sites = rows.site.to_numpy().copy()
        self.memory_sites = self.rows.site.to_numpy()[: self.nn.n_samples_fit_].copy()
        assert self.pca.n_samples_ == len(self.pca_fit_sites) == self.nn.n_samples_fit_

    def score(self, q: pd.DataFrame):
        d, i = self.nn.kneighbors(self.pca.transform(ml.emb_matrix(q).astype(np.float64)))
        return d.mean(axis=1), d, i


def summarise(nov: np.ndarray) -> tuple[float, float]:
    return float(np.median(nov)), float(np.percentile(nov, 90))


def run_method(E_ref: pd.DataFrame, E_q: pd.DataFrame, balanced: bool) -> dict:
    """One view × resolution × method. Returns per-site table, per-tile tables, nearest-neighbour rows for query
    tiles (full memory), matched-count sensitivity rows and memory composition."""
    ref_sites = list(dict.fromkeys(E_ref.site))
    q_sites = list(dict.fromkeys(E_q.site))
    if balanced:
        sel = {s: select_tiles(E_ref[E_ref.site == s], s) for s in ref_sites}
        fit_rows = pd.concat([E_ref[(E_ref.site == s) & E_ref.patch_id.isin(sel[s])] for s in ref_sites], ignore_index=True)
    else:
        fit_rows = E_ref.reset_index(drop=True)
    mem = Memory(fit_rows)
    nov_q, d_q, i_q = mem.score(E_q)
    per_tile_q = E_q[["batch", "site", "patch_id", "y0", "x0"]].copy()
    per_tile_q["novelty"] = nov_q
    per_tile_q["role"] = "query"
    nn_rows = pd.DataFrame({
        "site": np.repeat(E_q.site.to_numpy(), i_q.shape[1]), "patch_id": np.repeat(E_q.patch_id.to_numpy(), i_q.shape[1]),
        "rank": np.tile(np.arange(i_q.shape[1]), len(E_q)), "ref_site": mem.rows.site.to_numpy()[i_q.ravel()],
        "ref_patch_id": mem.rows.patch_id.to_numpy()[i_q.ravel()], "ref_y0": mem.rows.y0.to_numpy()[i_q.ravel()],
        "ref_x0": mem.rows.x0.to_numpy()[i_q.ravel()], "distance": d_q.ravel()})

    # balance / holdout verification rows (full memory first)
    verif = []
    pc, mc = pd.Series(mem.pca_fit_sites).value_counts(), pd.Series(mem.memory_sites).value_counts()
    for s in ref_sites:
        verif.append(dict(fold="full_reference", held_out_site="", site=s, n_tiles_complete=int((E_ref.site == s).sum()),
                          n_tiles_in_pca_fit=int(pc.get(s, 0)), n_tiles_in_memory=int(mc.get(s, 0)), n_tiles_scored=0))
    loo_tiles, loo_rows, matched = [], [], []
    loo_mems = {}
    for s in ref_sites:
        rows_s = fit_rows[fit_rows.site != s]
        mem_s = Memory(rows_s)
        loo_mems[s] = mem_s
        held = E_ref[E_ref.site == s]
        pc_s, mc_s = pd.Series(mem_s.pca_fit_sites).value_counts(), pd.Series(mem_s.memory_sites).value_counts()
        assert pc_s.get(s, 0) == 0 and mc_s.get(s, 0) == 0, f"held-out site {s} leaked into PCA fit or memory"
        for s2 in ref_sites:
            verif.append(dict(fold="loo", held_out_site=s, site=s2, n_tiles_complete=int((E_ref.site == s2).sum()),
                              n_tiles_in_pca_fit=int(pc_s.get(s2, 0)), n_tiles_in_memory=int(mc_s.get(s2, 0)),
                              n_tiles_scored=int(len(held)) if s2 == s else 0))
        nov_s, _, _ = mem_s.score(held)
        t = held[["batch", "site", "patch_id", "y0", "x0"]].copy()
        t["novelty"] = nov_s
        t["role"] = "reference_loo"
        loo_tiles.append(t)
        med, p90 = summarise(nov_s)
        loo_rows.append(dict(site=s, novelty_median=med, novelty_p90=p90, n_components=mem_s.n_components))
        # matched-count sensitivity: queries against this n_ref-1 memory
        nov_qs, _, _ = mem_s.score(E_q)
        for qs in q_sites:
            m = (E_q.site == qs).to_numpy()
            med_q, p90_q = summarise(nov_qs[m])
            matched.append(dict(query_site=qs, held_out_ref=s, novelty_median=med_q, novelty_p90=p90_q))
    ref_loo = pd.DataFrame(loo_rows)
    per_tile_ref = pd.concat(loo_tiles, ignore_index=True)

    rows = []
    for qs in q_sites:
        m = (E_q.site == qs).to_numpy()
        med, p90 = summarise(nov_q[m])
        rows.append(dict(site=qs, role="query", novelty_median=med, novelty_p90=p90,
                         pct_median=sps.percentileofscore(ref_loo.novelty_median, med, kind="weak"),
                         pct_p90=sps.percentileofscore(ref_loo.novelty_p90, p90, kind="weak"),
                         exceeds_ref_loo_max=bool(med > ref_loo.novelty_median.max()), n_tiles_in_fit=0))
    for r in ref_loo.itertuples(index=False):
        rows.append(dict(site=r.site, role="reference_loo", novelty_median=r.novelty_median, novelty_p90=r.novelty_p90,
                         pct_median=np.nan, pct_p90=np.nan, exceeds_ref_loo_max=False,
                         n_tiles_in_fit=int((fit_rows.site == r.site).sum())))
    per_site = pd.DataFrame(rows)
    # matched-count sensitivity summary: across-fold median of the site summary; percentile of that within ref LOO
    M = pd.DataFrame(matched)
    ms = M.groupby("query_site").agg(mc_median=("novelty_median", "median"), mc_p90=("novelty_p90", "median"),
                                     mc_median_min=("novelty_median", "min"), mc_median_max=("novelty_median", "max")).reset_index()
    ms["mc_pct_median"] = [sps.percentileofscore(ref_loo.novelty_median, v, kind="weak") for v in ms.mc_median]
    ms["mc_pct_p90"] = [sps.percentileofscore(ref_loo.novelty_p90, v, kind="weak") for v in ms.mc_p90]
    per_site = per_site.merge(ms.rename(columns={"query_site": "site"}), on="site", how="left")
    comp = fit_rows.groupby("site").size().rename("n_tiles_in_fit").reset_index()
    comp["share_of_memory"] = comp.n_tiles_in_fit / comp.n_tiles_in_fit.sum()
    return dict(per_site=per_site, per_tile=pd.concat([per_tile_q, per_tile_ref], ignore_index=True), nearest=nn_rows,
                matched=M, memory_composition=comp, n_components=mem.n_components, pca_explained=mem.explained,
                n_memory_rows=len(fit_rows), memory=mem, loo_memories=loo_mems, verification=pd.DataFrame(verif))


# ---- acquisition-only control -------------------------------------------------------------------------------
def loo_ridge_r2(df: pd.DataFrame, y_col: str, covs=ACQ_COVARIATES) -> dict:
    d = df.dropna(subset=[y_col]).reset_index(drop=True)
    X = d[covs].to_numpy(float)
    y = d[y_col].to_numpy(float)
    oof = np.full(len(y), np.nan)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        pipe.fit(X[tr], y[tr])
        oof[i] = pipe.predict(X[[i]])[0]
    r2 = 1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2)
    # in-sample fit on all rows, for the coefficient signs only (reporting, not evaluation)
    full = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))]).fit(X, y)
    coefs = dict(zip([f"coef_{c}" for c in covs], full["ridge"].coef_))
    return dict(n_sites=int(len(y)), heldout_r2=float(r2), **coefs)


def within_batch_correlations(df: pd.DataFrame, y_col: str, covs=ACQ_COVARIATES) -> list[dict]:
    out = []
    groups = [("pooled", df)] + [(b, df[df.batch == b]) for b in (REFERENCE, *COMPARED)] + [("ordinary_only", df[df.acquisition_group == "ordinary"])]
    for name, d in groups:
        for c in covs:
            v = pd.to_numeric(d[c], errors="coerce")
            m = v.notna() & d[y_col].notna()
            if m.sum() < 4 or v[m].nunique() < 2:
                out.append(dict(subset=name, covariate=c, n=int(m.sum()), rho=np.nan, p=np.nan))
                continue
            rho, p = sps.spearmanr(d.loc[m, y_col], v[m])
            out.append(dict(subset=name, covariate=c, n=int(m.sum()), rho=float(rho), p=float(p)))
    return out


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


# ---- crops ----------------------------------------------------------------------------------------------------
_IMG_CACHE: dict = {}


def raw_image(batch: str, site: str, det: str) -> np.ndarray:
    key = (batch, site, det)
    if key not in _IMG_CACHE:
        _IMG_CACHE[key] = ml.load_channel0(ml._find_image(os.path.join(DATASET, batch), site, det))
    return _IMG_CACHE[key]


def crop(batch: str, site: str, det: str, y0: int, x0: int, size: int = 512) -> np.ndarray:
    return raw_image(batch, site, det)[y0:y0 + size, x0:x0 + size]


def flag_caption(flags: pd.DataFrame, site: str) -> str:
    r = flags[flags.site == site].iloc[0]
    bits = [r.acquisition_group]
    if r.contrast_stretched_bse:
        bits.append("stretched")
    if r.bright_low_contrast:
        bits.append("low-contrast")
    if r.grey_pore:
        bits.append("grey-pore")
    if r.cracked_known:
        bits.append("cracked(known)")
    return f"{r.batch} {site}\n{'/'.join(bits)}; bse_p1 {r.bse_p1:.0f}, sep {r.bright_sep:.0f}, H {int(r.H)}"


def save_panel(path: str, q: dict, nns: list[dict], flags: pd.DataFrame, site_batch: dict, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [q] + nns
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 7.2))
    for j, c in enumerate(cols):
        b = site_batch[c["site"]]
        for i, det in enumerate(("BSE", "ETD")):
            ax = axes[i, j]
            ax.imshow(crop(b, c["site"], det, int(c["y0"]), int(c["x0"])), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                head = ("QUERY tile" if j == 0 else f"nearest #{j}") + f"  y0={int(c['y0'])} x0={int(c['x0'])}"
                head += f"\nnovelty {c['novelty']:.2f}" if j == 0 else f"\ndistance {c['distance']:.2f}"
                ax.set_title(head + "\n" + flag_caption(flags, c["site"]), fontsize=8)
            if j == 0:
                ax.set_ylabel(det + " (raw, 512 px)", fontsize=9)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---- figures ---------------------------------------------------------------------------------------------------
def strip_plot(site_tab: pd.DataFrame, flags: pd.DataFrame, path: str, res: int, view: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = site_tab[(site_tab.res == res) & (site_tab.view == view)].merge(
        flags[["site", "acquisition_group"]], on="site", how="left", suffixes=("", "_f"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, method in zip(axes, ("balanced", "unbalanced")):
        dd = d[d.method == method]
        rng = np.random.default_rng(1)
        for k, b in enumerate((*COMPARED, REFERENCE)):
            s = dd[dd.batch == b]
            x = k + rng.uniform(-0.18, 0.18, len(s))
            col = [GREY_PORE_COLOUR if g == "grey_pore" else BATCH_COLOURS[b] for g in s.acquisition_group]
            ax.scatter(x, s.novelty_median, c=col, s=34, zorder=3, edgecolors="none")
            lc = s.acquisition_group == "low_contrast"
            if lc.any():
                ax.scatter(x[lc.to_numpy()], s.novelty_median[lc], s=90, facecolors="none", edgecolors=LOW_CONTRAST_RING, linewidths=1.5, zorder=4)
            ax.hlines(np.median(s.novelty_median), k - 0.3, k + 0.3, color="k", lw=1.5, zorder=5)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{b}\n(query)" if b != REFERENCE else f"{b}\n(reference LOO)" for b in (*COMPARED, REFERENCE)])
        ax.set_title(f"{method} memory, {res} px, view: {view}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("site median tile novelty (mean distance to 5 NN, PCA space)")
    handles = [Line2D([], [], marker="o", ls="", color=BATCH_COLOURS[b], label=b) for b in (*COMPARED, REFERENCE)]
    handles += [Line2D([], [], marker="o", ls="", color=GREY_PORE_COLOUR, label="Batch 3 grey-pore group"),
                Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=LOW_CONTRAST_RING, label="low-contrast site"),
                Line2D([], [], color="k", lw=1.5, label="bar = batch median")]
    axes[1].legend(handles=handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def scatter_plot(a: pd.DataFrame, b: pd.DataFrame, labels: tuple[str, str], path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = a.merge(b, on=["site", "batch"], suffixes=("_a", "_b"))
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for bt in (*COMPARED, REFERENCE):
        s = d[d.batch == bt]
        ax.scatter(s.novelty_median_a, s.novelty_median_b, color=BATCH_COLOURS[bt], s=30, label=bt)
        for r in s.itertuples():
            ax.annotate(r.site, (r.novelty_median_a, r.novelty_median_b), fontsize=6, xytext=(2, 2), textcoords="offset points")
    rho = sps.spearmanr(d.novelty_median_a, d.novelty_median_b)[0]
    ax.set_xlabel(f"site median novelty, {labels[0]}"); ax.set_ylabel(f"site median novelty, {labels[1]}")
    ax.set_title(f"{title}\nSpearman rho = {rho:.2f} over {len(d)} sites", fontsize=9)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BATCH_COLOURS[bt], label=bt) for bt in (*COMPARED, REFERENCE)], fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def coverage_plot(cov: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = cov[cov.res == 518].sort_values("n_tiles_complete", ascending=False)
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x - 0.2, d.n_tiles_complete, width=0.4, label="unbalanced (all complete tiles)", color="#999999")
    axes[0].bar(x + 0.2, d.n_tiles_balanced, width=0.4, label="balanced (32 per site)", color=BATCH_COLOURS[REFERENCE])
    axes[0].set_ylabel("tiles contributed to PCA + memory"); axes[0].set_title("Reference-site rows in the memory, 518 px", fontsize=10)
    axes[1].bar(x - 0.2, 100 * d.nn_hit_share_unbalanced, width=0.4, label="unbalanced", color="#999999")
    axes[1].bar(x + 0.2, 100 * d.nn_hit_share_balanced, width=0.4, label="balanced", color=BATCH_COLOURS[REFERENCE])
    axes[1].set_ylabel("% of query 5-NN hits landing on the site"); axes[1].set_title("Where query tiles find their neighbours", fontsize=10)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(d.site, rotation=70, fontsize=7); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---- acquisition challenge ----------------------------------------------------------------------------------
def perturb(img: np.ndarray, kind: str, param) -> np.ndarray:
    x = img.astype(np.float64)
    if kind == "gamma":
        y = 255.0 * (x / 255.0) ** param
    elif kind == "affine":
        a, b = param
        y = a * x + b
    else:
        raise ValueError(kind)
    assert y.max() <= 255.0 + 1e-9 and y.min() >= 0, "perturbation must not clip"
    return np.rint(y).astype(np.uint8)


def embed_image(model, img: np.ndarray, res: int) -> pd.DataFrame:
    """Same pipeline as ml.patch_embeddings for one in-memory image (bands → normalise → 512 tiles → CLS)."""
    top, bot = ml.bright_bands(img)
    body = img[top: img.shape[0] - bot if bot else None]
    x, meta = ml.normalise_image(body, 3.0)
    coords = ml.tile_coords(*x.shape, 512, 512)
    P = np.stack([x[a:a + 512, b:b + 512] for _, a, b in coords])
    emb = ml.embed_patches(model, P, res)
    df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    df.insert(0, "x0", [c[2] for c in coords]); df.insert(0, "y0", [c[1] + top for c in coords]); df.insert(0, "patch_id", [c[0] for c in coords])
    df.attrs.update(meta, band_top=top, band_bottom=bot)
    return df


def run_challenge(E: dict, runs: dict, flags: pd.DataFrame) -> pd.DataFrame:
    model, used, _ = ml.load_embedding_model("dinov2_vits14")
    log(f"challenge model: {used}")
    rows = []
    for res in RESOLUTIONS:
        r = runs[("all_sites", res, "balanced")]
        ref_loo_medians = r["per_site"][r["per_site"].role == "reference_loo"].novelty_median
        for batch, sites in CHALLENGE_SITES.items():
            for site in sites:
                mem = r["loo_memories"][site] if batch == REFERENCE else r["memory"]
                orig_rows = E[res][(E[res].batch == batch) & (E[res].site == site)]
                nov0, d0, i0 = mem.score(orig_rows)
                key0 = {(int(y), int(x)): n for y, x, n in zip(orig_rows.y0, orig_rows.x0, range(len(orig_rows)))}
                nn0 = {k: set(map(tuple, mem.rows.iloc[i0[n]][["site", "patch_id"]].to_numpy())) for k, n in key0.items()}
                top0 = {k: tuple(mem.rows.iloc[i0[n][0]][["site", "patch_id"]]) for k, n in key0.items()}
                med0, p900 = summarise(nov0)
                pct0 = sps.percentileofscore(ref_loo_medians, med0, kind="weak")
                raw = raw_image(batch, site, "BSE")
                for pname, (kind, param) in CHALLENGE_PERTURBATIONS.items():
                    t0 = time.time()
                    pert = embed_image(model, perturb(raw, kind, param), res)
                    nov1, d1, i1 = mem.score(pert)
                    med1, p901 = summarise(nov1)
                    pct1 = sps.percentileofscore(ref_loo_medians, med1, kind="weak")
                    same_top, jac, matched = [], [], 0
                    for n, (y, x) in enumerate(zip(pert.y0, pert.x0)):
                        k = (int(y), int(x))
                        if k not in key0:
                            continue
                        matched += 1
                        s1 = set(map(tuple, mem.rows.iloc[i1[n]][["site", "patch_id"]].to_numpy()))
                        jac.append(len(s1 & nn0[k]) / len(s1 | nn0[k]))
                        same_top.append(tuple(mem.rows.iloc[i1[n][0]][["site", "patch_id"]]) == top0[k])
                    rows.append(dict(res=res, batch=batch, site=site, acquisition_group=flags.loc[flags.site == site, "acquisition_group"].iloc[0],
                                     perturbation=pname, memory="balanced LOO (site excluded)" if batch == REFERENCE else "balanced all-sites",
                                     n_tiles_orig=len(orig_rows), n_tiles_pert=len(pert), n_tiles_matched=matched,
                                     band_top_pert=pert.attrs["band_top"], band_bottom_pert=pert.attrs["band_bottom"],
                                     norm_median_pert=pert.attrs["median"], norm_iqr_pert=pert.attrs["iqr"], saturated_frac_pert=pert.attrs["saturated_frac"],
                                     median_orig=med0, median_pert=med1, median_delta=med1 - med0,
                                     p90_orig=p900, p90_pert=p901, p90_delta=p901 - p900,
                                     pct_median_orig=pct0, pct_median_pert=pct1,
                                     frac_top1_unchanged=float(np.mean(same_top)) if same_top else np.nan,
                                     mean_jaccard_5nn=float(np.mean(jac)) if jac else np.nan, seconds=round(time.time() - t0, 1)))
                    log(f"challenge {res} {batch}/{site} {pname}: median {med0:.2f} -> {med1:.2f}, top1 same {rows[-1]['frac_top1_unchanged']:.2f}")
    return pd.DataFrame(rows)


# ---- main -------------------------------------------------------------------------------------------------------
def main(challenge: bool = False):
    t_start = time.time()
    flags, _ = load_site_metadata()
    flags.to_csv(os.path.join(OUT, "site_acquisition_flags.csv"), index=False)
    site_batch = dict(zip(flags.site, flags.batch))
    E = {res: load_embeddings(res) for res in RESOLUTIONS}
    elig = eligibility(E[RESOLUTIONS[0]])
    for res in RESOLUTIONS[1:]:
        e2 = eligibility(E[res])
        assert (e2.n_tiles_complete.to_numpy() == elig.n_tiles_complete.to_numpy()).all(), "tile grids differ between resolutions"
        elig["all_finite"] &= e2.all_finite.to_numpy()
        elig["eligible"] &= e2.eligible.to_numpy()
    elig.to_csv(os.path.join(OUT, "eligibility.csv"), index=False)
    log(f"eligible sites: {int(elig.eligible.sum())}/{len(elig)}")

    ordinary = set(flags.loc[flags.acquisition_group == "ordinary", "site"])
    ref_all = [s for s in flags.loc[flags.batch == REFERENCE, "site"]]
    q_all = [s for s in flags.loc[flags.batch.isin(COMPARED), "site"]]
    views = {"all_sites": (ref_all, q_all),
             "quality_matched": ([s for s in ref_all if s in ordinary], [s for s in q_all if s in ordinary]),
             "ordinary_reference": ([s for s in ref_all if s in ordinary], q_all)}
    elig_sites = set(elig.loc[elig.eligible, "site"])

    runs, site_rows, matched_rows, comp_rows, verif_rows = {}, [], [], [], []
    for view, (rs, qs) in views.items():
        for res in RESOLUTIONS:
            Ed = E[res]
            rs_e = [s for s in rs if s in elig_sites]
            E_ref = Ed[Ed.site.isin(rs_e) & (Ed.batch == REFERENCE)].reset_index(drop=True)
            E_q = Ed[Ed.site.isin(qs) & Ed.batch.isin(COMPARED)].reset_index(drop=True)
            for method in ("balanced", "unbalanced"):
                t0 = time.time()
                r = run_method(E_ref, E_q, balanced=(method == "balanced"))
                runs[(view, res, method)] = r
                ps = r["per_site"].copy()
                ps.insert(0, "method", method); ps.insert(0, "res", res); ps.insert(0, "view", view)
                site_rows.append(ps)
                mm = r["matched"].copy(); mm.insert(0, "method", method); mm.insert(0, "res", res); mm.insert(0, "view", view)
                matched_rows.append(mm)
                cc = r["memory_composition"].copy(); cc.insert(0, "method", method); cc.insert(0, "res", res); cc.insert(0, "view", view)
                comp_rows.append(cc)
                vv = r["verification"].copy(); vv.insert(0, "method", method); vv.insert(0, "res", res); vv.insert(0, "view", view)
                verif_rows.append(vv)
                log(f"{view} {res} {method}: memory rows {r['n_memory_rows']}, PCA {r['n_components']} comps "
                    f"({r['pca_explained']:.2f} var), {time.time() - t0:.1f} s")
            # ineligible reference sites of this view are listed as abstained
            for s in rs:
                if s not in elig_sites:
                    for method in ("balanced", "unbalanced"):
                        site_rows.append(pd.DataFrame([dict(view=view, res=res, method=method, site=s, role="reference_not_eligible")]))

    site_tab = pd.concat(site_rows, ignore_index=True)
    site_tab = site_tab.merge(flags[["site", "batch", "acquisition_group"]], on="site", how="left")
    site_tab = site_tab.merge(elig[["site", "n_tiles_complete", "eligible"]], on="site", how="left")
    front = ["view", "res", "method", "site", "batch", "acquisition_group", "n_tiles_complete", "eligible", "role", "n_tiles_in_fit",
             "novelty_median", "novelty_p90", "pct_median", "pct_p90", "exceeds_ref_loo_max",
             "mc_median", "mc_p90", "mc_median_min", "mc_median_max", "mc_pct_median", "mc_pct_p90"]
    site_tab = site_tab[front]
    site_tab.to_csv(os.path.join(OUT, "site_table.csv"), index=False)
    pd.concat(matched_rows, ignore_index=True).to_csv(os.path.join(OUT, "matched_count_sensitivity_folds.csv"), index=False)
    pd.concat(comp_rows, ignore_index=True).to_csv(os.path.join(OUT, "memory_composition.csv"), index=False)
    # balance and whole-site holdout verification: full table + one summary row per (view, res, method)
    V = pd.concat(verif_rows, ignore_index=True)
    V.to_csv(os.path.join(OUT, "balance_holdout_verification.csv"), index=False)
    own = V[(V.fold == "loo") & (V.site == V.held_out_site)]
    others = V[(V.fold == "loo") & (V.site != V.held_out_site)]
    full = V[V.fold == "full_reference"]
    vs = []
    for key, g in V.groupby(["view", "res", "method"]):
        o, t, f = own.loc[g.index.intersection(own.index)], others.loc[g.index.intersection(others.index)], full.loc[g.index.intersection(full.index)]
        vs.append(dict(view=key[0], res=key[1], method=key[2], n_ref_sites=int(f.site.nunique()), n_loo_folds=int(o.held_out_site.nunique()),
                       max_own_tiles_in_pca_fit_when_held_out=int(o.n_tiles_in_pca_fit.max()), max_own_tiles_in_memory_when_held_out=int(o.n_tiles_in_memory.max()),
                       held_out_tiles_scored_total=int(o.n_tiles_scored.sum()),
                       min_tiles_per_site_in_pca_fit_full=int(f.n_tiles_in_pca_fit.min()), max_tiles_per_site_in_pca_fit_full=int(f.n_tiles_in_pca_fit.max()),
                       min_tiles_per_site_in_memory_full=int(f.n_tiles_in_memory.min()), max_tiles_per_site_in_memory_full=int(f.n_tiles_in_memory.max()),
                       min_tiles_per_remaining_site_in_loo_fits=int(t.n_tiles_in_pca_fit.min()), max_tiles_per_remaining_site_in_loo_fits=int(t.n_tiles_in_pca_fit.max()),
                       pca_fit_rows_equal_memory_rows=bool((g.n_tiles_in_pca_fit == g.n_tiles_in_memory).all()),
                       n_ref_sites_not_eligible_abstained=int(len(set(views[key[0]][0]) - elig_sites))))
    pd.DataFrame(vs).to_csv(os.path.join(OUT, "balance_holdout_verification_summary.csv"), index=False)
    for res in RESOLUTIONS:
        for method in ("balanced", "unbalanced"):
            runs[("all_sites", res, method)]["per_tile"].to_csv(os.path.join(OUT, f"per_tile_{method}_all_sites_r{res}.csv"), index=False)
            runs[("all_sites", res, method)]["nearest"].to_csv(os.path.join(OUT, f"nearest_ref_{method}_all_sites_r{res}.csv"), index=False)

    # ---- check: unbalanced reproduces polaron_qc.ml.novelty and the committed E07 table at 518 -------------------
    checks = {}
    Ed = E[518]
    R = Ed[Ed.batch == REFERENCE].reset_index(drop=True); Q = Ed[Ed.batch.isin(COMPARED)].reset_index(drop=True)
    nv = ml.novelty(ml.emb_matrix(R), ml.emb_matrix(Q), R.site, Q.site, patch_ids_ref=R.patch_id, patch_ids_query=Q.patch_id)
    mine = runs[("all_sites", 518, "unbalanced")]["per_site"]
    a = nv["per_site"].merge(mine[mine.role == "query"], on="site", suffixes=("_ml", "_f"))
    b = nv["ref_loo"].merge(mine[mine.role == "reference_loo"], on="site", suffixes=("_ml", "_f"))
    checks["max_abs_diff_vs_ml_novelty_query_median"] = float((a.novelty_median_ml - a.novelty_median_f).abs().max())
    checks["max_abs_diff_vs_ml_novelty_ref_loo_median"] = float((b.novelty_median_ml - b.novelty_median_f).abs().max())
    e07 = os.path.join(EMB_CACHE, "novelty_per_site.csv")
    if os.path.exists(e07):
        old = pd.read_csv(e07)[["site", "novelty_median", "novelty_p90"]]
        c = old.merge(mine, on="site", suffixes=("_e07", "_f"))
        checks["max_abs_diff_vs_committed_E07_median"] = float((c.novelty_median_e07 - c.novelty_median_f).abs().max())
        checks["n_sites_compared_with_E07"] = int(len(c))

    # ---- agreement statistics -------------------------------------------------------------------------------------
    def medians(view, res, method):
        d = site_tab[(site_tab.view == view) & (site_tab.res == res) & (site_tab.method == method) & site_tab.role.isin(["query", "reference_loo"])]
        return d[["site", "batch", "role", "novelty_median", "novelty_p90", "exceeds_ref_loo_max", "pct_median"]]

    agree = []
    for view in views:
        pairs = [("resolution", (view, 518, "balanced"), (view, 224, "balanced")), ("resolution", (view, 518, "unbalanced"), (view, 224, "unbalanced")),
                 ("method", (view, 518, "balanced"), (view, 518, "unbalanced")), ("method", (view, 224, "balanced"), (view, 224, "unbalanced"))]
        for kind, ka, kb in pairs:
            A, B = medians(*ka), medians(*kb)
            d = A.merge(B, on=["site", "batch", "role"], suffixes=("_a", "_b"))
            q = d[d.role == "query"]
            agree.append(dict(view=view, comparison=kind, a=f"{ka[1]}_{ka[2]}", b=f"{kb[1]}_{kb[2]}", n_sites=len(d),
                              spearman_median_all_sites=float(sps.spearmanr(d.novelty_median_a, d.novelty_median_b)[0]),
                              spearman_p90_all_sites=float(sps.spearmanr(d.novelty_p90_a, d.novelty_p90_b)[0]),
                              n_query_sites=len(q),
                              spearman_median_query_sites=float(sps.spearmanr(q.novelty_median_a, q.novelty_median_b)[0]) if len(q) > 2 else np.nan,
                              spearman_pct_median_query_sites=float(sps.spearmanr(q.pct_median_a, q.pct_median_b)[0]) if len(q) > 2 else np.nan,
                              n_query_exceeds_flag_flips=int((q.exceeds_ref_loo_max_a != q.exceeds_ref_loo_max_b).sum()),
                              n_query_exceeds_a=int(q.exceeds_ref_loo_max_a.sum()), n_query_exceeds_b=int(q.exceeds_ref_loo_max_b.sum())))
    agree = pd.DataFrame(agree)
    agree.to_csv(os.path.join(OUT, "agreement.csv"), index=False)

    # ---- full-reference vs LOO-reference (matched-count) sensitivity summary -------------------------------------
    loo_sens = site_tab[(site_tab.role == "query")].copy()
    loo_sens["median_shift_mc_minus_full"] = loo_sens.mc_median - loo_sens.novelty_median
    loo_sens["pct_shift_mc_minus_full"] = loo_sens.mc_pct_median - loo_sens.pct_median
    loo_sens = loo_sens[["view", "res", "method", "site", "batch", "acquisition_group", "novelty_median", "mc_median", "mc_median_min", "mc_median_max",
                         "median_shift_mc_minus_full", "pct_median", "mc_pct_median", "pct_shift_mc_minus_full", "exceeds_ref_loo_max"]]
    loo_sens.to_csv(os.path.join(OUT, "loo_vs_full_reference_sensitivity.csv"), index=False)
    ls_sum = loo_sens.groupby(["view", "res", "method"]).agg(
        n_query=("site", "size"), mean_median_shift=("median_shift_mc_minus_full", "mean"), max_abs_median_shift=("median_shift_mc_minus_full", lambda s: s.abs().max()),
        mean_pct_shift=("pct_shift_mc_minus_full", "mean"), max_abs_pct_shift=("pct_shift_mc_minus_full", lambda s: s.abs().max()),
        n_exceeds_full=("exceeds_ref_loo_max", "sum")).reset_index()
    ls_sum.to_csv(os.path.join(OUT, "loo_vs_full_reference_summary.csv"), index=False)

    # ---- acquisition-only control (all-sites view) --------------------------------------------------------------
    ctrl, wb = [], []
    for res in RESOLUTIONS:
        for method in ("balanced", "unbalanced"):
            d = medians("all_sites", res, method).merge(flags[["site", "acquisition_group"] + ACQ_COVARIATES], on="site", how="left")
            for stat in ("novelty_median", "novelty_p90"):
                for subset_name, dd in (("all_31", d), ("ordinary_only", d[d.acquisition_group == "ordinary"])):
                    r = loo_ridge_r2(dd, stat)
                    ctrl.append(dict(res=res, method=method, summary=stat, subset=subset_name, **r))
                for row in within_batch_correlations(d, stat):
                    wb.append(dict(res=res, method=method, summary=stat, **row))
    ctrl = pd.DataFrame(ctrl); ctrl.to_csv(os.path.join(OUT, "acquisition_control_r2.csv"), index=False)
    wb = pd.DataFrame(wb); wb.to_csv(os.path.join(OUT, "acquisition_control_correlations.csv"), index=False)
    # also the existing KPI/acquisition correlate table of the primary balanced summary (ml.novelty_correlates)
    site_df = ml.assemble_site_table(os.path.join(ROOT, "analysis_cache"))
    prim = medians("all_sites", 518, "balanced")
    ml.novelty_correlates(prim, site_df).to_csv(os.path.join(OUT, "novelty_correlates_balanced_r518.csv"), index=False)
    ml.novelty_correlates(medians("all_sites", 518, "unbalanced"), site_df).to_csv(os.path.join(OUT, "novelty_correlates_unbalanced_r518.csv"), index=False)

    # ---- coverage audit -----------------------------------------------------------------------------------------
    cov_rows = []
    for res in RESOLUTIONS:
        rb, ru = runs[("all_sites", res, "balanced")], runs[("all_sites", res, "unbalanced")]
        hb = rb["nearest"].ref_site.value_counts(normalize=True); hu = ru["nearest"].ref_site.value_counts(normalize=True)
        for s in ref_all:
            cov_rows.append(dict(res=res, site=s, acquisition_group=flags.loc[flags.site == s, "acquisition_group"].iloc[0],
                                 n_tiles_complete=int(elig.loc[elig.site == s, "n_tiles_complete"].iloc[0]),
                                 n_tiles_balanced=int((rb["memory_composition"].site == s).sum() and rb["memory_composition"].set_index("site").n_tiles_in_fit.get(s, 0)),
                                 share_memory_unbalanced=float(ru["memory_composition"].set_index("site").share_of_memory.get(s, 0)),
                                 share_memory_balanced=float(rb["memory_composition"].set_index("site").share_of_memory.get(s, 0)),
                                 nn_hit_share_unbalanced=float(hu.get(s, 0.0)), nn_hit_share_balanced=float(hb.get(s, 0.0))))
    cov = pd.DataFrame(cov_rows); cov.to_csv(os.path.join(OUT, "coverage_audit.csv"), index=False)
    cov_sum = cov.groupby("res").agg(n_ref_sites=("site", "size"), memory_rows_unbalanced=("n_tiles_complete", "sum"), memory_rows_balanced=("n_tiles_balanced", "sum"),
                                     max_share_memory_unbalanced=("share_memory_unbalanced", "max"), max_share_memory_balanced=("share_memory_balanced", "max"),
                                     max_nn_hit_share_unbalanced=("nn_hit_share_unbalanced", "max"), max_nn_hit_share_balanced=("nn_hit_share_balanced", "max"),
                                     gini_nn_hits_unbalanced=("nn_hit_share_unbalanced", gini), gini_nn_hits_balanced=("nn_hit_share_balanced", gini)).reset_index()
    cov_sum.to_csv(os.path.join(OUT, "coverage_audit_summary.csv"), index=False)

    # ---- nearest-reference evidence (balanced, all sites, 518) ---------------------------------------------------
    rb = runs[("all_sites", 518, "balanced")]
    pt = rb["per_tile"][rb["per_tile"].role == "query"]
    nn = rb["nearest"]
    top_rows = []
    for b in COMPARED:
        tb = pt[pt.batch == b].sort_values("novelty", ascending=False).head(TOP_TILES_PER_BATCH)
        for rank_in_batch, t in enumerate(tb.itertuples(index=False), 1):
            nns = nn[(nn.site == t.site) & (nn.patch_id == t.patch_id)].sort_values("rank").head(3)
            rec = dict(batch=b, rank_in_batch=rank_in_batch, site=t.site, acquisition_group=flags.loc[flags.site == t.site, "acquisition_group"].iloc[0],
                       patch_id=int(t.patch_id), y0=int(t.y0), x0=int(t.x0), novelty=float(t.novelty),
                       site_pct_median=float(rb["per_site"].set_index("site").pct_median[t.site]))
            for j, n_ in enumerate(nns.itertuples(index=False), 1):
                rec.update({f"nn{j}_site": n_.ref_site, f"nn{j}_group": flags.loc[flags.site == n_.ref_site, "acquisition_group"].iloc[0],
                            f"nn{j}_patch_id": int(n_.ref_patch_id), f"nn{j}_y0": int(n_.ref_y0), f"nn{j}_x0": int(n_.ref_x0), f"nn{j}_distance": float(n_.distance)})
            png = f"top{rank_in_batch}_{b}_{t.site}_p{int(t.patch_id)}_y{int(t.y0)}_x{int(t.x0)}_r518.png"
            rec["png"] = os.path.join("output", "crops", png)
            top_rows.append(rec)
            save_panel(os.path.join(CROPS, png), dict(site=t.site, y0=t.y0, x0=t.x0, novelty=t.novelty),
                       [dict(site=n_.ref_site, y0=n_.ref_y0, x0=n_.ref_x0, distance=n_.distance) for n_ in nns.itertuples(index=False)],
                       flags, site_batch, f"Balanced memory, 518 px, all-sites view. {b} query tile #{rank_in_batch} by novelty and its 3 nearest reference tiles "
                                          f"(reference {REFERENCE}). Raw crops, no normalisation; a tile localises a 512-px field only.")
    pd.DataFrame(top_rows).to_csv(os.path.join(OUT, "nearest_reference_top_tiles_r518.csv"), index=False)
    # top tile per query site (table only), both resolutions
    per_site_top = []
    for res in RESOLUTIONS:
        r = runs[("all_sites", res, "balanced")]
        q = r["per_tile"][r["per_tile"].role == "query"]
        idx = q.groupby("site").novelty.idxmax()
        for t in q.loc[idx].itertuples(index=False):
            nns = r["nearest"][(r["nearest"].site == t.site) & (r["nearest"].patch_id == t.patch_id)].sort_values("rank").head(3)
            per_site_top.append(dict(res=res, batch=t.batch, site=t.site, patch_id=int(t.patch_id), y0=int(t.y0), x0=int(t.x0), novelty=float(t.novelty),
                                     nn_sites=";".join(nns.ref_site), nn_groups=";".join(flags.set_index("site").acquisition_group[nns.ref_site]),
                                     nn_distances=";".join(f"{d:.2f}" for d in nns.distance)))
    pd.DataFrame(per_site_top).to_csv(os.path.join(OUT, "nearest_reference_top_tile_per_site.csv"), index=False)
    # which reference groups the query tiles' neighbours come from, per query site (balanced vs unbalanced, 518)
    grp_rows = []
    for method in ("balanced", "unbalanced"):
        n_ = runs[("all_sites", 518, method)]["nearest"].merge(flags[["site", "acquisition_group"]].rename(columns={"site": "ref_site", "acquisition_group": "ref_group"}), on="ref_site")
        g = n_.groupby("site").ref_group.value_counts(normalize=True).unstack(fill_value=0.0)
        g.insert(0, "method", method)
        grp_rows.append(g.reset_index())
    pd.concat(grp_rows, ignore_index=True).to_csv(os.path.join(OUT, "query_nn_reference_group_shares_r518.csv"), index=False)

    # ---- figures -------------------------------------------------------------------------------------------------
    for view in views:
        strip_plot(site_tab, flags, os.path.join(FIGS, f"strip_{view}_r518.png"), 518, view)
    strip_plot(site_tab, flags, os.path.join(FIGS, "strip_all_sites_r224.png"), 224, "all_sites")
    scatter_plot(medians("all_sites", 518, "balanced"), medians("all_sites", 518, "unbalanced"), ("balanced 518", "unbalanced 518"),
                 os.path.join(FIGS, "scatter_balanced_vs_unbalanced_r518.png"), "Balanced vs unbalanced memory (all-sites view)")
    scatter_plot(medians("all_sites", 518, "balanced"), medians("all_sites", 224, "balanced"), ("balanced 518", "balanced 224"),
                 os.path.join(FIGS, "scatter_r518_vs_r224_balanced.png"), "Resolution sensitivity, balanced memory (all-sites view)")
    coverage_plot(cov, os.path.join(FIGS, "coverage_audit_r518.png"))

    # ---- optional challenge --------------------------------------------------------------------------------------
    challenge_done = False
    if challenge:
        ch = run_challenge(E, runs, flags)
        ch.to_csv(os.path.join(OUT, "acquisition_challenge.csv"), index=False)
        challenge_done = True

    # ---- manifest ------------------------------------------------------------------------------------------------
    emb_files = sorted(f for f in os.listdir(EMB_CACHE) if f.startswith("emb_") and f.endswith("_n1c3.npz"))
    def sha1(p):
        h = hashlib.sha1()
        with open(p, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()[:12]
    manifest = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), runtime_s=round(time.time() - t_start, 1),
                    settings=dict(F_SEED=F_SEED, tiles_per_site=TILES_PER_SITE, n_components_cap=N_COMP, k=K, resolutions=list(RESOLUTIONS),
                                  reference=REFERENCE, compared=list(COMPARED), covariates=ACQ_COVARIATES, top_tiles_per_batch=TOP_TILES_PER_BATCH,
                                  challenge_sites=CHALLENGE_SITES, challenge_perturbations={k: str(v) for k, v in CHALLENGE_PERTURBATIONS.items()}),
                    pca=({f"{v}_{r}_{m}": dict(n_components=runs[(v, r, m)]["n_components"], explained=runs[(v, r, m)]["pca_explained"], memory_rows=runs[(v, r, m)]["n_memory_rows"])
                          for (v, r, m) in runs}),
                    views={v: dict(n_ref=len(rs), n_query=len(qs), ref=rs, query=qs) for v, (rs, qs) in views.items()},
                    checks=checks, challenge_done=challenge_done, feature_version=features.FEATURE_VERSION,
                    embedding_cache_files={f: sha1(os.path.join(EMB_CACHE, f)) for f in emb_files},
                    versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, sklearn=__import__("sklearn").__version__))
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    log(f"done in {time.time() - t_start:.0f} s; checks {checks}")


if __name__ == "__main__":
    main(challenge="--challenge" in sys.argv)
