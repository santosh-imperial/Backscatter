"""E20 — the actual per-particle Inlens texture measurand under local-contrast normalisation (31 known sites).

Protocol: analysis/e20_inlens/findings.md §1 (pre-registered). Measurement support is imported from
polaron_qc.features (loader, edge-band trimming, histogram-anchored thresholds, segmentation, interior erosion,
area floor) and is NOT re-implemented. Sites are n (7/7/17); particles and tiles never are.

Usage (from the repository root, with Dataset/ and analysis_cache/features present):
    python3 analysis/e20_inlens/run_e20.py                 # extract -> analyse -> perturb -> figures
    python3 analysis/e20_inlens/run_e20.py --stage analyse # re-run analysis/figures from particles.csv.gz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats as sps
from skimage.feature import local_binary_pattern
from skimage.measure import label, find_contours

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from polaron_qc import BATCH_COLORS, GREY_PORE_SITES, LOW_CONTRAST_SITES, CRACKED_SITES  # noqa: E402
from polaron_qc import features as F  # noqa: E402
from polaron_qc.report import derive_flags  # noqa: E402
from polaron_qc.stats import permutation_test, robust_shift  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
VARIANTS = ["cur", "a_cv", "b_aff", "c_loc64", "c_loc32", "d_lbp", "d_lbp_ent"]
PRIMARY_VARIANTS = ["cur", "a_cv", "b_aff", "c_loc64", "d_lbp"]
VARIANT_LABEL = {"cur": "current: SD / image IQR", "a_cv": "a: CV = SD / mean", "b_aff": "b: SD / particle IQR",
                 "c_loc64": "c: SD after local z (σ 64 px)", "c_loc32": "c′: SD after local z (σ 32 px)",
                 "d_lbp": "d: LBP non-uniform fraction", "d_lbp_ent": "d′: LBP code entropy"}
COVARIATES = ["inlens_p50", "inlens_std", "bse_p1", "bse_std", "bright_sep", "H"]
PERTURB_SITES = {"Batch_1": ["f1vzngrs", "4ih2ggld"], "Batch_2": ["3806gxp0"], "Batch_3": ["9luzk4jm"]}
PERTURBATIONS = {"gamma_0.75": ("gamma", 0.75), "gamma_1.25": ("gamma", 1.25), "affine_0.6x+40": ("affine", (0.6, 40.0))}
EVIDENCE_SITES = {"Batch_1": ["4ih2ggld", "uhdslk0o"], "Batch_2": ["3806gxp0", "rxax5ozo"], "Batch_3": ["0grcilhi", "xgj4xftb"]}
INTERIOR_EROSION, MIN_INTERIOR_AREA = 6, 400          # exactly multichannel_features
LOCAL_FLOOR_FRAC = 0.05                                # floor = 0.05 x image IQR (E25 convention)
LBP_P, LBP_R = 8, 1
GREY_COLOR, LOWC_COLOR = "#eda100", "#e34948"


# ---------------------------------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------------------------------
def load_site(batch_dir, site):
    """Trimmed (bse, etd, inlens) uint8 arrays, trimmed by the BSE edge bands exactly as features.extract_site."""
    raw = {d: F.load_image(batch_dir, site, d) for d in F.DETECTORS}
    t, bo = F.bright_bands(raw["BSE"]); sl = slice(t, -bo if bo else None)
    return raw["BSE"][sl], raw["ETD"][sl], raw["Inlens"][sl], (t, bo)


def interiors(bse):
    """(label image of bright-particle interiors with area >= 400 px, thresholds dict). Reuses features.*"""
    sm = F.smooth_bse(bse)
    th = F.phase_thresholds(sm); th.pop("hist")
    _, bright, _ = F.segment(bse, th["th_lo"], th["th_hi"])
    inner = ndi.binary_erosion(bright, iterations=INTERIOR_EROSION)
    lab = label(inner)
    idx = np.arange(1, lab.max() + 1)
    if len(idx) == 0:
        return lab, th, idx
    area = ndi.sum(np.ones_like(lab), lab, idx)
    keep = idx[area >= MIN_INTERIOR_AREA]
    lab = np.where(np.isin(lab, keep), lab, 0)
    return lab, th, keep


def local_z(il_f, sigma, iqr_img):
    """N = (I - G(I)) / sqrt(max(G(I^2) - G(I)^2, 0) + floor^2), floor = 0.05 x image IQR. Returns (N, floor_mask)."""
    g1 = ndi.gaussian_filter(il_f, sigma, truncate=3.0)
    g2 = ndi.gaussian_filter(il_f * il_f, sigma, truncate=3.0)
    var = np.maximum(g2 - g1 * g1, 0.0)
    floor = (LOCAL_FLOOR_FRAC * iqr_img) ** 2
    return (il_f - g1) / np.sqrt(var + floor), var < floor


def particle_table(il_u8, lab, keep):
    """All variants per particle for one Inlens image (uint8, trimmed) and one interior label image."""
    il_f = il_u8.astype(float)
    iqr_img = np.subtract(*np.percentile(il_f, [75, 25])) + 1e-6
    sd = ndi.standard_deviation(il_f, lab, keep)
    mean = ndi.mean(il_f, lab, keep)
    area = ndi.sum(np.ones_like(lab), lab, keep)
    cur = sd / iqr_img
    a_cv = sd / np.maximum(mean, 1.0)
    # per-particle median / IQR
    objs = ndi.find_objects(lab)
    med = np.empty(len(keep)); iqr_p = np.empty(len(keep)); n_sat = np.empty(len(keep))
    for k, lbl in enumerate(keep):
        sl = objs[lbl - 1]; v = il_f[sl][lab[sl] == lbl]
        q25, q50, q75 = np.percentile(v, [25, 50, 75]); med[k] = q50; iqr_p[k] = q75 - q25
        n_sat[k] = np.mean((v <= 0) | (v >= 255))
    b_floor = iqr_p < 1.0
    b_aff = sd / np.maximum(iqr_p, 1.0)
    # local z-normalisation (whole image, then restricted to interiors)
    out_c = {}
    for sig, name in ((64, "c_loc64"), (32, "c_loc32")):
        N, fl = local_z(il_f, sig, iqr_img)
        out_c[name] = ndi.standard_deviation(N, lab, keep)
        out_c[name + "_floor_frac"] = ndi.mean(fl.astype(float), lab, keep)
        del N, fl
    # LBP (rotation-invariant uniform, P=8, R=1) on the uint8 image
    lbp = local_binary_pattern(il_u8, LBP_P, LBP_R, method="uniform").astype(np.int64)   # codes 0..9, 9 = non-uniform
    d_lbp = ndi.mean((lbp == LBP_P + 1).astype(float), lab, keep)
    ncode = LBP_P + 2
    comb = np.bincount((lab.astype(np.int64) * ncode + lbp)[lab > 0], minlength=(lab.max() + 1) * ncode).reshape(-1, ncode)
    hist = comb[keep].astype(float); hist /= np.maximum(hist.sum(1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(hist > 0, hist * np.log(hist), 0.0)).sum(1) / np.log(ncode)
    del lbp, comb
    return pd.DataFrame(dict(label=keep, area=area, inlens_mean=mean, inlens_median=med, inlens_sd=sd, inlens_iqr_p=iqr_p,
                             sat_frac=n_sat, img_iqr=iqr_img, cur=cur, a_cv=a_cv, b_aff=b_aff, b_floor=b_floor,
                             c_loc64=out_c["c_loc64"], c_loc64_floor_frac=out_c["c_loc64_floor_frac"],
                             c_loc32=out_c["c_loc32"], c_loc32_floor_frac=out_c["c_loc32_floor_frac"],
                             d_lbp=d_lbp, d_lbp_ent=ent))


def perturb(il_u8, kind, par):
    x = il_u8.astype(float)
    if kind == "gamma":
        y = 255.0 * (x / 255.0) ** par
    else:
        a, b = par; y = a * x + b
    assert y.min() >= 0 and y.max() <= 255, "perturbation must not clip"
    return np.round(y).astype(np.uint8)


def site_summary(pt):
    row = dict(n_particles=int(len(pt)))
    for v in VARIANTS:
        row[f"{v}_median"] = float(np.median(pt[v])) if len(pt) else np.nan
        row[f"{v}_p90"] = float(np.percentile(pt[v], 90)) if len(pt) else np.nan
    row["cur_speckled_frac"] = float((pt.cur > 0.5).mean()) if len(pt) else np.nan
    row["b_floor_frac_particles"] = float(pt.b_floor.mean()) if len(pt) else np.nan
    row["c_loc64_floor_frac_median"] = float(pt.c_loc64_floor_frac.median()) if len(pt) else np.nan
    row["sat_frac_median"] = float(pt.sat_frac.median()) if len(pt) else np.nan
    return row


# ---------------------------------------------------------------------------------------------------------------
# stage 1: extraction over all sites (+ perturbations on the prespecified sites, + evidence crops)
# ---------------------------------------------------------------------------------------------------------------
def covariates():
    tabs = {b: F.extract_batch(os.path.join(ROOT, "Dataset", b), cache_dir=os.path.join(ROOT, "analysis_cache", "features"), verbose=False)
            for b in BATCHES}
    sites = pd.concat([tabs[b]["sites"] for b in BATCHES], ignore_index=True)
    images = pd.concat([tabs[b]["images"] for b in BATCHES], ignore_index=True)
    fl = derive_flags(sites, images)
    il = images[images.det == "Inlens"].set_index("site")
    fl["inlens_std"] = fl.site.map(il["std"]).astype(float)
    fl["inlens_p50_images"] = fl.site.map(il["p50"]).astype(float)
    assert np.allclose(fl.inlens_p50.astype(float), fl.inlens_p50_images), "derive_flags inlens_p50 must match images table"
    fl["cache_inlens_particle_texture"] = fl.site.map(sites.set_index("site")["inlens_particle_texture"]).astype(float)
    fl["cache_inlens_particles_measured"] = fl.site.map(sites.set_index("site")["inlens_particles_measured"]).astype(int)
    fl["cache_th_lo"] = fl.site.map(sites.set_index("site")["th_lo"]).astype(float)
    fl["cache_th_hi"] = fl.site.map(sites.set_index("site")["th_hi"]).astype(float)
    keep = ["batch", "site", "acquisition_group", "bright_low_contrast", "grey_pore", "cracked", "contrast_stretched_any",
            "inlens_p50", "inlens_std", "inlens_empty_bin_frac", "bse_p1", "bse_std", "bright_sep", "H",
            "cache_inlens_particle_texture", "cache_inlens_particles_measured", "cache_th_lo", "cache_th_hi"]
    return fl[keep].copy()


def stage_extract():
    os.makedirs(FIG, exist_ok=True)
    cov = covariates()
    rows, parts, pert_rows, evidence = [], [], [], []
    t_all = time.time()
    for b in BATCHES:
        bdir = os.path.join(ROOT, "Dataset", b)
        for s in F.list_sites(bdir):
            t0 = time.time()
            bse, etd, il, bands = load_site(bdir, s)
            lab, th, keep = interiors(bse)
            pt = particle_table(il, lab, keep); pt.insert(0, "site", s); pt.insert(0, "batch", b)
            parts.append(pt)
            row = dict(batch=b, site=s, th_lo=float(th["th_lo"]), th_hi=float(th["th_hi"]), band_top=bands[0], band_bottom=bands[1],
                       H_trimmed=int(bse.shape[0]), **site_summary(pt))
            c = cov[cov.site == s].iloc[0]
            row["matches_cache"] = bool(abs(row["cur_median"] - c.cache_inlens_particle_texture) < 1e-9 and row["n_particles"] == c.cache_inlens_particles_measured
                                        and abs(row["th_lo"] - c.cache_th_lo) < 1e-9 and abs(row["th_hi"] - c.cache_th_hi) < 1e-9)
            rows.append(row)
            print(f"[e20] {b} {s}: {row['n_particles']} particles | cur {row['cur_median']:.3f} (cache {c.cache_inlens_particle_texture:.3f}, "
                  f"match={row['matches_cache']}) a {row['a_cv_median']:.3f} b {row['b_aff_median']:.3f} c {row['c_loc64_median']:.3f} "
                  f"d {row['d_lbp_median']:.3f} | {time.time()-t0:.0f}s", flush=True)
            if s in PERTURB_SITES.get(b, []):
                for pname, (kind, par) in PERTURBATIONS.items():
                    ptp = particle_table(perturb(il, kind, par), lab, keep)
                    pert_rows.append(dict(batch=b, site=s, perturbation=pname, **site_summary(ptp)))
                print(f"[e20]   perturbations done for {s}", flush=True)
            if s in EVIDENCE_SITES.get(b, []):
                evidence.append(evidence_crops(b, s, bse, il, lab, keep, pt))
    sites = cov.merge(pd.DataFrame(rows), on=["batch", "site"], how="right")
    sites.to_csv(os.path.join(OUT, "sites.csv"), index=False)
    pd.concat(parts, ignore_index=True).to_csv(os.path.join(OUT, "particles.csv.gz"), index=False, compression="gzip")
    pd.DataFrame(pert_rows).to_csv(os.path.join(OUT, "perturbation_raw.csv"), index=False)
    np.save(os.path.join(OUT, "evidence_crops.npy"), np.array(evidence, dtype=object), allow_pickle=True)
    print(f"[e20] extraction: {len(rows)} sites in {time.time()-t_all:.0f}s; all match cache: {sites.matches_cache.all()}", flush=True)


def evidence_crops(batch, site, bse, il, lab, keep, pt):
    """Largest interior component of the site: crops of BSE, raw Inlens, b-standardised, local-z and LBP mask."""
    k = int(pt.area.idxmax()); lbl = int(pt.label.iloc[k])
    sl = ndi.find_objects(lab)[lbl - 1]; m = 24
    r0, r1 = max(sl[0].start - m, 0), min(sl[0].stop + m, lab.shape[0]); c0, c1 = max(sl[1].start - m, 0), min(sl[1].stop + m, lab.shape[1])
    win = (slice(r0, r1), slice(c0, c1))
    mask = lab[win] == lbl
    il_f = il.astype(float); iqr_img = np.subtract(*np.percentile(il_f, [75, 25])) + 1e-6
    # local z on a padded window (sigma 64 needs context): pad 3 sigma
    pad = 3 * 64
    pw = (slice(max(r0 - pad, 0), min(r1 + pad, lab.shape[0])), slice(max(c0 - pad, 0), min(c1 + pad, lab.shape[1])))
    N, _ = local_z(il_f[pw], 64, iqr_img)
    N = N[r0 - pw[0].start: r1 - pw[0].start, c0 - pw[1].start: c1 - pw[1].start]
    v = il_f[win][mask]; q25, q50, q75 = np.percentile(v, [25, 50, 75])
    B = (il_f[win] - q50) / max(q75 - q25, 1.0)
    lbp = local_binary_pattern(il[pw], LBP_P, LBP_R, method="uniform")[r0 - pw[0].start: r1 - pw[0].start, c0 - pw[1].start: c1 - pw[1].start]
    vals = {vn: float(pt[vn].iloc[k]) for vn in VARIANTS}
    return dict(batch=batch, site=site, label=lbl, area=float(pt.area.iloc[k]), bbox=(r0, r1, c0, c1), mask=mask,
                bse=bse[win].copy(), il=il[win].copy(), b_img=B, c_img=N, lbp_nonuni=(lbp == LBP_P + 1), vals=vals)


# ---------------------------------------------------------------------------------------------------------------
# stage 2: analysis
# ---------------------------------------------------------------------------------------------------------------
def spearman(x, y):
    d = pd.DataFrame(dict(x=x, y=y)).dropna()
    if len(d) < 4:
        return np.nan, np.nan, len(d)
    r, p = sps.spearmanr(d.x, d.y); return float(r), float(p), int(len(d))


def analyse(sites):
    views = {"all31": sites, "bright_usable": sites[~sites.bright_low_contrast.astype(bool)]}
    corr_rows, test_rows, ctrl_rows = [], [], []
    for vname, df in views.items():
        n_by = df.groupby("batch").size().reindex(BATCHES).fillna(0).astype(int).to_dict()
        for v in VARIANTS:
            for stat in ("median", "p90"):
                col = f"{v}_{stat}"
                for cov in ("inlens_p50", "inlens_std"):
                    r, p, n = spearman(df[col], df[cov])
                    corr_rows.append(dict(view=vname, variant=v, statistic=stat, covariate=cov, scope="all", rho=r, p=p, n=n))
                    for b in BATCHES:
                        sub = df[df.batch == b]; r, p, n = spearman(sub[col], sub[cov])
                        corr_rows.append(dict(view=vname, variant=v, statistic=stat, covariate=cov, scope=b, rho=r, p=p, n=n))
                groups = [df[df.batch == b][col].dropna().to_numpy() for b in BATCHES]
                kw = sps.kruskal(*groups)
                meds = {b: float(np.median(g)) for b, g in zip(BATCHES, groups)}
                order = " > ".join(b.replace("Batch_", "B") for b in sorted(BATCHES, key=lambda b: -meds[b]))
                ref = groups[2]
                row = dict(view=vname, variant=v, statistic=stat, n_B1=n_by["Batch_1"], n_B2=n_by["Batch_2"], n_B3=n_by["Batch_3"],
                           med_B1=meds["Batch_1"], med_B2=meds["Batch_2"], med_B3=meds["Batch_3"], kruskal_H=float(kw.statistic),
                           kruskal_p=float(kw.pvalue), ordering=order, mad_B3=float(sps.median_abs_deviation(ref, scale="normal")))
                for b, g in (("B1", groups[0]), ("B2", groups[1])):
                    pt = permutation_test(ref, g, statistic="hl_shift")
                    rs = robust_shift(ref, g, n_boot=2000, seed=0)
                    row.update({f"{b}_vs_B3_hl": float(pt["observed"]), f"{b}_vs_B3_p": float(pt["p"]), f"{b}_vs_B3_method": pt["method"],
                                f"{b}_vs_B3_shift_mad": float(rs["shift_mad"]), f"{b}_vs_B3_shift_ci_low": float(rs["ci_low"]),
                                f"{b}_vs_B3_shift_ci_high": float(rs["ci_high"]), f"{b}_vs_B3_cliffs": float(rs["cliffs_delta"])})
                test_rows.append(row)
                # acquisition-only control (LOO ridge, fold-local imputation/scaling)
                r2, kw_res, n_used = acquisition_control(df, col)
                ctrl_rows.append(dict(view=vname, variant=v, statistic=stat, covariates="+".join(COVARIATES), n=n_used, loo_r2=r2,
                                      residual_kruskal_p=kw_res))
    return pd.DataFrame(corr_rows), pd.DataFrame(test_rows), pd.DataFrame(ctrl_rows)


def acquisition_control(df, col):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = df[[col, "batch"] + COVARIATES].copy(); d = d[d[col].notna()]
    X = d[COVARIATES].astype(float).to_numpy(); y = d[col].astype(float).to_numpy()
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    yhat = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    res = y - yhat
    kw = sps.kruskal(*[res[(d.batch == b).to_numpy()] for b in BATCHES])
    return float(r2), float(kw.pvalue), int(len(y))


def perturbation_table(sites):
    raw = pd.read_csv(os.path.join(OUT, "perturbation_raw.csv"))
    base = sites.set_index("site")
    gap = {v: abs(sites[sites.batch == "Batch_1"][f"{v}_median"].median() - sites[sites.batch == "Batch_3"][f"{v}_median"].median()) for v in VARIANTS}
    rows = []
    for _, r in raw.iterrows():
        for v in VARIANTS:
            b0 = base.loc[r.site, f"{v}_median"]; b1 = r[f"{v}_median"]
            rows.append(dict(batch=r.batch, site=r.site, perturbation=r.perturbation, variant=v, unperturbed=b0, perturbed=b1,
                             delta=b1 - b0, gap_B1_B3=gap[v], delta_over_gap=(b1 - b0) / gap[v] if gap[v] > 0 else np.nan,
                             rel_change=(b1 - b0) / b0 if b0 else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------
def _style(s):
    col = GREY_COLOR if s.site in GREY_PORE_SITES else BATCH_COLORS[s.batch]
    return col, (s.site in LOW_CONTRAST_SITES), (s.site in CRACKED_SITES)


def fig_strips(sites, tests):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(PRIMARY_VARIANTS), figsize=(4.4 * len(PRIMARY_VARIANTS), 4.6))
    t = tests[(tests.view == "all31") & (tests.statistic == "median")].set_index("variant")
    for ax, v in zip(axes, PRIMARY_VARIANTS):
        for i, b in enumerate(BATCHES):
            sub = sites[sites.batch == b]
            x = i + rng.uniform(-0.18, 0.18, len(sub))
            for xi, (_, s) in zip(x, sub.iterrows()):
                col, lowc, crk = _style(s)
                ax.scatter(xi, s[f"{v}_median"], s=42, color=col, edgecolor=LOWC_COLOR if lowc else "none", linewidth=2 if lowc else 0, zorder=3,
                           marker="s" if crk else "o")
            ax.hlines(sub[f"{v}_median"].median(), i - 0.3, i + 0.3, color="black", linewidth=2, zorder=4)
        ax.set_xticks(range(3)); ax.set_xticklabels(["B1\nn=7", "B2\nn=7", "B3\nn=17"])
        ax.set_title(f"{VARIANT_LABEL[v]}\nKruskal p = {t.loc[v, 'kruskal_p']:.3g}\nexact perm. p: B1 vs B3 {t.loc[v, 'B1_vs_B3_p']:.2g}, B2 vs B3 {t.loc[v, 'B2_vs_B3_p']:.2g}",
                     fontsize=8)
        ax.set_ylabel("site median over particles", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    handles = [Line2D([], [], marker="o", linestyle="", color=BATCH_COLORS[b], label=b.replace("_", " ")) for b in BATCHES]
    handles += [Line2D([], [], marker="o", linestyle="", color=GREY_COLOR, label="B3 grey-pore group"),
                Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=LOWC_COLOR, markeredgewidth=2, label="low-contrast (B1)"),
                Line2D([], [], marker="s", linestyle="", color=BATCH_COLORS["Batch_3"], label="cracked (B3)"),
                Line2D([], [], color="black", linewidth=2, label="bar = batch median")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("E20 — per-particle Inlens texture, current measurand and normalised variants (sites are n; all31 view)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(os.path.join(FIG, "strips_by_batch.png"), dpi=150); plt.close(fig)


def fig_brightness(sites, corr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    c = corr[(corr.view == "all31") & (corr.statistic == "median") & (corr.covariate == "inlens_p50")].set_index(["variant", "scope"])
    fig, axes = plt.subplots(1, len(PRIMARY_VARIANTS), figsize=(4.4 * len(PRIMARY_VARIANTS), 4.4))
    for ax, v in zip(axes, PRIMARY_VARIANTS):
        for _, s in sites.iterrows():
            col, lowc, crk = _style(s)
            ax.scatter(s.inlens_p50, s[f"{v}_median"], s=42, color=col, edgecolor=LOWC_COLOR if lowc else "none", linewidth=2 if lowc else 0,
                       marker="s" if crk else "o", zorder=3)
        within = " / ".join(f"{c.loc[(v, b), 'rho']:+.2f}" for b in BATCHES)
        ax.set_title(f"{VARIANT_LABEL[v]}\nSpearman ρ, all 31 sites: {c.loc[(v, 'all'), 'rho']:+.2f}\nwithin B1 / B2 / B3 (n 7/7/17): {within}", fontsize=8)
        ax.set_xlabel("Inlens median grey level (acquisition covariate)", fontsize=8); ax.set_ylabel("site median", fontsize=8)
        ax.grid(alpha=0.25)
    handles = [Line2D([], [], marker="o", linestyle="", color=BATCH_COLORS[b], label=b.replace("_", " ")) for b in BATCHES]
    handles += [Line2D([], [], marker="o", linestyle="", color=GREY_COLOR, label="B3 grey-pore group"),
                Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=LOWC_COLOR, markeredgewidth=2, label="low-contrast (B1)"),
                Line2D([], [], marker="s", linestyle="", color=BATCH_COLORS["Batch_3"], label="cracked (B3)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("E20 — each variant against Inlens brightness (Spearman; the confound E03 found for the current measurand)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(os.path.join(FIG, "variant_vs_inlens_p50.png"), dpi=150); plt.close(fig)


def fig_perturbation(pert):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = pert[pert.variant.isin(PRIMARY_VARIANTS)]
    sites = [s for b in BATCHES for s in PERTURB_SITES[b]]
    fig, axes = plt.subplots(1, len(PERTURBATIONS), figsize=(4.2 * len(PERTURBATIONS), 4.0), sharey=True)
    hatch = {"f1vzngrs": "", "4ih2ggld": "//", "3806gxp0": "", "9luzk4jm": ""}
    for ax, pn in zip(axes, PERTURBATIONS):
        sub = p[p.perturbation == pn]
        w = 0.8 / len(sites)
        for j, s in enumerate(sites):
            d = sub[sub.site == s].set_index("variant").reindex(PRIMARY_VARIANTS)
            b = d.batch.iloc[0]
            ax.bar(np.arange(len(PRIMARY_VARIANTS)) + (j - 1.5) * w, d.delta_over_gap, width=w * 0.92, color=BATCH_COLORS[b],
                   edgecolor=LOWC_COLOR if s in LOW_CONTRAST_SITES else "none", linewidth=1.5 if s in LOW_CONTRAST_SITES else 0,
                   hatch=hatch[s], label=f"{b.replace('_', ' ')} {s}" + (" (low-contrast)" if s in LOW_CONTRAST_SITES else ""))
        ax.axhline(0, color="black", linewidth=0.8); ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8); ax.axhline(-0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(PRIMARY_VARIANTS))); ax.set_xticklabels([v for v in PRIMARY_VARIANTS], fontsize=8)
        ax.set_title(pn.replace("_", " "), fontsize=9); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Δ site median / |B1 − B3 gap| of that variant", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("E20 — movement of each variant under intensity remaps of the real Inlens image (BSE masks fixed; dashed = ±0.5 gap)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(FIG, "perturbation.png"), dpi=150); plt.close(fig)


def fig_evidence():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = list(np.load(os.path.join(OUT, "evidence_crops.npy"), allow_pickle=True))
    cols = ["BSE (interior outline)", "Inlens raw (0–255)", "b: per-particle standardised (±3)", "c: local z, σ 64 px (±3)", "d: LBP non-uniform px"]
    fig, axes = plt.subplots(len(ev), len(cols), figsize=(3.0 * len(cols), 3.0 * len(ev)))
    for i, e in enumerate(ev):
        imgs = [(e["bse"], dict(cmap="gray", vmin=0, vmax=255)), (e["il"], dict(cmap="gray", vmin=0, vmax=255)),
                (np.where(e["mask"], e["b_img"], np.nan), dict(cmap="gray", vmin=-3, vmax=3)),
                (e["c_img"], dict(cmap="gray", vmin=-3, vmax=3)), (e["lbp_nonuni"].astype(float), dict(cmap="gray", vmin=0, vmax=1))]
        for j, (im, kw) in enumerate(imgs):
            ax = axes[i, j]; ax.imshow(im, interpolation="nearest", **kw)
            for cc in find_contours(e["mask"].astype(float), 0.5):
                ax.plot(cc[:, 1], cc[:, 0], color=BATCH_COLORS[e["batch"]], linewidth=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(cols[j], fontsize=8.5)
        v = e["vals"]
        axes[i, 0].set_ylabel(f"{e['batch'].replace('_', ' ')} {e['site']}\ninterior {e['area']:.0f} px\ncur {v['cur']:.2f} a {v['a_cv']:.2f} b {v['b_aff']:.2f}\nc {v['c_loc64']:.2f} d {v['d_lbp']:.2f}",
                              fontsize=7.5)
    fig.suptitle("E20 evidence — largest interior particle of the first/last site by id per batch: raw vs normalised Inlens (crops are illustrations, not n)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(FIG, "evidence_particles.png"), dpi=100); plt.close(fig)


def stage_analyse():
    sites = pd.read_csv(os.path.join(OUT, "sites.csv"))
    assert len(sites) == 31 and sites.matches_cache.all(), "integrity: 31 sites and cur must reproduce the cached measurand"
    corr, tests, ctrl = analyse(sites)
    corr.to_csv(os.path.join(OUT, "correlations.csv"), index=False)
    tests.to_csv(os.path.join(OUT, "batch_tests.csv"), index=False)
    ctrl.to_csv(os.path.join(OUT, "control_r2.csv"), index=False)
    pert = perturbation_table(sites); pert.to_csv(os.path.join(OUT, "perturbation.csv"), index=False)
    fig_strips(sites, tests); fig_brightness(sites, corr); fig_perturbation(pert); fig_evidence()
    # console summary for the findings
    t = tests[(tests.view == "all31") & (tests.statistic == "median")].set_index("variant")
    c = corr[(corr.view == "all31") & (corr.statistic == "median") & (corr.covariate == "inlens_p50") & (corr.scope == "all")].set_index("variant")
    r = ctrl[(ctrl.view == "all31") & (ctrl.statistic == "median")].set_index("variant")
    print("\nvariant | rho(p50) | kruskal p | B1/B2/B3 med | B1-B3 shift(MAD) p | B2-B3 shift(MAD) p | LOO R2")
    for v in VARIANTS:
        print(f"{v:9s} | {c.loc[v,'rho']:+.2f} | {t.loc[v,'kruskal_p']:.3g} | {t.loc[v,'med_B1']:.3f}/{t.loc[v,'med_B2']:.3f}/{t.loc[v,'med_B3']:.3f} | "
              f"{t.loc[v,'B1_vs_B3_shift_mad']:+.2f} [{t.loc[v,'B1_vs_B3_shift_ci_low']:+.2f},{t.loc[v,'B1_vs_B3_shift_ci_high']:+.2f}] p {t.loc[v,'B1_vs_B3_p']:.3g} | "
              f"{t.loc[v,'B2_vs_B3_shift_mad']:+.2f} [{t.loc[v,'B2_vs_B3_shift_ci_low']:+.2f},{t.loc[v,'B2_vs_B3_shift_ci_high']:+.2f}] p {t.loc[v,'B2_vs_B3_p']:.3g} | {r.loc[v,'loo_r2']:+.2f}")
    print("\nperturbation (delta/gap, all primary variants):")
    print(pert[pert.variant.isin(PRIMARY_VARIANTS)].pivot_table(index=["variant"], columns=["perturbation", "site"], values="delta_over_gap").round(2).to_string())
    json.dump(dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), feature_version=F.FEATURE_VERSION, n_sites=int(len(sites)),
                   all_match_cache=bool(sites.matches_cache.all())), open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["all", "extract", "analyse"], default="all")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    if a.stage in ("all", "extract"):
        stage_extract()
    if a.stage in ("all", "analyse"):
        stage_analyse()
