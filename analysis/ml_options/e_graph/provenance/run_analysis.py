"""E18: exploratory morphology analysis, separate from the frozen QC pipeline.

Run: /opt/anaconda3/bin/python3 -m analysis.morphology.run_analysis
Outputs stay in analysis/morphology/output. Sites, never components, are n.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/polaron-morphology-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from skimage.measure import label, regionprops_table
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from polaron_qc import BATCH_COLORS, MATERIAL_KPIS, PRIMARY_KPIS
from polaron_qc import features, stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
VERSION = "E18.1"
SEED = 20261003
N_BOOT = 4000
N_PERM = 5000
N_CSR = 200
MIN_ASPECT_ORIENTATION = 1.5
PROPS = ("label", "area", "equivalent_diameter_area", "major_axis_length", "minor_axis_length",
         "orientation", "centroid", "bbox", "solidity", "perimeter")
MORPH = ["pore_elong", "pore_width_d90_d50", "pore_horizontal_alignment", "pore_alignment_strength",
         "bright_circ", "bright_solidity", "bright_width_d90_d10", "bright_aspect_aw",
         "bright_horizontal_alignment", "bright_alignment_strength", "bright_nn_csr_ratio", "bright_nn_mean_px",
         "profile_pore_abs_slope", "profile_bright_abs_slope", "corr_len_px", "fft_slope"]
GEOMETRY = [k for k in MORPH if k not in ("corr_len_px", "fft_slope")]
ACQ = ["H", "bright_sep", "bse_p1", "bse_std", "bse_empty_bin_frac", "etd_curtain_frac", "etd_boundary_sharpness"]
LABELS = {
    "pore_elong": "Void aspect ratio (area weighted)",
    "pore_width_d90_d50": "Void size width D90 / D50",
    "pore_horizontal_alignment": "Void horizontal alignment",
    "pore_alignment_strength": "Void axial alignment strength",
    "bright_circ": "Bright-particle circularity",
    "bright_solidity": "Bright-particle solidity",
    "bright_width_d90_d10": "Bright size width D90 / D10",
    "bright_aspect_aw": "Bright-particle aspect ratio",
    "bright_horizontal_alignment": "Bright horizontal alignment",
    "bright_alignment_strength": "Bright axial alignment strength",
    "bright_nn_csr_ratio": "Bright centroid spacing / random-point mean",
    "bright_nn_mean_px": "Bright centroid nearest spacing (px)",
    "profile_pore_abs_slope": "Void image-depth gradient magnitude",
    "profile_bright_abs_slope": "Bright image-depth gradient magnitude",
    "corr_len_px": "BSE correlation length (px; contrast sensitive)",
    "fft_slope": "BSE spectrum slope (contrast sensitive)",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def component_table(mask, floor):
    table = pd.DataFrame(regionprops_table(label(mask), properties=PROPS))
    table = table[table.area >= floor].copy()
    h, w = mask.shape
    table["touches_edge"] = ((table["bbox-0"] == 0) | (table["bbox-1"] == 0) |
                              (table["bbox-2"] == h) | (table["bbox-3"] == w))
    table["aspect"] = table.major_axis_length / np.maximum(table.minor_axis_length, 1)
    table["angle_horizontal_rad"] = (np.pi / 2 - table.orientation) % np.pi
    return table


def orientation_summary(table):
    """Area-weighted axial statistics of non-clipped, elongated components.

    Horizontal alignment: +1 horizontal, -1 vertical, 0 balanced. Strength is
    |mean(exp(2i theta))|. Round components have no reliable major-axis angle.
    """
    use = table[(~table.touches_edge) & (table.aspect >= MIN_ASPECT_ORIENTATION)]
    if use.empty:
        return dict(horizontal_alignment=np.nan, alignment_strength=np.nan, n_oriented=0,
                    oriented_area_share=0.0, hist=np.full(12, np.nan))
    theta = use.angle_horizontal_rad.to_numpy()
    weights = use.area.to_numpy()
    moment = np.average(np.exp(2j * theta), weights=weights)
    bins = np.histogram(theta, bins=np.linspace(0, np.pi, 13), weights=weights)[0]
    return dict(horizontal_alignment=float(moment.real), alignment_strength=float(abs(moment)),
                n_oriented=len(use), oriented_area_share=float(weights.sum() / table.area.sum()),
                hist=bins / bins.sum())


def guarded_nn(points, shape):
    """Nearest-centroid distances whose circles fit in the observed rectangle."""
    if len(points) < 2:
        return np.array([]), np.zeros(len(points), dtype=bool), np.array([], dtype=int)
    distances, indices = cKDTree(points).query(points, k=2)
    d, other = distances[:, 1], indices[:, 1]
    h, w = shape
    edge = np.minimum.reduce([points[:, 0], points[:, 1], h - 1 - points[:, 0], w - 1 - points[:, 1]])
    eligible = edge >= d
    return d[eligible], eligible, other


def spacing_summary(table, shape, seed, n_csr=N_CSR):
    points = table[["centroid-0", "centroid-1"]].to_numpy()
    distances, eligible, _ = guarded_nn(points, shape)
    if len(distances) < 10:
        return dict(bright_nn_mean_px=np.nan, bright_nn_csr_ratio=np.nan,
                    bright_nn_n_guarded=len(distances), bright_nn_csr_mean_px=np.nan,
                    bright_nn_csr_mc_se_px=np.nan)
    rng = np.random.default_rng(seed)
    random_means = []
    for _ in range(n_csr):
        random_points = rng.uniform([0, 0], np.array(shape) - 1, size=(len(points), 2))
        rd, _, _ = guarded_nn(random_points, shape)
        if len(rd):
            random_means.append(rd.mean())
    expected = float(np.mean(random_means))
    return dict(bright_nn_mean_px=float(distances.mean()), bright_nn_csr_ratio=float(distances.mean() / expected),
                bright_nn_n_guarded=int(eligible.sum()), bright_nn_csr_mean_px=expected,
                bright_nn_csr_mc_se_px=float(np.std(random_means, ddof=1) / np.sqrt(len(random_means))))


def weighted_d(table, quantile):
    if table.empty:
        return np.nan
    ordered = table.sort_values("equivalent_diameter_area")
    cumulative = ordered.area.cumsum() / ordered.area.sum()
    return float(ordered.equivalent_diameter_area.iloc[np.searchsorted(cumulative, quantile)])


def profile_slope(mask):
    edges = np.linspace(0, len(mask), 11).astype(int)
    profile = np.array([mask[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    return float(np.polyfit(np.arange(10) / 10 + 0.05, profile, 1)[0])


def measure_masks(pore, bright, seed):
    pt, bt = component_table(pore, features.MIN_PORE_AREA_PX), component_table(bright, features.MIN_BRIGHT_AREA_PX)
    po, bo = orientation_summary(pt), orientation_summary(bt)
    interior = bt[~bt.touches_edge]
    weighted = lambda table, values: float(np.average(values, weights=table.area)) if len(table) else np.nan
    pc, bc = profile_slope(pore), profile_slope(bright)
    metrics = dict(
        pore_frac_recomputed=float(pore.mean()), bright_frac_recomputed=float(bright.mean()),
        pore_elong=weighted(pt, pt.aspect),
        pore_width_d90_d50=weighted_d(pt, 0.9) / weighted_d(pt, 0.5),
        bright_width_d90_d10=weighted_d(bt, 0.9) / weighted_d(bt, 0.1),
        bright_circ=weighted(bt, 4 * np.pi * bt.area / np.maximum(bt.perimeter, 1) ** 2),
        bright_solidity=weighted(bt, bt.solidity), bright_aspect_aw=weighted(interior, interior.aspect),
        pore_horizontal_alignment=po["horizontal_alignment"], pore_alignment_strength=po["alignment_strength"],
        bright_horizontal_alignment=bo["horizontal_alignment"], bright_alignment_strength=bo["alignment_strength"],
        profile_pore_slope=pc, profile_pore_abs_slope=abs(pc), profile_bright_slope=bc, profile_bright_abs_slope=abs(bc),
        pore_n_components=len(pt), bright_n_components=len(bt),
        pore_n_oriented=po["n_oriented"], bright_n_oriented=bo["n_oriented"],
        pore_oriented_area_share=po["oriented_area_share"], bright_oriented_area_share=bo["oriented_area_share"],
        bright_edge_component_share=float(bt.touches_edge.mean()) if len(bt) else np.nan,
        **spacing_summary(bt, bright.shape, seed),
    )
    return metrics, {"pore": pt, "bright": bt}, {"pore": po["hist"], "bright": bo["hist"]}


def load_sites():
    sites = pd.read_csv(ROOT / "analysis_cache/site_features.csv")
    etd = pd.read_csv(ROOT / "analysis_cache/etd_inlens_features.csv")
    sites = sites.merge(etd, on=["batch", "site"], validate="one_to_one")
    sites["etd_crack_density_particles"] = sites[f"crack_p_{features.RIDGE_T}"]
    flags = pd.read_csv(ROOT / "analysis_cache/acquisition_sites.csv")
    carry = [c for c in flags if c not in sites or c in ("batch", "site")]
    sites = sites.merge(flags[carry], on=["batch", "site"], validate="one_to_one")
    assert not sites.duplicated(["batch", "site"]).any()
    assert sites.site.nunique() == len(sites), "site IDs must be globally unique for this analysis"
    return sites


def extract_geometry(sites, manifest, force=False):
    cache_key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    key_path = OUT / "extraction_key.txt"
    if not force and key_path.exists() and key_path.read_text().strip() == cache_key:
        return pd.read_csv(OUT / "threshold_variants.csv"), pd.read_csv(OUT / "orientation_histograms.csv")
    variants, components, histograms = [], [], []
    for i, r in enumerate(sites.itertuples()):
        bse = features.load_image(str(ROOT / "Dataset" / r.batch), r.site)
        bse = features.trimmed(bse)
        assert bse.shape == (r.H, r.W), (r.site, bse.shape, r.H, r.W)
        smoothed = features.smooth_bse(bse)
        for offset in (-5, 0, 5):
            pore = ndi.binary_opening(smoothed < r.th_lo + offset, iterations=1)
            bright = ndi.binary_opening(smoothed > r.th_hi + offset, iterations=1)
            m, tables, hist = measure_masks(pore, bright, SEED + i)
            variants.append(dict(batch=r.batch, site=r.site, threshold_offset=offset, **m))
            if offset == 0:
                for cached, recomputed in (("pore_frac", "pore_frac_recomputed"), ("bright_frac", "bright_frac_recomputed"),
                                          ("pore_elong", "pore_elong"), ("bright_circ", "bright_circ"), ("bright_solidity", "bright_solidity")):
                    assert np.isclose(getattr(r, cached), m[recomputed], atol=1e-10), (r.site, cached)
                for phase, table in tables.items():
                    components.append(table.assign(batch=r.batch, site=r.site, phase=phase))
                    for j, share in enumerate(hist[phase]):
                        histograms.append(dict(batch=r.batch, site=r.site, phase=phase,
                                               angle_bin_deg=7.5 + 15 * j, area_share=share))
        print(f"geometry {i + 1}/{len(sites)}: {r.batch}/{r.site}", flush=True)
    variants = pd.DataFrame(variants)
    histograms = pd.DataFrame(histograms)
    variants.to_csv(OUT / "threshold_variants.csv", index=False)
    histograms.to_csv(OUT / "orientation_histograms.csv", index=False)
    pd.concat(components, ignore_index=True).to_csv(OUT / "component_geometry.csv.gz", index=False, compression="gzip")
    key_path.write_text(cache_key + "\n")
    return variants, histograms


def family(kpi):
    if kpi.startswith(("pore_", "crack_", "profile_pore")):
        return "pore"
    if kpi.startswith(("bright_", "profile_bright")):
        return "bright"
    return "etd" if kpi.startswith("etd_") else "texture"


def view_sites(sites, batch, view, is_reference=False):
    sub = sites[sites.batch == batch].copy()
    if view != "full_usable":
        sub = sub[~sub.bright_low_contrast & ~sub.grey_pore]
    if view == "ordinary_reference" and is_reference:
        sub = sub[~sub.cracked_known]
    return sub


def usable(sub, kpi):
    sub = sub[~sub.bright_low_contrast] if family(kpi) == "bright" else sub
    return sub.loc[sub[kpi].notna(), ["site", kpi]]


def median_interval(a, b, seed):
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(b, (N_BOOT, len(b))), axis=1) - np.median(rng.choice(a, (N_BOOT, len(a))), axis=1)
    return np.quantile(draws, [0.025, 0.975])


def compare(sites, variants):
    pairs = [("Batch_3", "Batch_1"), ("Batch_3", "Batch_2"), ("Batch_1", "Batch_2")]
    kpis = list(dict.fromkeys([*MATERIAL_KPIS, *MORPH, "profile_pore_slope", "profile_bright_slope"]))
    rows, multi = [], []
    bounds = variants.groupby(["site"])[GEOMETRY + ["profile_pore_slope", "profile_bright_slope"]].agg(["min", "max"])
    for view in ("full_usable", "quality_matched", "ordinary_reference"):
        for reference, batch in pairs:
            if view == "ordinary_reference" and reference != "Batch_3":
                continue
            rr, bb = view_sites(sites, reference, view, True), view_sites(sites, batch, view)
            for k in kpis:
                r, b = usable(rr, k), usable(bb, k)
                a, q = r[k].to_numpy(), b[k].to_numpy()
                if min(len(a), len(q)) < 3:
                    continue
                shift = stats.robust_shift(a, q, n_boot=N_BOOT, seed=SEED)
                perm = stats.permutation_test(a, q, statistic="hl_shift", n_mc=N_PERM, seed=SEED,
                                              exact_max=20000)
                delta = float(np.median(q) - np.median(a))
                low, high = median_interval(a, q, SEED)
                signs = [np.sign(np.median(q) - np.median(np.delete(a, i))) for i in range(len(a))]
                signs += [np.sign(np.median(np.delete(q, i)) - np.median(a)) for i in range(len(q))]
                row = dict(view=view, reference=reference, batch=batch, kpi=k, family=family(k),
                           n_ref=len(a), n_batch=len(q), ref_median=np.median(a), batch_median=np.median(q),
                           median_delta=delta, delta_ci_low=low, delta_ci_high=high,
                           shift_mad=shift["shift_mad"], shift_ci_low=shift["ci_low"], shift_ci_high=shift["ci_high"],
                           cliffs_delta=shift["cliffs_delta"], hl_shift=perm["observed"], p_exploratory=perm["p"],
                           p_method=perm["method"], n_perm=perm["n_perm"],
                           loo_sign_share=float(np.mean(np.array(signs) == np.sign(delta))) if delta else np.nan,
                           threshold_delta_low=np.nan, threshold_delta_high=np.nan, threshold_sign_survives=None)
                if k in bounds.columns.get_level_values(0):
                    lo = float(bounds.loc[b.site, (k, "min")].median() - bounds.loc[r.site, (k, "max")].median())
                    hi = float(bounds.loc[b.site, (k, "max")].median() - bounds.loc[r.site, (k, "min")].median())
                    row.update(threshold_delta_low=lo, threshold_delta_high=hi, threshold_sign_survives=bool(lo > 0 or hi < 0))
                rows.append(row)
            for panel, columns in (("existing_material", list(MATERIAL_KPIS)), ("morphology", MORPH)):
                r, b = rr.copy(), bb.copy()
                for k in columns:
                    if family(k) == "bright":
                        r.loc[r.bright_low_contrast, k] = np.nan
                        b.loc[b.bright_low_contrast, k] = np.nan
                families = [family(k) for k in columns]
                weights = [1 / families.count(family(k)) for k in columns]
                e = stats.energy_distance_test(r[columns], b[columns], weights=weights, n_mc=N_PERM, seed=SEED, chunk=250)
                multi.append(dict(view=view, reference=reference, batch=batch, panel=panel, **e))
            print(f"comparisons: {view} {batch} vs {reference}", flush=True)
    result = pd.DataFrame(rows)
    # Multiplicity context for an exploratory screen; no promotion into verdicts.
    result["p_holm_screen"] = np.nan
    for view, idx in result.groupby("view").groups.items():
        result.loc[idx, "p_holm_screen"] = stats.holm(result.loc[idx, "p_exploratory"])
    return result, pd.DataFrame(multi)


def confounds(sites):
    correlations, predictive = [], []
    for view in ("all", "quality_matched", "ordinary"):
        sub = sites if view == "all" else sites[~sites.bright_low_contrast & ~sites.grey_pore]
        if view == "ordinary":
            sub = sub[~sub.cracked_known]
        for k in MORPH:
            for cov in ACQ:
                pairs = usable(sub, k).merge(sub[["site", "batch", cov]], on="site")
                for group in ("pooled", "Batch_1", "Batch_2", "Batch_3"):
                    a = pairs if group == "pooled" else pairs[pairs.batch == group]
                    a = a.dropna(subset=[k, cov])
                    if len(a) >= 5 and a[k].nunique() > 1 and a[cov].nunique() > 1:
                        rho = float(spearmanr(a[k], a[cov]).statistic)
                        correlations.append(dict(view=view, group=group, kpi=k, covariate=cov, n=len(a), spearman_rho=rho))
    ordinary = sites[~sites.bright_low_contrast & ~sites.grey_pore & ~sites.cracked_known]
    covariates = ["H", "bright_sep", "bse_p1", "etd_boundary_sharpness"]
    for k in MORPH:
        sub = ordinary[[k, *covariates]].dropna()
        X, y = sub[covariates].to_numpy(), sub[k].to_numpy()
        if len(y) < 10 or np.var(y) == 0:
            continue
        pred = np.zeros(len(y))
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(X[keep], y[keep])
            pred[i] = model.predict(X[i:i+1])[0]
        predictive.append(dict(kpi=k, n=len(y), acquisition_loo_r2=float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
                               covariates=";".join(covariates), ridge_alpha=1.0))
    return pd.DataFrame(correlations), pd.DataFrame(predictive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sites = load_sites()
    paths = [ROOT / "analysis_cache" / name for name in ("site_features.csv", "etd_inlens_features.csv", "acquisition_sites.csv")]
    paths += [ROOT / "polaron_qc" / name for name in ("features.py", "stats.py", "__init__.py")]
    paths += [Path(__file__)]
    paths += [ROOT / "Dataset" / r.batch / f"img_{r.site}_BSE.tif" for r in sites.itertuples()]
    manifest = dict(version=VERSION, seed=SEED, n_boot=N_BOOT, n_perm=N_PERM, n_csr=N_CSR,
                    min_aspect_orientation=MIN_ASPECT_ORIENTATION, morphology_kpis=MORPH,
                    existing_material_kpis=list(MATERIAL_KPIS), frozen_primary_kpis=list(PRIMARY_KPIS),
                    inputs={str(p.relative_to(ROOT)): sha256(p) for p in paths})
    variants, histograms = extract_geometry(sites, manifest, args.force_extract)
    nominal = variants[variants.threshold_offset == 0].drop(columns=["threshold_offset"])
    nominal = nominal.drop(columns=[c for c in nominal if c in sites and c not in ("batch", "site")])
    sites = sites.merge(nominal, on=["batch", "site"], validate="one_to_one")
    sites.to_csv(OUT / "morphology_sites.csv", index=False)
    comparison, multi = compare(sites, variants)
    comparison.to_csv(OUT / "pairwise_comparisons.csv", index=False)
    multi.to_csv(OUT / "multivariate_comparisons.csv", index=False)
    correlations, predictive = confounds(sites)
    correlations.to_csv(OUT / "acquisition_correlations.csv", index=False)
    predictive.to_csv(OUT / "acquisition_predictability.csv", index=False)
    changed = [str(p.relative_to(ROOT)) for p in paths if manifest["inputs"][str(p.relative_to(ROOT))] != sha256(p)]
    manifest["inputs_changed_during_run"] = changed
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if changed:
        raise RuntimeError(f"Input files changed during run; rerun from the stable revision: {changed}")
    print("Tables complete. Build figures/report with analysis.morphology.make_report.", flush=True)


if __name__ == "__main__":
    main()
