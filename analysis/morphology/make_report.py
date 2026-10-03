"""Render E18's tables as scientific figures and a standalone review report."""
from __future__ import annotations

import base64
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
from matplotlib.lines import Line2D

from analysis.morphology.run_analysis import (
    OUT, ROOT, MORPH, LABELS, BATCH_COLORS, features, guarded_nn,
)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def effects(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.8), sharey=True)
    pairs = [("Batch_3", "Batch_1"), ("Batch_3", "Batch_2"), ("Batch_1", "Batch_2")]
    for ax, view in zip(axes, ("quality_matched", "ordinary_reference")):
        # The ordinary-reference view changes Batch 3 only; the B2/B1 pair is the same matched comparison.
        cols, labels = [], []
        for ref, batch in pairs:
            v = "quality_matched" if view == "ordinary_reference" and ref != "Batch_3" else view
            sub = comparison[(comparison.view == v) & (comparison.reference == ref) & (comparison.batch == batch)].set_index("kpi")
            cols.append([sub.loc[k, "shift_mad"] for k in MORPH])
            n = sub.loc["bright_circ"]
            labels.append(f"{batch.replace('Batch_', 'B')} − {ref.replace('Batch_', 'B')}\n(n={int(n.n_batch)} vs {int(n.n_ref)})")
        values = np.array(cols).T
        im = ax.imshow(np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        for i in range(len(MORPH)):
            for j in range(3):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(values[i, j]) > 1.8 else "black")
        ax.set_xticks(range(3), labels, fontsize=9)
        ax.set_yticks(range(len(MORPH)), [LABELS[k] for k in MORPH], fontsize=9)
        ax.set_title("Quality matched (cracked reference retained)" if view == "quality_matched" else "Ordinary Batch 3 reference only", fontsize=11)
    cax = fig.add_axes([0.925, 0.20, 0.016, 0.60])
    fig.colorbar(im, cax=cax, label="Median shift / reference MAD (colour clipped at ±3)")
    fig.suptitle("Exploratory morphology differences — uncertainty and threshold bands are in the tables", fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.27, right=0.90, bottom=0.12, top=0.92, wspace=0.14)
    save(fig, "morphology_effects.png")


def site_strips(sites, comparison):
    keys = ["bright_width_d90_d10", "bright_aspect_aw", "bright_nn_csr_ratio", "bright_circ",
            "pore_horizontal_alignment", "pore_width_d90_d50", "profile_pore_abs_slope", "corr_len_px"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for ax, k in zip(axes.ravel(), keys):
        for i, batch in enumerate(("Batch_1", "Batch_2", "Batch_3")):
            sub = sites[sites.batch == batch].copy()
            if k.startswith(("bright_", "profile_bright")):
                sub = sub[~sub.bright_low_contrast]
            sub = sub.sort_values("site")
            jitter = np.linspace(-0.11, 0.11, len(sub))
            for (_, r), x in zip(sub.iterrows(), i + jitter):
                color = "#eda100" if r.grey_pore else BATCH_COLORS[batch]
                ax.scatter(x, r[k], color=color, marker="^" if r.cracked_known else "o", s=35,
                           facecolors="none" if r.grey_pore else color, zorder=3)
            normal = sub[~sub.grey_pore]
            median = normal[k].median()
            ax.plot([i - 0.18, i + 0.18], [median, median], color=BATCH_COLORS[batch], lw=2.8)
        ax.set_xticks(range(3), ["Batch 1", "Batch 2", "Batch 3"])
        ax.set_title(LABELS[k], fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.15)
        if "alignment" in k:
            ax.axhline(0, lw=0.6, color="gray")
            ax.set_ylim(-1.06, 1.06)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color="gray", label="Ordinary site"),
                        Line2D([], [], marker="^", linestyle="none", color="gray", label="Known cracked site"),
                        Line2D([], [], marker="o", markerfacecolor="none", linestyle="none", color="#eda100", label="Grey-pore / fallback site")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Each point is one site; bars are medians excluding grey-pore sites\nBright plots exclude low-contrast sites. Cracked sites remain visible.", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, "site_distributions.png")


def angle_distributions(sites, hist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    groups = [("Batch 1 usable", sites[(sites.batch == "Batch_1") & ~sites.bright_low_contrast], BATCH_COLORS["Batch_1"], "-"),
              ("Batch 2", sites[sites.batch == "Batch_2"], BATCH_COLORS["Batch_2"], "-"),
              ("Batch 3 ordinary", sites[(sites.batch == "Batch_3") & ~sites.grey_pore & ~sites.cracked_known], BATCH_COLORS["Batch_3"], "-"),
              ("Batch 3 cracked", sites[sites.cracked_known], BATCH_COLORS["Batch_3"], "--"),
              ("Batch 3 grey pores", sites[sites.grey_pore], "#eda100", ":")]
    for ax, phase in zip(axes, ("pore", "bright")):
        for name, sub, color, style in groups:
            h = hist[(hist.phase == phase) & hist.site.isin(sub.site)]
            mean = h.groupby("angle_bin_deg").area_share.mean()
            ax.plot(mean.index, mean.values, color=color, linestyle=style, label=f"{name} (n={len(sub)})")
        ax.set(xlabel="Major-axis direction relative to image horizontal (°)", ylabel="Mean per-site area share",
               title=f"{'Void' if phase == 'pore' else 'Bright-particle'} axial orientations")
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.grid(alpha=0.15)
    axes[1].legend(fontsize=8)
    fig.suptitle("Elongated components only (aspect ≥1.5); edge-clipped components excluded", fontsize=11)
    fig.tight_layout()
    save(fig, "orientation_distributions.png")


def evidence(sites):
    components = pd.read_csv(OUT / "component_geometry.csv.gz")
    clean = sites[~sites.bright_low_contrast & ~sites.grey_pore]
    selections = []
    for k, phase in (("bright_nn_csr_ratio", "bright"), ("pore_horizontal_alignment", "pore")):
        sub = clean.dropna(subset=[k]).sort_values(k)
        selections += [(sub.iloc[0], k, phase, "low"), (sub.iloc[-1], k, phase, "high")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6))
    records = []
    for ax, (r, k, phase, side) in zip(axes.ravel(), selections):
        bse = features.trimmed(features.load_image(str(ROOT / "Dataset" / r.batch), r.site))
        ct = components[(components.site == r.site) & (components.phase == phase) & ~components.touches_edge]
        ct = ct.sort_values("area", ascending=False).head(30)
        centre = ct.iloc[0]
        h, w = bse.shape
        y0 = int(np.clip(centre["centroid-0"] - 400, 0, h - 800))
        x0 = int(np.clip(centre["centroid-1"] - 800, 0, w - 1600))
        ax.imshow(bse[y0:y0+800, x0:x0+1600], cmap="gray", vmin=0, vmax=255)
        ct = components[(components.site == r.site) & (components.phase == phase) & ~components.touches_edge]
        ct = ct[(ct["centroid-0"] >= y0) & (ct["centroid-0"] < y0+800) &
                (ct["centroid-1"] >= x0) & (ct["centroid-1"] < x0+1600)]
        for _, c in ct.iterrows():
            x, y = c["centroid-1"] - x0, c["centroid-0"] - y0
            ax.plot(x, y, ".", color="#ffcc55", markersize=3)
            if c.aspect >= 1.5:
                dx = np.cos(c.angle_horizontal_rad) * c.major_axis_length * 0.35
                dy = np.sin(c.angle_horizontal_rad) * c.major_axis_length * 0.35
                ax.plot([x-dx, x+dx], [y-dy, y+dy], color="#5ddbe8", lw=0.9)
        if phase == "bright":
            all_centres = components[(components.site == r.site) & (components.phase == phase)][["centroid-0", "centroid-1"]].to_numpy()
            _, eligible, neighbours = guarded_nn(all_centres, bse.shape)
            for i in np.flatnonzero(eligible):
                a, b = all_centres[i], all_centres[neighbours[i]]
                if all(y0 <= p[0] < y0+800 and x0 <= p[1] < x0+1600 for p in (a, b)):
                    ax.plot([a[1]-x0, b[1]-x0], [a[0]-y0, b[0]-y0], color="#ffcc55", lw=0.5, alpha=0.7)
        ax.set_title(f"{r.batch}/{r.site}: {side} {k}\nsite value {r[k]:.3f}; trimmed crop x={x0}, y={y0}", fontsize=9)
        ax.set_xlim(-0.5, 1599.5)
        ax.set_ylim(799.5, -0.5)
        ax.set_axis_off()
        records.append(dict(batch=r.batch, site=r.site, kpi=k, value=r[k], selection=side, x0=x0, y0=y0,
                            crop_width=1600, crop_height=800, coordinates="trimmed BSE", cracked_known=bool(r.cracked_known)))
    fig.suptitle("Inspection examples: centroids, nearest neighbours (gold), major axes (cyan)\nExtremes selected after analysis; these crops do not validate a manufacturing defect.", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, hspace=0.30, wspace=0.03)
    save(fig, "geometry_examples.png")
    pd.DataFrame(records).to_csv(OUT / "evidence_examples.csv", index=False)


def image(name):
    data = base64.b64encode((OUT / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(name)}">'


def table(df):
    return '<div class="table">' + df.to_html(index=False, border=0, float_format=lambda x: f"{x:.3g}", escape=True) + '</div>'


def main():
    sites = pd.read_csv(OUT / "morphology_sites.csv")
    comparison = pd.read_csv(OUT / "pairwise_comparisons.csv")
    multi = pd.read_csv(OUT / "multivariate_comparisons.csv")
    corr = pd.read_csv(OUT / "acquisition_correlations.csv")
    predictive = pd.read_csv(OUT / "acquisition_predictability.csv")
    hist = pd.read_csv(OUT / "orientation_histograms.csv")
    floor_path = OUT / "spacing_area_floor_comparisons.csv"
    floor = pd.read_csv(floor_path) if floor_path.exists() else None
    effects(comparison)
    site_strips(sites, comparison)
    angle_distributions(sites, hist)
    evidence(sites)
    matched = comparison[(comparison.view == "quality_matched") & comparison.kpi.isin(MORPH)].copy()
    ranked = matched.assign(abs_effect=lambda x: x.shift_mad.abs()).sort_values("abs_effect", ascending=False)
    top = ranked.head(12)
    # A broad screen is exploratory even after this multiplicity context.
    holm_min = float(matched.p_holm_screen.min())
    strongest = ranked.iloc[0]
    strongest_available = top.dropna(subset=["threshold_sign_survives"])
    summary = dict(n_sites=int(len(sites)), n_morphology_kpis=len(MORPH),
                   min_holm_screen_morphology_matched=holm_min,
                   strongest_matched=dict(reference=strongest.reference, batch=strongest.batch, kpi=strongest.kpi,
                                          shift_mad=float(strongest.shift_mad), p_exploratory=float(strongest.p_exploratory),
                                          p_holm_screen=float(strongest.p_holm_screen),
                                          threshold_sign_survives=bool(strongest.threshold_sign_survives) if pd.notna(strongest.threshold_sign_survives) else None),
                   top12_with_threshold_band=len(strongest_available),
                   top12_sign_surviving=int(strongest_available.threshold_sign_survives.astype(bool).sum()),
                   max_acquisition_loo_r2=float(predictive.acquisition_loo_r2.max()),
                   acquisition_best_predictable_kpi=predictive.loc[predictive.acquisition_loo_r2.idxmax(), "kpi"],
                   interpretation="Exploratory morphology screen; no primary KPIs or verdict thresholds changed.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ntable = sites.groupby("batch").agg(total_sites=("site", "size"), low_contrast=("bright_low_contrast", "sum"),
                                       grey_pore=("grey_pore", "sum"), cracked_known=("cracked_known", "sum")).reset_index()
    qc = sites[~sites.bright_low_contrast & ~sites.grey_pore]
    ntable["quality_matched"] = ntable.batch.map(qc.groupby("batch").size()).fillna(0).astype(int)
    ordinary = comparison[(comparison.view == "ordinary_reference") & comparison.kpi.isin(MORPH)]
    compare_cols = ["reference", "batch", "kpi", "n_ref", "n_batch", "median_delta", "delta_ci_low", "delta_ci_high",
                    "shift_mad", "p_exploratory", "p_holm_screen", "loo_sign_share", "threshold_delta_low", "threshold_delta_high", "threshold_sign_survives"]
    cc = corr[(corr.view == "ordinary") & (corr.group == "pooled")].copy()
    cc["abs_rho"] = cc.spearman_rho.abs()
    cc = cc.sort_values("abs_rho", ascending=False).groupby("kpi", sort=False).head(1)
    floor_html = ("<p>After inspecting the overlays, spacing was audited at minimum component areas of 50, 500 and 2,000 pixels. "
                  "This post-screen measurement audit does not choose a preferred floor or run new significance tests.</p>" + table(floor)) if floor is not None else ""
    links = " · ".join(f'<a href="{n}">{n}</a>' for n in ("morphology_sites.csv", "pairwise_comparisons.csv", "multivariate_comparisons.csv",
              "threshold_variants.csv", "acquisition_correlations.csv", "acquisition_predictability.csv", "orientation_histograms.csv", "manifest.json"))
    conclusion = ("No morphology comparison survives the broad exploratory Holm screen." if holm_min >= 0.05 else
                  "Some morphology comparisons survive the exploratory Holm screen; they still require measurement and acquisition review.")
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>E18 morphology analysis</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{font:16px/1.55 system-ui;color:#23313b;background:#f3f5f6;margin:0}} main{{max-width:1200px;margin:auto;padding:28px;background:white}}
h1,h2{{line-height:1.2}}h2{{margin-top:32px}} .note{{background:#fff6dc;padding:16px;border-left:4px solid #d9a32e}}
img{{display:block;width:100%;height:auto;margin:18px 0}} .table{{overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}
td,th{{padding:7px;border-bottom:1px solid #dce2e5;text-align:left}} th{{background:#edf2f4}}code{{font-size:13px}}a{{color:#176398}}</style><main>
<h1>Morphology differences across the three supplier batches</h1>
<p>E18 · {len(sites)} sites · {len(MORPH)} morphology descriptors · exploratory analysis developed after inspecting Batches 1–3.</p>
<div class="note"><b>{conclusion}</b> This is a description of observed 2-D geometry, with sampling and segmentation sensitivity.
It does not change the five primary QC KPIs, set new release limits, or establish accuracy on the unseen batch.</div>
<h2>What is compared</h2>
<p>“Full usable” excludes low-contrast sites only for bright-phase measurements and retains flagged pore fallback sites.
“Quality matched” excludes low-contrast and grey-pore sites from both groups, retaining known cracked sites.
“Ordinary reference” additionally removes the three known cracked sites from Batch 3 only. It is a sensitivity view, not a cleaner truth.
Batch 3 remains the working reference; Batch 2 minus Batch 1 is also shown.</p>{table(ntable)}
<h2>Where differences appear</h2>{image('morphology_effects.png')}
<p>Shifts use each comparison’s reference MAD; values across panels are not directly interchangeable.
The largest effects were selected after analysis for inspection. Intervals below are approximate 95% bootstrap-over-site intervals,
conditional on independent sites. They omit specimen clustering and measurement error.</p>{table(top[compare_cols])}
<p>Unadjusted <code>p_exploratory</code> values are exploratory site-label permutation results for the Hodges–Lehmann shift.
Exact enumeration is used up to 20,000 allocations; otherwise 5,000 Monte Carlo allocations use the plus-one convention.
<code>p_holm_screen</code> corrects all tested KPIs and pairs within that view, including the existing material panel and signed profiles.
No secondary descriptor becomes a verdict driver through this screen. LOO sign share is sign stability after deleting each site from either group,
not decision stability or probability of correctness.</p>
<h2>Site distributions and reference heterogeneity</h2>{image('site_distributions.png')}
<h2>Do the expanded panels distinguish the batches?</h2>
<p>Energy distance uses the existing material panel and the exploratory morphology panel separately.
Each family (void, bright phase, texture, ETD) has equal total weight, so adding correlated descriptors does not give one family extra votes.
Reference median/MAD scaling is refitted inside every permutation. Complete-case counts are reported; these panel p-values are also exploratory.</p>
{table(multi[['view','reference','batch','panel','n_ref','n_batch','n_cols','statistic','p','n_perm']])}
<p>A nonsignificant comparison means insufficient evidence at these site counts, not equivalence or acceptance.
Only one observed production batch per folder is represented; site variation does not estimate variation across future production batches.</p>
<h2>Orientation evidence</h2>{image('orientation_distributions.png')}
<p>Orientations are axial (0° and 180° coincide), measured against image horizontal, not an assumed collector direction.
Only components with aspect ratio ≥1.5 have an orientation claim; components touching the image edge are excluded.
Horizontal alignment is area-weighted cos(2θ): +1 horizontal, −1 vertical. Alignment strength is the magnitude of the area-weighted axial moment.
Each site has equal weight in these batch curves; thousands of components never become thousands of independent observations.</p>
<h2>How sensitive are these measurements?</h2>
<p>Every mask-derived descriptor is recomputed at its site thresholds ±5 grey levels. The threshold interval for a median batch difference
uses the smallest batch median minus the largest reference median, and vice versa, across per-site measurement envelopes.
It is a conservative perturbation range, not a confidence interval. Texture metrics have no mask-threshold band.</p>
{table(ordinary.sort_values('shift_mad', key=lambda x: x.abs(), ascending=False).head(10)[compare_cols])}
<h2>Acquisition confounds</h2>
<p>Correlations are available for all sites, quality-matched sites, ordinary sites, and separately within each batch.
The table shows the largest absolute ordinary-site correlation for each descriptor. Correlation does not establish an acquisition cause.</p>
{table(cc[['kpi','covariate','n','spearman_rho']])}
<p>A fixed ridge regression (α=1) predicts each descriptor from frame height, bright-phase separation, BSE black level and ETD boundary sharpness.
Scaling and regression are refitted in each held-out-site fold on the ordinary sites. LOO R² is descriptive predictive ability;
negative values mean worse than a constant mean. It does not tell us what fraction of material change is caused by the instrument.</p>
{table(predictive.sort_values('acquisition_loo_r2', ascending=False))}
<h2>Geometry to review on images</h2>{image('geometry_examples.png')}
<p>Yellow marks are component centroids, gold lines join eligible nearest centroids, and cyan lines are major axes of elongated components in the trimmed BSE frame.
Spacing is a centroid-pattern descriptor: measured nearest-neighbour distances divided by the mean from 200 uniform random point patterns
with the same count and rectangular window, applying the same boundary censoring. Real finite particles cannot overlap;
therefore a ratio above one is not proof of dispersion, and a ratio below one is not proof of harmful agglomeration.
The original 50-pixel component floor also retains small bright fragments around particle rims; the spacing result is not yet a validated additive-particle arrangement measurement.
Phase identity, segmentation merging and the 2-D sectioning process still matter. The full object table retains coordinates and edge flags.</p>
{floor_html}
<h2>Definitions, limits and reproduction</h2>
<p>Size-width descriptors use area-weighted equivalent-diameter quantiles. Aspect ratio is major/minor second-moment axis length;
the bright aspect summary excludes edge-clipped components. Profile slopes use ten equal image-depth bins; absolute slopes enter the morphology panel
because collector orientation is unconfirmed. No raw intensity is used as a material descriptor. BSE texture remains contrast sensitive.
Lengths are pixels. Specimen independence, phase chemistry and tolerance limits are unresolved.</p>
<p>Component geometry follows <a href="https://scikit-image.org/docs/stable/api/skimage.measure.html">scikit-image region properties</a>;
centroid distances use <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html">SciPy nearest-neighbour queries</a>.
The analysis version and SHA-256 input/code hashes are recorded in the manifest; no verdict thresholds were retuned.</p>
<p><code>/opt/anaconda3/bin/python3 -m analysis.morphology.run_analysis</code><br>
<code>/opt/anaconda3/bin/python3 -m analysis.morphology.make_report</code></p><p>{links}</p></main></html>'''
    (OUT / "report.html").write_text(text)
    print(json.dumps(summary, indent=2))
    print(f"Report: {OUT / 'report.html'}")


if __name__ == "__main__":
    main()
