#!/usr/bin/env python3
"""Audit fixed B01/B04 candidate geometry on all sites; raw data/caches untouched.

Run from repository root with /opt/anaconda3/bin/python3. Outputs are reserved
under analysis/battery/output/neighbourhood_*. Sites, not components/windows,
are the statistical unit. No verdicts or acceptance thresholds are generated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from polaron_qc.battery_metrics import (  # noqa: E402
    BATTERY_METRIC_VERSION, KPI_NAMES, KPI_DEFINITIONS, DIAGNOSTIC_NAMES,
    measure_neighbourhoods, threshold_sensitivity,
)
from polaron_qc.features import (  # noqa: E402
    FEATURE_VERSION, bright_bands, list_sites, load_image, phase_thresholds, smooth_bse,
)

OUT = ROOT / "analysis/battery/output"
SEED = 20261003
BOOTSTRAPS = 4000
COLORS = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a"}
EXAMPLES = (("Batch_2", "3806gxp0"), ("Batch_3", "hzumfsms"),
            ("Batch_1", "4ih2ggld"), ("Batch_3", "71vgq3fw"))
ACQUISITION = ("H", "bright_sep", "bse_p1", "bse_p50", "bse_std", "bse_empty_bin_frac",
               "etd_boundary_sharpness", "etd_curtain_frac", "inlens_p50")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quality(table, kpi, ordinary=False):
    ok = np.isfinite(pd.to_numeric(table[kpi], errors="coerce"))
    for flag, desired in KPI_DEFINITIONS[kpi]["required_flags"].items():
        # A missing/unknown quality flag cannot certify a usable mask.
        ok &= table[flag].eq(desired).fillna(False) if flag in table else False
    if ordinary:
        ok &= ~table["cracked_known"].fillna(True)
        ok &= table["grey_pore"].eq(False).fillna(False)
    return np.asarray(ok, dtype=bool)


def interval_median(a, b=None, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    if not a.size or (b is not None and not len(b)):
        return np.nan, np.nan
    am = np.median(a[rng.integers(0, len(a), size=(BOOTSTRAPS, len(a)))], axis=1)
    if b is not None:
        b = np.asarray(b, dtype=float)
        bm = np.median(b[rng.integers(0, len(b), size=(BOOTSTRAPS, len(b)))], axis=1)
        am = bm-am
    return tuple(np.quantile(am, [.025, .975]))


def overlay(batch, site, a, pore, bright, details, top, flags):
    comps = details["components"]
    valid = comps[~comps.edge_clipped & comps.complete_ring]
    # Deterministic example: component nearest trimmed-frame centre, never one
    # selected to maximise between-batch or mask agreement.
    if valid.empty:
        cy, cx = np.asarray(a.shape)//2
    else:
        score = ((valid.y0+valid.y1)/2-a.shape[0]/2)**2 + ((valid.x0+valid.x1)/2-a.shape[1]/2)**2
        c = valid.loc[score.idxmin()]
        cy, cx = int((c.y0+c.y1)/2), int((c.x0+c.x1)/2)
    y0 = min(max(cy-256, 0), max(a.shape[0]-512, 0))
    x0 = min(max(cx-256, 0), max(a.shape[1]-512, 0))
    y1, x1 = min(y0+512, a.shape[0]), min(x0+512, a.shape[1])
    sl = (slice(y0, y1), slice(x0, x1))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(a[sl], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Real BSE crop")
    axes[1].imshow(a[sl], cmap="gray", vmin=0, vmax=255)
    rgba = np.zeros((*pore[sl].shape, 4))
    rgba[pore[sl]] = (0, .8, 1, .75); rgba[bright[sl]] = (1, .55, .05, .70)
    axes[1].imshow(rgba)
    lab = details["labels"]
    if not valid.empty:
        comp = lab[sl] == int(c.component_id)
        # Draw complete ring from full object, then crop (avoids crop truncation).
        r = 16; sy = slice(int(c.y0)-r, int(c.y1)+r); sx = slice(int(c.x0)-r, int(c.x1)+r)
        d = ndi.distance_transform_edt(lab[sy, sx] != int(c.component_id))
        ring_full = np.zeros_like(bright)
        ring_full[sy, sx] = (d > 0) & (d <= r)
        ring_overlay = np.zeros((*pore[sl].shape, 4)); ring_overlay[ring_full[sl]] = (.95, .1, .8, .3)
        axes[1].imshow(ring_overlay)
        axes[1].contour(comp, levels=[.5], colors=["#fff400"], linewidths=.8)
        info = f"ID {int(c.component_id)}: void distance {c.void_distance_px:.2f} px; ring pore {c.ring_void_frac:.3f}\nring pixels: void {int(c.ring_void_pixels)}, other bright {int(c.ring_other_bright_pixels)}, residual {int(c.ring_residual_solid_pixels)}"
    else:
        info = "No retained component with complete 16 px ring"
    axes[1].set_title("Pore cyan / bright orange / ring magenta")
    axes[2].imshow(a, cmap="gray", vmin=0, vmax=255, aspect="auto")
    windows = details["windows"]
    for row in windows[windows.scale_px == 1024].itertuples():
        axes[2].add_patch(plt.Rectangle((row.x0, row.y0), 1024, 1024, fill=False, edgecolor="#ffa400", linewidth=.7))
        axes[2].text(row.x0+512, row.y0+512, f"{row.bright_frac:.3f}", fontsize=6, color="white", ha="center",
                     bbox=dict(facecolor="black", alpha=.35, pad=.5))
    axes[2].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#fff400", linewidth=1.3))
    axes[2].set_title("1024 px windows: bright fractions")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{batch}/{site} — raw y={top+y0}:{top+y1}, x={x0}:{x1}; trimmed y={y0}:{y1}\n{flags}; geometry unreviewed", fontsize=10)
    fig.text(.01, .01, info+"\nRaster adjacency ≠ electrical contact; local void rings ≠ 3-D expansion volume. Partial windows / clipped components excluded.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .9))
    path = OUT / f"neighbourhood_overlay_{batch}_{site}.png"
    fig.savefig(path, dpi=160); plt.close(fig)
    return dict(batch=batch, site=site, file=path.relative_to(ROOT).as_posix(),
                raw_y0=int(top+y0), raw_y1=int(top+y1), x0=int(x0), x1=int(x1),
                trimmed_y0=int(y0), trimmed_y1=int(y1), flags=flags)


def summaries(sites):
    rows, comparisons, correlations, predictions = [], [], [], []
    variable = [k for k in KPI_NAMES if sites.loc[quality(sites, k), k].nunique() > 1]
    for ik, k in enumerate(KPI_NAMES):
        for view in ("quality_gated", "ordinary_sensitivity"):
            ok = quality(sites, k, ordinary=view == "ordinary_sensitivity")
            usable = sites.loc[ok]
            for ib, batch in enumerate(sorted(sites.batch.unique())):
                t = usable[usable.batch == batch]
                v = t[k].to_numpy()
                lo, hi = interval_median(v, seed=SEED+ik*30+ib)
                rows.append(dict(kpi=k, view=view, batch=batch, n_sites=len(t), median=float(np.median(v)) if len(v) else np.nan,
                                 median_ci_low=lo, median_ci_high=hi,
                                 threshold_min_median=t[k+"_sensitivity_min"].median(),
                                 threshold_max_median=t[k+"_sensitivity_max"].median(),
                                 pore_fallback_sites=int((~t.pore_mode_resolved).sum())))
            for ref, batch in (("Batch_3", "Batch_1"), ("Batch_3", "Batch_2"), ("Batch_1", "Batch_2")):
                a, b = usable[usable.batch == ref], usable[usable.batch == batch]
                lo, hi = interval_median(a[k], b[k], seed=SEED+ik*40+len(comparisons))
                # Conservative independent per-site endpoint envelope. NOT a CI
                # and NOT a directly measured single shared threshold setting.
                tl = b[k+"_sensitivity_min"].median()-a[k+"_sensitivity_max"].median()
                th = b[k+"_sensitivity_max"].median()-a[k+"_sensitivity_min"].median()
                comparisons.append(dict(kpi=k, view=view, reference=ref, batch=batch, n_reference=len(a), n_batch=len(b),
                    median_difference=b[k].median()-a[k].median(), difference_ci_low=lo, difference_ci_high=hi,
                    threshold_difference_min=tl, threshold_difference_max=th,
                    globally_variable=k in variable))
        usable = sites.loc[quality(sites, k)]
        if k not in variable:
            continue
        groups = [("pooled", usable)] + [(b, usable[usable.batch == b]) for b in sorted(sites.batch.unique())]
        groups.append(("Batch_3_ordinary", sites.loc[quality(sites, k, ordinary=True) & sites.batch.eq("Batch_3")]))
        for group, t in groups:
            for cov in ACQUISITION:
                valid = t[[k, cov]].replace([np.inf, -np.inf], np.nan).dropna()
                rho = np.nan
                if len(valid) >= 4 and valid[k].nunique() > 1 and valid[cov].nunique() > 1:
                    rho = float(spearmanr(valid[k], valid[cov]).statistic)
                correlations.append(dict(kpi=k, group=group, acquisition_covariate=cov, n_sites=len(valid), rho=rho))
        covs = [c for c in ACQUISITION if usable[c].notna().any()]
        x, y = usable[covs].to_numpy(dtype=float), usable[k].to_numpy(dtype=float)
        pred = np.full(len(y), np.nan)
        if len(y) >= 5:
            for i in range(len(y)):
                train = np.arange(len(y)) != i
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
                model.fit(x[train], y[train]); pred[i] = model.predict(x[i:i+1])[0]
        finite = np.isfinite(pred)
        r2 = 1-np.sum((y[finite]-pred[finite])**2)/np.sum((y[finite]-y[finite].mean())**2) if finite.sum() >= 5 and np.ptp(y[finite]) else np.nan
        for i, row in enumerate(usable.itertuples()):
            predictions.append(dict(kpi=k, batch=row.batch, site=row.site, observed=y[i], acquisition_only_prediction=pred[i],
                                    pooled_loo_r2=r2, n_sites=len(y), covariates=";".join(covs)))
    return pd.DataFrame(rows), pd.DataFrame(comparisons), pd.DataFrame(correlations), pd.DataFrame(predictions), variable


def findings(sites, summary, comparison, confounds, predictions, variable, manifest):
    constant = [k for k in KPI_NAMES if k not in variable]
    lines = ["# Fixed battery neighbourhood candidates (E25)", "",
        "Fresh graphite–Si/SiOx material is user-confirmed; pixelwise identity and these measurement masks remain expert-unvalidated. These are secondary, exploratory measurements, not QC verdict inputs or performance predictions.", "",
        "Methods were prespecified before this run: existing Gaussian/histogram/opening masks; 8-connected bright components >=50 px; image-clipped components excluded; 4-neighbour raster boundary faces; Euclidean 16 px external rings; complete top-left 512/1024 px windows. Rings classify void, other bright and residual solid separately. Window/components are summarised per site and never add material n.", "",
        "Quality views exclude low-contrast sites for all quantities and data-derived grey-pore sites for joint quantities. The separate ordinary sensitivity also excludes the three previously identified long-void reference sites. Missing flags are unusable. All31 nominal pore_mode_resolved flags are false: this is the common fallback-threshold method, not evidence that all pore masks are unusable or accurate. Grey-pore exclusion does not validate ordinary masks.", "",
        "Intervals are 4000-resample percentile site-bootstrap intervals under a working site-independence assumption; specimen independence remains unknown. Threshold envelopes perturb both thresholds together by -5/0/+5 gray levels, keep opening fixed, and are not confidence intervals or an exhaustive independent threshold grid. They do not sample uncertainty in phase chemistry, sample preparation or the mask definition.", "",
        f"Run time: {manifest['elapsed_seconds']:.1f} s; {len(sites)} sites. Raw BSE hashes and input caches checked unchanged. No cache/dataset writes.", "",
        "## Coverage and definitions", "",
        f"Excluded from between-batch comparisons because constant/non-variable on usable sites: {', '.join(constant) or 'none'}.", "",
        "| KPI | units | usable sites B1/B2/B3 | median B1/B2/B3 |", "|---|---|---|---|",
    ]
    for k in KPI_NAMES:
        t = summary[(summary.kpi == k) & summary.view.eq("quality_gated")].sort_values("batch")
        counts = "/".join(str(n) for n in t.n_sites)
        med = "/".join(f"{v:.5g}" for v in t["median"])
        lines.append(f"| `{k}` | {KPI_DEFINITIONS[k]['units']} | {counts} | {med} |")
    lines += ["", "Component coverage (median site counts by folder; counts are diagnostics):", "",
              "| batch | retained | edge-clipped | used | full rings | incomplete rings |", "|---|---|---|---|---|---|"]
    for batch, t in sites.groupby("batch", sort=True):
        keys = ("neighbourhood_bright_components_retained", "neighbourhood_bright_components_clipped", "neighbourhood_bright_components_used", "neighbourhood_ring_components_used", "neighbourhood_ring_components_clipped")
        lines.append("| "+batch+" | "+" | ".join(f"{t[k].median():g}" for k in keys)+" |")
    lines += ["", "## Exploratory differences and acquisition sensitivity", "",
              "Every pair and both quality views are in neighbourhood_comparisons.csv. Below, differences are batch minus reference in raw units; bootstrap intervals and threshold bands describe different uncertainty sources. No acceptance/equivalence or significance claim is made.", "",
              "| KPI | pair | difference | site-bootstrap 95% interval | threshold endpoint envelope |", "|---|---|---|---|---|"]
    for row in comparison[comparison.view.eq("quality_gated") & comparison.globally_variable].itertuples():
        lines.append(f"| `{row.kpi}` | {row.batch} − {row.reference} | {row.median_difference:.4g} | [{row.difference_ci_low:.4g}, {row.difference_ci_high:.4g}] | [{row.threshold_difference_min:.4g}, {row.threshold_difference_max:.4g}] |")
    lines += ["", "Acquisition-only leave-one-site-out ridge predictions (fixed alpha=1, within-fold imputation/scaling; pooled covariates may also encode batch/session):", "",
              "| KPI | n sites | held-out R² | strongest within-batch |rho| (n≥4) |", "|---|---|---|---|"]
    for k in variable:
        p = predictions[predictions.kpi.eq(k)]
        c = confounds[confounds.kpi.eq(k) & confounds.group.isin(["Batch_1", "Batch_2", "Batch_3"])].dropna(subset=["rho"])
        text = "unavailable"
        if len(c):
            r = c.loc[c.rho.abs().idxmax()]
            text = f"{r.group}, {r.acquisition_covariate}: {r.rho:.3f} (n={r.n_sites})"
        lines.append(f"| `{k}` | {len(p)} | {p.pooled_loo_r2.iloc[0]:.3f} | {text} |")
    lines += ["", "Correlations/prediction are confound screens, not attribution. Weak or negative held-out prediction does not establish acquisition invariance; strong prediction is grounds to investigate the measurement and spatial sampling. Single-section chance placement, clipped-object selection, ring overlap, mean bright loading and compositional closure can affect these quantities.", "",
              "## Interpretation limits and next review", "",
              "Zero exact boundary faces can result from the residual-solid gray-level transition between widely separated pore/bright thresholds, especially after Gaussian smoothing/opening. It cannot establish full binder/electronic contact. Bright-to-void distance and rings retain measured geometric variation without treating that transition as a material contact network.", "",
              "Local bright dispersion describes mask heterogeneity at two chosen pixel scales. A bright–pore correlation is a section association, not causation or three-dimensional co-location. A visible external void ring does not prove available expansion volume for Si/SiOx; use chemical validation, matched section/preparation and reviewed annotations before mechanics claims.", "",
              "Review phase boundaries and representative neighbourhoods in the four real overlays, then test changed definitions against independent annotations and process/electrochemical outcomes. Keep source masks and excluded objects visible; do not select new scales to maximise folder separation. No morphology accuracy is claimed with zero expert reference masks.", "",
              "Applied pre-presentation checks: C01/C02 numeric and constant-feature audit; C03 acquisition/within-batch/held-out screen; C04/C05/C13 observable and wording scope; C06/C09/C14 site units and quality counts; C07/C21 no acceptance or primary promotion; C16 exploratory provenance; C22 suppress non-variable KPI comparisons; C28 source coordinates/excluded objects and unreviewed masks.", ""]
    (OUT / "neighbourhood_findings.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cache_paths = [ROOT/"analysis_cache"/name for name in ("site_features.csv", "acquisition_sites.csv")]
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in cache_paths}
    source_paths = [Path(__file__), ROOT/"polaron_qc/battery_metrics.py", ROOT/"polaron_qc/features.py"]
    sources_before = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    base = pd.read_csv(cache_paths[0]); acq = pd.read_csv(cache_paths[1])
    acq = acq.drop(columns=[k for k in acq if k in base and k not in ("batch", "site")])
    context = base.merge(acq, on=["batch", "site"], validate="one_to_one").set_index(["batch", "site"])
    site_rows, variant_rows, component_rows, window_rows, overlays = [], [], [], [], []
    raw_hashes = {}
    for folder in sorted((ROOT/"Dataset").glob("Batch_*")):
        for site in list_sites(folder):
            batch = folder.name
            t = time.perf_counter()
            path = folder / f"img_{site}_BSE.tif"
            raw_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            raw = load_image(folder, site)
            top, bottom = bright_bands(raw)
            a = raw[top:len(raw)-bottom if bottom else None]
            sm = smooth_bse(a); th = phase_thresholds(sm); th.pop("hist")
            pore = ndi.binary_opening(sm < th["th_lo"]); bright = ndi.binary_opening(sm > th["th_hi"])
            example = (batch, site) in EXAMPLES
            result = measure_neighbourhoods(pore, bright, return_details=True)
            f, details = result
            envelope, variants = threshold_sensitivity(sm, th["th_lo"], th["th_hi"], nominal=f, return_variants=True)
            source = context.loc[(batch, site)].to_dict()
            assert np.isclose(pore.mean(), source["pore_frac"], rtol=0, atol=1e-10), (batch, site, "pore mask drift")
            assert np.isclose(bright.mean(), source["bright_frac"], rtol=0, atol=1e-10), (batch, site, "bright mask drift")
            site_rows.append({"batch": batch, "site": site, **source, **th,
                              "trim_top": top, "trim_bottom": bottom, **f, **envelope})
            variant_rows.extend(dict(batch=batch, site=site, **v) for v in variants)
            component_rows.append(details["components"].assign(batch=batch, site=site))
            window_rows.append(details["windows"].assign(batch=batch, site=site))
            if example:
                flags = f"bright_low_contrast={source['bright_low_contrast']}; grey_pore={source['grey_pore']}; pore_mode_resolved={th['pore_mode_resolved']}"
                overlays.append(overlay(batch, site, a, pore, bright, details, top, flags))
            print(f"{batch}/{site} {time.perf_counter()-t:.2f}s; retained {f['neighbourhood_bright_components_retained']}, clipped {f['neighbourhood_bright_components_clipped']}", flush=True)
            del raw, a, sm, pore, bright, details, result
    sites = pd.DataFrame(site_rows)
    assert len(sites) == len(context) == 31
    summary, comparison, confounds, predictions, variable = summaries(sites)
    # Constant values remain in site tables/definitions for audit, not a
    # between-batch test. Suppress their comparison rows (C22).
    comparison = comparison[comparison.globally_variable].reset_index(drop=True)
    for name, frame in (("sites", sites), ("threshold_variants", pd.DataFrame(variant_rows)),
                        ("components", pd.concat(component_rows, ignore_index=True)),
                        ("windows", pd.concat(window_rows, ignore_index=True)),
                        ("summary", summary), ("comparisons", comparison),
                        ("acquisition_correlations", confounds), ("acquisition_predictions", predictions)):
        frame.to_csv(OUT/f"neighbourhood_{name}.csv", index=False)
    caches_after = {p.relative_to(ROOT).as_posix(): sha(p) for p in cache_paths}
    raw_after = {name: sha(ROOT/name) for name in raw_hashes}
    assert inputs == caches_after and raw_hashes == raw_after, "read-only input changed during run"
    manifest = dict(experiment="E25", status="secondary_exploratory_unvalidated", seed=SEED, bootstrap_resamples=BOOTSTRAPS,
        statistical_unit="site; specimen independence unconfirmed", material_state="fresh graphite-Si/SiOx confirmed by user",
        battery_metric_version=BATTERY_METRIC_VERSION, feature_version_at_import=FEATURE_VERSION,
        input_cache_sha256=inputs, raw_bse_sha256=raw_hashes, input_hashes_unchanged=True,
        source_sha256_at_start=sources_before, source_sha256_at_end={p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
        source_masks_match_cached_fractions=True, threshold_offsets_paired=[-5,0,5], ring_radius_px=16,
        window_sizes_px=[512,1024], min_bright_area_px=50, primary_kpis_unchanged=True,
        pore_mode_resolved_counts=sites.pore_mode_resolved.value_counts().to_dict(),
        variable_kpis=variable, constant_kpis=[k for k in KPI_NAMES if k not in variable],
        definitions=KPI_DEFINITIONS, diagnostics=DIAGNOSTIC_NAMES, overlays=overlays,
        elapsed_seconds=time.perf_counter()-start)
    # JSON booleans cannot be mapping keys under strict serialization.
    manifest["pore_mode_resolved_counts"] = {str(k): int(v) for k,v in manifest["pore_mode_resolved_counts"].items()}
    (OUT/"neighbourhood_manifest.json").write_text(json.dumps(manifest, indent=2))
    findings(sites, summary, comparison, confounds, predictions, variable, manifest)
    print(f"Completed {len(sites)} sites in {manifest['elapsed_seconds']:.1f}s; variable KPIs {variable}", flush=True)


if __name__ == "__main__":
    main()
