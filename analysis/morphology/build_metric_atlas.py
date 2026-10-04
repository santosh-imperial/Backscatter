"""Build the visual metric atlas from the canonical morphology register.

Run from the repository root:
    /opt/anaconda3/bin/python3 -m analysis.morphology.build_metric_atlas
    /opt/anaconda3/bin/python3 -m analysis.morphology.build_metric_atlas --verify-only

The atlas is a measurement reference, not a new batch comparison. Image crops are
illustrations; scalar values come only from saved full-site tables. Raw TIFFs are
read-only. New candidate visualisations are explicitly marked as illustrations.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
from html.parser import HTMLParser
import shutil
import subprocess
import os
from pathlib import Path
import re
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/polaron-metric-atlas-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label, find_contours, regionprops
from skimage.morphology import medial_axis, convex_hull_image, reconstruction

from analysis.morphology.run_analysis import ROOT, OUT, guarded_nn
from analysis.morphology.benchmark_methods import local_width
from polaron_qc import features
from polaron_qc.battery_metrics import measure_neighbourhoods
from polaron_qc.void_metrics import long_void_context

REGISTER = ROOT / "analysis/morphology/metric_register.json"
DEST = OUT / "metric_atlas.html"
ASSETS = OUT / "atlas_assets"
PALETTE = dict(pore="#54d7e4", bright="#ffc65b", crack="#f96791", seed="#88f6a8")
EXAMPLES = {
    "ordinary": ("Batch_1", "ffwubibz"),
    "bright": ("Batch_2", "3806gxp0"),
    "crack": ("Batch_3", "hzumfsms"),
    "reference": ("Batch_3", "xgj4xftb"),
    "grey": ("Batch_3", "71vgq3fw"),
    "low": ("Batch_1", "4ih2ggld"),
    "collector": ("Batch_2", "epqdaau9"),
    "comb": ("Batch_3", "ufdvpb81"),
}
TABLE_PATHS = [OUT / "morphology_sites.csv", ROOT / "analysis_cache/site_features.csv",
               ROOT / "analysis_cache/etd_inlens_features.csv", ROOT / "analysis_cache/physics_sites.csv",
               ROOT / "analysis/morphology/benchmark/local_width_sites.csv",
               ROOT / "analysis/battery/output/neighbourhood_sites.csv",
               ROOT / "analysis/battery/output/void_sites.csv",
               ROOT / "analysis/battery/output/void_threshold_envelopes.csv",
               ROOT / "analysis/ml_options/e_graph/sites.csv",
               ROOT / "analysis/ml_options/j_gabor/sites.csv",
               ROOT / "analysis/morphology/k_pilot/sites.csv",
               ROOT / "analysis/classification_m1_m2/output/appearance_features.csv"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar(v):
    if isinstance(v, (np.integer, np.floating)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


class Atlas:
    def __init__(self):
        self.tables = []
        self.input_hashes = {}
        for p in TABLE_PATHS:
            if p.exists():
                self.input_hashes[str(p.relative_to(ROOT))] = digest(p)
                table = pd.read_csv(p)
                if "threshold_offset" in table:
                    table = table[table.threshold_offset == 0].copy()
                self.tables.append(table)
        self.components = pd.read_csv(OUT / "component_geometry.csv.gz")
        self.histograms = pd.read_csv(OUT / "orientation_histograms.csv")
        self.sites = self.tables[0]
        self.cache = {}
        self.assets = {}
        self.selected_sources = {}
        ASSETS.mkdir(parents=True, exist_ok=True)

    def row(self, batch, site):
        result = {}
        # Earlier tables have priority: morphology output includes data-derived flags.
        for t in reversed(self.tables):
            selected = t[(t.batch == batch) & (t.site == site)]
            if len(selected):
                result.update({k: scalar(v) for k, v in selected.iloc[0].items()})
        if "crack_g_0.4" in result:
            result.setdefault("etd_crack_density_graphite", result["crack_g_0.4"])
        return result

    def load(self, example):
        if example in self.cache:
            return self.cache[example]
        batch, site = EXAMPLES[example]
        raw = features.load_image(ROOT / "Dataset" / batch, site)
        top, bottom = features.bright_bands(raw)
        bse = raw[top:len(raw)-bottom if bottom else None]
        row = self.row(batch, site)
        pore, bright, sm = features.segment(bse, row["th_lo"], row["th_hi"])
        ct = self.components[(self.components.batch == batch) & (self.components.site == site)].copy()
        data = dict(batch=batch, site=site, row=row, raw=raw, bse=bse, pore=pore, bright=bright, sm=sm,
                    top=top, bottom=bottom, ct=ct, labels={"pore": label(pore), "bright": label(bright)})
        self.cache[example] = data
        for det in ("BSE",):
            path = features._image_path(ROOT / "Dataset" / batch, site, det)
            self.selected_sources[str(Path(path).relative_to(ROOT))] = digest(path)
        return data

    def channel(self, d, det):
        if det not in d:
            raw = features.load_image(ROOT / "Dataset" / d["batch"], d["site"], det)
            d[det] = raw[d["top"]:len(raw)-d["bottom"] if d["bottom"] else None]
            path = features._image_path(ROOT / "Dataset" / d["batch"], d["site"], det)
            self.selected_sources[str(Path(path).relative_to(ROOT))] = digest(path)
        return d[det]

    def window(self, d, phase="bright", largest_axis=False, centre=None, width=1100, height=650):
        ct = d["ct"][d["ct"].phase == phase]
        use = ct[~ct.touches_edge]
        if use.empty:
            use = ct
        if centre is not None:
            y,x=centre
        elif len(use):
            p = use.sort_values("major_axis_length" if largest_axis else "area", ascending=False).iloc[0]
            y, x = p["centroid-0"], p["centroid-1"]
        else:
            y, x = np.array(d["bse"].shape) / 2
        height, width = min(height, len(d["bse"])), min(width, d["bse"].shape[1])
        y0 = int(np.clip(y - height / 2, 0, len(d["bse"]) - height))
        x0 = int(np.clip(x - width / 2, 0, d["bse"].shape[1] - width))
        return (slice(y0, y0+height), slice(x0, x0+width)), dict(x=x0, y_trimmed=y0, y_raw=y0+d["top"],
                                                                       width=width, height=height)

    @staticmethod
    def image(ax, a, title):
        ax.imshow(a, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10, loc="left", pad=8)
        ax.set_axis_off()

    @staticmethod
    def overlay(ax, mask, color, alpha=.34):
        rgba = np.zeros((*mask.shape, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(color)
        rgba[..., 3] = mask * alpha
        ax.imshow(rgba)
        if mask.any() and (~mask).any():
            ax.contour(mask, levels=[.5], colors=[color], linewidths=.6)

    @staticmethod
    def components_in_crop(d, win, phase, edge=False):
        ct = d["ct"][d["ct"].phase == phase]
        if not edge:
            ct = ct[~ct.touches_edge]
        return ct[(ct["centroid-0"] >= win[0].start) & (ct["centroid-0"] < win[0].stop) &
                  (ct["centroid-1"] >= win[1].start) & (ct["centroid-1"] < win[1].stop)]

    def axes(self, ax, d, win, phase, elongated_only=False):
        ct = self.components_in_crop(d, win, phase)
        if elongated_only:
            ct = ct[ct.aspect >= 1.5]
        for _, p in ct.iterrows():
            x, y = p["centroid-1"]-win[1].start, p["centroid-0"]-win[0].start
            theta = p.angle_horizontal_rad
            dx, dy = np.cos(theta)*p.major_axis_length/2, np.sin(theta)*p.major_axis_length/2
            ax.plot([x-dx, x+dx], [y-dy, y+dy], color=PALETTE[phase], lw=1)
            if not elongated_only:
                dx, dy = -np.sin(theta)*p.minor_axis_length/2, np.cos(theta)*p.minor_axis_length/2
                ax.plot([x-dx, x+dx], [y-dy, y+dy], color="#a0afff", lw=.8)
            ax.plot(x, y, ".", color="white", ms=2)
        ax.set_xlim(-.5, win[1].stop-win[1].start-.5)
        ax.set_ylim(win[0].stop-win[0].start-.5, -.5)

    def save(self, name, fig, d, crop, legend, kind, note="", channel="BSE"):
        fig.patch.set_facecolor("#ffffff")
        for ax in fig.axes:
            ax.title.set_fontsize(10)
            ax.xaxis.label.set_fontsize(9)
            ax.yaxis.label.set_fontsize(9)
            ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=.015, right=.98, top=.86, bottom=.22 if kind in ("local_width","crack_local_width") else .19, wspace=.23)
        fig.text(.018, .028, textwrap.fill(legend, width=155), fontsize=8, color="#415567")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="jpg", dpi=140, pil_kwargs={"quality":88}, facecolor="white")
        plt.close(fig)
        data = buffer.getvalue()
        (ASSETS / f"{name}.jpg").write_bytes(data)
        self.assets[name] = dict(data="data:image/jpeg;base64,"+base64.b64encode(data).decode(),
                                 batch=d["batch"], site=d["site"], channel=channel, crop=crop,
                                 legend=legend, kind=kind, note=note, path=f"atlas_assets/{name}.jpg")
        return name

    def battery_details(self, d):
        if "battery_details" not in d:
            measured, detail = measure_neighbourhoods(d["pore"], d["bright"], return_details=True)
            for key, value in measured.items():
                if key in d["row"] and isinstance(value, (float, int)):
                    saved = d["row"][key]
                    assert (saved is None and not np.isfinite(value)) or np.isclose(value, saved, equal_nan=True), (d["site"], key)
            d["battery_details"] = detail
            for p in (ROOT / "polaron_qc/battery_metrics.py", ROOT / "analysis/battery/output/neighbourhood_manifest.json"):
                self.input_hashes[str(p.relative_to(ROOT))] = digest(p)
        return d["battery_details"]

    def render_battery(self, kind):
        """E25 geometry illustrated on source masks; saved scalars stay full-site."""
        is_void = kind.startswith("void_")
        d = self.load("crack" if is_void else "bright")
        fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4))
        note = "Crop illustrates predicted-mask geometry. Card values are saved full-site E25 measurements; expert review is pending."
        if is_void:
            metrics, detail = long_void_context(d["pore"], return_details=True)
            for key, value in metrics.items():
                if key in d["row"]:
                    saved = d["row"][key]
                    assert (saved is None and not np.isfinite(value)) or np.isclose(value, saved, equal_nan=True), (d["site"], key)
            p = ROOT / "polaron_qc/void_metrics.py"
            self.input_hashes[str(p.relative_to(ROOT))] = digest(p)
            ct = detail["components"]
            retained = ct[~ct.touches_edge]
            chosen = retained.sort_values("major_axis_length", ascending=False).iloc[0]
            win, crop = self.window(d, "pore", centre=(chosen["centroid-0"], chosen["centroid-1"]), width=900, height=500)
            self.image(axs[0], d["bse"][win], "Source crop centred on retained internal long void")
            self.image(axs[1], d["bse"][win], "Internal versus clipped full-site component labels")
            self.overlay(axs[1], detail["internal_mask"][win], PALETTE["pore"])
            self.overlay(axs[1], (detail["long_mask"] & ~detail["internal_mask"])[win], PALETTE["crack"])
            centres = (detail["depth_edges"][:-1]+detail["depth_edges"][1:])/2/(len(d["bse"])-1)
            axs[2].barh(centres, detail["internal_long_void_depth_area_share"], height=.075, color=PALETTE["pore"], label="Internal area share")
            axs[2].axhline(metrics["long_void_y_centroid_norm"], color="#4a658e", ls="--", label="Area-weighted row centroid")
            axs[2].set(xlabel="Share of internal long-void area", ylabel="Normalised image row", ylim=(1, 0), title="Full-site depth distribution; bins are not n")
            axs[2].legend(fontsize=7, loc="best")
            legend = "Cyan: internal >500 px major-axis voids. Pink: clipped long voids omitted from internal KPIs and local width. Image rows are not collector depth."
            if kind == "void_coverage":
                axs[2].clear()
                vals = [metrics["long_void_internal_area_share"], metrics["long_void_edge_clipped_area_share"]]
                axs[2].bar(["Internal", "Clipped"], vals, color=[PALETTE["pore"], PALETTE["crack"]])
                axs[2].set(ylabel="Share of all long-void area", ylim=(0, 1), title="Full-site observability; clipped area is excluded")
            note += " The original crack fraction includes clipped components. Internal burden and widths do not describe every visible cavity; collector geometry is unconfirmed."
            if kind in ("void_width_depth", "void_crack_width_depth", "void_width_depth_both"):
                path = ROOT / "analysis/battery/output/void_width_depth_profiles.csv"
                self.input_hashes[str(path.relative_to(ROOT))] = digest(path)
                profiles = pd.read_csv(path)
                profiles = profiles[(profiles.site == d["site"]) & (profiles.threshold_offset == 0)].sort_values("depth_bin")
                crack_only = kind == "void_crack_width_depth"
                column = "crack_width_d50_px" if crack_only else "void_width_d50_px"
                axs[2].clear()
                axs[2].plot(profiles.depth_mid_norm, profiles[column], "o-", color="#5f809a", lw=1.5, ms=4)
                if kind == "void_width_depth_both":
                    axs[2].lines[0].set_label("All retained voids")
                    axs[2].plot(profiles.depth_mid_norm, profiles.crack_width_d50_px, "s--", color="#dc7896", lw=1.2, ms=3, label="Retained long voids")
                    axs[2].legend(fontsize=7,loc="best")
                axs[2].set(xlabel="Normalised image row", ylabel="Retained medial width D50 (px)", title="Full-site width medians by image-depth bin")
                retained = detail["internal_mask"] if crack_only else np.isin(d["labels"]["pore"], d["ct"][(d["ct"].phase == "pore") & ~d["ct"].touches_edge].label)
                axs[1].clear();self.image(axs[1], d["bse"][win], "Full-site retained component boundaries on crop")
                self.overlay(axs[1], retained[win], PALETTE["pore"], .2)
                legend = "Cyan: retained non-edge-clipped void scope. Graph: saved nominal full-site width medians; undefined bins omitted. Depth bins are not independent sites."
                note += " Width is twice distance-to-solid on the deterministic medial axis. The slope fits at least three defined bin medians; image-depth trend is not calibrated thickness or transport."
        else:
            detail = self.battery_details(d)
            components, windows, labs = detail["components"], detail["windows"], detail["labels"]
            if kind == "battery_threshold_envelopes":
                win, crop = self.window(d, centre=np.array(d["bse"].shape)/2, width=900, height=500)
                self.image(axs[0], d["bse"][win], "Nominal source crop · threshold perturbation context")
                self.overlay(axs[0], d["pore"][win], PALETTE["pore"], .22)
                self.overlay(axs[0], d["bright"][win], PALETTE["bright"], .22)
                p_lo = ndi.binary_opening(d["sm"] < d["row"]["th_lo"]-5)
                p_hi = ndi.binary_opening(d["sm"] < d["row"]["th_lo"]+5)
                b_lo = ndi.binary_opening(d["sm"] > d["row"]["th_hi"]-5)
                b_hi = ndi.binary_opening(d["sm"] > d["row"]["th_hi"]+5)
                self.image(axs[1], d["bse"][win], "Mask pixels that vary across paired ±5 thresholds")
                self.overlay(axs[1], (p_lo ^ p_hi)[win], PALETTE["pore"], .65)
                self.overlay(axs[1], (b_lo ^ b_hi)[win], PALETTE["bright"], .65)
                path = ROOT / "analysis/battery/output/neighbourhood_threshold_variants.csv"
                self.input_hashes[str(path.relative_to(ROOT))] = digest(path)
                variants = pd.read_csv(path);variants = variants[variants.site == d["site"]].sort_values("threshold_offset")
                axs[2].plot(variants.threshold_offset, variants.bright_ring_void_frac_16px, "o-", color="#a078ba", lw=1.5)
                axs[2].set(xlabel="Paired pore + bright threshold offset", ylabel="Full-site median 16 px band pore fraction", xticks=[-5,0,5], title="One saved full-site measurement envelope")
                legend = "Cyan/gold: pore/bright predictions or changed pixels. Graph uses three saved full-site E25 variants; deterministic sensitivity is separate from site-bootstrap uncertainty."
                note += " Envelopes use paired ±5 offsets, not an exhaustive independent 3×3 grid. Finite-variant counts remain explicit; a sensitivity range is not a confidence interval."
            elif kind.startswith("battery_windows") or kind == "battery_coverage":
                scale = 1024 if "1024" in kind or kind == "battery_coverage" else 512
                ww = windows[windows.scale_px == scale]
                win, crop = self.window(d, centre=(scale/2, scale/2), width=scale, height=scale)
                self.image(axs[0], d["bse"][win], f"First complete {scale}×{scale} px source window")
                self.overlay(axs[0], d["bright"][win], PALETTE["bright"], .22)
                self.overlay(axs[0], d["pore"][win], PALETTE["pore"], .22)
                self.image(axs[1], d["bse"], f"Full-site {scale} px windows · partial edges omitted")
                max_frac = max(float(ww.bright_frac.max()), .01)
                cmap = matplotlib.colormaps["YlOrBr"]
                for row in ww.itertuples(index=False):
                    axs[1].add_patch(Rectangle((row.x0, row.y0), scale, scale, facecolor=cmap(row.bright_frac/max_frac), alpha=.65, ec="white", lw=.4))
                if kind == "battery_coverage":
                    counts = [int((~components.edge_clipped).sum()), int(components.edge_clipped.sum())]
                    axs[2].bar(["Usable", "Frame clipped"], counts, color=[PALETTE["bright"], PALETTE["crack"]])
                    axs[2].set(ylabel="Retained bright components", title="Component coverage · area ≥50 px²")
                    fraction = d["row"][f"neighbourhood_window_coverage_{scale}px"]
                    axs[2].text(.02, .98, f"{len(ww)} complete windows; frame coverage {fraction:.1%}", transform=axs[2].transAxes, va="top", fontsize=8)
                else:
                    axs[2].scatter(ww.bright_frac, ww.pore_frac, color="#698fc5", s=16, alpha=.8)
                    axs[2].set(xlabel="Window bright-mask area fraction", ylabel="Window pore-mask area fraction", title=f"{len(ww)} windows form ONE site measurement")
                    value = d["row"].get(f"local_bright_pore_spearman_{scale}px")
                    label_text = "Undefined association" if value is None else f"Saved full-site Spearman ρ={value:.3f}"
                    axs[2].text(.02, .98, label_text, transform=axs[2].transAxes, va="top", fontsize=8)
                legend = "Gold/cyan: predicted bright/void masks. Overview fill darkens with bright fraction; uncoloured partial edges are unsampled. Windows never increase site n."
                note += " Bright-fraction variability depends on loading, section sizes and window scale. Within-site association is not electrical contact or independent batch evidence."
            else:
                manifest_path = ROOT / "analysis/battery/output/neighbourhood_manifest.json"
                source = next(x for x in json.loads(manifest_path.read_text())["overlays"] if x["site"] == d["site"])
                win = (slice(source["trimmed_y0"], source["trimmed_y1"]), slice(source["x0"], source["x1"]))
                crop = dict(x=win[1].start, y_trimmed=win[0].start, y_raw=source["raw_y0"], width=win[1].stop-win[1].start, height=win[0].stop-win[0].start)
                use = components[(~components.edge_clipped) & components.complete_ring].copy()
                cy, cx = np.array(d["bse"].shape)/2
                use["centre_distance"] = ((use.y0+use.y1)/2-cy)**2 + ((use.x0+use.x1)/2-cx)**2
                selected = use.sort_values("centre_distance").iloc[0]
                assert selected.y0 > win[0].start+16 and selected.y1 < win[0].stop-16 and selected.x0 > win[1].start+16 and selected.x1 < win[1].stop-16
                obj = labs[win] == selected.component_id
                self.image(axs[0], d["bse"][win], "Same source crop as E25 neighbourhood review")
                self.image(axs[1], d["bse"][win], "One retained component · operation illustrated")
                self.overlay(axs[1], d["pore"][win], PALETTE["pore"], .3)
                self.overlay(axs[1], obj, PALETTE["bright"], .3)
                if kind.endswith("ring") or kind == "battery_neighbourhood":
                    dist = ndi.distance_transform_edt(~obj)
                    ring = (dist > 0) & (dist <= 16)
                    assert int(ring.sum()) == int(selected.ring_pixels)
                    self.overlay(axs[1], ring, "#c079ed", .38)
                    vals = use.ring_void_frac.dropna()
                    axs[2].hist(vals, bins=20, color="#ad8ac9")
                    axs[2].set(xlabel="Component 16 px band pore fraction", ylabel="Retained components", title="Full-site component values; not statistical n")
                    legend = "Gold: selected bright component. Cyan: pore mask. Violet: external 16 px distance band, including threshold holes. Complete bands only; bands can overlap."
                elif kind.endswith("distance"):
                    dist, indices = ndi.distance_transform_edt(~d["pore"][win], return_indices=True)
                    vals = np.where(obj, dist, np.inf)
                    yy, xx = np.unravel_index(np.argmin(vals), vals.shape)
                    py, px = indices[:, yy, xx]
                    assert np.isclose(vals[yy,xx], selected.void_distance_px)
                    axs[1].annotate("", xy=(px,py), xytext=(xx,yy), arrowprops=dict(arrowstyle="<->", color="white", lw=1.4))
                    axs[1].text(.03,.97,f"Illustrated component minimum = {selected.void_distance_px:.2f} px",transform=axs[1].transAxes,ha="left",va="top",color="white",fontsize=8,bbox=dict(facecolor="black",alpha=.65,edgecolor="none"))
                    axs[2].hist(components.loc[~components.edge_clipped,"void_distance_px"].dropna(), bins=20, color="#7a95bd")
                    axs[2].set(xlabel="Minimum bright-to-pore distance (px)", ylabel="Retained components", title="Full-site component minima; median is site KPI")
                    legend = "Gold: selected retained bright component. Cyan: pore mask. White arrow joins nearest pixel centres. Minimum distance is geometric, not expansion clearance."
                else:
                    boundary = obj & ~ndi.binary_erosion(obj)
                    self.overlay(axs[1], boundary, "#fa98bd", .7)
                    axs[2].bar(["Pore-facing", "Residual-solid-facing"], [selected.void_faces/selected.boundary_faces, selected.residual_solid_faces/selected.boundary_faces], color=[PALETTE["pore"],"#9eadd7"])
                    axs[2].set(ylim=(0,1), ylabel="Selected component exposed raster-face fraction", title="Pore-facing fraction is constant zero on all31")
                    legend = "Pink: component boundary pixels (display only); measured boundary uses four-neighbour raster faces. Pore-facing fraction is uninformative on these threshold masks."
                    note += " Zero adjacency follows from separated bright/pore thresholds plus smoothing/opening; it cannot establish electrical contact or particle isolation."
        return self.save(kind, fig, d, crop, legend, kind, note)

    def render_gabor(self):
        """Saved four-window site summary; display one measured valid interior."""
        d = self.load("bright")
        source = ROOT / "analysis/ml_options/j_gabor"
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        crop = next(c for c in manifest["crops"] if (c["batch"], c["site"]) == (d["batch"], d["site"]))
        crop = {k: crop[k] for k in ("x", "y_raw", "y_trimmed", "width", "height")}
        y, x = crop["y_trimmed"], crop["x"]
        image = d["bse"][y:y+512, x:x+512]
        map_path = source / f'maps_{d["batch"]}_{d["site"]}.npz'
        energy_path = source / "window_energies.csv"
        maps = np.load(map_path)
        powers = pd.read_csv(energy_path)
        powers = powers[(powers.batch == d["batch"]) & (powers.site == d["site"]) & (powers["mode"] == "nominal")]
        pooled = powers.groupby("period_px").mean_power.mean()
        fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4))
        self.image(axs[0], image, "Measured quarter-frame BSE window")
        axs[0].add_patch(Rectangle((96,96),320,320,fill=False,color="#ffc65b",lw=1.4))
        axs[1].imshow(np.log1p(maps["coarse_power"]), cmap="magma", extent=(96,416,416,96))
        axs[1].set(xlim=(0,512),ylim=(512,0),title="64px energy · valid interior only")
        axs[1].set_xticks([]);axs[1].set_yticks([])
        axs[2].bar([str(int(p)) for p in pooled.index], pooled/pooled.sum(), color="#648acc")
        axs[2].set(xlabel="Fixed modulation period (px)",ylabel="Filter-energy share",title="Four-window site energy profile")
        for p in (map_path,energy_path,manifest_path):
            self.input_hashes[str(p.relative_to(ROOT))] = digest(p)
        legend = "Gold square: valid interior after 96px border removal. Middle: coarse Gabor image response, not a particle/defect mask. Bars pool four fixed windows equally; wavevectors are normal to stripes, not graphite plate axes."
        note = "Card scalars summarise four sampled windows, not the entire frame. Site coverage, contrast/resolution, acquisition and FFT/geometry/tensor controls are in the J audit. Expert validation is pending."
        return self.save("gabor_texture",fig,d,crop,legend,"gabor_texture",note)

    def render_k_morphology(self):
        """Fixed image reference and full-site shape/phase summaries from E30K."""
        d = self.load("bright")
        source = ROOT / "analysis/morphology/k_pilot"
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        example = next(c for c in manifest["crops"] if (c["batch"],c["site"]) == (d["batch"],d["site"]))
        crop = {k: example[k] for k in ("x","y_raw","y_trimmed","width","height")}
        path = source / example["image"]
        fig,ax = plt.subplots(figsize=(15,8.2))
        ax.imshow(plt.imread(path)); ax.set_axis_off()
        for p in (path,manifest_path):
            self.input_hashes[str(p.relative_to(ROOT))] = digest(p)
        legend = "Fixed central BSE crop; bright cyan, void red, eligible bright-object bounds gold. Histograms and x/y correlation bars use the full trimmed frame. Red dashed lines mark prespecified quantiles; arrows show image-axis lags, not collector coordinates."
        note = "Scalars use full-site saved K tables. Shape quantiles count components equally; raster phase correlations use finite signed-lag pair domains and their own marginals. Components/pixels are coverage, not n. All material/QC interpretation and expert validation are deferred."
        return self.save("k_morphology",fig,d,crop,legend,"k_morphology",note)

    def render(self, kind):
        if kind in self.assets:
            return kind
        if kind == "m2_appearance":
            # Reuse the measured source panel, not a segmentation proxy or invented map.
            directory = ROOT / "analysis/classification_m1_m2"
            path = directory / "assets/appearance_responses_3806gxp0.jpg"
            receipt = json.loads((directory / "output/appearance_extraction_receipt.json").read_text())
            panel = next(r for r in receipt["response_maps"] if r["path"] == str(path.relative_to(ROOT)))
            assert digest(path) == panel["sha256"]
            diag = pd.read_csv(directory / "output/appearance_diagnostics.csv")
            row = diag[diag.site == "3806gxp0"].iloc[0]
            y, x = json.loads(row.tile_origins_raw_yx)[4]
            for source in receipt["source_images"]["3806gxp0"].values():
                source_path = ROOT / source["path"]
                assert digest(source_path) == source["sha256"]
                self.selected_sources[source["path"]] = source["sha256"]
            self.input_hashes[str(path.relative_to(ROOT))] = digest(path)
            self.assets[kind] = dict(data="data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode(),
                batch="Batch_2", site="3806gxp0", channel="BSE + ETD + Inlens",
                crop=dict(x=x, y_raw=y, y_trimmed=y-int(row.trim_top), width=512, height=512),
                legend="Actual central tile: raw detector image, G2-G8 response and sigma8 gradient magnitude. Fine-band and gradient maps illustrate the filters, not phase labels. Other band energies and spatial IQR are defined in the evidence files.",
                kind=kind, note="Saved scalar pools nine fixed windows; measured union coverage is 14.6-19.3% across supplied crops. Image-gradient axes are not plate or coating axes. Gamma sensitivity remains; no classifier or QC promotion.",
                path=os.path.relpath(path, OUT))
            return kind
        if kind == "k_morphology":
            return self.render_k_morphology()
        if kind == "gabor_texture":
            return self.render_gabor()
        if kind.startswith("battery_") or kind in ("void_internal_context", "void_coverage", "void_width_depth", "void_crack_width_depth", "void_width_depth_both"):
            return self.render_battery(kind)
        phase = "pore" if "pore" in kind or kind in ("local_width", "crack_local_width", "tortuosity", "percolation") else "bright"
        example = "ordinary" if phase == "pore" else "bright"
        if kind in ("crack_mask", "void_span", "columns", "delamination", "crack_orientation", "local_width", "crack_local_width"):
            example, phase = "crack", "pore"
        elif kind in ("etd_curtain", "etd_orientation", "binder", "pending"):
            example = "grey"
        elif kind in ("fft", "correlation", "depth_profile_pore", "depth_profile_bright", "tortuosity", "percolation"):
            example = "reference"
        elif kind == "acquisition_histogram":
            example = "comb"
        elif kind == "low_contrast":
            example = "low"
        elif kind == "edge_band":
            example = "collector"
        d = self.load(example)
        win, crop = self.window(d, phase, largest_axis=example == "crack")
        a = d["bse"][win]
        fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4))
        self.image(axs[0], a, "BSE source crop · display range 0–255")
        note = "Crop illustrates the operation; values on the card come from saved full-site tables."
        legend = "Cyan: segmented void phase. Gold: segmented bright phase. No chemistry is inferred from these labels."
        channel = "BSE"
        if kind == "bright_graph":
            from matplotlib.collections import LineCollection
            graph_root = ROOT / "analysis/ml_options/e_graph"
            nodes = pd.read_csv(graph_root / "nodes.csv.gz")
            edges = pd.read_csv(graph_root / "edges.csv.gz")
            def select(t):
                return t[(t.batch == d["batch"]) & (t.site == d["site"]) &
                         (t.threshold_offset == 0) & (t.area_floor_px2 == 50)]
            nodes, edges = select(nodes).sort_values("node_id"), select(edges)
            points = nodes[["centroid-1", "centroid-0"]].to_numpy(float)
            self.image(axs[1], d["bse"], "Full-site graph cropped for display")
            segments = points[edges[["node_a", "node_b"]].to_numpy(int)]
            axs[1].add_collection(LineCollection(segments, colors="#49d7ed", linewidths=.65))
            axs[1].scatter(points[:,0], points[:,1], s=8, color=PALETTE["bright"])
            axs[1].set_xlim(crop["x"], crop["x"]+crop["width"])
            axs[1].set_ylim(crop["y_trimmed"]+crop["height"], crop["y_trimmed"])
            axs[2].hist(edges.length_px, bins=25, color="#668ddd")
            axs[2].axvline(edges.length_px.median(), color="#ffc65b", label="Median")
            axs[2].set(xlabel="Edge length (px)", ylabel="Unique undirected edges", title="Full-site graph distances")
            axs[2].legend(fontsize=8)
            legend="Gold: retained bright centroid; cyan: undirected union of three nearest distinct neighbours. Graph built on whole frame; retained components ≥50 px²; image-clipped components excluded."
            note="Four card scalars use full-site tables. Proximity is not physical/electrical contact; masks and meanings remain expert-unreviewed. Threshold/object-floor and acquisition/size/loading checks are in the graph report."
        elif kind in ("pore_mask", "bright_mask", "pore_count", "bright_count", "crack_mask", "graphite_mask", "void_span", "columns", "delamination", "crack_orientation"):
            mask = d[phase]
            color = PALETTE[phase]
            if kind in ("pore_count", "bright_count"):
                ct = d["ct"][d["ct"].phase == phase]
                mask = np.isin(d["labels"][phase], ct.label)
                legend = f"{phase.capitalize()} outlines: retained full-site components only; minimum area {30 if phase == chr(112)+chr(111)+chr(114)+chr(101) else 50} px². Count is divided by full-site area in Mpx."
            if kind == "graphite_mask":
                mask, color = ~(d["pore"] | d["bright"]), "#9eadd7"
            if kind in ("crack_mask", "void_span", "columns", "delamination", "crack_orientation"):
                ct = d["ct"][(d["ct"].phase == "pore") & (d["ct"].major_axis_length > features.CRACK_MAJOR_AXIS_PX)]
                mask = np.isin(d["labels"]["pore"], ct.label)
                color = PALETTE["crack"]
            self.image(axs[1], a, "Full-site labels overlaid on the crop")
            self.overlay(axs[1], mask[win], color)
            if kind in ("void_span", "crack_orientation"):
                self.axes(axs[1], d, win, "pore", elongated_only=True)
            axs[2].imshow(d["bse"], cmap="gray", vmin=0, vmax=255, aspect="auto")
            axs[2].set_title("Full site · crop position and masks", fontsize=10, loc="left")
            self.overlay(axs[2], mask, color, .45)
            axs[2].add_patch(Rectangle((crop["x"], crop["y_trimmed"]), crop["width"], crop["height"], fill=False, ec="white", lw=1))
            if kind in ("columns", "delamination"):
                yy = mask.any(axis=0)
                axs[2].fill_between(np.arange(len(yy)), 0, len(mask)-1, where=yy, color=color, alpha=.14)
                legend = "Pink: full-site void labels with major axis >500 px. Shaded columns intersect at least one selected void."
            elif kind in ("crack_mask", "void_span", "crack_orientation"):
                legend = "Pink: full-site void labels with major axis >500 px. Axes describe 2-D connected regions, not confirmed cracks in 3-D."
            axs[2].set_axis_off()
        elif kind in ("conditional_fractions", "conditional_mass"):
            self.image(axs[1],a,"Observed 2-D phase masks · model inputs")
            self.overlay(axs[1],d["pore"][win],PALETTE["pore"],.35)
            self.overlay(axs[1],d["bright"][win],PALETTE["bright"],.35)
            if kind == "conditional_fractions":
                vals=[d["row"]["pore_frac"],d["row"]["bright_frac"],d["row"]["graphite_frac"]]
                axs[2].bar(["Void","Bright","Residual solid"],vals,color=[PALETTE["pore"],PALETTE["bright"],"#9eadd7"])
                axs[2].set(ylabel="Observed 2-D area fraction",ylim=(0,1),title="Observed 2-D model inputs")
            else:
                keys=["nominal_mass_frac_additive_if_Si","nominal_mass_frac_additive_if_SiOx"]
                axs[2].bar(["If Si","If SiOx"],[d["row"][k] for k in keys],color=[PALETTE["bright"],"#db9e42"])
                axs[2].set(ylabel="Conditional bright-phase mass fraction",ylim=(0,.2),title="Conditional density conversion")
            legend="Cyan: measured void mask. Gold: measured bright mask. The 3-D and mass interpretations require unverified sampling, phase and density assumptions."
            note="The image shows measured 2-D model inputs. Any volume/mass scalar is conditional on unvalidated assumptions and excluded from verdicts."
        elif kind in ("pore_size", "bright_size", "pore_shape", "bright_shape", "bright_solidity", "bright_circularity"):
            self.image(axs[1], a, "Retained components · outlines and fitted axes")
            ct = d["ct"][d["ct"].phase == phase]
            retained = np.isin(d["labels"][phase], ct.label)
            self.overlay(axs[1], retained[win], PALETTE[phase], .10)
            if kind.endswith("size"):
                for _, p in self.components_in_crop(d, win, phase).sort_values("area", ascending=False).head(12).iterrows():
                    axs[1].add_patch(Circle((p["centroid-1"]-win[1].start, p["centroid-0"]-win[0].start),
                                            p.equivalent_diameter_area/2, fill=False, ec="white", lw=.8))
                ct = ct.sort_values("equivalent_diameter_area")
                axs[2].plot(ct.equivalent_diameter_area, ct.area.cumsum()/ct.area.sum(), color=PALETTE[phase], lw=2)
                for q in ([.5,.9] if phase == "pore" else [.1,.5,.9]):
                    cumulative=ct.area.cumsum()/ct.area.sum()
                    value=ct.equivalent_diameter_area.iloc[np.searchsorted(cumulative,q)]
                    axs[2].axvline(value,color="#82939f",ls="--",lw=.8)
                    axs[2].text(value,.04,f"D{int(q*100)}",rotation=90,fontsize=7,color="#526879",ha="right")
                axs[2].set(xlabel="Equivalent circle diameter (px)", ylabel="Cumulative component area", ylim=(0,1.02),
                           title="Full-site area-weighted size distribution")
                axs[2].grid(alpha=.18)
                legend = "White circles: same area as each component, not fitted particle boundaries. Curve: all retained full-site components."
            elif kind == "bright_solidity":
                for _, p in self.components_in_crop(d, win, phase).sort_values("area", ascending=False).head(8).iterrows():
                    m = d["labels"][phase][win] == int(p.label)
                    if m.any():
                        hull = convex_hull_image(m)
                        axs[1].contour(hull, [.5], colors=["#aaadff"], linewidths=.8)
                axs[2].hist(ct.solidity, bins=24, weights=ct.area/ct.area.sum(), color=PALETTE[phase])
                axs[2].set(xlabel="Area / convex-hull area", ylabel="Component-area share", title="Full-site solidity distribution")
                legend = "Gold: observed outline. Lavender: convex hull of the visible component; crop boundary can truncate the illustrated hull."
            else:
                self.axes(axs[1], d, win, phase)
                if kind == "bright_circularity":
                    vals = 4*np.pi*ct.area/np.maximum(ct.perimeter,1)**2
                    axs[2].set(xlabel="4π area / perimeter²", ylabel="Component-area share", title="Full-site circularity distribution")
                    legend = "Gold outlines show perimeter; circularity uses pixelated perimeter, not the ellipse fit. Axes are shown for geometric context."
                else:
                    vals = ct.aspect
                    axs[2].set(xlabel="Major / max(minor, 1) axis", ylabel="Component-area share", title="Full-site aspect distribution")
                    legend = "Gold: region major axis. Lavender: minor axis. These are moment-based region axes, not physical platelet dimensions."
                axs[2].hist(vals, bins=24, weights=ct.area/ct.area.sum(), color=PALETTE[phase])
        elif kind in ("pore_orientation", "bright_orientation"):
            self.image(axs[1], a, "Axial major directions · aspect ≥1.5")
            self.axes(axs[1], d, win, phase, elongated_only=True)
            hist = self.histograms[(self.histograms.batch == d["batch"]) & (self.histograms.site == d["site"]) & (self.histograms.phase == phase)]
            axs[2].bar(hist.angle_bin_deg, hist.area_share, width=13, color=PALETTE[phase])
            axs[2].set(xlabel="Axial angle from horizontal (degrees)", ylabel="Oriented component-area share", xlim=(0,180),
                       title="Full-site orientation distribution")
            axs[2].set_xticks([0,45,90,135,180])
            legend = "Horizontal=0°/180°, vertical=90°. Round and edge-clipped components are excluded. Orientation is axial (no arrow direction)."
        elif kind == "bright_spacing":
            self.image(axs[1], a, "Centroids and eligible nearest neighbours")
            ct = d["ct"][d["ct"].phase == "bright"]
            points = ct[["centroid-0", "centroid-1"]].to_numpy()
            distances, eligible, neighbours = guarded_nn(points, d["bse"].shape)
            for i in np.flatnonzero(eligible):
                p, q = points[i], points[neighbours[i]]
                if all(win[0].start <= z[0] < win[0].stop and win[1].start <= z[1] < win[1].stop for z in (p,q)):
                    axs[1].plot([p[1]-win[1].start,q[1]-win[1].start], [p[0]-win[0].start,q[0]-win[0].start], color=PALETTE["bright"], lw=.7)
                    axs[1].plot(p[1]-win[1].start,p[0]-win[0].start,".",color="white",ms=3)
            axs[2].hist(distances, bins=25, color=PALETTE["bright"])
            axs[2].axvline(d["row"]["bright_nn_csr_mean_px"], color="#668ddd", ls="--", label="Saved uniform-point mean")
            axs[2].set(xlabel="Guarded nearest-centroid distance (px)", ylabel="Retained component count", title="Full-site distance distribution")
            axs[2].legend(fontsize=8)
            legend = "Gold: nearest-centroid pair. Edge-censored on the full image, not the crop. Fragmentation strongly affects this comparator."
        elif kind in ("depth_profile_pore", "depth_profile_bright"):
            phase = "pore" if kind.endswith("pore") else "bright"
            axs[0].clear()
            self.image(axs[0], d["bse"], "Full trimmed BSE site · ten image-depth bins")
            axs[0].set_aspect("equal")
            for y in np.linspace(0,len(d["bse"]),11):
                axs[0].axhline(y,color=PALETTE[phase],lw=.5)
            axs[1].imshow(d[phase], cmap="gray", aspect="equal")
            axs[1].set_title(f"Full-site {phase} mask",fontsize=10,loc="left");axs[1].set_axis_off()
            profiles=[d["row"][f"profile_{phase}_{k}"] for k in range(10)]
            depth=np.arange(10)/10+.05
            axs[2].plot(profiles, depth, "o-", color=PALETTE[phase], lw=2)
            slope=d["row"].get(f"profile_{phase}_slope")
            axs[2].set(xlabel=f"{phase.capitalize()} area fraction", ylabel="Normalised image depth", ylim=(1,0), title="Saved full-site ten-bin profile")
            axs[2].grid(alpha=.18)
            crop = dict(x=0,y_trimmed=0,y_raw=d["top"],width=d["bse"].shape[1],height=len(d["bse"]))
            legend = "Depth follows image rows from top to bottom; separator/current-collector orientation is unconfirmed. Profile bins are not independent samples."
        elif kind in ("fft", "correlation"):
            # Exact documented first random window (seed=0), visualised alongside full-site summary curve.
            rng=np.random.default_rng(0);y,x=rng.integers(0,len(d["bse"])-512),rng.integers(0,d["bse"].shape[1]-512)
            patch=d["bse"][y:y+512,x:x+512]
            axs[0].clear();self.image(axs[0],patch,"First seeded 512 × 512 px analysis window")
            crop=dict(x=int(x),y_trimmed=int(y),y_raw=int(y)+d["top"],width=512,height=512)
            p=patch.astype(float);p-=p.mean();rr=features._radial_index(512)
            if kind == "fft":
                P=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(p*np.hanning(512)[:,None]*np.hanning(512)[None,:])))**2)
                axs[1].imshow(P,cmap="magma");axs[1].set_title("Window power spectrum · log display",fontsize=10,loc="left");axs[1].set_axis_off()
                spec=features.radial_psd(d["bse"]);k=np.arange(1,len(spec)+1)
                axs[2].loglog(k,spec,color="#668ddd");axs[2].axvspan(5,120,alpha=.12,color="#668ddd")
                axs[2].set(xlabel="Spatial frequency bin",ylabel="Radial mean power",title="Full-site 24-window mean spectrum")
                legend = "Spectrum display is illustrative. Slope fits log power vs log frequency in bins 5–120; contrast, focus and preparation remain confounds."
            else:
                ac=np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(p))**2).real);ac/=ac.max()
                axs[1].imshow(ac[192:320,192:320],cmap="viridis");axs[1].set_title("Window autocorrelation · central lags",fontsize=10,loc="left");axs[1].set_axis_off()
                rng=np.random.default_rng(0); curves=[]
                for _ in range(16):
                    yy,xx=rng.integers(0,len(d["bse"])-512),rng.integers(0,d["bse"].shape[1]-512)
                    pp=d["bse"][yy:yy+512,xx:xx+512].astype(float);pp-=pp.mean();curves.append(features._acf_radial(pp,rr,512))
                mean=np.mean(curves,axis=0)
                axs[2].plot(np.arange(80),mean[:80],color="#668ddd");axs[2].axhline(np.exp(-1),color="#8c98a4",ls="--")
                axs[2].axvline(d["row"]["corr_len_px"],color="#df788e",ls="--")
                axs[2].set(xlabel="Radial lag (px)",ylabel="Normalised correlation",title="Full-site 16-window mean · 1/e crossing")
                legend = "Correlation length is the first radial lag below 1/e; it is a BSE texture scale, not a measured particle diameter."
        elif kind in ("etd_ridges", "etd_graphite", "etd_curtain", "etd_orientation", "etd_boundary", "etd_energy", "inlens_texture", "inlens_energy"):
            etd=self.channel(d,"ETD");il=self.channel(d,"Inlens")
            _,maps=features.multichannel_features(d["bse"],etd,il,d["row"]["th_lo"],d["row"]["th_hi"],return_maps=True)
            if kind.startswith("inlens"):
                channel="Inlens + aligned BSE mask"
                self.image(axs[1],il[win],"Aligned Inlens crop · particle interiors")
                self.overlay(axs[1],maps["inner_p"][win],PALETTE["bright"],.15)
                lab=label(maps["inner_p"]);idx=np.arange(1,lab.max()+1)
                area=ndi.sum(np.ones_like(lab),lab,idx);iqr=np.subtract(*np.percentile(il,[75,25]))+1e-6
                sd=ndi.standard_deviation(il,lab,idx)/iqr;keep=area>=400
                axs[2].hist(sd[keep],bins=24,color=PALETTE["bright"])
                axs[2].axvline(.5,color="#df788e",ls="--")
                axs[2].set(xlabel="Within-particle SD / image IQR",ylabel="Interior component count",title="Full-site raw Inlens texture · confounded")
                legend="Gold: BSE bright mask eroded by 6 px. Raw Inlens texture remains charging/acquisition-confounded; speckled threshold=0.5."
            else:
                channel="ETD + aligned BSE mask"
                self.image(axs[1],etd[win],"Aligned ETD source with computed map")
                if kind == "etd_boundary":
                    solid=~d["pore"];mask=solid ^ ndi.binary_erosion(solid,iterations=2)
                    self.overlay(axs[1],mask[win],PALETTE["pore"],.50)
                    from skimage.filters import sobel
                    vals=sobel(etd.astype(float))[mask]/(np.subtract(*np.percentile(etd,[75,25]))+1e-6)
                    axs[2].hist(vals,bins=30,color=PALETTE["pore"])
                    axs[2].set(xlabel="ETD Sobel gradient / image IQR",ylabel="Boundary-pixel count",title="Full-site 2-px boundary measurement")
                    legend="Cyan: 2-px boundary on the solid side of the BSE pore mask. Median gradient/IQR is a preparation/focus flag."
                elif kind == "etd_energy":
                    from skimage.filters import sobel
                    grad=sobel(etd.astype(float))/(np.subtract(*np.percentile(etd,[75,25]))+1e-6)
                    vals=grad[maps["inner_p"]|maps["inner_g"]]
                    axs[2].hist(vals,bins=30,color="#668ddd")
                    axs[2].set(xlabel="ETD Sobel gradient / image IQR",ylabel="Interior-pixel count",title="Full-site interior gradient distribution")
                    legend="ETD texture energy is the median normalised Sobel gradient in eroded BSE-defined interiors; it is an acquisition descriptor."
                else:
                    interior=maps["inner_p"]|maps["inner_g"]
                    if kind in ("etd_curtain","etd_orientation"):
                        mask=maps["curtain"]&interior;color="#e8adff"
                    elif kind == "etd_graphite":
                        mask=maps["crack"]&maps["inner_g"];color=PALETTE["crack"]
                    else:
                        mask=maps["crack"]&maps["inner_p"];color=PALETTE["crack"]
                    if kind in ("etd_ridges","etd_graphite"):
                        measurement_region=maps["inner_p"] if kind == "etd_ridges" else maps["inner_g"]
                        self.overlay(axs[1],measurement_region[win],PALETTE["bright"],.08)
                    self.overlay(axs[1],ndi.binary_dilation(mask[win],iterations=1),color,.75)
                    theta=maps["theta"][(maps["ridge"]>d["row"]["ridge_p97"])&interior]
                    axs[2].hist(theta,bins=36,range=(-90,90),color="#9e95cc")
                    dom=d["row"]["etd_ridge_dom_angle"];axs[2].axvline(dom,color="#eb6834",ls="--")
                    axs[2].set(xlabel="Hessian-derived ridge direction (degrees)",ylabel="Strong interior-ridge pixels",title="Full-site direction histogram")
                    legend="Gold: BSE-defined eroded measurement interior. Pink/lavender: detected ridges, widened 1 px for display. Ridge threshold=0.4; ±15° aligned band removed."
                    if kind in ("etd_curtain", "etd_orientation"):
                        legend="Lavender: ETD ridges >0.4 within ±15° of the dominant angle, widened 1 px for display. Histogram: strong interior ridges >saved p97; dashed line: dominant angle."
                        note="Aligned ridges are shown as an acquisition/preparation diagnostic. They are excluded from crack-density masks; this is not a graphite plate-orientation measurement."
        elif kind in ("local_width","crack_local_width"):
            if "width_maps" not in d:
                _,d["width_maps"]=local_width(d["pore"],return_maps=True)
            maps=d["width_maps"];mask=maps["retained"];dist=maps["distance"]
            skel=maps["crack_skeleton"] if kind == "crack_local_width" else maps["skeleton"]
            yy,xx=np.unravel_index(np.argmax(dist*skel),dist.shape)
            win,crop=self.window(d,phase="pore",centre=(yy,xx),width=850,height=500)
            a=d["bse"][win];axs[0].clear();self.image(axs[0],a,"BSE crop centred on retained width sample")
            self.image(axs[1],a,"Illustrative distance-map centreline width")
            display=ndi.binary_dilation(skel,iterations=2)
            widths=np.ma.masked_where(~display[win],ndi.maximum_filter(2*dist*skel,size=5)[win])
            im=axs[1].imshow(widths,cmap="turbo",vmin=0,vmax=150)
            self.overlay(axs[1],mask[win],PALETTE["pore"],.04)
            cy,cx=yy-win[0].start,xx-win[1].start;radius=float(dist[yy,xx])
            axs[1].add_patch(Circle((cx,cy),radius,fill=False,ec="white",lw=1.2))
            axs[1].annotate("",xy=(cx-radius,cy),xytext=(cx+radius,cy),arrowprops=dict(arrowstyle="<->",color="white",lw=1))
            axs[1].text(cx,cy+radius+15,f"2r={2*radius:.0f} px",ha="center",color="white",fontsize=8,bbox=dict(facecolor="#152d3e",alpha=.8,edgecolor="none",pad=2))
            fig.colorbar(im,ax=axs[1],orientation="horizontal",fraction=.08,pad=.08,label="2 × EDT radius at centreline (px)")
            vals=2*dist[skel];axs[2].hist(vals,bins=30,color=PALETTE["pore"])
            axs[2].set(xlabel="2 × distance to mask exterior (px)",ylabel="Centreline-pixel count",title="Crack-like-void width samples" if kind == "crack_local_width" else "Retained-void width samples")
            legend="Colour: 2× distance on deterministic medial axis, widened 2 px for display. Non-edge-clipped voids ≥30 px²"+("; major axis >500 px." if kind == "crack_local_width" else ".")
            note="Map follows the candidate medial-axis operation. Cropping centres on a retained interior sample; image-edge-clipped components are excluded even if a large adjacent dark cavity is visible. Scalars come from the saved nominal full-site table. Skeleton pixels are not independent observations."
        elif kind == "hysteresis":
            seeds=d["sm"]>d["row"]["th_hi"]+5;support=d["sm"]>d["row"]["th_hi"]-5
            grown=reconstruction(seeds.astype(np.uint8),support.astype(np.uint8),method="dilation",footprint=ndi.generate_binary_structure(2,1)).astype(bool)
            grown=ndi.binary_opening(grown)
            self.image(axs[1],a,"Illustrative strict bright seeds (+5 levels)")
            self.overlay(axs[1],seeds[win],PALETTE["seed"],.40)
            self.image(axs[2],a,"Seeds grown within weak support (−5 levels)")
            self.overlay(axs[2],grown[win],PALETTE["bright"],.35)
            legend="Green: strict seeds. Gold: 4-connected geodesic reconstruction within weak support, followed by default opening. ±5 offsets illustrate the operation; no parameter choice is validated."
            note="Illustrative candidate segmentation, computed independently for the atlas; does not replace the frozen mask or benchmark implementation."
        elif kind in ("segmentation_sensitivity","connectivity"):
            self.image(axs[1],a,"Pore-mask sensitivity: threshold −5 / +5")
            lo=ndi.binary_opening(d["sm"]<d["row"]["th_lo"]-5)
            hi=ndi.binary_opening(d["sm"]<d["row"]["th_lo"]+5)
            self.overlay(axs[1],lo[win],PALETTE["pore"],.30);self.overlay(axs[1],(hi&~lo)[win],"#ed9ccd",.55)
            self.image(axs[2],a,"Bright-mask sensitivity: threshold +5 / −5")
            lo=ndi.binary_opening(d["sm"]>d["row"]["th_hi"]+5)
            hi=ndi.binary_opening(d["sm"]>d["row"]["th_hi"]-5)
            self.overlay(axs[2],lo[win],PALETTE["bright"],.30);self.overlay(axs[2],(hi&~lo)[win],"#ed9ccd",.55)
            legend="Cyan/gold: stricter mask. Pink: pixels included only at the looser threshold. This range is sensitivity, not a confidence interval."
            if kind == "connectivity":
                note="Threshold sensitivity visualises the mask being labelled. No 4-vs-8 connectivity result has been validated or asserted here."
        elif kind in ("tortuosity","percolation"):
            axs[0].clear();self.image(axs[0],d["bse"],"Full-site BSE section");axs[0].set_aspect("equal")
            axs[1].imshow(d["pore"],cmap="gray",aspect="equal");axs[1].set_axis_off();axs[1].set_title("Full-site macro-void mask",fontsize=10,loc="left")
            # Component connectivity is drawn only on the real 2-D mask, never as an invented path.
            lab=d["labels"]["pore"];toplabels=set(lab[0])-{0};bottomlabels=set(lab[-1])-{0};shared=toplabels&bottomlabels
            connected=np.isin(lab,list(shared)) if shared else np.zeros_like(lab,dtype=bool)
            axs[2].imshow(connected,cmap="gray",aspect="equal");axs[2].set_axis_off();axs[2].set_title("Full-resolution top–bottom connected labels",fontsize=10,loc="left")
            axs[2].text(.5,.5,"No spanning label in this 2-D mask" if not shared else "Spanning labels highlighted",transform=axs[2].transAxes,
                        ha="center",va="center",color="#e8adff",fontsize=10,bbox=dict(facecolor="black",alpha=.65,edgecolor="none"))
            crop=dict(x=0,y_trimmed=0,y_raw=d["top"],width=d["bse"].shape[1],height=len(d["bse"]))
            legend="No path is invented for an undefined index. Full-resolution labels illustrate connectivity; cached geodesic calculations use ×4 masks."
            note="Disconnected macro-voids in a 2-D section do not imply absent 3-D electrolyte transport. The penalised solid-cost index is only a numerical comparator."
        elif kind in ("acquisition_histogram","low_contrast"):
            self.image(axs[1],a,"BSE crop · segmentation-quality context")
            if kind == "low_contrast":
                self.overlay(axs[1],d["bright"][win],PALETTE["bright"],.3)
            hist=np.bincount(d["bse"].ravel(),minlength=256)
            axs[2].bar(np.arange(256),hist,width=1,color="#668ddd")
            axs[2].axvline(d["row"]["th_lo"],color=PALETTE["pore"],ls="--",label="Pore threshold")
            axs[2].axvline(d["row"]["th_hi"],color=PALETTE["bright"],ls="--",label="Bright threshold")
            axs[2].set(xlabel="Exported BSE grey level",ylabel="Full-site pixel count",title="Full-site histogram · acquisition diagnostic")
            axs[2].legend(fontsize=8)
            legend="Histogram gaps show remapping of exported values; they do not prove when remapping occurred or recover lost contrast. Raw intensity is not a material KPI."
        elif kind == "edge_band":
            raw=d["raw"];win2=(slice(max(0,len(raw)-220),len(raw)),slice(0,min(1100,raw.shape[1])))
            axs[0].clear();self.image(axs[0],raw[win2],"Raw bottom-edge crop · band visible")
            axs[1].clear();self.image(axs[1],raw[win2],"BSE band trim boundary")
            if d["bottom"]:
                axs[1].axhline(220-d["bottom"],color="#f96791",lw=2)
            axs[2].plot(raw.mean(axis=1),np.arange(len(raw)),color="#668ddd")
            axs[2].axhline(len(raw)-d["bottom"],color="#f96791",ls="--")
            axs[2].set(xlabel="Raw row mean intensity",ylabel="Raw y-coordinate",ylim=(len(raw),len(raw)-250),title="Raw row profile · acquisition diagnostic")
            crop=dict(x=0,y_raw=win2[0].start,y_trimmed=None,width=win2[1].stop,height=win2[0].stop-win2[0].start)
            legend="Pink: bright-band removal boundary. Continuous bright lower band is consistent with collector/stitching; copper chemistry is unconfirmed."
        else:
            # Deliberately avoid inventing masks for unresolved plate or binder classes.
            self.image(axs[1],self.channel(d,"ETD")[win],"Aligned ETD crop · unresolved morphology")
            self.image(axs[2],self.channel(d,"Inlens")[win],"Aligned Inlens crop · acquisition-sensitive")
            channel="BSE + ETD + Inlens"
            legend="Source-only visual reference. No validated binder/plate mask or candidate value is asserted. BSE residual solid includes unresolved binder/carbon black."
            note="Pending candidate: images identify where expert review is needed; no segmentation or numeric result is claimed."
        return self.save(kind,fig,d,crop,legend,kind,note,channel)


def kind_for(metric):
    """Map semantic registry kinds and individual ids to faithful illustrations."""
    key=" ".join([metric["id"],metric.get("visual_kind","")]+metric.get("keys",[])).lower()
    visual=metric.get("visual_kind","")
    if visual == "m2_appearance":return "m2_appearance"
    if visual == "k_morphology":return "k_morphology"
    if visual == "gabor_texture":return "gabor_texture"
    if visual == "bright_graph": return "bright_graph"
    supported={"pore_mask","bright_mask","pore_count","bright_count","crack_mask","graphite_mask","pore_size","bright_size","pore_shape","bright_shape",
               "bright_solidity","bright_circularity","pore_orientation","bright_orientation","bright_spacing","depth_profile_pore",
               "depth_profile_bright","fft","correlation","etd_ridges","etd_graphite","etd_curtain","etd_orientation","etd_boundary",
               "etd_energy","inlens_texture","inlens_energy","local_width","crack_local_width","hysteresis","segmentation_sensitivity","connectivity",
               "tortuosity","percolation","acquisition_histogram","low_contrast","edge_band","void_span","columns","delamination",
               "crack_orientation","binder","pending","conditional_fractions","conditional_mass"}
    supported |= {"battery_neighbourhood", "battery_neighbourhood_boundary", "battery_neighbourhood_distance", "battery_neighbourhood_ring",
                  "battery_windows_512", "battery_windows_1024", "void_internal_context", "void_coverage", "battery_coverage", "battery_threshold_envelopes", "void_width_depth", "void_crack_width_depth", "void_width_depth_both"}
    # Exact keys refine broad visual kinds so circularity, solidity and size use distinct plots.
    if metric["id"] == "battery_geometry_threshold_envelopes":return "battery_threshold_envelopes"
    if "crack_width_depth_slope" in key and "void_width_depth_slope" in key:return "void_width_depth_both"
    if "crack_width_depth_slope" in key:return "void_crack_width_depth"
    if "void_width_depth_slope" in key:return "void_width_depth"
    if "long_void" in key:
        return "void_coverage" if "coverage" in key or "clipped" in key else "void_internal_context"
    if "bright_void_boundary_frac" in key:return "battery_neighbourhood_boundary"
    if "bright_void_distance_d50_px" in key:return "battery_neighbourhood_distance"
    if "bright_ring_void_frac_16px" in key:return "battery_neighbourhood_ring"
    if visual in supported and visual.startswith("battery_"):return visual
    if "stereology_volume_fractions" in key:return "conditional_fractions"
    if "nominal_additive_mass_fractions" in key:return "conditional_mass"
    if "saltykov" in key:return "bright_size"
    if "anisotropy_index" in key:return "pore_shape"
    if "hysteresis" in key:return "hysteresis"
    if "crack_local_width" in key:return "crack_local_width"
    if "local_width" in key or "local width" in key or "edt_width" in key:return "local_width"
    if "columns_interrupted" in key:return "columns"
    if "delamination_index" in key:return "delamination"
    if "crack_orientation" in key:return "crack_orientation"
    if "longest_void" in key:return "void_span"
    if "tortuosity_fraction_connected" in key:return "percolation"
    if "tortuosity" in key:return "tortuosity"
    if "percolation" in key or "connected" in key:return "percolation"
    if "binder" in key or "plate_orientation" in key or "phase_boundary_distance" in key or "watershed" in key or "normalized" in key or "normalised" in key:return "pending"
    if "bright_solidity" in key:return "bright_solidity"
    if "bright_circ" in key:return "bright_circularity"
    if "bright_nn" in key or "bright_spacing" in key:return "bright_spacing"
    if "profile_pore" in key:return "depth_profile_pore"
    if "profile_bright" in key:return "depth_profile_bright"
    if "etd_crack_density_graphite" in key:return "etd_graphite"
    if "etd_crack_density_particles" in key or "crack_p_" in key:return "etd_ridges"
    if "etd_ridge_dom_angle" in key or "ridge_orientation" in key:return "etd_orientation"
    if "etd_curtain" in key:return "etd_curtain"
    if "etd_boundary" in key:return "etd_boundary"
    if "etd_grad_energy" in key:return "etd_energy"
    if "inlens_grad_energy" in key:return "inlens_energy"
    if "inlens" in key:return "inlens_texture"
    if "bright_low_contrast" in key:return "low_contrast"
    if "band" in key or "collector" in key:return "edge_band"
    if "histogram" in key or "graphite_mode" in key or "bright_sep" in key or "th_lo" in key or "empty_bin" in key:return "acquisition_histogram"
    if "fft" in key:return "fft"
    if "corr_len" in key:return "correlation"
    if "alignment" in key or "orientation" in key:return "pore_orientation" if "pore" in key else "bright_orientation"
    if "pore_d" in key or "pore_max_d" in key or "pore_width_d" in key:return "pore_size"
    if "bright_d" in key or "bright_max_d" in key or "bright_width_d" in key:return "bright_size"
    if "aspect" in key or "elong" in key:return "pore_shape" if "pore" in key else "bright_shape"
    if "crack_frac" in key or "crack_count" in key:return "crack_mask"
    if "graphite_frac" in key:return "graphite_mask"
    if "pore_count" in key:return "pore_count"
    if "bright_count" in key:return "bright_count"
    if "pore_frac" in key:return "pore_mask"
    if "bright_frac" in key:return "bright_mask"
    return visual if visual in supported else "pending"


def list_text(value):
    return " · ".join(map(str,value)) if isinstance(value,list) else str(value or "")


def esc(value):
    return html.escape(list_text(value),quote=True)


def card(metric, asset, atlas, index):
    batch,site=asset["batch"],asset["site"]
    row=atlas.row(batch,site)
    values=[]
    for key in metric.get("keys",[]):
        value=row.get(key)
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            values.append(dict(key=key,value=value))
    status=str(metric.get("implementation_status","unspecified"))
    evidence=str(metric.get("evidence_status","unspecified"))
    role=str(metric.get("qc_role","exploratory"))
    pending=asset["kind"]=="pending" or any(word in status.lower() for word in ("planned","not implemented","pending","deferred"))
    # A pending candidate must never acquire a proxy's number through a broad key mapping.
    if pending:values=[]
    value_html="".join(f'<span class="value"><code>{esc(v["key"])}</code><b>{v["value"]:.4g}</b></span>' for v in values)
    if not value_html:
        value_html='<span class="empty-value">Full-site scalar unavailable or undefined for the selected site; see status and limits.</span>'
    links=[]
    for path in metric.get("evidence_paths",[]):
        p=ROOT/path
        if p.exists():
            rel=os.path.relpath(p,OUT)
            links.append(f'<a href="{esc(rel)}">{esc(path)}</a>')
        else:
            links.append(f'<span class="unavailable">{esc(path)} — pending artifact</span>')
    coords=asset["crop"]
    crop=f'x={coords["x"]}, raw y={coords["y_raw"]}, w={coords["width"]}, h={coords["height"]}'
    if coords.get("y_trimmed") is not None:crop+=f'; trimmed y={coords["y_trimmed"]}'
    flag_names=[]
    for key,label_text in (("bright_low_contrast","bright low contrast"),("grey_pore","grey-pore / fallback"),("cracked_known","known crack-like-void site")):
        if row.get(key):flag_names.append(label_text)
    flags="; ".join(flag_names) or "no known subgroup flag on selected site"
    limit=metric.get("limitations",[])
    details="".join(f"<li>{esc(item)}</li>" for item in (limit if isinstance(limit,list) else [limit]))
    search=" ".join(list_text(metric.get(k,"")) for k in ("id","label","family","keys","definition","implementation_status","evidence_status","qc_role","battery_check_ids","battery_implication"))
    battery_paragraph=""
    if metric.get("battery_implication") or metric.get("battery_check_ids"):
        checks=esc(metric.get("battery_check_ids",[]))
        heading="Battery implication / hypothesis"+(" · "+checks if checks else "")
        battery_paragraph=f'<p class="battery-hypothesis"><strong>{heading}</strong><br>{esc(metric.get("battery_implication","Interpretation pending materials review."))} <a href="../../../docs/battery_microstructure_review.md">Battery review and check definitions</a>.</p>'
    display_note=asset["note"]
    saved_heading = ("Saved site summary (four sampled windows)" if asset["kind"] == "gabor_texture" else
                     "Saved crop summary (nine sampled windows)" if asset["kind"] == "m2_appearance" else "Saved full-site example")
    if metric["id"]=="saltykov_unfolded_size_distribution":
        display_note="Figure shows observed 2-D input sizes only. No cached unfolded 3-D distribution or scalar is asserted in this atlas."
    body=f'''<article class="metric-card" id="{esc(metric["id"])}" data-family="{esc(metric.get("family","Other"))}" data-role="{esc(role)}" data-status="{esc(status)}" data-evidence="{esc(evidence)}" data-search="{esc(search.lower())}">
<div class="card-head"><span class="ordinal">{index:02d}</span><div><p class="eyebrow">{esc(metric.get("family","Other"))} · {esc(metric.get("channels",[]))}</p><h2>{esc(metric["label"])}</h2><code class="metric-id">{esc(metric["id"])}</code></div><a class="anchor" href="#{esc(metric["id"])}" aria-label="Link to this metric">#</a></div>
<div class="badges"><span class="badge role">{esc(role)}</span><span class="badge">Implementation: {esc(status)}</span><span class="badge evidence">Evidence: {esc(evidence)}</span><span class="badge">Expert review: {esc(metric.get("review_status","unreviewed"))}</span></div>
<figure><img data-asset="{esc(asset["kind"])}" alt="{esc(metric["label"])}: {esc(asset["legend"])}" width="1764" height="616"><figcaption><strong>{esc(batch)} / {esc(site)}</strong> · {esc(asset["channel"])} · {esc(crop)}<br>{esc(flags)}. Selected for illustration, not an independent validation example.</figcaption></figure>
<div class="card-grid"><div><h3>What is measured <span>{esc(metric.get("units",""))}</span></h3><p>{esc(metric.get("definition","Definition pending."))}</p><p class="interpretation">{esc(metric.get("interpretation",""))}</p>{battery_paragraph}<h3>{esc(saved_heading)} <span>{esc(batch)} / {esc(site)}</span></h3><div class="values">{value_html}</div><p class="small">{esc(display_note)}</p></div><div><h3>Interpretation limits</h3><ul>{details}</ul><p class="legend">{esc(asset["legend"])}</p><details><summary>Evidence and provenance · {esc(metric.get("experiments",[]))}</summary><div class="links">{"".join(links)}</div><p class="small">The register is the status source. Rebuild this report after changing its entries. Source TIFF hashes are embedded in the report payload.</p></details></div></div></article>'''
    return body,dict(id=metric["id"],asset=asset["kind"],values=values,batch=batch,site=site,pending=pending,crop=coords)


CSS='''
:root{--ink:#182b3b;--muted:#536a7b;--line:#dce5eb;--paper:#f3f6f8;--blue:#285cd5;--gold:#aa7300}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:var(--paper)}a{color:var(--blue)}header{background:#102a3b;color:#f6f9fc;padding:38px 30px 30px}.header-inner{max-width:1350px;margin:auto}.kicker{font:700 12px/1.5 system-ui;text-transform:uppercase;letter-spacing:.14em;color:#91bdd4;margin:0 0 12px}h1{font-size:clamp(28px,4vw,44px);font-weight:650;line-height:1.12;letter-spacing:-.03em;margin:0 0 14px}.intro{max-width:1000px;color:#d3e2eb;font-size:17px}.summary{display:flex;gap:18px;flex-wrap:wrap;margin-top:24px}.summary div{padding:12px 18px;border:1px solid #476074;border-radius:9px;min-width:140px}.summary b{font-size:26px;display:block;line-height:1.15}.summary span{font-size:12px;color:#c2d4df}.context{max-width:1350px;margin:22px auto 18px;padding:17px 21px;background:#fff6df;border:1px solid #ecd7a1;border-radius:9px}.context p{margin:0 0 7px}.context p:last-child{margin:0}.layout{display:grid;grid-template-columns:228px minmax(0,1fr);gap:24px;max-width:1350px;margin:0 auto;padding:0 0 36px}.sidebar{align-self:start;position:sticky;top:16px;border:1px solid var(--line);background:white;border-radius:11px;padding:18px;max-height:calc(100vh - 32px);overflow:auto}.sidebar label{font-size:12px;font-weight:650;display:block;margin:12px 0 5px}.sidebar label:first-child{margin-top:0}input,select{font:inherit;width:100%;padding:9px;border:1px solid #c7d6e0;border-radius:6px;background:white;color:var(--ink)}input:focus,select:focus{outline:2px solid #a8c3ff;outline-offset:1px}.count{color:var(--muted);font-size:12px;margin:16px 0}.navlinks{display:flex;flex-direction:column;gap:7px}.navlinks a{text-decoration:none;font-size:12px;color:var(--muted);line-height:1.4;padding:3px 0}.navlinks a:hover{color:var(--blue)}.metric-card{background:white;border:1px solid var(--line);border-radius:12px;margin:0 0 22px;overflow:hidden;box-shadow:0 4px 14px #213a4910;scroll-margin-top:18px}.card-head{display:flex;gap:13px;align-items:flex-start;padding:22px 24px 10px}.ordinal{background:#edf3f8;color:#467086;border-radius:7px;font-size:14px;font-weight:650;padding:7px 9px}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:0 0 3px}.card-head h2{font-size:23px;line-height:1.2;margin:0 0 4px;letter-spacing:-.015em}.metric-id{font-size:11px;color:#668092}.anchor{margin-left:auto;text-decoration:none;font-size:22px;color:#98b0c0}.badges{display:flex;flex-wrap:wrap;gap:6px;padding:0 24px 16px}.badge{font-size:10px;padding:4px 8px;border-radius:5px;border:1px solid #dce6ed;background:#f5f8fa;line-height:1.4}.badge.role{background:#eaf0ff;border-color:#c7d6ff;color:#224ca4}.badge.evidence{background:#fff5e1;border-color:#ebdab5;color:#825d1f}figure{margin:0;background:#f7f9fa;border-top:1px solid #e8eef2;border-bottom:1px solid #e8eef2}figure img{width:100%;height:auto;display:block;min-height:140px;background:#f5f7f9}figcaption{padding:9px 24px 11px;font-size:10px;color:var(--muted);line-height:1.5}.card-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;padding:20px 24px 23px}h3{font-size:12px;margin:0 0 7px;color:#335063}h3 span{font-weight:400;color:#6c8292;font-size:10px;display:block}.card-grid p{margin:0 0 13px;font-size:12px}.interpretation{color:#467086}.battery-hypothesis{background:#eef4f8;border-left:3px solid #9bbace;padding:9px 11px;color:#3d5b70}.battery-hypothesis strong{font-size:11px}.card-grid ul{margin:0 0 12px;padding-left:16px;font-size:11px;color:#526878}.card-grid li{margin-bottom:4px}.values{display:flex;flex-wrap:wrap;gap:7px;margin:9px 0 10px}.value{display:flex;flex-direction:column;padding:8px 10px;border-radius:6px;border:1px solid #d9e6ef;background:#f7fafc}.value code{font-size:10px;color:#678092}.value b{font-size:18px;line-height:1.25;color:#243e51}.empty-value{font-size:11px;color:#826326;background:#fff5df;padding:8px 10px;border-radius:5px}.card-grid .small{font-size:10px;color:#677e8f;line-height:1.5}.legend{background:#f0f5f8;border-left:3px solid #9bbace;padding:8px 10px;color:#3d5b70}.links{display:flex;flex-direction:column;gap:5px;font-size:10px;word-break:break-word;margin:9px 0}summary{cursor:pointer;font-size:11px;color:var(--blue)}.unavailable{color:#987728}footer{max-width:1350px;margin:0 auto 30px;border-top:1px solid var(--line);padding-top:17px;font-size:11px;color:var(--muted)}.hidden{display:none!important}.empty-results{padding:30px;color:#536a7b;border:1px dashed #bacbd6;border-radius:10px;background:white}noscript{display:block;background:#fff2d3;padding:15px}.method-section{background:white;border:1px solid var(--line);padding:20px 24px;border-radius:12px;margin-bottom:22px}.method-section h2{font-size:20px;margin:0 0 10px}.method-section p{font-size:12px}.method-table{width:100%;border-collapse:collapse;font-size:11px}.method-table td,.method-table th{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #e4ebef}.method-table th{background:#f3f7fa} @media(max-width:1000px){.layout{margin:0 16px;grid-template-columns:190px minmax(0,1fr);gap:16px}.context{margin:16px}.card-grid{grid-template-columns:1fr;gap:8px}.summary{gap:9px}}@media(max-width:700px){header{padding:26px 20px}.layout{display:block}.sidebar{position:relative;top:0;max-height:none;margin-bottom:18px}.navlinks{display:none}.card-head{padding:18px 16px 10px}.badges{padding:0 16px 12px}.card-grid{padding:18px 16px}.card-head h2{font-size:20px}figcaption{padding:8px 16px}footer{margin:15px}.summary div{min-width:105px;padding:9px 12px}.summary b{font-size:22px}}@media print{.sidebar{display:none}.layout{display:block;max-width:none}.metric-card{break-inside:avoid;box-shadow:none}.context{max-width:none}header{background:white;color:#152c3d}.intro,.kicker{color:#405b70}.summary{display:none}.method-section{break-inside:avoid}}
'''

JS='''
const payload=JSON.parse(document.getElementById('atlas-data').textContent);
const cards=[...document.querySelectorAll('.metric-card')];
const search=document.getElementById('search');const family=document.getElementById('family');const role=document.getElementById('role');const status=document.getElementById('status');const evidence=document.getElementById('evidence');
function populate(el,key){const vals=[...new Set(cards.map(c=>c.dataset[key]))].sort();for(const v of vals){let o=document.createElement('option');o.value=v;o.textContent=v;el.append(o)}}
populate(family,'family');populate(role,'role');populate(status,'status');populate(evidence,'evidence');
for(const img of document.querySelectorAll('img[data-asset]')){img.src=payload.assets[img.dataset.asset].data;img.loading='lazy';}
function filter(){const q=search.value.trim().toLowerCase();let n=0;for(const c of cards){const show=(!q||c.dataset.search.includes(q))&&(!family.value||c.dataset.family===family.value)&&(!role.value||c.dataset.role===role.value)&&(!status.value||c.dataset.status===status.value)&&(!evidence.value||c.dataset.evidence===evidence.value);c.classList.toggle('hidden',!show);document.querySelector(`[data-nav="${CSS.escape(c.id)}"]`).classList.toggle('hidden',!show);n+=show;}document.getElementById('shown').textContent=n+' of '+cards.length+' metrics shown';document.getElementById('empty-results').classList.toggle('hidden',n>0)}
for(const el of [search,family,role,status,evidence])el.addEventListener('input',filter);filter();
document.getElementById('clear').addEventListener('click',()=>{for(const el of [search,family,role,status,evidence])el.value='';filter()});
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--verify-only",action="store_true");args=parser.parse_args()
    if args.verify_only:
        verify();return
    register_text=REGISTER.read_text()
    register_hash=hashlib.sha256(register_text.encode()).hexdigest()
    register=json.loads(register_text)
    metrics=register["metrics"]
    material_context=""
    material=register.get("material_confirmation")
    if material:
        if isinstance(material,dict):
            premise=material.get("statement") or material.get("text") or "Fresh graphite–Si/SiOx electrodes, confirmed by Santosh."
        elif isinstance(material,str):
            premise=material
        else:
            premise="Fresh graphite–Si/SiOx electrodes, confirmed by Santosh."
        material_context=f'<p><strong>Confirmed material premise:</strong> {esc(premise)} The image masks still do not distinguish Si from SiOx, resolve binder/carbon black, validate 3-D geometry, or measure cycling history, transport or battery performance. Battery implications on cards are hypotheses for expert review, with <a href="../../../docs/battery_microstructure_review.md">battery check definitions</a>.</p>'
    baseline=register.get("supplier_baseline_confirmation")
    if baseline:
        material_context+=f'<p><strong>Challenge reference:</strong> {esc(baseline["statement"])} <a href="../../../docs/qc_plan.md">D49P morphology OOD extension</a> is planned, not activated; current roles and expert-review states below are unchanged. Distribution evidence does not require proof of battery harm, but still needs measurement, acquisition and alert validation.</p>'
    if register.get("dataset_construction"):
        material_context+=f'<p><strong>Crop provenance / dependence (D59S):</strong> {esc(register["dataset_construction"]["interpretation"])} Crop-held-out scores do not establish source-held-out accuracy or independent specimen counts.</p>'
    assert metrics and len({m["id"] for m in metrics})==len(metrics)
    atlas=Atlas(); cards=[];coverage=[]
    for i,m in enumerate(metrics,1):
        kind=kind_for(m);asset_name=atlas.render(kind);body,entry=card(m,atlas.assets[asset_name],atlas,i)
        cards.append(body);coverage.append(entry)
        print(f'{i}/{len(metrics)} {m["id"]} -> {kind}',flush=True)
    nav="".join(f'<a href="#{esc(m["id"])}" data-nav="{esc(m["id"])}">{esc(m["label"])}</a>' for m in metrics)
    methods=register.get("methods",[])
    atlas.render("hysteresis")
    context_figures=[]
    for kind,title in (("low_contrast","Low bright-phase contrast"),("acquisition_histogram","Combed exported histogram"),("edge_band","Bright lower edge band")):
        atlas.render(kind)
        a=atlas.assets[kind]
        context_figures.append(f'<details class="context-example"><summary>{esc(title)} · {esc(a["batch"])}/{esc(a["site"])}</summary><figure><img data-asset="{kind}" alt="{esc(a["legend"])}"><figcaption>{esc(a["legend"])}</figcaption></figure></details>')
    context_section='<section class="method-section"><h2>Acquisition context</h2><p>These examples show why quality flags accompany morphology measurements. The images are real; interpretation of collector chemistry and remapping timing remains unconfirmed.</p>'+"".join(context_figures)+'</section>'
    methodrows=""
    for m in methods:
        methodrows+=f'<tr><td><strong>{esc(m.get("label",m.get("id","Method")))}</strong></td><td>{esc(m.get("status",m.get("implementation_status","unspecified")))}</td><td>{esc(m.get("purpose",m.get("definition",m.get("description",""))))}</td></tr>'
    hysteresis_example='<details><summary>Visual method reference · bright hysteresis</summary><figure><img data-asset="hysteresis" alt="Strict bright seeds and 4-connected growth within weaker support"><figcaption>Candidate operation illustrated on Batch_2/3806gxp0. No ground-truth correction is claimed.</figcaption></figure></details>'
    methodsection=f'<section class="method-section"><h2>Measurement development methods</h2><p>Methods change how a KPI is measured; they are not additional independent samples or extra verdict votes.</p><table class="method-table"><thead><tr><th>Method</th><th>Status</th><th>Purpose</th></tr></thead><tbody>{methodrows}</tbody></table>{hysteresis_example}</section>' if methods else ""
    payload=dict(schema_version=1,register_sha256=register_hash,register=register,coverage=coverage,assets=atlas.assets,
                 selected_source_sha256=atlas.selected_sources,input_table_sha256=atlas.input_hashes,
                 builder_sha256=digest(__file__),generated_date="2026-10-03",scope="Exploratory known-batch measurement development; unseen-batch accuracy not evaluated.")
    embedded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace("</","<\\/")
    nprimary=sum("primary" in str(m.get("qc_role","")).lower() for m in metrics)
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Polaron · visual morphology metric atlas</title><style>{CSS}</style></head><body>
<header><div class="header-inner"><p class="kicker">Polaron / measurement development / 03 October 2026</p><h1>Visual morphology metric atlas</h1><p class="intro">Every metric we are testing, connected to the image operation that creates it. Read the definition, inspect the markup, and see what is implemented, uncertain or awaiting review.</p><div class="summary"><div><b>{len(metrics)}</b><span>tracked metric entries</span></div><div><b>{len(atlas.assets)}</b><span>image-derived visual references</span></div><div><b>{len({(a['batch'],a['site']) for a in atlas.assets.values()})}</b><span>illustration sites</span></div><div><b>{nprimary}</b><span>entries marked primary</span></div></div></div></header>
<div class="context"><p><strong>Interpretation boundary:</strong> developed using exploratory analysis of Batches 1–3. No unseen-batch measurement or accuracy is claimed. Batch 3 is a heterogeneous working reference.</p>{material_context}<p>All lengths are <strong>pixels</strong>; export scale is unverified. Statistical units are <strong>sites</strong>, with specimen independence unknown. Crops illustrate full-site operations; crop pixels, particles and profile bins do not increase statistical n. Colour overlays are computed masks, not expert ground truth.</p><p>Candidate and unavailable measurements remain visible. Status describes implementation and evidence separately; a working extractor does not imply validated material interpretation. This atlas changes no frozen primary KPI or QC threshold.</p></div>
<noscript>This report needs JavaScript to attach its embedded image assets and enable search. Definitions and status remain readable.</noscript>
<div class="layout"><aside class="sidebar"><label for="search">Search metrics</label><input id="search" type="search" placeholder="width, void, Inlens…"><label for="family">Metric family</label><select id="family"><option value="">All families</option></select><label for="role">QC role</label><select id="role"><option value="">All roles</option></select><label for="status">Implementation</label><select id="status"><option value="">All statuses</option></select><label for="evidence">Evidence status</label><select id="evidence"><option value="">All evidence states</option></select><button id="clear" style="margin-top:12px;border:0;background:none;color:#285cd5;cursor:pointer;padding:0;font:inherit;font-size:12px">Clear filters</button><p class="count" id="shown"></p><nav class="navlinks" aria-label="Metric index">{nav}</nav></aside><main>{methodsection}{context_section}<div id="empty-results" class="empty-results hidden">No metrics match these filters.</div>{''.join(cards)}</main></div>
<footer>Canonical status source: <a href="../metric_register.json">metric_register.json</a> · <a href="report.html">E18 comparison report</a> · <a href="../../../docs/assumption_register.html">Assumption register</a><br>Rebuild: python3 -m analysis.morphology.build_metric_atlas. Provenance hashes and coverage are embedded in this standalone HTML. Source images remain read-only; no raw TIFFs are embedded.</footer>
<script id="atlas-data" type="application/json">{embedded}</script><script>{JS}</script></body></html>'''
    DEST.write_text(text)
    verify()
    print(json.dumps(dict(report=str(DEST),bytes=DEST.stat().st_size,metrics=len(metrics),assets=len(atlas.assets),illustration_sites=len({(a['batch'],a['site']) for a in atlas.assets.values()})),indent=2))



def verify_controls(text, payload):
    """Execute actual filter/image-binding JS with a minimal DOM model, no browser."""
    class Parser(HTMLParser):
        def __init__(self):
            super().__init__();self.cards=[];self.images=[]
        def handle_starttag(self, tag, attrs):
            a=dict(attrs)
            if tag=="article" and a.get("class")=="metric-card":
                self.cards.append(dict(id=a["id"],dataset={k[5:]:v for k,v in a.items() if k.startswith("data-")}))
            if tag=="img" and "data-asset" in a:self.images.append(dict(dataset=dict(asset=a["data-asset"])))
    parser=Parser();parser.feed(text)
    executable=shutil.which("node")
    if not executable:
        print("Node unavailable: interactive-control logic smoke test not run.");return
    reduced=dict(assets={k:dict(data=f"data:image/jpeg;base64,ASSET_{k}") for k in payload["assets"]})
    setup=json.dumps(dict(cards=parser.cards,images=parser.images,payload=reduced),ensure_ascii=False)
    harness=r"""
const fixture=FIXTURE;
const assert=(p,m)=>{if(!p)throw Error(m)};
class E {constructor(x={}){Object.assign(this,x);this.value='';this.textContent='';this.options=[];this.listeners={};this.hidden=false;this.classList={toggle:(c,on)=>{if(c==='hidden')this.hidden=on;},contains:c=>c==='hidden'&&this.hidden};}append(v){this.options.push(v)}addEventListener(k,f){this.listeners[k]=f}}
const metricCards=fixture.cards.map(x=>new E(x));const images=fixture.images.map(x=>new E(x));
const controls={};for(const k of ['search','family','role','status','evidence','shown','empty-results','clear'])controls[k]=new E();
controls['atlas-data']=new E({textContent:JSON.stringify(fixture.payload)});controls['atlas-data'].textContent=JSON.stringify(fixture.payload);
const nav=Object.fromEntries(metricCards.map(c=>[c.id,new E()]));
global.document={getElementById:k=>controls[k],createElement:()=>new E(),querySelectorAll:q=>q==='.metric-card'?metricCards:images,querySelector:q=>nav[q.match(/data-nav=\"(.*?)\"/)[1]]};
global.CSS={escape:s=>s};
new Function(SCRIPT)();
assert(metricCards.every(c=>!c.hidden),'Initial cards hidden');
assert(images.every(i=>i.src===fixture.payload.assets[i.dataset.asset].data),'Image binding incomplete');
const fire=id=>controls[id].listeners.input();
const check=pred=>{assert(metricCards.every(c=>c.hidden===!pred(c)),'Filter mismatch');const n=metricCards.filter(c=>!c.hidden).length;assert(controls.shown.textContent===`${n} of ${metricCards.length} metrics shown`,'Count mismatch');assert(controls['empty-results'].hidden===(n>0),'Empty state mismatch');assert(metricCards.every(c=>nav[c.id].hidden===c.hidden),'Index/card visibility mismatch');};
for(const q of ['void','corr_len','b01','zz_no_metric_exists_zz']){controls.search.value=q;fire('search');check(c=>c.dataset.search.includes(q));}
controls.clear.listeners.click();
for(const id of ['family','role','status','evidence']){const key=id;const wanted=metricCards[0].dataset[key];controls[id].value=wanted;fire(id);check(c=>c.dataset[key]===wanted);controls.clear.listeners.click();}
controls.family.value=metricCards[0].dataset.family;controls.search.value='pore';fire('search');check(c=>c.dataset.family===controls.family.value&&c.dataset.search.includes('pore'));
controls.clear.listeners.click();check(()=>true);
console.log(`Control logic verified: ${metricCards.length} cards, ${images.length} bound images, text/family/role/implementation/evidence/combined/empty/clear filters.`);
"""
    harness=harness.replace("FIXTURE",setup).replace("SCRIPT",json.dumps(JS))
    result=subprocess.run([executable,"-"],input=harness,text=True,capture_output=True,check=True)
    print(result.stdout.strip(),flush=True)

def verify():
    text=DEST.read_text()
    match=re.search(r'<script id="atlas-data" type="application/json">(.*?)</script>',text,re.S)
    assert match,"Missing embedded payload"
    payload=json.loads(match.group(1))
    register=json.loads(REGISTER.read_text())
    assert payload["register_sha256"]==digest(REGISTER),"Register changed: rebuild atlas"
    ids=[m["id"] for m in register["metrics"]]
    assert [c["id"] for c in payload["coverage"]]==ids,"Coverage does not match register"
    assert text.count('class="metric-card"')==len(ids)
    lookup={}
    for path in reversed(TABLE_PATHS):
        if not path.exists():continue
        table=pd.read_csv(path)
        if "threshold_offset" in table:table=table[table.threshold_offset==0]
        for _,row in table.iterrows():
            key=(row.batch,row.site);lookup.setdefault(key,{}).update({k:scalar(v) for k,v in row.items()})
    for row in lookup.values():
        if "crack_g_0.4" in row:row.setdefault("etd_crack_density_graphite",row["crack_g_0.4"])
    assert len(ids)==len(set(ids))
    for metric in register["metrics"]:
        if metric.get("battery_implication") or metric.get("battery_check_ids"):
            assert esc(metric.get("battery_implication","Interpretation pending materials review.")) in text
            for check in metric.get("battery_check_ids",[]):assert check.lower() in text.lower()
    if register.get("material_confirmation"):
        assert "Confirmed material premise:" in text
        assert "../../../docs/battery_microstructure_review.md" in text
    for c in payload["coverage"]:
        assert c["asset"] in payload["assets"]
        asset=payload["assets"][c["asset"]]
        assert asset["batch"]==c["batch"] and asset["site"]==c["site"]
        assert asset["crop"]["width"]>0 and asset["crop"]["height"]>0
        assert asset["crop"]["x"]>=0 and asset["crop"]["y_raw"]>=0
        assert asset["data"].startswith("data:image/jpeg;base64,")
        from PIL import Image
        raw=base64.b64decode(asset["data"].split(",",1)[1]);im=Image.open(io.BytesIO(raw));im.verify()
        assert (OUT/asset["path"]).exists()
        assert len(raw)<2_000_000
        if c["pending"]:assert not c["values"],"Pending metric has a fabricated scalar"
        for v in c["values"]:
            assert v["value"] is None or np.isfinite(v["value"])
            assert v["value"]==lookup[(c["batch"],c["site"])][v["key"]],"Displayed value does not match full-site table"
        if c["id"] in {m["id"] for m in register["metrics"] if "E25" in m.get("experiments", [])} and not c["pending"]:
            metric = next(m for m in register["metrics"] if m["id"] == c["id"])
            expected = {k for k in metric.get("keys", []) if isinstance(lookup[(c["batch"],c["site"])].get(k), (float,int))}
            assert {v["key"] for v in c["values"]} == expected, "Finite E25 saved scalar omitted"
        import tifffile
        path=features._image_path(ROOT/"Dataset"/c["batch"],c["site"],"BSE")
        with tifffile.TiffFile(path) as tif:shape=tif.pages[0].shape
        assert c["crop"]["x"]+c["crop"]["width"]<=shape[1]
        assert c["crop"]["y_raw"]+c["crop"]["height"]<=shape[0]
    assert DEST.stat().st_size<20_000_000
    for path,h in payload["input_table_sha256"].items():assert digest(ROOT/path)==h,"Input table changed during generation"
    for path,h in payload["selected_source_sha256"].items():assert digest(ROOT/path)==h,"Illustration source image changed"
    assert payload["builder_sha256"]==digest(__file__),"Builder changed: rebuild atlas"
    verify_controls(text,payload)
    print(f'Atlas verification passed: {len(ids)} registry entries covered; images decode; no pending scalar; provenance current; file under 20 MB.',flush=True)


if __name__=="__main__":main()
