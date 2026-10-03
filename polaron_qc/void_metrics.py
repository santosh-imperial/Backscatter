"""Cheap, secondary 2-D long-void context (E25); no collector or adhesion inference.

The original crack_frac retains all qualifying components, including clipped
ones. This module instead makes the fully observable internal scope explicit.
Local-width/skeleton extraction stays in the exploratory analysis, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table

KPI_NAMES = ("long_void_internal_area_frac", "long_void_y_centroid_norm")
KPI_DEFINITIONS = {
    "long_void_internal_area_frac": {
        "label": "Internal long-void area fraction", "units": "fraction",
        "definition": "Area of 8-connected pore components with area >=30 px² and moment-based major axis >500 px that touch no image edge, divided by the complete trimmed image area (default parameters).",
        "interpretation": "Fully observable internal long-void burden in one 2-D image; valid zero means no retained component, not material acceptance.",
        "limitations": "Pore segmentation must be usable; grey/fallback sites remain quality flagged. Excludes edge-clipped long voids and is not a replacement for crack_frac. No adhesion, electrical-disconnection or 3-D claim.",
    },
    "long_void_y_centroid_norm": {
        "label": "Internal long-void image-row centroid", "units": "normalised image row (0–1)",
        "definition": "Area-weighted pixel-row centroid of fully observable internal long-void components, divided by H-1. Undefined when no internal long void exists.",
        "interpretation": "Location relative to image top and bottom; collector/separator direction and physical coating thickness are unconfirmed.",
        "limitations": "Image-row location is not collector-interface distance. Field of view and excluded components affect sampling. No fixed alarm threshold; descriptive secondary only.",
    },
}
DIAGNOSTIC_NAMES = ("long_void_n_components", "long_void_internal_n_components", "long_void_edge_clipped_n_components",
                    "long_void_edge_clipped_area_share", "long_void_internal_area_share")


def long_void_context(pore_mask, *, major_axis_cutoff_px=500., min_area_px=30, n_depth_bins=10,
                      return_details=False):
    """Return two secondary KPIs and separate observability diagnostics.

    By default the return is a flat dict and contains no large arrays. With
    return_details=True return (metrics, details), where details carries a
    component DataFrame and long/internal masks for visual audit. Components
    and depth bins are never independent statistical observations.
    """
    a = np.asarray(pore_mask)
    if a.ndim != 2:
        raise ValueError("pore_mask must be two-dimensional")
    if a.dtype != bool and not np.isin(a, [0, 1]).all():
        raise ValueError("pore_mask must be a binary mask")
    if major_axis_cutoff_px < 0 or min_area_px < 1 or int(n_depth_bins) != n_depth_bins or n_depth_bins < 1:
        raise ValueError("positive component area and depth-bin count, and nonnegative axis cutoff required")
    mask = a.astype(bool, copy=False)
    h, w = mask.shape
    if not h or not w:
        raise ValueError("pore_mask must contain an observed image field")
    metrics = dict(long_void_internal_area_frac=0., long_void_y_centroid_norm=np.nan,
                   long_void_n_components=0, long_void_internal_n_components=0,
                   long_void_edge_clipped_n_components=0, long_void_edge_clipped_area_share=np.nan,
                   long_void_internal_area_share=np.nan)
    columns = ("label", "area", "major_axis_length", "minor_axis_length", "orientation", "centroid", "bbox")
    if h and w:
        labs = label(mask, connectivity=2)
        components = pd.DataFrame(regionprops_table(labs, properties=columns))
        components = components[(components.area >= min_area_px) & (components.major_axis_length > major_axis_cutoff_px)].copy()
    else:
        labs = np.zeros_like(mask, dtype=np.int32)
        components = pd.DataFrame(columns=["label", "area", "major_axis_length", "minor_axis_length", "orientation",
                                            "centroid-0", "centroid-1", "bbox-0", "bbox-1", "bbox-2", "bbox-3"])
    components["touches_edge"] = ((components["bbox-0"] == 0) | (components["bbox-1"] == 0) |
                                    (components["bbox-2"] == h) | (components["bbox-3"] == w))
    components["depth_centroid_norm"] = components["centroid-0"] / max(h-1, 1)
    components["edge_top_px"] = components["bbox-0"]
    components["edge_bottom_px"] = h-components["bbox-2"]
    components["edge_left_px"] = components["bbox-1"]
    components["edge_right_px"] = w-components["bbox-3"]
    components["nearest_frame_edge_px"] = components[["edge_top_px", "edge_bottom_px", "edge_left_px", "edge_right_px"]].min(axis=1)
    components["tilt_from_horizontal_deg"] = np.abs(90-np.degrees(np.abs(components.orientation.astype(float))))
    internal = components[~components.touches_edge]
    area = float(components.area.sum())
    internal_area = float(internal.area.sum())
    metrics.update(long_void_n_components=int(len(components)), long_void_internal_n_components=int(len(internal)),
                   long_void_edge_clipped_n_components=int(components.touches_edge.sum()))
    if h*w:
        metrics["long_void_internal_area_frac"] = internal_area/(h*w)
    if area:
        metrics["long_void_edge_clipped_area_share"] = (area-internal_area)/area
        metrics["long_void_internal_area_share"] = internal_area/area
    if internal_area:
        metrics["long_void_y_centroid_norm"] = float(np.average(internal["centroid-0"], weights=internal.area)/max(h-1, 1))
    if not return_details:
        return metrics
    long_mask = np.isin(labs, components.label.to_numpy(dtype=int))
    internal_mask = np.isin(labs, internal.label.to_numpy(dtype=int))
    edges = np.linspace(0, h, int(n_depth_bins)+1).astype(int)
    profile = np.array([long_mask[lo:hi].sum()/area if area else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
    internal_profile = np.array([internal_mask[lo:hi].sum()/internal_area if internal_area else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
    return metrics, dict(components=components, long_mask=long_mask, internal_mask=internal_mask,
                         depth_edges=edges, long_void_depth_area_share=profile,
                         internal_long_void_depth_area_share=internal_profile)
