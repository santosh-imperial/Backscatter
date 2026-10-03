"""Secondary B01/B04 measurements from existing disjoint 2-D phase masks.

These quantify raster geometry, not electrical contact, free expansion volume or
3-D electrolyte transport. No quality flag or QC decision is inferred here.
Callers must exclude low-contrast bright masks from every KPI and require a
non-grey-pore acquisition for joint bright/pore KPIs. All scales are fixed in pixels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.measure import label

BATTERY_METRIC_VERSION = "1.0.0"
MIN_BRIGHT_AREA_PX = 50
WINDOW_SIZES = (512, 1024)
RING_RADIUS_PX = 16
THRESHOLD_OFFSETS = (-5, 0, 5)

KPI_DEFINITIONS = {
    "bright_void_boundary_frac": {
        "units": "fraction", "battery_check_ids": ["B01"],
        "definition": "Pore-facing 4-neighbour raster faces / all exposed faces of retained, non-edge-clipped 8-connected bright components (area >=50 px). Faces are pooled by boundary length; diagonal adjacency is not a face.",
        "interpretation": "Observed void adjacency at bright-mask boundaries; residual-solid adjacency is its complement, not electrical contact.",
        "required_flags": {"bright_low_contrast": False, "grey_pore": False},
    },
    "bright_void_distance_d50_px": {
        "units": "px", "battery_check_ids": ["B01"],
        "definition": "Median over retained, non-edge-clipped bright components of the minimum Euclidean centre-to-centre distance from any component pixel to any pore-mask pixel. Adjacent face centres are 1 px apart.",
        "interpretation": "2-D bright-to-visible-void proximity; not a 3-D pore throat, expansion clearance or wetting distance.",
        "required_flags": {"bright_low_contrast": False, "grey_pore": False},
    },
    "bright_ring_void_frac_16px": {
        "units": "fraction", "battery_check_ids": ["B01"],
        "definition": "Median component-wise pore fraction in Euclidean pixel-centre distance bands outside retained bright-mask components (distance >0 and <=16 px). This includes threshold holes within the silhouette, not only the physical outer neighbourhood. Only complete bands are used; overlapping bands are allowed. Other bright and residual-solid pixels are separately counted.",
        "interpretation": "Visible local void neighbourhood of a segmented component, including threshold holes; not validated outer clearance or free volume for Si/SiOx expansion.",
        "required_flags": {"bright_low_contrast": False, "grey_pore": False},
    },
}
for _scale in WINDOW_SIZES:
    KPI_DEFINITIONS[f"local_bright_std_{_scale}px"] = {
        "units": "fraction", "battery_check_ids": ["B04"],
        "definition": f"Population standard deviation of original bright-mask area fractions in complete {_scale}x{_scale} px non-overlapping windows anchored at trimmed-image top-left; >=2 windows required. Partial edge windows excluded.",
        "interpretation": "Spatial bright-mask heterogeneity at a fixed observed scale; also depends on mean loading, particle sizes and sampling.",
        "required_flags": {"bright_low_contrast": False},
    }
    KPI_DEFINITIONS[f"local_bright_pore_spearman_{_scale}px"] = {
        "units": "correlation", "battery_check_ids": ["B01", "B04"],
        "definition": f"Spearman correlation of bright- and pore-mask fractions across the same complete {_scale}x{_scale} px windows; >=3 windows and both variables nonconstant required.",
        "interpretation": "Within-section spatial co-location; compositional closure, segmentation and common depth structure can induce association. Windows are not independent material samples.",
        "required_flags": {"bright_low_contrast": False, "grey_pore": False},
    }
KPI_NAMES = tuple(KPI_DEFINITIONS)
JOINT_KPI_NAMES = tuple(k for k, d in KPI_DEFINITIONS.items() if "grey_pore" in d["required_flags"])
BRIGHT_ONLY_KPI_NAMES = tuple(k for k in KPI_NAMES if k not in JOINT_KPI_NAMES)
DIAGNOSTIC_NAMES = (
    "neighbourhood_bright_pixels", "neighbourhood_pore_pixels",
    "neighbourhood_bright_components_all", "neighbourhood_bright_components_small",
    "neighbourhood_bright_components_retained", "neighbourhood_bright_components_clipped",
    "neighbourhood_bright_components_used", "neighbourhood_bright_boundary_faces",
    "neighbourhood_bright_void_faces", "neighbourhood_bright_residual_solid_faces",
    "neighbourhood_bright_other_bright_faces", "neighbourhood_ring_components_used",
    "neighbourhood_ring_components_clipped",
) + tuple(k for s in WINDOW_SIZES for k in (
    f"neighbourhood_windows_{s}px", f"neighbourhood_window_coverage_{s}px",
    f"neighbourhood_windows_bright_present_{s}px", f"neighbourhood_windows_pore_present_{s}px",
))


def _masks(pore_mask, bright_mask):
    masks = [np.asarray(x) for x in (pore_mask, bright_mask)]
    if masks[0].ndim != 2 or masks[1].ndim != 2 or masks[0].shape != masks[1].shape:
        raise ValueError("pore and bright masks must be 2-D with the same shape")
    for m in masks:
        if m.dtype != bool and (not np.issubdtype(m.dtype, np.number) or not np.all((m == 0) | (m == 1))):
            raise ValueError("masks must contain only boolean/0/1 values")
    pore, bright = [m.astype(bool, copy=False) for m in masks]
    if np.any(pore & bright):
        raise ValueError("pore and bright masks must not overlap")
    return pore, bright


def _face_counts(lab, pore, bright, nlabels):
    """Count exposed unit faces with in-frame neighbours; clipped objects excluded later."""
    total = np.zeros(nlabels + 1, dtype=np.int64)
    void = total.copy(); residual = total.copy(); other_bright = total.copy()
    pairs = ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[1:, :], np.s_[:-1, :]),
             (np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:, 1:], np.s_[:, :-1]))
    for src, dst in pairs:
        a, b = lab[src], lab[dst]
        exposed = (a != 0) & (a != b)
        ids = a[exposed]
        total += np.bincount(ids, minlength=nlabels + 1)
        void += np.bincount(a[exposed & pore[dst]], minlength=nlabels + 1)
        other_bright += np.bincount(a[exposed & bright[dst]], minlength=nlabels + 1)
        residual += np.bincount(a[exposed & ~pore[dst] & ~bright[dst]], minlength=nlabels + 1)
    return total, void, residual, other_bright


def measure_neighbourhoods(pore_mask, bright_mask, *, return_details=False):
    """Return a flat KPI/count dict; optional ``(dict, details)`` for audit overlays.

    ``details`` contains component/window DataFrames and the 8-connected label
    image. No quality gates are applied: caller retains raw candidates, then gates
    them using ``KPI_DEFINITIONS[key]['required_flags']`` before comparisons.
    Empty/absent phase or insufficient observation produces NaN, never an
    invented zero. True measured zero (e.g. no boundary void adjacency) is valid.
    """
    pore, bright = _masks(pore_mask, bright_mask)
    h, w = bright.shape
    bright_present, pore_present = bool(bright.any()), bool(pore.any())
    out = {k: np.nan for k in KPI_NAMES}
    out.update({k: 0 for k in DIAGNOSTIC_NAMES})
    out["neighbourhood_bright_pixels"] = int(bright.sum())
    out["neighbourhood_pore_pixels"] = int(pore.sum())
    window_rows, component_rows = [], []
    for s in WINDOW_SIZES:
        ny, nx = h // s, w // s
        n = ny * nx
        out[f"neighbourhood_windows_{s}px"] = n
        out[f"neighbourhood_window_coverage_{s}px"] = n * s * s / bright.size if bright.size else np.nan
        if not n:
            continue
        bf = bright[:ny*s, :nx*s].reshape(ny, s, nx, s).mean(axis=(1, 3)).ravel()
        pf = pore[:ny*s, :nx*s].reshape(ny, s, nx, s).mean(axis=(1, 3)).ravel()
        out[f"neighbourhood_windows_bright_present_{s}px"] = int((bf > 0).sum())
        out[f"neighbourhood_windows_pore_present_{s}px"] = int((pf > 0).sum())
        if (bf > 0).any() and n >= 2:
            out[f"local_bright_std_{s}px"] = float(np.std(bf, ddof=0))
        if bright_present and pore_present and n >= 3 and np.ptp(bf) > 0 and np.ptp(pf) > 0:
            out[f"local_bright_pore_spearman_{s}px"] = float(spearmanr(bf, pf).statistic)
        if return_details:
            for j, (b, p) in enumerate(zip(bf, pf)):
                window_rows.append(dict(scale_px=s, y0=(j//nx)*s, x0=(j%nx)*s, bright_frac=b, pore_frac=p))
    if bright.size and bright_present:
        lab = label(bright, connectivity=2)
        nlabels = int(lab.max())
        areas = np.bincount(lab.ravel(), minlength=nlabels + 1)
        clipped_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        clipped = np.zeros(nlabels + 1, dtype=bool); clipped[clipped_ids] = True
        retained = areas >= MIN_BRIGHT_AREA_PX; retained[0] = False
        used = retained & ~clipped
        out.update(neighbourhood_bright_components_all=nlabels,
                   neighbourhood_bright_components_small=int((~retained[1:]).sum()),
                   neighbourhood_bright_components_retained=int(retained.sum()),
                   neighbourhood_bright_components_clipped=int((retained & clipped).sum()),
                   neighbourhood_bright_components_used=int(used.sum()))
        faces, vf, sf, of = _face_counts(lab, pore, bright, nlabels)
        out.update(neighbourhood_bright_boundary_faces=int(faces[used].sum()),
                   neighbourhood_bright_void_faces=int(vf[used].sum()),
                   neighbourhood_bright_residual_solid_faces=int(sf[used].sum()),
                   neighbourhood_bright_other_bright_faces=int(of[used].sum()))
        # With 8-connected objects, other bright labels cannot share a raster
        # face. Ring counts below still distinguish nearby bright from residual.
        if pore_present and faces[used].sum():
            out["bright_void_boundary_frac"] = float(vf[used].sum() / faces[used].sum())
        if pore_present and used.any():
            dp = ndi.distance_transform_edt(~pore)
            distances = ndi.minimum(dp, labels=lab, index=np.flatnonzero(used))
            out["bright_void_distance_d50_px"] = float(np.median(distances))
            del dp
        else:
            distances = np.full(int(used.sum()), np.nan)
        distance_lookup = dict(zip(np.flatnonzero(used), distances))
        boxes = ndi.find_objects(lab)
        ring_values = []
        r = RING_RADIUS_PX
        for obj in np.flatnonzero(retained):
            sy, sx = boxes[obj-1]
            edge = bool(clipped[obj])
            full_ring = sy.start >= r and sx.start >= r and sy.stop + r <= h and sx.stop + r <= w
            ring_n = ring_void = ring_bright = ring_solid = 0
            if not edge and full_ring:
                sub = (slice(sy.start-r, sy.stop+r), slice(sx.start-r, sx.stop+r))
                component = lab[sub] == obj
                d = ndi.distance_transform_edt(~component)
                ring = (d > 0) & (d <= r)
                ring_n = int(ring.sum())
                ring_void = int((ring & pore[sub]).sum())
                ring_bright = int((ring & bright[sub]).sum())
                ring_solid = ring_n - ring_void - ring_bright
                out["neighbourhood_ring_components_used"] += 1
                if pore_present and ring_n:
                    ring_values.append(ring_void/ring_n)
            elif not edge:
                out["neighbourhood_ring_components_clipped"] += 1
            if return_details:
                component_rows.append(dict(component_id=int(obj), area_px=int(areas[obj]),
                    y0=sy.start, y1=sy.stop, x0=sx.start, x1=sx.stop, edge_clipped=edge,
                    complete_ring=bool(full_ring and not edge), boundary_faces=int(faces[obj]) if not edge else np.nan,
                    void_faces=int(vf[obj]) if not edge else np.nan,
                    residual_solid_faces=int(sf[obj]) if not edge else np.nan,
                    void_boundary_frac=float(vf[obj]/faces[obj]) if not edge and faces[obj] and pore_present else np.nan,
                    void_distance_px=float(distance_lookup.get(obj, np.nan)),
                    ring_pixels=ring_n, ring_void_pixels=ring_void, ring_other_bright_pixels=ring_bright,
                    ring_residual_solid_pixels=ring_solid,
                    ring_void_frac=ring_void/ring_n if ring_n and pore_present else np.nan))
        if ring_values:
            out["bright_ring_void_frac_16px"] = float(np.median(ring_values))
    else:
        lab = np.zeros(bright.shape, dtype=np.int32)
    if return_details:
        columns = ("component_id", "area_px", "y0", "y1", "x0", "x1", "edge_clipped", "complete_ring",
                   "boundary_faces", "void_faces", "residual_solid_faces", "void_boundary_frac", "void_distance_px",
                   "ring_pixels", "ring_void_pixels", "ring_other_bright_pixels", "ring_residual_solid_pixels", "ring_void_frac")
        return out, {"components": pd.DataFrame(component_rows, columns=columns),
                     "windows": pd.DataFrame(window_rows, columns=("scale_px", "y0", "x0", "bright_frac", "pore_frac")),
                     "labels": lab}
    return out


def threshold_sensitivity(sm, th_lo, th_hi, *, nominal=None, return_variants=False):
    """Fixed paired offsets -5, 0, +5 applied to BOTH existing thresholds.

    A deterministic method envelope, not a statistical interval. This paired
    sweep is not the exhaustive independent 3x3 threshold grid. Same default
    cross opening as production; sm is already Gaussian-smoothed. ``nominal``
    avoids recomputing the zero variant. Invalid thresholds raise ValueError.
    """
    sm = np.asarray(sm)
    if sm.ndim != 2 or not np.isfinite(sm).all():
        raise ValueError("sm must be a finite 2-D smoothed intensity image")
    if not np.isfinite(th_lo) or not np.isfinite(th_hi) or th_lo >= th_hi:
        raise ValueError("finite pore threshold must be below bright threshold")
    variants = []
    for offset in THRESHOLD_OFFSETS:
        if offset == 0 and nominal is not None:
            values = dict(nominal)
        else:
            pore = ndi.binary_opening(sm < th_lo + offset, iterations=1)
            bright = ndi.binary_opening(sm > th_hi + offset, iterations=1)
            values = measure_neighbourhoods(pore, bright)
        variants.append(dict(threshold_offset=offset, **values))
    out = {}
    for k in KPI_NAMES:
        v = np.asarray([row[k] for row in variants], dtype=float)
        finite = v[np.isfinite(v)]
        out[f"{k}_sensitivity_min"] = float(finite.min()) if finite.size else np.nan
        out[f"{k}_sensitivity_max"] = float(finite.max()) if finite.size else np.nan
        out[f"{k}_sensitivity_n"] = int(finite.size)
    for k in DIAGNOSTIC_NAMES:
        v = np.asarray([row[k] for row in variants], dtype=float)
        finite = v[np.isfinite(v)]
        out[f"{k}_sensitivity_min"] = float(finite.min()) if finite.size else np.nan
        out[f"{k}_sensitivity_max"] = float(finite.max()) if finite.size else np.nan
    return (out, variants) if return_variants else out
