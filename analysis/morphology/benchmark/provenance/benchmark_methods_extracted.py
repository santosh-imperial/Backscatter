"""E23 measurement candidates. These functions do not alter production QC."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops_table
from skimage.morphology import medial_axis

SEED = 20261003


def retained_voids(mask, min_area=30, connectivity=2):
    """Remove small and image-edge-clipped components, with explicit connectivity."""
    labels = label(np.asarray(mask, bool), connectivity=connectivity)
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep[edge_ids] = False
    keep[0] = False
    return keep[labels], labels, keep


def local_width(mask, min_area=30, crack_axis=500, return_maps=False):
    """2*distance at deterministic medial-axis pixels, a local 2-D width proxy.

    Samples pool centreline pixels (approximately length weighted). They are
    NOT independent observations. This is not a maximal-disk thickness map.
    No branch pruning or 3-D pore-throat interpretation is applied.
    """
    retained, labels, keep = retained_voids(mask, min_area)
    skeleton, distance = medial_axis(retained, return_distance=True, rng=SEED)
    width = 2.0 * distance[skeleton]
    props = regionprops_table(labels, properties=("label", "area", "major_axis_length"))
    crack_ids = props["label"][(props["major_axis_length"] > crack_axis) & keep[props["label"]]]
    is_crack = np.zeros(len(keep), bool)
    is_crack[crack_ids] = True
    crack_skeleton = skeleton & is_crack[labels]
    crack_width = 2 * distance[crack_skeleton]
    out = dict(
        void_local_width_d50_px=float(np.median(width)) if len(width) else np.nan,
        void_local_width_d90_px=float(np.quantile(width, .9)) if len(width) else np.nan,
        crack_local_width_d50_px=float(np.median(crack_width)) if len(crack_width) else np.nan,
        n_retained_voids=int(keep.sum()), n_crack_width_voids=len(crack_ids),
        n_width_samples=int(skeleton.sum()), n_crack_width_samples=int(crack_skeleton.sum()),
        retained_void_area_frac=float(retained.mean()),
        excluded_void_area_frac=float((np.asarray(mask, bool) & ~retained).mean()),
    )
    if return_maps:
        return out, dict(retained=retained, skeleton=skeleton, distance=distance,
                         crack_skeleton=crack_skeleton)
    return out


def bright_hysteresis(smoothed, threshold, half_window=5):
    """Grow >t+h seeds within >t-h, 4-connected, then the existing cross opening."""
    if half_window < 0:
        raise ValueError("half_window must be non-negative")
    strong = smoothed > threshold + half_window
    weak = smoothed > threshold - half_window
    grown = ndi.binary_propagation(strong, structure=ndi.generate_binary_structure(2, 1), mask=weak)
    return ndi.binary_opening(grown, iterations=1)


def component_summary(mask, min_area=50, connectivity=2):
    p = regionprops_table(label(mask, connectivity=connectivity),
                          properties=("area", "equivalent_diameter_area", "perimeter", "solidity"))
    use = p["area"] >= min_area
    area = p["area"][use]
    diameter = p["equivalent_diameter_area"][use]
    if not len(area):
        return dict(frac=float(mask.mean()), count_per_Mpx=0., d50=np.nan, circ=np.nan, solidity=np.nan)
    order = np.argsort(diameter)
    index = np.searchsorted(np.cumsum(area[order]) / area.sum(), .5)
    return dict(frac=float(mask.mean()), count_per_Mpx=len(area) / (mask.size / 1e6),
                d50=float(diameter[order[index]]),
                circ=float(np.average(4*np.pi*area / np.maximum(p["perimeter"][use], 1)**2, weights=area)),
                solidity=float(np.average(p["solidity"][use], weights=area)))


def semantic_mask(pore, bright):
    if np.any(pore & bright):
        raise ValueError("Pore and bright classes must be disjoint")
    out = np.ones(pore.shape, np.uint8)
    out[pore] = 0
    out[bright] = 2
    return out


def semantic_errors(prediction, manual):
    """Evaluate only manually labelled pixels; absent classes have undefined IoU."""
    use = manual != 255
    out = dict(n_labelled_pixels=int(use.sum()), labelled_fraction=float(use.mean()))
    for k, name in enumerate(("void", "solid", "bright")):
        pred = (prediction == k) & use
        truth = (manual == k) & use
        intersection = int((pred & truth).sum())
        union = int((pred | truth).sum())
        total = int(pred.sum() + truth.sum())
        out[name + "_iou"] = intersection / union if union else np.nan
        out[name + "_dice"] = 2*intersection / total if total else np.nan
        out[name + "_frac_error_pp"] = 100*(pred.sum()-truth.sum()) / use.sum() if use.any() else np.nan
    return out
