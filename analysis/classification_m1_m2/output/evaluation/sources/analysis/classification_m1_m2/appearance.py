"""Fixed, mask-independent SEM appearance descriptors for E39S.

These are image measurements, not validated material or battery-performance KPIs.
Nine fixed-size spatial windows are pooled to one vector per supplied crop/site;
windows are never observations for classification or uncertainty estimates.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from polaron_qc.features import bright_bands


VERSION = "m2-mask-independent-1.0"
CHANNELS = ("BSE", "ETD", "Inlens")
TILE_SIZE = 512
SCALES = (2.0, 8.0, 32.0)
TRUNCATE = 3.0
CONTEXT = 96
SUFFIXES = (
    "dog_fine_fraction_median", "dog_mid_fraction_median",
    "dog_coarse_fraction_median", "dog_fine_fraction_iqr",
    "dog_coarse_fraction_iqr", "gradient_axis_cos2_median",
    "gradient_axis_sin2_median", "gradient_axis_coherence_median",
)
FEATURE_NAMES = tuple(f"appearance_{channel.lower()}_{name}"
                      for channel in CHANNELS for name in SUFFIXES)


def tile_coordinates(shape: tuple[int, int], tile_size: int = TILE_SIZE,
                     context: int = CONTEXT) -> list[tuple[int, int]]:
    """Nine deterministic measurement-tile origins in the trimmed field.

The first and last tile have fixed filter context at each edge; the middle tile
is centred. Coordinates depend on geometry, never the filename or batch label.
No image resizing or scale conversion is performed.
"""
    if len(shape) != 2:
        raise ValueError("a two-dimensional grayscale image is required")
    starts = []
    for extent in shape:
        if extent < tile_size + 2 * context + 2:
            raise ValueError("field too small for nine distinct tiles with filter context")
        axis = (context, int(round((extent - tile_size) / 2)),
                extent - tile_size - context)
        if len(set(axis)) != 3:
            raise ValueError("field too small for nine distinct tile origins")
        starts.append(axis)
    return [(y, x) for y in starts[0] for x in starts[1]]


def union_area(coordinates: list[tuple[int, int]], tile_size: int = TILE_SIZE) -> int:
    """Exact union of axis-aligned measurement windows, including overlap."""
    y_edges = sorted({y for y, _ in coordinates} |
                     {y + tile_size for y, _ in coordinates})
    area = 0
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        intervals = sorted((x, x + tile_size) for y, x in coordinates
                           if y <= y0 and y + tile_size >= y1)
        length, right = 0, None
        for left, end in intervals:
            if right is None:
                length += end - left
            else:
                length += max(0, end - max(left, right))
            right = end if right is None else max(right, end)
        area += (y1 - y0) * length
    return int(area)


def measure_tile(patch: np.ndarray, *, return_maps: bool = False):
    """Measure one tile with 96px context and Gaussian sigma 2/8/32px.

Each contextual patch is centred and divided by its standard deviation before
filtering. Positive gain and an offset cancel absent clipping or rounding. The
three band energies are mean squares of G2-G8, G8-G32 and centred G32, divided
by their sum. These are overlapping Gaussian bands, not orthogonal power bins.
Gradient moments at sigma8 are energy-weighted cos(2 theta), sin(2 theta), and
their resultant magnitude. Theta describes image gradients, not plate axes.
"""
    a = np.asarray(patch, dtype=np.float64)
    expected = TILE_SIZE + 2 * CONTEXT
    if a.shape != (expected, expected):
        raise ValueError(f"contextual tile must have shape {(expected, expected)}")
    if not np.isfinite(a).all():
        raise ValueError("nonfinite image pixels are unsupported")
    a = a - float(a.mean())
    scale = float(a.std())
    if scale == 0.0:
        return np.full(6, np.nan), None
    a /= scale
    smooth = [gaussian_filter(a, sigma=s, mode="reflect", truncate=TRUNCATE)
              for s in SCALES]
    core = (slice(CONTEXT, CONTEXT + TILE_SIZE),) * 2
    fine = (smooth[0] - smooth[1])[core]
    mid = (smooth[1] - smooth[2])[core]
    coarse = smooth[2][core]
    coarse = coarse - coarse.mean()
    energy = np.asarray([np.mean(v * v) for v in (fine, mid, coarse)])
    total = float(energy.sum())
    fraction = energy / total if total > 0 else np.full(3, np.nan)
    gx = gaussian_filter(a, sigma=8.0, order=(0, 1), mode="reflect",
                         truncate=TRUNCATE)[core]
    gy = gaussian_filter(a, sigma=8.0, order=(1, 0), mode="reflect",
                         truncate=TRUNCATE)[core]
    ex, ey, exy = np.mean(gx * gx), np.mean(gy * gy), np.mean(gx * gy)
    grad_total = float(ex + ey)
    if grad_total > 0:
        cos2, sin2 = float((ex - ey) / grad_total), float(2 * exy / grad_total)
        coherence = float(np.clip(np.hypot(cos2, sin2), 0, 1))
    else:
        cos2 = sin2 = coherence = np.nan
    values = np.r_[fraction, cos2, sin2, coherence]
    maps = None if not return_maps else {
        "raw": np.asarray(patch)[core].copy(), "fine_band": fine,
        "mid_band": mid, "coarse_band": coarse,
        "gradient_magnitude": np.hypot(gx, gy),
    }
    return values, maps


def aggregate_tiles(values: np.ndarray) -> np.ndarray:
    """Equal-window medians/IQR; invalid tiles remain unavailable, never zero."""
    v = np.asarray(values, dtype=float)
    if v.shape != (9, 6):
        raise ValueError("exactly nine tile vectors of length six are required")
    # A fixed count of nine is part of the method, so fail closed rather than
    # quietly changing the set of sampled windows on constant/broken data.
    if not np.isfinite(v).all():
        return np.full(8, np.nan)
    med = np.median(v, axis=0)
    iqr = np.percentile(v[:, [0, 2]], 75, axis=0) - np.percentile(v[:, [0, 2]], 25, axis=0)
    return np.r_[med[:3], iqr, med[3:]]


def extract_site(images: dict[str, np.ndarray], *, trim: tuple[int, int] | None = None,
                 return_maps: bool = False):
    """Return (24 feature dict, diagnostics dict, central-tile response maps).

    A single BSE edge-band trim is applied to every aligned detector. Pass the
    original trim for acquisition perturbations: the invariance guarantee is
    conditional on a fixed field of view, not on the acquisition-sensitive band
    detector. Filter context lies outside the measured tiles but inside the
    trimmed field. No phase segmentation or intensity/geometry flags are inputs.
    """
    if set(images) != set(CHANNELS):
        raise ValueError(f"exactly the channels {CHANNELS} are required")
    shapes = {np.asarray(a).shape for a in images.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("detector images must be aligned two-dimensional arrays")
    height, width = next(iter(shapes))
    top, bottom = bright_bands(np.asarray(images["BSE"])) if trim is None else trim
    if top < 0 or bottom < 0 or top + bottom >= height:
        raise ValueError("invalid shared trim")
    field_shape = (height - top - bottom, width)
    coordinates = tile_coordinates(field_shape)
    area = union_area(coordinates)
    diagnostics = {
        "appearance_version": VERSION, "trim_top": int(top), "trim_bottom": int(bottom),
        "field_height": int(field_shape[0]), "field_width": int(width),
        "measurement_tile_size": TILE_SIZE, "filter_context": CONTEXT,
        "n_tiles_per_detector": 9, "measurement_union_pixels": area,
        "measurement_coverage": area / (field_shape[0] * width),
        "overlap_pixels": 9 * TILE_SIZE**2 - area,
        "tile_origins_trimmed_yx": coordinates,
        "tile_origins_raw_yx": [(y + top, x) for y, x in coordinates],
    }
    output, maps = {}, {}
    for channel in CHANNELS:
        field = np.asarray(images[channel])[top: height - bottom if bottom else None]
        measurements = []
        for index, (y, x) in enumerate(coordinates):
            patch = field[y - CONTEXT:y + TILE_SIZE + CONTEXT,
                          x - CONTEXT:x + TILE_SIZE + CONTEXT]
            values, response = measure_tile(patch, return_maps=return_maps and index == 4)
            measurements.append(values)
            if response is not None:
                maps[channel] = response
        pooled = aggregate_tiles(np.asarray(measurements))
        for name, value in zip(SUFFIXES, pooled):
            output[f"appearance_{channel.lower()}_{name}"] = float(value)
        diagnostics[f"{channel.lower()}_finite_tiles"] = int(
            np.isfinite(measurements).all(axis=1).sum())
    diagnostics["available_features"] = int(np.isfinite(list(output.values())).sum())
    return output, diagnostics, maps


def definition() -> dict:
    """Machine-readable preregistered panel, not a learned feature selector."""
    return {
        "version": VERSION, "feature_names": list(FEATURE_NAMES),
        "channels": list(CHANNELS), "tile_size_px": TILE_SIZE,
        "grid": "3 by 3; edge/centre/edge origins with fixed Gaussian context",
        "filter_context_px": CONTEXT, "gaussian_sigmas_px": list(SCALES),
        "gaussian_truncate": TRUNCATE,
        "bands": ["G2-G8", "G8-G32", "G32 minus measurement-tile mean"],
        "energy": "mean square within 512px measurement tile; divided by sum of three band energies",
        "pooling": "equal-window median; fine/coarse fractions also have tile IQR",
        "gradient": "sigma8 Gaussian derivatives; energy-weighted cos2/sin2 and axial resultant; tile medians",
        "input_normalization": "contextual patch centred and divided by its standard deviation",
        "shared_trim": "existing bright_bands(BSE) top/bottom applied to all aligned detectors",
        "constant_policy": "any nonfinite tile makes all eight channel descriptors unavailable",
        "provenance": "31 labelled supplied crops; parent mapping unknown; about15 source images and artificial batches per organiser",
        "interpretation": "mask-independent acquisition-sensitive image appearance; no phase chemistry, plate axes or battery-harm validation",
        "invariance": "positive affine intensity transform absent clipping/rounding and conditional on fixed trim; gamma and quantisation not invariant",
        "independent_unit": "one supplied crop/site vector; tiles and channels are not additional observations",
    }
