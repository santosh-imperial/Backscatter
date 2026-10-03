"""polaron_qc.features — batch folder -> per-site / per-image / per-particle / per-patch tables.

Refactor of the extraction cells of ``notebooks/_build_01_dataset_analysis.py`` (sections 4, 6, 6c) into a
reusable module so an unseen batch folder can be processed with one call::

    from polaron_qc.features import extract_batch
    tables = extract_batch("Dataset/Batch_2", cache_dir="analysis_cache/features")
    tables["sites"], tables["images"], tables["particles"], tables["patches"]

Numerics are kept identical to notebook 01 (same smoothing, same histogram-anchored thresholds, same seeds for
the sampled texture descriptors) so that ``analysis_cache/site_features.csv`` and ``etd_inlens_features.csv``
are reproduced exactly. Do not "improve" a definition here without changing FEATURE_VERSION and re-running
``tests/test_features.py``; the notebook caches would then have to be regenerated as well.

Rules carried from CLAUDE.md: all lengths in px; no raw-intensity material KPI (intensity statistics live in the
``images`` table as instrument flags); detector label ``SE`` == ``ETD``; bright edge bands are trimmed before
measuring; the ETD ridge threshold is the absolute ``RIDGE_T``, never a per-image percentile.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.feature import hessian_matrix
from skimage.filters import gaussian, sobel
from skimage.measure import label, regionprops_table

from . import GREY_PORE_SITES, LOW_CONTRAST_SITES, NM_PER_PX  # noqa: F401  (shared constants; do not redefine)

__all__ = [
    "FEATURE_VERSION", "RIDGE_T", "RIDGE_GRID", "CRACK_MAJOR_AXIS_PX",
    "list_sites", "load_image", "bright_bands", "trimmed", "phase_thresholds", "segment", "ridge_maps",
    "corr_length", "radial_psd", "image_quality", "multichannel_features", "extract_site", "extract_batch",
]

# ---------------------------------------------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------------------------------------------
FEATURE_VERSION = "1.1.0"          # E25 adds separate secondary geometry; original measurements unchanged
RIDGE_T = 0.4                      # absolute ETD ridge threshold chosen in notebook 01 (T* = grid value nearest the
                                   # median per-site p97 = 0.425); one value for every site, never a percentile
RIDGE_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]   # sensitivity grid kept from notebook 01 (column names use repr)
CRACK_MAJOR_AXIS_PX = 500          # crack-like void: pore component with major axis > 500 px (≈ 12.5 µm nominal)
MIN_BRIGHT_AREA_PX = 50            # bright-phase particle size floor
MIN_PORE_AREA_PX = 30              # pore component size floor
DET_ALIASES = {"SE": "ETD"}        # detector label normalisation (confirmed same detector)
DETECTORS = ("BSE", "ETD", "Inlens")
_FILE_RE = re.compile(r"img_(\w+)_(\w+)\.tif$")

IMAGE_COLUMNS = ["batch", "site", "det", "H", "mean", "p1", "p50", "p99", "std", "gray_levels", "empty_bin_frac",
                 "band_top", "band_bottom"]
PARTICLE_COLUMNS = ["area", "equivalent_diameter_area", "eccentricity", "solidity", "perimeter", "major_axis_length",
                    "minor_axis_length", "circ", "batch", "site"]
PATCH_COLUMNS = ["batch", "site", "patch_id", "y0", "x0", "pore_frac", "bright_frac", "bright_count", "crack_area_frac",
                 "pore_max_d", "corr_len_px", "graphite_mode_local", "bright_low_contrast", "grey_pore"]


# ---------------------------------------------------------------------------------------------------------------
# folder access
# ---------------------------------------------------------------------------------------------------------------
def _index_files(batch_dir):
    """One row per TIFF: site, det (as labelled), det_norm (SE -> ETD), path, size, mtime."""
    rows = []
    for f in sorted(glob.glob(os.path.join(batch_dir, "*.tif"))):
        m = _FILE_RE.match(os.path.basename(f))
        if not m:
            continue
        det = m.group(2)
        st = os.stat(f)
        rows.append(dict(site=m.group(1), det=det, det_norm=DET_ALIASES.get(det, det), path=f, size=st.st_size, mtime=st.st_mtime))
    return pd.DataFrame(rows, columns=["site", "det", "det_norm", "path", "size", "mtime"])


def batch_name(batch_dir):
    return os.path.basename(os.path.normpath(batch_dir))


def list_sites(batch_dir):
    """Sorted site ids found in ``batch_dir`` (files named ``img_<site>_<det>.tif``)."""
    return sorted(_index_files(batch_dir).site.unique().tolist())


def _image_path(batch_dir, site, det="BSE"):
    det_norm = DET_ALIASES.get(det, det)
    idx = _index_files(batch_dir)
    hit = idx[(idx.site == site) & (idx.det_norm == det_norm)]
    if hit.empty:
        raise FileNotFoundError(f"no {det} image for site {site!r} in {batch_dir}")
    return hit.path.iloc[0]


def load_image(batch_dir, site, det="BSE"):
    """Grayscale uint8 image. The TIFFs store identical R,G,B planes; plane 0 is used. ``det='ETD'`` falls back
    to a file labelled ``SE`` (and vice versa)."""
    return _read_gray(_image_path(batch_dir, site, det))


def _read_gray(path):
    a = tifffile.imread(path)
    return a[..., 0] if a.ndim == 3 else a


# ---------------------------------------------------------------------------------------------------------------
# edge bands (verbatim from notebook 01)
# ---------------------------------------------------------------------------------------------------------------
def bright_bands(img, margin=0.2, delta=40, minrows=4):
    """Rows at the top/bottom that are far brighter than the body (current collector / stitching border)."""
    rm = img.mean(1); body = np.median(rm); H = len(rm)
    hot = rm > body + delta
    top = 0
    while top < H*margin and hot[top]: top += 1
    bot = 0
    while bot < H*margin and hot[H-1-bot]: bot += 1
    # allow a few dark rows before the band (bands are not always flush with the edge)
    idx = np.where(hot)[0]
    if (idx >= H*(1-margin)).any(): bot = max(bot, H - idx[idx >= H*(1-margin)].min())
    if (idx <  H*margin).any():     top = max(top, idx[idx < H*margin].max()+1)
    return (top if top >= minrows else 0), (bot if bot >= minrows else 0)


def trimmed(img):
    """Image with the bright edge bands removed."""
    t, b = bright_bands(img); return img[t: img.shape[0]-b if b else None]


# ---------------------------------------------------------------------------------------------------------------
# per-image acquisition statistics (instrument flags, never material KPIs)
# ---------------------------------------------------------------------------------------------------------------
def image_quality(a):
    """Row of ``analysis_cache/image_quality.csv`` for one untrimmed uint8 image (without batch/site/det)."""
    lo, hi = np.percentile(a, [1, 99]).astype(int)
    h = np.bincount(a.ravel()[::3], minlength=256)
    t, bo = bright_bands(a)
    return dict(H=a.shape[0], mean=a.mean(), p1=lo, p50=np.median(a), p99=hi, std=a.std(),
                gray_levels=int((h > 0).sum()), empty_bin_frac=(h[lo:hi+1] == 0).mean(), band_top=t, band_bottom=bo)


# ---------------------------------------------------------------------------------------------------------------
# BSE segmentation (verbatim from notebook 01)
# ---------------------------------------------------------------------------------------------------------------
def _smooth_hist(sm):
    """Smoothed 256-bin histogram of a (float) BSE image, sub-sampled 3x3 as in notebook 01."""
    h = np.bincount(sm[::3, ::3].astype(int).ravel(), minlength=256).astype(float)
    return ndi.gaussian_filter1d(h, 3) + 1e-9


def _graphite_mode(hs):
    return int(np.argmax(hs[10:200])) + 10


def phase_thresholds(sm):
    """Per-image pore/graphite/bright thresholds anchored on the graphite mode of a smoothed BSE histogram.
    Bright threshold = valley between graphite mode and a resolved second (bright) mode when one exists,
    otherwise graphite mode + 3.5 sigma_right (and the image is flagged as 'bright mode not resolved').
    Pore threshold   = valley between the dark mode and graphite mode when one exists, otherwise mode - 3.5 sigma_left.
    ``sm`` is the gaussian(sigma=1)-smoothed BSE image (float, preserve_range)."""
    hs = _smooth_hist(sm)
    m = _graphite_mode(hs); half = hs[m] / 2
    l = m
    while l > 0 and hs[l] > half: l -= 1
    r = m
    while r < 255 and hs[r] > half: r += 1
    sl, sr = (m - l) / 1.177, (r - m) / 1.177
    # bright side
    pk, _ = find_peaks(np.log(hs[m+15:220]), prominence=0.4); pk = pk + m + 15
    if len(pk):
        p2 = int(pk[np.argmax(hs[pk])]); thi = int(m + np.argmin(hs[m:p2])); bright_resolved = True
    else:
        p2 = np.nan; thi = m + 3.5 * sr; bright_resolved = False
    bright_sep = p2 - m if bright_resolved else np.nan          # contrast between bright phase and graphite, in gray levels
    bright_low_contrast = (not bright_resolved) or (bright_sep < 45)
    # dark side
    pkl, _ = find_peaks(np.log(hs[:max(m-10, 1)]), prominence=0.4)
    if len(pkl) or hs[0] > hs[max(m-10,1):m].min():
        dark_peak = pkl[np.argmax(hs[pkl])] if len(pkl) else 0
        tlo = int(dark_peak + np.argmin(hs[dark_peak:m])); pore_resolved = True
    else:
        tlo = m - 3.5 * sl; pore_resolved = False
    return dict(th_lo=tlo, th_hi=thi, graphite_mode=m, sigma_l=sl, sigma_r=sr, bright_mode=p2, bright_sep=bright_sep,
                bright_mode_resolved=bright_resolved, bright_low_contrast=bool(bright_low_contrast), pore_mode_resolved=pore_resolved,
                hist=hs/hs.sum())


def smooth_bse(bse_uint8):
    """The gaussian(sigma=1, preserve_range) float image every BSE measurement is made on."""
    return gaussian(bse_uint8, sigma=1.0, preserve_range=True)


def segment(bse_uint8, th_lo, th_hi):
    """(pore_mask, bright_mask, smoothed) for a trimmed uint8 BSE image at the given site thresholds.
    Masks are opened once with the default 3x3 cross, exactly as in notebook 01."""
    sm = smooth_bse(bse_uint8)
    pore = ndi.binary_opening(sm < th_lo, iterations=1)
    bright = ndi.binary_opening(sm > th_hi, iterations=1)
    return pore, bright, sm


# ---------------------------------------------------------------------------------------------------------------
# texture descriptors (verbatim from notebook 01; seeded sampling)
# ---------------------------------------------------------------------------------------------------------------
def _radial_index(size):
    y, x = np.indices((size, size)); return np.hypot(y-size/2, x-size/2).astype(int)


def _acf_radial(p, rr, size):
    """Radially averaged, max-normalised autocorrelation of one zero-mean float patch (first size//2 lags)."""
    Fp = np.fft.fft2(p); ac = np.fft.fftshift(np.real(np.fft.ifft2(Fp*np.conj(Fp)))); ac /= ac.max()
    return (np.bincount(rr.ravel(), ac.ravel()) / np.maximum(np.bincount(rr.ravel()), 1))[:size//2]


def _one_over_e(acc):
    idx = np.where(acc < np.exp(-1))[0]
    return float(idx[0]) if len(idx) else np.nan


def corr_length(img, n=16, size=512, seed=0):
    """1/e decay length (px) of the radially averaged autocorrelation of BSE texture, averaged over n seeded
    random 512-px windows."""
    rng = np.random.default_rng(seed); H, W = img.shape; acc = np.zeros(size//2)
    rr = _radial_index(size)
    for _ in range(n):
        r0, c0 = rng.integers(0, H-size), rng.integers(0, W-size); p = img[r0:r0+size, c0:c0+size].astype(float); p -= p.mean()
        acc += _acf_radial(p, rr, size)
    acc /= n
    return _one_over_e(acc)


def radial_psd(img, n=24, size=512, seed=0):
    """Radially averaged power spectrum of n seeded random Hann-windowed 512-px windows (bins 1 .. size//2-1)."""
    rng = np.random.default_rng(seed); H, W = img.shape
    acc = None
    r = _radial_index(size)
    for _ in range(n):
        r0, c0 = rng.integers(0, H-size), rng.integers(0, W-size)
        p = img[r0:r0+size, c0:c0+size].astype(float); p -= p.mean(); p /= (p.std()+1e-9)
        P = np.abs(np.fft.fftshift(np.fft.fft2(p*np.hanning(size)[:,None]*np.hanning(size)[None,:])))**2
        rad = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
        acc = rad if acc is None else acc + rad
    return acc[1:size//2] / n


def fft_slope(spec):
    k = np.arange(1, len(spec)+1); sl = slice(4, 120)
    return float(np.polyfit(np.log(k[sl]), np.log(spec[sl]), 1)[0])


# ---------------------------------------------------------------------------------------------------------------
# BSE site features (verbatim arithmetic from the notebook-01 site loop)
# ---------------------------------------------------------------------------------------------------------------
def _area_weighted_d(d, wts, q):
    order = np.argsort(d); cw = np.cumsum(wts[order])/wts.sum()
    return d[order][np.searchsorted(cw, q)]


def bse_site_features(a):
    """All BSE site features for one *trimmed* uint8 BSE image.

    Returns (features: dict, particles: DataFrame, state: dict). ``state`` carries the masks and label images
    needed for the patch table so the (expensive) segmentation is done once per site."""
    H, W = a.shape
    sm = smooth_bse(a)
    th = phase_thresholds(sm); hist = th.pop("hist")
    pore = ndi.binary_opening(sm < th["th_lo"], iterations=1); bright = ndi.binary_opening(sm > th["th_hi"], iterations=1)
    f = dict(H=H, W=W, **th, pore_frac=pore.mean(), bright_frac=bright.mean(), graphite_frac=1-pore.mean()-bright.mean())
    # bright-phase particles
    lab = label(bright)
    rp = pd.DataFrame(regionprops_table(lab, properties=("label", "area", "equivalent_diameter_area", "eccentricity", "solidity",
                                                         "perimeter", "major_axis_length", "minor_axis_length")))
    rp = rp[rp.area >= MIN_BRIGHT_AREA_PX]; rp["circ"] = 4*np.pi*rp.area/np.maximum(rp.perimeter, 1)**2
    d = rp.equivalent_diameter_area.values; wts = rp.area.values  # area-weighted PSD, like laser-diffraction D-values
    if len(rp):
        f.update(bright_count_per_Mpx=len(rp)/(H*W/1e6), bright_d10=_area_weighted_d(d, wts, .1), bright_d50=_area_weighted_d(d, wts, .5),
                 bright_d90=_area_weighted_d(d, wts, .9), bright_circ=np.average(rp.circ, weights=wts),
                 bright_solidity=np.average(rp.solidity, weights=wts), bright_max_d=d.max())
    else:  # notebook 01 would have raised here; an unseen batch must not crash the run
        f.update(bright_count_per_Mpx=0.0, bright_d10=np.nan, bright_d50=np.nan, bright_d90=np.nan, bright_circ=np.nan,
                 bright_solidity=np.nan, bright_max_d=np.nan)
    # pores
    labp = label(pore)
    pp = pd.DataFrame(regionprops_table(labp, properties=("label", "area", "equivalent_diameter_area", "major_axis_length", "minor_axis_length")))
    pp = pp[pp.area >= MIN_PORE_AREA_PX]
    dp, wp = pp.equivalent_diameter_area.values, pp.area.values
    big = pp.major_axis_length > CRACK_MAJOR_AXIS_PX   # crack-like voids: long axis > 500 px (~1/4 of the coating thickness)
    if len(pp):
        f.update(pore_count_per_Mpx=len(pp)/(H*W/1e6), pore_d50=_area_weighted_d(dp, wp, .5), pore_d90=_area_weighted_d(dp, wp, .9),
                 pore_elong=np.average(pp.major_axis_length/np.maximum(pp.minor_axis_length, 1), weights=wp),
                 pore_max_d=dp.max(), crack_frac=pp.area[big].sum()/(H*W), crack_count_per_Mpx=big.sum()/(H*W/1e6))
    else:
        f.update(pore_count_per_Mpx=0.0, pore_d50=np.nan, pore_d90=np.nan, pore_elong=np.nan, pore_max_d=np.nan, crack_frac=0.0, crack_count_per_Mpx=0.0)
    # through-thickness profiles (10 bins; image orientation kept, separator side unknown)
    for k, (lo, hi) in enumerate(zip(np.linspace(0, H, 11)[:-1].astype(int), np.linspace(0, H, 11)[1:].astype(int))):
        f[f"profile_pore_{k}"] = pore[lo:hi].mean(); f[f"profile_bright_{k}"] = bright[lo:hi].mean()
    # texture scale (seeded random windows; same seeds as notebook 01)
    spec = radial_psd(a)
    f["fft_slope"] = fft_slope(spec); f["corr_len_px"] = corr_length(a)
    particles = rp.drop(columns=["label"]).reset_index(drop=True)
    state = dict(a=a, sm=sm, pore=pore, bright=bright, lab_bright=lab, lab_pore=labp, bright_props=rp, pore_props=pp, hist=hist, spec=spec)
    return f, particles, state


# ---------------------------------------------------------------------------------------------------------------
# ETD / Inlens features inside the BSE masks (verbatim arithmetic from notebook 01 §6c)
# ---------------------------------------------------------------------------------------------------------------
def ridge_maps(etd, sigma=2.0):
    """(ridge_strength, theta_deg) of a uint8 ETD image: largest Hessian eigenvalue (dark ridges positive),
    scaled by sigma^2 and by the image's inter-quartile range so that the absolute RIDGE_T is comparable across
    sites; theta is the ridge orientation in degrees (-90, 90]."""
    e = etd.astype(float); scale = np.subtract(*np.percentile(e, [75, 25])) + 1e-6
    Hrr, Hrc, Hcc = hessian_matrix(e, sigma=sigma, order="rc", use_gaussian_derivatives=False)
    tr = Hrr + Hcc; det = Hrr*Hcc - Hrc**2; lam1 = tr/2 + np.sqrt(np.maximum(tr**2/4 - det, 0))
    ridge = np.maximum(lam1, 0) * sigma**2 / scale
    theta = np.degrees(0.5*np.arctan2(2*Hrc, Hrr - Hcc))
    return ridge, theta


def multichannel_features(bse, etd, il, th_lo, th_hi, sigma=2.0, return_maps=False, T=None):
    """ETD ridge / Inlens texture features inside the BSE phase masks. ``bse``, ``etd``, ``il`` are the three
    pixel-aligned channels of one site, trimmed by the BSE edge bands; ``th_lo``/``th_hi`` are the site's BSE
    thresholds. Columns match ``analysis_cache/etd_inlens_features.csv`` (``crack_p_<t>`` etc. over RIDGE_GRID)."""
    pore, bright, _ = segment(bse, th_lo, th_hi); solid = ~pore
    inner_p = ndi.binary_erosion(bright, iterations=6); inner_g = ndi.binary_erosion(solid & ~bright, iterations=8); interior = inner_p | inner_g
    e = etd.astype(float); scale = np.subtract(*np.percentile(e, [75, 25])) + 1e-6
    ridge, theta = ridge_maps(etd, sigma=sigma)
    p97 = np.percentile(ridge[interior], 97)
    ang = theta[(ridge > p97) & interior]; hist, edges = np.histogram(ang, bins=36, range=(-90, 90)); dom = edges[np.argmax(hist)] + 2.5
    dtheta = np.abs((theta - dom + 90) % 180 - 90); aligned = dtheta < 15
    out = dict(ridge_p97=p97, etd_ridge_dom_angle=dom, etd_curtain_anisotropy=hist.max()/max(np.median(hist), 1))
    for t in RIDGE_GRID:
        strong = ridge > t
        out[f"crack_p_{t}"] = (strong & ~aligned)[inner_p].mean() if inner_p.any() else np.nan
        out[f"crack_g_{t}"] = (strong & ~aligned)[inner_g].mean()
        out[f"curtain_{t}"] = (strong & aligned)[interior].mean()
    g = sobel(e); bnd = solid ^ ndi.binary_erosion(solid, iterations=2)
    out["etd_boundary_sharpness"] = np.median(g[bnd]) / scale; out["etd_grad_energy"] = np.median(g[interior]) / scale
    il_f = il.astype(float); il_scale = np.subtract(*np.percentile(il_f, [75, 25])) + 1e-6
    out["inlens_grad_energy"] = np.median(sobel(il_f)[interior]) / il_scale
    lab = label(inner_p); idx = np.arange(1, lab.max()+1)
    if len(idx):
        area = ndi.sum(np.ones_like(lab), lab, idx); sd = ndi.standard_deviation(il_f, lab, idx) / il_scale; keep = area >= 400
        out.update(inlens_particle_texture=np.median(sd[keep]) if keep.any() else np.nan, inlens_particle_texture_p90=np.percentile(sd[keep], 90) if keep.any() else np.nan,
                   inlens_speckled_particle_frac=(sd[keep] > 0.5).mean() if keep.any() else np.nan, inlens_particles_measured=int(keep.sum()))
    else:
        out.update(inlens_particle_texture=np.nan, inlens_particle_texture_p90=np.nan, inlens_speckled_particle_frac=np.nan, inlens_particles_measured=0)
    if return_maps:
        T = RIDGE_T if T is None else T
        strong = ridge > T; return out, dict(curtain=strong & aligned, crack=strong & ~aligned, inner_p=inner_p, inner_g=inner_g, ridge=ridge, theta=theta)
    return out


def _ridge_t_key(t):
    """Column suffix used by notebook 01 for a grid value (repr of the float: 0.1 not 0.10)."""
    return f"{t}"


# ---------------------------------------------------------------------------------------------------------------
# 512-px patch table (new; site thresholds, site-level masks)
# ---------------------------------------------------------------------------------------------------------------
def _patch_grid(H, W, size, stride):
    ys = list(range(0, H - size + 1, stride)); xs = list(range(0, W - size + 1, stride))
    return [(y, x) for y in ys for x in xs]


def patch_features(state, H, W, patch_size=512, patch_stride=512):
    """Per-patch KPIs on the site-level masks (``state`` from ``bse_site_features``). Patches are full windows
    only, laid on the trimmed BSE image. Definitions:
      pore_frac / bright_frac   mean of the site pore / bright mask inside the patch
      bright_count              number of bright particles (site labelling, area >= 50 px) touching the patch
      crack_area_frac           pixels inside the patch belonging to crack-like voids (pore components with major
                                axis > 500 px, measured on the whole site) / patch area
      pore_max_d                largest equivalent diameter (whole-component, px) of any pore component (area >= 30)
                                touching the patch; NaN when no pore touches the patch
      corr_len_px               1/e autocorrelation length of the raw BSE patch itself (same arithmetic as corr_length, n=1)
      graphite_mode_local       argmax of the patch's smoothed BSE histogram (same histogram as phase_thresholds)
    Patch-level values are for within-site variability and localisation only; they never count as n."""
    a, sm, pore, bright = state["a"], state["sm"], state["pore"], state["bright"]
    lab_b, lab_p = state["lab_bright"], state["lab_pore"]
    rp, pp = state["bright_props"], state["pore_props"]
    keep_b = np.zeros(lab_b.max()+1, bool); keep_b[rp.label.values.astype(int)] = True
    eqd_p = np.full(lab_p.max()+1, np.nan); eqd_p[pp.label.values.astype(int)] = pp.equivalent_diameter_area.values
    is_crack = np.zeros(lab_p.max()+1, bool); is_crack[pp.label.values[(pp.major_axis_length > CRACK_MAJOR_AXIS_PX).values].astype(int)] = True
    rr = _radial_index(patch_size); area = float(patch_size*patch_size)
    rows = []
    for pid, (y0, x0) in enumerate(_patch_grid(H, W, patch_size, patch_stride)):
        win = (slice(y0, y0+patch_size), slice(x0, x0+patch_size))
        lb = np.unique(lab_b[win]); lb = lb[(lb > 0) & keep_b[lb]]
        lp = np.unique(lab_p[win]); lp = lp[lp > 0]; dmax = eqd_p[lp]; dmax = dmax[~np.isnan(dmax)]
        p = a[win].astype(float); p -= p.mean()   # raw trimmed BSE, same input as the site corr_length
        cl = _one_over_e(_acf_radial(p, rr, patch_size)) if p.std() > 0 else np.nan
        hs = _smooth_hist(sm[win])
        rows.append(dict(patch_id=pid, y0=y0, x0=x0, pore_frac=float(pore[win].mean()), bright_frac=float(bright[win].mean()),
                         bright_count=int(len(lb)), crack_area_frac=float(is_crack[lab_p[win]].sum()/area),
                         pore_max_d=float(dmax.max()) if len(dmax) else np.nan, corr_len_px=cl, graphite_mode_local=_graphite_mode(hs)))
    return pd.DataFrame(rows, columns=PATCH_COLUMNS[2:12])


# ---------------------------------------------------------------------------------------------------------------
# one site, one batch
# ---------------------------------------------------------------------------------------------------------------
def extract_site(batch_dir, site, batch=None, patch_size=512, patch_stride=512):
    """All tables for one site: dict(site=dict, images=list[dict], particles=DataFrame, patches=DataFrame)."""
    batch = batch or batch_name(batch_dir)
    paths = {d: _image_path(batch_dir, site, d) for d in DETECTORS}
    images = []
    raw = {}
    for d in DETECTORS:
        a = _read_gray(paths[d]); raw[d] = a
        images.append(dict(batch=batch, site=site, det=d, **image_quality(a)))
    t, bo = bright_bands(raw["BSE"]); sl = slice(t, -bo if bo else None)
    bse, etd, il = raw["BSE"][sl], raw["ETD"][sl], raw["Inlens"][sl]
    del raw
    f, particles, state = bse_site_features(bse)
    # New battery hypotheses are extracted alongside the original measurements,
    # but do not join the frozen primary/classifier/decision feature families.
    from . import secondary
    f.update(secondary.extract(state["sm"], f["th_lo"], f["th_hi"],
                               pore=state["pore"], bright=state["bright"]))
    row = dict(batch=batch, site=site, **f)
    mc = multichannel_features(bse, etd, il, row["th_lo"], row["th_hi"])
    row.update(mc)
    k = _ridge_t_key(RIDGE_T)
    row["etd_crack_density_particles"] = mc[f"crack_p_{k}"]; row["etd_crack_density_graphite"] = mc[f"crack_g_{k}"]; row["etd_curtain_frac"] = mc[f"curtain_{k}"]
    row["grey_pore"] = site in GREY_PORE_SITES
    patches = patch_features(state, row["H"], row["W"], patch_size, patch_stride)
    patches.insert(0, "site", site); patches.insert(0, "batch", batch)
    patches["bright_low_contrast"] = bool(row["bright_low_contrast"]); patches["grey_pore"] = bool(row["grey_pore"])
    row["n_patches"] = int(len(patches)); row["batch_dir"] = os.path.abspath(batch_dir)
    particles["batch"], particles["site"] = batch, site
    return dict(site=row, images=images, particles=particles[PARTICLE_COLUMNS], patches=patches[PATCH_COLUMNS])


def folder_hash(batch_dir, patch_size=512, patch_stride=512):
    """sha1 of sorted (relative path, size, mtime) of the TIFFs + FEATURE_VERSION + patch geometry."""
    idx = _index_files(batch_dir)
    items = sorted((os.path.relpath(p, batch_dir), int(s), int(m)) for p, s, m in zip(idx.path, idx["size"], idx.mtime))
    payload = json.dumps(dict(files=items, version=FEATURE_VERSION, patch_size=patch_size, patch_stride=patch_stride), sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _have_pyarrow():
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def _cache_paths(cache_dir, name, h, fmt):
    return {k: os.path.join(cache_dir, f"{name}_{h}.{k}.{fmt}") for k in ("sites", "images", "particles", "patches")}


def _write_table(df, path, fmt):
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _read_table(path, fmt):
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)


def extract_batch(batch_dir, cache_dir=None, patch_size=512, patch_stride=512, force=False, verbose=True):
    """Process one batch folder (``img_<site>_<det>.tif``) into four DataFrames:

      sites      one row per site: notebook-01 BSE features + ETD/Inlens features + the RIDGE_T-derived
                 etd_crack_density_particles / etd_crack_density_graphite / etd_curtain_frac, grey_pore, n_patches, batch_dir
      images     one row per (site, detector) with the acquisition statistics of analysis_cache/image_quality.csv
      particles  one row per bright-phase particle (area >= 50 px)
      patches    one row per full patch_size window (stride patch_stride) on the trimmed BSE image

    With ``cache_dir`` the tables are written as <cache_dir>/<batch>_<hash>.<table>.parquet (csv if pyarrow is
    missing) and reused when the folder content hash and FEATURE_VERSION match, unless ``force``.
    Sites are processed one at a time to keep memory bounded (~1-2 GB peak per 7000x2300 site)."""
    batch_dir = os.path.normpath(batch_dir)
    name = batch_name(batch_dir)
    sites = list_sites(batch_dir)
    if not sites:
        raise FileNotFoundError(f"no img_<site>_<det>.tif files in {batch_dir}")
    fmt = "parquet" if _have_pyarrow() else "csv"
    paths = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        paths = _cache_paths(cache_dir, name, folder_hash(batch_dir, patch_size, patch_stride), fmt)
        if not force and all(os.path.exists(p) for p in paths.values()):
            if verbose:
                print(f"[features] {name}: cache hit ({os.path.basename(paths['sites'])})", flush=True)
            return {k: _read_table(p, fmt) for k, p in paths.items()}
    site_rows, image_rows, parts, patch_tabs = [], [], [], []
    t_start = time.time()
    for s in sites:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = extract_site(batch_dir, s, batch=name, patch_size=patch_size, patch_stride=patch_stride)
        site_rows.append(r["site"]); image_rows.extend(r["images"]); parts.append(r["particles"]); patch_tabs.append(r["patches"])
        if verbose:
            f = r["site"]
            print(f"[features] {name} {s}: thr {f['th_lo']:.0f}/{f['th_hi']:.0f} low-contrast={f['bright_low_contrast']} | "
                  f"pore {f['pore_frac']:.3f} bright {f['bright_frac']:.3f} d50 {f['bright_d50']:.0f}px crack {f['crack_frac']:.4f} "
                  f"corr {f['corr_len_px']:.0f}px | {f['n_patches']} patches | {time.time()-t0:.1f}s", flush=True)
    sites_df = pd.DataFrame(site_rows)
    for c in ("th_lo", "th_hi", "bright_mode", "bright_sep"):   # int on resolved sites, float on fallback: fix one schema
        sites_df[c] = sites_df[c].astype(float)
    out = dict(sites=sites_df, images=pd.DataFrame(image_rows, columns=IMAGE_COLUMNS),
               particles=pd.concat(parts, ignore_index=True)[PARTICLE_COLUMNS] if parts else pd.DataFrame(columns=PARTICLE_COLUMNS),
               patches=pd.concat(patch_tabs, ignore_index=True)[PATCH_COLUMNS])
    if verbose:
        print(f"[features] {name}: {len(sites)} sites in {time.time()-t_start:.0f}s", flush=True)
    if paths:
        for k, p in paths.items():
            _write_table(out[k], p, fmt)
    return out
