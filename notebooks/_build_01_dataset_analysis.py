import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 01 · Dataset analysis — SEM cross-sections, three batches

**Purpose.** Understand the data before designing any QC logic: what was imaged, how consistently it was acquired, which measurable microstructure quantities vary between and within batches, and which artifacts would mislead a naive comparison.

**Scope.** Descriptive only. No baseline is assumed, no verdicts are produced. Everything here feeds the KPI design in the next notebook.

**Data.** `Dataset/Batch_{1,2,3}/img_<site>_<detector>.tif`. Grayscale stored as three identical RGB planes (39 images have a single differing edge column, nothing inside the frame). Each *site* is one stitched cross-section imaged with three detectors: backscattered electrons (**BSE**, contrast ∝ atomic number), secondary electrons (**ETD**, labelled **SE** on four sites; topography), and the in-column **Inlens** detector (surface-sensitive, strong charging contrast).""")

code(r'''import os, re, glob, warnings, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage as ndi, stats
from skimage.filters import threshold_multiotsu, gaussian
from skimage.measure import label, regionprops_table
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)

ROOT  = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
DATA  = os.path.join(ROOT, "Dataset")
CACHE = os.path.join(ROOT, "analysis_cache"); os.makedirs(CACHE, exist_ok=True)

BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
COL = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a", "Batch_3 (grey-pore group)": "#eda100"}
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.frameon": False})

# ---- index every file -------------------------------------------------------
rows = []
for f in sorted(glob.glob(os.path.join(DATA, "*", "*.tif"))):
    b = os.path.basename(os.path.dirname(f)); m = re.match(r"img_(\w+)_(\w+)\.tif", os.path.basename(f))
    det = m.group(2); rows.append(dict(batch=b, site=m.group(1), det=det, det_norm={"SE": "ETD"}.get(det, det), path=f, mb=os.path.getsize(f)/1e6))
index = pd.DataFrame(rows)
SITES = index[["batch", "site"]].drop_duplicates().reset_index(drop=True)

def load(batch, site, det="BSE"):
    """Grayscale uint8 image (the TIFFs store identical R,G,B planes)."""
    p = index.query("batch==@batch and site==@site and det_norm==@det").path.iloc[0]
    return tifffile.imread(p)[..., 0]

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
    t, b = bright_bands(img); return img[t: img.shape[0]-b if b else None]

print(f"{len(index)} files, {len(SITES)} sites, {index.mb.sum():.0f} MB total")''')

md("""## 1 · Inventory

How many sites per batch, which detectors exist for each, and whether anything is missing.""")

code(r'''inv = index.pivot_table(index=["batch","site"], columns="det", values="mb", aggfunc="size").fillna(0).astype(int)
summary = pd.DataFrame({
    "sites": SITES.groupby("batch").size(),
    "files": index.groupby("batch").size(),
    "MB":    index.groupby("batch").mb.sum().round(0),
    "sites with ETD": index[index.det=="ETD"].groupby("batch").site.nunique(),
    "sites with SE (alt. label)": index[index.det=="SE"].groupby("batch").site.nunique(),
}).fillna(0).astype(int)
display(summary)
print("Detector counts per site (1 = present):"); display(inv)
assert (inv.sum(1) == 3).all(), "every site should have exactly three detector images"''')

md("""Every site has exactly three detector images. The secondary-electron image is labelled `ETD` on 27 sites and `SE` on 4 (one in Batch 2, three in Batch 3). These are the same physical signal and are merged under `ETD` below, but the label change is itself a hint that acquisition sessions differed.

Batch 3 is 2.4× larger than the others. A reference batch is often the largest, but size alone does not make it the baseline.""")

md("""## 2 · Image geometry

All images are stitched strips ~7000 px wide. Height varies per site. Since each strip spans the full coating cross-section, **height is a proxy for coating thickness** — but only if magnification was constant. Microscope metadata was stripped when the files were re-saved (`Software = tifffile.py`), but the TIFF resolution tags survived and encode a **nominal 25.0 nm/pixel** on every image (24.9992–25.0005). That is export metadata, not verified calibration; at face value a strip is ~175 µm wide and the coating ~40–58 µm thick. We keep all KPIs in pixels and quote µm equivalents only as "nominal".""")

code(r'''geo = []
for _, r in SITES.iterrows():
    with tifffile.TiffFile(index.query("batch==@r.batch and site==@r.site and det=='BSE'").path.iloc[0]) as t:
        p = t.pages[0]; geo.append(dict(batch=r.batch, site=r.site, H=p.imagelength, W=p.imagewidth, dtype=str(p.dtype), spp=p.samplesperpixel))
geo = pd.DataFrame(geo)
display(geo.groupby("batch").agg(sites=("site","size"), H_min=("H","min"), H_median=("H","median"), H_max=("H","max"), H_cv=("H", lambda x: round(x.std()/x.mean(),3)), W_values=("W", lambda x: sorted(set(x)))))

fig, ax = plt.subplots(figsize=(8, 3.2))
for i, b in enumerate(BATCHES):
    y = geo[geo.batch==b].H.values
    ax.scatter(np.full_like(y, i, dtype=float) + np.random.uniform(-.12,.12,len(y)), y, s=28, color=COL[b], label=b, zorder=3)
    ax.hlines(np.median(y), i-.25, i+.25, color=COL[b], lw=2)
ax.set_xticks(range(3)); ax.set_xticklabels(BATCHES); ax.set_ylabel("image height (px)"); ax.set_title("Strip height per site (bar = batch median)")
plt.show()''')

md("""Median heights are similar across batches (~2060–2150 px) and the spread within a batch (CV 4–10 %) is as large as any difference between batches, so height carries no batch signal. Several exact heights recur across *different* batches (2080, 2148, 2156, 2272) and four Batch 3 sites share 2060 px while two Batch 1 sites share 2316 px. Repeated heights are more plausibly an acquisition-session fingerprint (same crop/stage settings) than coincidental equal thickness, which is why we treat height as a session indicator as much as a thickness proxy.""")

md("""## 3 · What the images look like

One site per batch, all three detectors, plus a 1600×800 px full-resolution crop.""")

code(r'''picks = [("Batch_1","5n1q8atc"), ("Batch_2","epqdaau9"), ("Batch_3","9luzk4jm")]
fig, axes = plt.subplots(3, 3, figsize=(20, 5.2))
for i, (b, s) in enumerate(picks):
    for j, d in enumerate(["BSE","ETD","Inlens"]):
        axes[i,j].imshow(load(b,s,d)[::6, ::6], cmap="gray", vmin=0, vmax=255, aspect="auto"); axes[i,j].axis("off"); axes[i,j].set_title(f"{b} · {s} · {d}", fontsize=9)
plt.suptitle("Full strips (6× downsampled)", y=1.01); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(3, 3, figsize=(20, 9))
for i, (b, s) in enumerate(picks):
    for j, d in enumerate(["BSE","ETD","Inlens"]):
        a = load(b,s,d); h, w = a.shape; cy, cx = h//2, w//2
        axes[i,j].imshow(a[cy-400:cy+400, cx-800:cx+800], cmap="gray", vmin=0, vmax=255); axes[i,j].axis("off"); axes[i,j].set_title(f"{b} · {s} · {d}  (1600×800 px crop)", fontsize=9)
plt.tight_layout(); plt.show()''')

md("""**Reading the images.** Ion-polished cross-sections of a porous electrode coating. The bulk phase is plate-like graphite (medium grey in BSE, with polishing striations in ETD). Scattered particles appear distinctly brighter in BSE, i.e. higher mean atomic number than carbon — consistent with silicon / silicon-oxide additive in a Si–graphite anode, though the chemistry is not confirmed. Black regions in BSE are open pores. Fine "spongy" texture between particles is the conductive-carbon/binder network.

The three detectors carry different information, and each contributes specific measurements below:
- **BSE** separates pore / graphite / bright phase by composition → phase fractions, particle and pore sizes, crack-like voids, through-thickness profiles (§6).
- **ETD** shows topography and polishing relief → dark-ridge density *inside* bright particles (intra-particle cracking), curtaining fraction and boundary sharpness as sample-preparation flags (§6c).
- **Inlens** is dominated by charging and surface contamination; its brightness swings wildly between sites → intra-particle texture only (speckled vs smooth particle interiors, §6c), never intensity.
The channels are recorded simultaneously and are pixel-aligned (phase-correlation shifts ≤ 0.2 px, checked on six sites), so BSE masks can be applied directly to ETD and Inlens.""")

md("""## 4 · Acquisition quality: histograms, contrast stretching, black level

Before trusting any intensity-derived feature, check whether images were acquired and processed consistently. Three diagnostics per image:
- `gray_levels` / `empty_bin_frac`: a comb-shaped histogram (many empty bins between p1 and p99) means a contrast stretch was applied after acquisition.
- `p1`: the 1st percentile. In BSE of a porous electrode, open pores should be essentially black (p1 ≈ 0). A raised p1 means either pores are filled (resin) or the detector black level/brightness was offset.
- top/bottom bright bands: copper current collector or stitching borders inside the frame.""")

code(r'''qpath = os.path.join(CACHE, "image_quality.csv")
if os.path.exists(qpath):
    Q = pd.read_csv(qpath)
else:
    rows = []
    for _, r in index.iterrows():
        a = tifffile.imread(r.path)[..., 0]
        lo, hi = np.percentile(a, [1, 99]).astype(int)
        h = np.bincount(a.ravel()[::3], minlength=256)
        t, bo = bright_bands(a)
        rows.append(dict(batch=r.batch, site=r.site, det=r.det_norm, H=a.shape[0], mean=a.mean(), p1=lo, p50=np.median(a), p99=hi, std=a.std(),
                         gray_levels=int((h > 0).sum()), empty_bin_frac=(h[lo:hi+1] == 0).mean(), band_top=t, band_bottom=bo))
    Q = pd.DataFrame(rows); Q.to_csv(qpath, index=False)
Q["contrast_stretched"] = Q.empty_bin_frac > 0.2
Q["raised_black_level"] = (Q.det == "BSE") & (Q.p1 > 10)

fig, axes = plt.subplots(1, 3, figsize=(20, 4), sharey=True)
for ax, d in zip(axes, ["BSE","ETD","Inlens"]):
    for _, r in index[index.det_norm == d].iterrows():
        a = tifffile.imread(r.path)[..., 0]
        h = np.bincount(a.ravel()[::9], minlength=256) / (a.size/9)
        ax.plot(np.arange(256), h, color=COL[r.batch], lw=0.8, alpha=0.6)
    ax.set_yscale("log"); ax.set_ylim(1e-7, 1); ax.set_title(f"{d} — per-image histograms"); ax.set_xlabel("gray level")
    for b in BATCHES: ax.plot([], [], color=COL[b], label=b)
    ax.legend()
axes[0].set_ylabel("fraction of pixels")
plt.tight_layout(); plt.show()

display(Q.groupby(["det","batch"]).agg(images=("site","size"), mean_gray=("mean","mean"), mean_sd=("mean","std"), p1_median=("p1","median"),
                                      stretched=("contrast_stretched","sum"), raised_black=("raised_black_level","sum")).round(1))''')

code(r'''bse = Q[Q.det=="BSE"].sort_values(["batch","p1","mean"])
display(bse[["batch","site","H","mean","p1","p50","p99","gray_levels","empty_bin_frac","contrast_stretched","raised_black_level"]].round(2).reset_index(drop=True))
GREY_PORE = sorted(bse[bse.raised_black_level].site.tolist()); print("BSE images with raised black level (p1 > 10):", GREY_PORE)''')

md("""Two acquisition findings:

1. **Contrast stretching is widespread and uneven.** About half of all images have comb histograms; Inlens almost always, BSE in roughly a third of sites, with up to 65 % of gray levels empty (`9luzk4jm`, `hzumfsms`). Absolute gray values are therefore not comparable across sites even within one batch. Any KPI built on raw intensity measures the operator's display settings as much as the material.
2. **Four Batch 3 sites have no black pixels in BSE** (`p1 ≈ 24` vs 0 everywhere else) and are the only images with *no* contrast stretching. Their histograms are a different shape, not a shifted copy. These are the same four sites with identical 2060 px height. Either their pores are filled with a grey phase (resin infiltration) or they were recorded in a different session with a different black level. We call them the **grey-pore group** and track them separately from here on.""")

md("""## 5 · Frame-edge artifacts

Bright bands at the top or bottom of a strip are copper current collector or stitching borders. They would inflate the bright-phase fraction if left in.""")

code(r'''bands = Q[(Q.band_top > 0) | (Q.band_bottom > 0)][["batch","site","det","H","band_top","band_bottom"]]
display(bands.reset_index(drop=True))
fig, axes = plt.subplots(len(bands), 1, figsize=(18, 1.6*len(bands)))
for ax, (_, r) in zip(np.atleast_1d(axes), bands.iterrows()):
    a = load(r.batch, r.site, r.det)
    strip = a[-200:, 1000:4500] if r.band_bottom else a[:200, 1000:4500]
    ax.imshow(strip, cmap="gray", vmin=0, vmax=255, aspect="auto"); ax.axis("off")
    ax.set_title(f"{r.batch} {r.site} {r.det} — {'bottom' if r.band_bottom else 'top'} 200 rows, band = {max(r.band_top, r.band_bottom)} px", fontsize=9)
plt.tight_layout(); plt.show()''')

md("""Only a handful of images are affected and the bands are thin (≤ 56 rows of ~2000). Trimming them is cheap insurance; the segmentation below does so.""")

md("""## 6 · Phase segmentation on BSE: pore / graphite / bright phase

Thresholds are set per image from the smoothed BSE histogram, anchored on the graphite mode (the dominant peak). Where a second, brighter mode is resolved, the bright-phase threshold is the valley between the two modes — the most defensible choice. Where no second mode exists, the threshold falls back to *graphite mode + 3.5 σ* and the image is flagged `bright_mode_resolved = False`. The same logic sets the pore threshold on the dark side. (A plain multi-Otsu was tried first and placed the upper threshold inside the graphite peak on low-contrast images, inflating the bright fraction 3×; that failure is documented in §4b below.)

This gives the first set of candidate material KPIs:

| KPI | Meaning for the material |
|---|---|
| `pore_frac` | area fraction of open porosity — electrolyte access, calendering density |
| `bright_frac` | area fraction of the high-Z additive (Si / SiOx) — formulation loading |
| `bright_count_per_Mpx`, `bright_d10/d50/d90` | number density and size distribution of additive particles (px units) — supplier powder PSD, agglomeration |
| `bright_circ` | mean circularity of additive particles — fractured vs intact |
| `pore_d50`, `pore_elong` | pore size and elongation — crack-like vs equiaxed porosity |
| `profile_*` | pore and bright fraction in 10 depth bins — through-thickness gradients (binder migration, additive settling) |
| `fft_slope`, `corr_len_px` | radial power-spectrum slope (edge sharpness / fine texture) and autocorrelation 1/e length (px) — scale-free texture descriptors independent of the segmentation |""")

code(r'''fpath = os.path.join(CACHE, "site_features.csv"); ppath = os.path.join(CACHE, "bright_particles.csv"); spath = os.path.join(CACHE, "fft_spectra.npz")

from scipy.signal import find_peaks

def phase_thresholds(sm):
    """Per-image pore/graphite/bright thresholds anchored on the graphite mode of a smoothed BSE histogram.
    Bright threshold = valley between graphite mode and a resolved second (bright) mode when one exists,
    otherwise graphite mode + 3.5 sigma_right (and the image is flagged as 'bright mode not resolved').
    Pore threshold   = valley between the dark mode and graphite mode when one exists, otherwise mode - 3.5 sigma_left."""
    h = np.bincount(sm[::3, ::3].astype(int).ravel(), minlength=256).astype(float); hs = ndi.gaussian_filter1d(h, 3) + 1e-9
    m = int(np.argmax(hs[10:200])) + 10; half = hs[m] / 2
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
                bright_mode_resolved=bright_resolved, bright_low_contrast=bright_low_contrast, pore_mode_resolved=pore_resolved, hist=hs/hs.sum())

def corr_length(img, n=16, size=512, seed=0):
    """1/e decay length (px) of the radially averaged autocorrelation of BSE texture."""
    rng = np.random.default_rng(seed); H, W = img.shape; acc = np.zeros(size//2)
    y, x = np.indices((size, size)); rr = np.hypot(y-size/2, x-size/2).astype(int)
    for _ in range(n):
        r0, c0 = rng.integers(0, H-size), rng.integers(0, W-size); p = img[r0:r0+size, c0:c0+size].astype(float); p -= p.mean()
        Fp = np.fft.fft2(p); ac = np.fft.fftshift(np.real(np.fft.ifft2(Fp*np.conj(Fp)))); ac /= ac.max()
        acc += (np.bincount(rr.ravel(), ac.ravel()) / np.maximum(np.bincount(rr.ravel()), 1))[:size//2]
    acc /= n; idx = np.where(acc < np.exp(-1))[0]
    return float(idx[0]) if len(idx) else np.nan

def radial_psd(img, n=24, size=512, seed=0):
    rng = np.random.default_rng(seed); H, W = img.shape
    acc = None
    y, x = np.indices((size, size)); r = np.hypot(y-size/2, x-size/2).astype(int)
    for _ in range(n):
        r0, c0 = rng.integers(0, H-size), rng.integers(0, W-size)
        p = img[r0:r0+size, c0:c0+size].astype(float); p -= p.mean(); p /= (p.std()+1e-9)
        P = np.abs(np.fft.fftshift(np.fft.fft2(p*np.hanning(size)[:,None]*np.hanning(size)[None,:])))**2
        rad = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
        acc = rad if acc is None else acc + rad
    return acc[1:size//2] / n

hpath = os.path.join(CACHE, "bse_histograms.npz")
if os.path.exists(fpath):
    F = pd.read_csv(fpath); P = pd.read_csv(ppath); SPEC = dict(np.load(spath)); HIST = dict(np.load(hpath))
else:
    feats, parts, SPEC, HIST = [], [], {}, {}
    for _, r in SITES.iterrows():
        a = trimmed(load(r.batch, r.site, "BSE")); H, W = a.shape
        sm = gaussian(a, sigma=1.0, preserve_range=True)
        th = phase_thresholds(sm); HIST[f"{r.batch}/{r.site}"] = th.pop("hist")
        pore, bright = sm < th["th_lo"], sm > th["th_hi"]
        pore = ndi.binary_opening(pore, iterations=1); bright = ndi.binary_opening(bright, iterations=1)
        f = dict(batch=r.batch, site=r.site, H=H, W=W, **th, pore_frac=pore.mean(), bright_frac=bright.mean(), graphite_frac=1-pore.mean()-bright.mean())
        # bright-phase particles
        lab = label(bright); rp = pd.DataFrame(regionprops_table(lab, properties=("area","equivalent_diameter_area","eccentricity","solidity","perimeter","major_axis_length","minor_axis_length")))
        rp = rp[rp.area >= 50]; rp["circ"] = 4*np.pi*rp.area/np.maximum(rp.perimeter,1)**2; rp["batch"], rp["site"] = r.batch, r.site
        parts.append(rp)
        d = rp.equivalent_diameter_area.values; wts = rp.area.values  # area-weighted PSD, like a laser-diffraction D-values
        order = np.argsort(d); cw = np.cumsum(wts[order])/wts.sum()
        f.update(bright_count_per_Mpx=len(rp)/(H*W/1e6), bright_d10=d[order][np.searchsorted(cw,.1)], bright_d50=d[order][np.searchsorted(cw,.5)], bright_d90=d[order][np.searchsorted(cw,.9)],
                 bright_circ=np.average(rp.circ, weights=wts), bright_solidity=np.average(rp.solidity, weights=wts), bright_max_d=d.max())
        # pores
        labp = label(pore); pp = pd.DataFrame(regionprops_table(labp, properties=("area","equivalent_diameter_area","major_axis_length","minor_axis_length"))); pp = pp[pp.area >= 30]
        dp, wp = pp.equivalent_diameter_area.values, pp.area.values; o = np.argsort(dp); cwp = np.cumsum(wp[o])/wp.sum()
        big = pp.major_axis_length > 500   # crack-like voids: long axis > 500 px (~¼ of the coating thickness)
        f.update(pore_count_per_Mpx=len(pp)/(H*W/1e6), pore_d50=dp[o][np.searchsorted(cwp,.5)], pore_d90=dp[o][np.searchsorted(cwp,.9)],
                 pore_elong=np.average(pp.major_axis_length/np.maximum(pp.minor_axis_length,1), weights=wp),
                 pore_max_d=dp.max(), crack_frac=pp.area[big].sum()/(H*W), crack_count_per_Mpx=big.sum()/(H*W/1e6))
        # through-thickness profiles (10 bins, top = separator side? unknown; keep image orientation)
        for k, (lo, hi) in enumerate(zip(np.linspace(0,H,11)[:-1].astype(int), np.linspace(0,H,11)[1:].astype(int))):
            f[f"profile_pore_{k}"] = pore[lo:hi].mean(); f[f"profile_bright_{k}"] = bright[lo:hi].mean()
        # texture scale
        spec = radial_psd(a); SPEC[f"{r.batch}/{r.site}"] = spec
        k = np.arange(1, len(spec)+1); sl = slice(4, 120)
        f["fft_slope"] = np.polyfit(np.log(k[sl]), np.log(spec[sl]), 1)[0]; f["corr_len_px"] = corr_length(a)
        feats.append(f); print(f"{r.batch} {r.site}: thr {f['th_lo']:.0f}/{f['th_hi']:.0f} bright sep {f['bright_sep']} low-contrast={f['bright_low_contrast']} | pore {f['pore_frac']:.3f} bright {f['bright_frac']:.3f} d50 {f['bright_d50']:.0f}px crack {f['crack_frac']:.4f} corr {f['corr_len_px']:.0f}px", flush=True)
    F = pd.DataFrame(feats); P = pd.concat(parts, ignore_index=True)
    F.to_csv(fpath, index=False); P.to_csv(ppath, index=False); np.savez(spath, **SPEC); np.savez(hpath, **HIST)
F["group"] = np.where(F.site.isin(GREY_PORE), "Batch_3 (grey-pore group)", F.batch)
groups = BATCHES + ["Batch_3 (grey-pore group)"]
print(F.shape, "sites × features")''')

code(r'''# §4b (placed here because it needs the per-image histograms): BSE histogram shapes aligned on the graphite mode
fig, axes = plt.subplots(1, 2, figsize=(17, 4.5))
for _, r in F.iterrows():
    hs = HIST[f"{r.batch}/{r.site}"]; x = np.arange(256) - r.graphite_mode
    ax = axes[1] if r.bright_low_contrast else axes[0]
    ax.plot(x, hs, color=COL[r.group], lw=1, alpha=.8, label=f"{r.site} (sep {r.bright_sep if r.bright_mode_resolved else 'n/a'})" if r.bright_low_contrast else None)
for ax, t in zip(axes, [f"normal bright-phase contrast ({int((~F.bright_low_contrast).sum())} sites)", f"LOW bright-phase contrast ({int(F.bright_low_contrast.sum())} sites): second mode < 45 levels above graphite or absent"]):
    ax.set_yscale("log"); ax.set_ylim(1e-6, .1); ax.set_xlim(-60, 120); ax.set_xlabel("gray level − graphite mode"); ax.set_title(t); ax.axvline(0, color="#777", lw=.8)
for g in groups: axes[0].plot([], [], color=COL[g], label=g)
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
plt.suptitle("§4b · Smoothed BSE histograms aligned on the graphite peak: how well is the bright phase separated?", y=1.02); plt.tight_layout(); plt.show()
print("Bright-phase separation (bright mode − graphite mode, gray levels) per group:")
display(F.groupby("group").bright_sep.describe()[["count","min","50%","max"]].round(0))
LOW_CONTRAST = sorted(F[F.bright_low_contrast].site.tolist()); print("Low-contrast sites:", LOW_CONTRAST)
display(F[F.bright_low_contrast][["batch","site","H","graphite_mode","bright_mode","bright_sep","th_lo","th_hi","bright_frac","bright_count_per_Mpx"]].round(3).reset_index(drop=True))

# segmentation overlays: one site per batch + one grey-pore site + one low-contrast site
show = picks + [("Batch_3", GREY_PORE[0])] + ([tuple(F[F.bright_low_contrast][["batch","site"]].iloc[0])] if F.bright_low_contrast.any() else [])
fig, axes = plt.subplots(len(show), 2, figsize=(20, 3.6*len(show)))
for i, (b, s) in enumerate(show):
    a = trimmed(load(b, s, "BSE")); h, w = a.shape; cy, cx = h//2, w//2; crop = a[cy-350:cy+350, cx-700:cx+700]
    r = F[(F.batch==b)&(F.site==s)].iloc[0]; sm = gaussian(crop, sigma=1.0, preserve_range=True)
    seg = np.zeros(crop.shape+(3,), float); seg[sm < r.th_lo] = (0.1,0.1,0.1); seg[(sm>=r.th_lo)&(sm<=r.th_hi)] = (0.55,0.55,0.55); seg[sm > r.th_hi] = (0.92,0.63,0.0)
    axes[i,0].imshow(crop, cmap="gray", vmin=0, vmax=255); axes[i,0].set_title(f"{b} · {s} · BSE  [{r.group}]", fontsize=10)
    axes[i,1].imshow(seg); axes[i,1].set_title(f"pore / graphite / bright · thresholds {r.th_lo:.0f}, {r.th_hi:.0f} · bright-phase separation: {r.bright_sep if r.bright_mode_resolved else 'unresolved'} levels", fontsize=10)
    for ax in axes[i]: ax.axis("off")
plt.tight_layout(); plt.show()''')

md("""The segmentation is clean wherever the bright phase is a well-separated mode (29 of 31 sites, separation 50–70 gray levels above graphite). Two kinds of site need care:
- **Low bright-phase contrast** (red rings in the KPI plots): on two Batch 1 sites the bright phase sits only ~25–35 levels above graphite, on the shoulder of the graphite peak. Any threshold there also picks up particle rims and the binder network, so their bright-phase fraction and particle count are inflated 2–8×. These two sites share an identical frame height (2316 px) that no other site has, which points to a separate acquisition session with different detector contrast rather than a different material. Their bright-phase KPIs must be shown as uncertain, not silently averaged in.
- **Grey-pore group**: the dark side has a broad plateau rather than a black mode, so the pore threshold falls back to *mode − 3.5 σ* and pore KPIs are not directly comparable with the rest.""")

code(r'''kpis = ["pore_frac","pore_d50","pore_elong","pore_max_d","crack_frac","crack_count_per_Mpx",
        "bright_frac","bright_count_per_Mpx","bright_d50","bright_d90","bright_circ",
        "fft_slope","corr_len_px","H"]
labels = {"pore_frac":"pore area fraction","pore_d50":"pore D50 (px, area-wtd)","pore_elong":"pore elongation (major/minor)","pore_max_d":"largest pore, equiv. diameter (px)",
          "crack_frac":"crack-like void area fraction (long axis > 500 px)","crack_count_per_Mpx":"crack-like voids per Mpx",
          "bright_frac":"bright-phase area fraction","bright_count_per_Mpx":"bright particles per Mpx","bright_d50":"bright-phase D50 (px, area-wtd)",
          "bright_d90":"bright-phase D90 (px)","bright_circ":"bright-phase circularity",
          "fft_slope":"radial PSD slope","corr_len_px":"texture correlation length (px)","H":"strip height (px)"}
np.random.seed(0)
fig, axes = plt.subplots(4, 4, figsize=(20, 14)); axes = axes.ravel()
for ax, k in zip(axes, kpis):
    for i, g in enumerate(groups):
        sub = F[F.group==g]
        y = sub[k].values; x = np.full(len(y), i, float) + np.random.uniform(-.14,.14,len(y))
        ax.scatter(x, y, s=26, color=COL[g], zorder=3); ax.hlines(np.median(y), i-.3, i+.3, color=COL[g], lw=2)
        lc = sub.bright_low_contrast.values
        if lc.any(): ax.scatter(x[lc], y[lc], s=110, facecolors="none", edgecolors="#e34948", lw=1.2, zorder=4)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(["B1","B2","B3","B3 grey"], fontsize=9); ax.set_title(labels[k], fontsize=10)
for ax in axes[len(kpis):]: ax.axis("off")
for g in groups: axes[-1].scatter([], [], color=COL[g], label=g)
axes[-1].scatter([], [], s=110, facecolors="none", edgecolors="#e34948", label="low bright-phase contrast\n(bright KPIs unreliable)")
axes[-1].legend(loc="center", fontsize=10)
plt.suptitle("Candidate KPIs per site, by batch (bar = median; Batch 3 split into ordinary and grey-pore sites)", y=1.0); plt.tight_layout(); plt.show()

tab = F.groupby("group")[kpis].agg(["median", "std"]).round(3)
display(tab.T)''')

md("""### 6b · Crack-like voids

The pore-structure KPIs single out a few sites with unusually large, elongated voids. Below: the three sites with the largest `crack_frac` and the three with the smallest, full strips at 6× downsampling, with voids whose long axis exceeds 500 px painted red. This is the most defect-like feature in the dataset and the kind of thing a materials engineer would want shown, not scored.""")

code(r'''top = F.sort_values("crack_frac", ascending=False); show = list(top.head(3)[["batch","site"]].itertuples(index=False, name=None)) + list(top.tail(3)[["batch","site"]].itertuples(index=False, name=None))
fig, axes = plt.subplots(len(show), 1, figsize=(20, 1.9*len(show)))
for ax, (b, s) in zip(axes, show):
    a = trimmed(load(b, s, "BSE")); r = F[(F.batch==b)&(F.site==s)].iloc[0]
    sm = gaussian(a, 1.0, preserve_range=True); pore = ndi.binary_opening(sm < r.th_lo, iterations=1)
    lab = label(pore); rp = pd.DataFrame(regionprops_table(lab, properties=("label","major_axis_length")))
    bigmask = np.isin(lab, rp.label[rp.major_axis_length > 500].values)
    rgb = np.stack([a]*3, -1).astype(float)/255; rgb[bigmask] = (0.89, 0.29, 0.28)
    ax.imshow(rgb[::6, ::6], aspect="auto"); ax.axis("off")
    ax.set_title(f"{b} · {s} [{r.group}] — crack-like void fraction {r.crack_frac:.4f}, {int(round(r.crack_count_per_Mpx*(r.H*r.W/1e6)))} voids > 500 px, largest pore ⌀ {r.pore_max_d:.0f} px", fontsize=9)
plt.tight_layout(); plt.show()''')

md("""### 6c · ETD and Inlens features inside the BSE masks

Everything so far is measured on BSE. The ETD and Inlens channels are pixel-aligned with it, so the BSE masks can be reused to ask questions BSE cannot answer:

| KPI | Channel | Meaning |
|---|---|---|
| `etd_crack_density_particles` | ETD | fraction of bright-particle *interior* covered by dark ridges that are **not** aligned with the ion-polishing curtaining direction — intra-particle cracking of the additive |
| `etd_crack_density_graphite` | ETD | same, inside graphite interiors |
| `etd_curtain_frac`, `etd_curtain_anisotropy` | ETD | fraction of solid interior covered by ridges aligned with the dominant orientation, and how dominant that orientation is — a **sample-preparation** flag (curtaining severity) |
| `etd_boundary_sharpness` | ETD | median ETD gradient on BSE solid/pore boundaries, normalised by ETD contrast — focus / polishing-quality flag |
| `inlens_particle_texture`, `inlens_speckled_particle_frac` | Inlens | per-particle Inlens intensity spread inside eroded bright particles, normalised by Inlens contrast — distinguishes speckled (porous / composite) from smooth (dense) particle interiors |
| `etd_grad_energy`, `inlens_grad_energy` | ETD / Inlens | normalised texture energy over the solid phase, used only for the three-channel agreement check below |

Ridges are detected as the positive principal curvature of the Gaussian-smoothed ETD image (σ = 2 px), normalised by the image's interquartile range. The curtaining direction is the modal ridge orientation inside solid interiors; ridges within ±15° of it are counted as curtaining, the rest as cracks. The ridge threshold is a single absolute value chosen once across all sites (the grid value nearest the median per-site 97th percentile), *not* a per-image percentile — a per-image percentile would make every density equal by construction.""")

code(r'''from skimage.filters import sobel
from skimage.feature import hessian_matrix
epath = os.path.join(CACHE, "etd_inlens_features.csv")
RIDGE_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

def multichannel_features(bse, etd, il, th_lo, th_hi, sigma=2.0, return_maps=False, T=None):
    sm = gaussian(bse, 1.0, preserve_range=True)
    bright = ndi.binary_opening(sm > th_hi, iterations=1); pore = ndi.binary_opening(sm < th_lo, iterations=1); solid = ~pore
    inner_p = ndi.binary_erosion(bright, iterations=6); inner_g = ndi.binary_erosion(solid & ~bright, iterations=8); interior = inner_p | inner_g
    e = etd.astype(float); scale = np.subtract(*np.percentile(e, [75, 25])) + 1e-6
    Hrr, Hrc, Hcc = hessian_matrix(e, sigma=sigma, order="rc", use_gaussian_derivatives=False)
    tr = Hrr + Hcc; det = Hrr*Hcc - Hrc**2; lam1 = tr/2 + np.sqrt(np.maximum(tr**2/4 - det, 0))
    ridge = np.maximum(lam1, 0) * sigma**2 / scale
    theta = np.degrees(0.5*np.arctan2(2*Hrc, Hrr - Hcc))
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
    if return_maps:
        strong = ridge > T; return out, dict(curtain=strong & aligned, crack=strong & ~aligned, inner_p=inner_p, inner_g=inner_g)
    return out

if os.path.exists(epath):
    E = pd.read_csv(epath)
else:
    rows = []
    for _, r in F.iterrows():
        t, bo = bright_bands(load(r.batch, r.site, "BSE")); sl = slice(t, -bo if bo else None)
        bse, etd, il = [load(r.batch, r.site, d)[sl] for d in ("BSE", "ETD", "Inlens")]
        o = multichannel_features(bse, etd, il, r.th_lo, r.th_hi); o.update(batch=r.batch, site=r.site); rows.append(o)
        print(f"{r.batch} {r.site}: p97 {o['ridge_p97']:.3f} dom {o['etd_ridge_dom_angle']:.0f}° sharp {o['etd_boundary_sharpness']:.2f} inlens tex {o.get('inlens_particle_texture', np.nan):.2f}", flush=True)
    E = pd.DataFrame(rows); E.to_csv(epath, index=False)

T_STAR = min(RIDGE_GRID, key=lambda t: abs(t - E.ridge_p97.median()))
E["etd_crack_density_particles"] = E[f"crack_p_{T_STAR}"]; E["etd_crack_density_graphite"] = E[f"crack_g_{T_STAR}"]; E["etd_curtain_frac"] = E[f"curtain_{T_STAR}"]
F = F.drop(columns=[c for c in F.columns if c.startswith(("etd_", "inlens_", "ridge_"))], errors="ignore").merge(E, on=["batch", "site"])
print(f"ridge threshold T* = {T_STAR} (median per-site p97 = {E.ridge_p97.median():.3f}, range {E.ridge_p97.min():.3f}–{E.ridge_p97.max():.3f})")''')

code(r'''# overlays at T*: one site per batch
fig, axes = plt.subplots(3, 3, figsize=(22, 10))
for i, (b, s_) in enumerate(picks):
    r = F[(F.batch==b)&(F.site==s_)].iloc[0]
    t, bo = bright_bands(load(b, s_, "BSE")); sl = slice(t, -bo if bo else None)
    bse, etd, il = [load(b, s_, d)[sl] for d in ("BSE", "ETD", "Inlens")]
    _, M = multichannel_features(bse, etd, il, r.th_lo, r.th_hi, return_maps=True, T=T_STAR)
    h, w = bse.shape; c = (slice(h//2-300, h//2+300), slice(w//2-500, w//2+500))
    axes[i,0].imshow(etd[c], cmap="gray"); axes[i,0].set_title(f"{b} · {s_} · ETD", fontsize=9)
    rgb = np.stack([etd[c]]*3, -1).astype(float)/255; rgb[M["curtain"][c]] = (0.2, 0.5, 1.0); rgb[M["crack"][c]] = (1.0, 0.3, 0.2)
    axes[i,1].imshow(rgb); axes[i,1].contour(M["inner_p"][c], levels=[.5], colors="yellow", linewidths=.6)
    axes[i,1].set_title(f"blue = curtaining ({r.etd_ridge_dom_angle:.0f}°) · red = other ridges · yellow = particle interiors · crack density in particles {r.etd_crack_density_particles:.4f}", fontsize=9)
    axes[i,2].imshow(il[c], cmap="gray"); axes[i,2].contour(M["inner_p"][c], levels=[.5], colors="yellow", linewidths=.6); axes[i,2].set_title(f"Inlens · particle texture {r.inlens_particle_texture:.2f}", fontsize=9)
    for ax in axes[i]: ax.axis("off")
plt.tight_layout(); plt.show()

mc = ["etd_crack_density_particles","etd_crack_density_graphite","etd_curtain_frac","etd_curtain_anisotropy","etd_boundary_sharpness","inlens_particle_texture","inlens_speckled_particle_frac","inlens_particles_measured"]
mlab = {"etd_crack_density_particles":"ETD crack density in bright particles","etd_crack_density_graphite":"ETD crack density in graphite","etd_curtain_frac":"ETD curtaining fraction (prep flag)",
        "etd_curtain_anisotropy":"ETD curtaining anisotropy (prep flag)","etd_boundary_sharpness":"ETD boundary sharpness (prep/focus flag)","inlens_particle_texture":"Inlens intra-particle texture (median)",
        "inlens_speckled_particle_frac":"fraction of speckled particles (Inlens)","inlens_particles_measured":"particles measured (n ≥ 400 px interior)"}
np.random.seed(1)
fig, axes = plt.subplots(2, 4, figsize=(20, 7.5)); axes = axes.ravel()
for ax, k in zip(axes, mc):
    for i, g in enumerate(groups):
        sub = F[F.group==g]; y = sub[k].values; x = np.full(len(y), i, float) + np.random.uniform(-.14, .14, len(y))
        ax.scatter(x, y, s=26, color=COL[g], zorder=3); ax.hlines(np.nanmedian(y), i-.3, i+.3, color=COL[g], lw=2)
        lc = sub.bright_low_contrast.values
        if lc.any(): ax.scatter(x[lc], y[lc], s=110, facecolors="none", edgecolors="#e34948", lw=1.2, zorder=4)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(["B1","B2","B3","B3 grey"], fontsize=9); ax.set_title(mlab[k], fontsize=10)
plt.suptitle("ETD / Inlens KPIs per site (red ring = low bright-phase contrast, particle masks unreliable)", y=1.0); plt.tight_layout(); plt.show()
display(F.groupby("group")[mc].agg(["median","std"]).round(4).T)''')

md("""**Is the Inlens texture KPI material or instrument?** It orders the batches more strongly than any BSE feature, which is exactly when to be suspicious of a charging-sensitive channel. The cell below checks how much of it is explained by Inlens acquisition statistics alone.""")

code(r"""from scipy import stats as _st
import statsmodels.formula.api as smf
qi = Q[Q.det=="Inlens"][["batch","site","mean","p50","std","empty_bin_frac"]].rename(columns={"mean":"il_mean","p50":"il_p50","std":"il_std","empty_bin_frac":"il_stretch"})
Dc = F.merge(qi, on=["batch","site"])
print("Spearman ρ of inlens_particle_texture with …")
for c, lab_ in [("il_p50","Inlens median brightness"),("il_mean","Inlens mean"),("il_std","Inlens std"),("il_stretch","Inlens contrast-stretch fraction"),("H","frame height (session fingerprint)"),("bright_d50","bright-phase D50 (material)"),("inlens_grad_energy","Inlens texture energy (same channel)")]:
    r_, p_ = _st.spearmanr(Dc[c], Dc.inlens_particle_texture, nan_policy="omit"); print(f"   {lab_:38s} ρ = {r_:+.2f}   p = {p_:.3f}")
m = smf.ols("inlens_particle_texture ~ il_p50 + il_std + il_stretch", data=Dc).fit(); Dc["resid"] = m.resid
Fr, pr = _st.f_oneway(*[g.resid.values for _, g in Dc.groupby("group")]); Fw, pw = _st.f_oneway(*[g.inlens_particle_texture.values for _, g in Dc.groupby("group")])
print(f"\nR² of an acquisition-only model (Inlens median, std, stretch): {m.rsquared:.2f}")
print(f"Group effect on the raw KPI:        F = {Fw:.1f}, p = {pw:.4f}")
print(f"Group effect on the residual KPI:   F = {Fr:.1f}, p = {pr:.4f}")
print("Within-group ρ(texture, Inlens median):", {g: round(_st.spearmanr(d.il_p50, d.inlens_particle_texture)[0], 2) for g, d in Dc.groupby("group") if len(d) >= 5})""")

md("""**Reading §6c.**
- *Intra-particle cracking is essentially absent.* ETD ridge density inside bright-particle interiors is 0.02–0.5 % of interior area in every group; the overlays show intact, smoothly polished particle faces. The KPI is a null result on this dataset, but it is exactly the measurement that would catch fractured additive particles in an unseen batch, so it stays in the pipeline.
- *Ridge density in graphite and curtaining fraction are highest in the grey-pore group* (0.041 vs 0.021–0.030; 0.0105 vs 0.006–0.008). Together with that group's missing black level, this reads as a different polishing / imaging session rather than a different material.
- *Boundary sharpness* is lower across Batch 3 (0.77 vs 0.92–0.97), a mild focus / polish-quality difference that the QC report should surface as a flag.
- *Inlens intra-particle texture* orders the groups cleanly (B1 0.32 > B2 0.24 > B3 0.18 > grey 0.13) and is the single strongest between-group feature in §10 — **but it is heavily confounded.** It correlates ρ ≈ 0.8 with Inlens median brightness and ρ ≈ 0.7 with frame height, an acquisition-only model explains ~75 % of its variance, and inside each batch it still tracks Inlens brightness (ρ ≈ 0.7 in Batches 2 and 3). A weak group effect survives on the residuals (p ≈ 0.02), so there may be a real difference in particle interiors (speckled vs smooth), but it cannot be separated from charging contrast with this normalisation. Treat it as *promising, not trusted*; in the QC pipeline it needs local-contrast or rank normalisation and must never drive a verdict on its own.""")

md("""### 6d · Three-channel agreement

A material change should move all three detectors in a consistent way; an instrument or display change typically moves one channel's intensity without moving texture, or one channel without the others. The table below standardises six per-site quantities — median intensity of each channel (from §4) and one texture measure per channel (BSE correlation length, ETD and Inlens gradient energy) — and reports, per site, the spread of the intensity z-scores across channels and the spread of the texture z-scores across channels. Large spread = channels disagree = lower confidence that the shift is material.""")

code(r'''q = Q.pivot_table(index=["batch","site"], columns="det", values="p50").reset_index().rename(columns={"BSE":"BSE_p50","ETD":"ETD_p50","Inlens":"Inlens_p50"})
A = F[["batch","site","group","corr_len_px","etd_grad_energy","inlens_grad_energy","bright_low_contrast"]].merge(q, on=["batch","site"])
cols_int = ["BSE_p50","ETD_p50","Inlens_p50"]; cols_tex = ["corr_len_px","etd_grad_energy","inlens_grad_energy"]
Z = A.copy()
for c in cols_int + cols_tex: Z[c] = (A[c] - A[c].median()) / (1.4826 * np.median(np.abs(A[c] - A[c].median())) + 1e-9)   # robust z
Z["intensity_disagreement"] = Z[cols_int].std(axis=1); Z["texture_disagreement"] = Z[cols_tex].std(axis=1)
Z = Z.sort_values(["group","site"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 10))
M = Z[cols_int + cols_tex].values
im = ax.imshow(np.clip(M, -4, 4), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
ax.set_xticks(range(6)); ax.set_xticklabels(["BSE median","ETD median","Inlens median","BSE corr. length","ETD texture energy","Inlens texture energy"], rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(Z))); ax.set_yticklabels([f"{r.site}" for _, r in Z.iterrows()], fontsize=8)
for lbl, (_, r) in zip(ax.get_yticklabels(), Z.iterrows()): lbl.set_color(COL[r.group])
ax.axvline(2.5, color="k", lw=1); ax.grid(False); ax.set_title("Robust z-scores per site (rows coloured by group); left block = intensity, right block = texture")
plt.colorbar(im, ax=ax, shrink=.6, label="robust z"); plt.tight_layout(); plt.show()

print("Sites whose channels disagree most (spread of z across the three channels):")
display(Z.sort_values("texture_disagreement", ascending=False).head(8)[["batch","site","group","intensity_disagreement","texture_disagreement","bright_low_contrast"]].round(2).reset_index(drop=True))
display(Z.groupby("group")[["intensity_disagreement","texture_disagreement"]].median().round(2))''')

md("""**Reading §6d.** The grey-pore group is the clearest case of channel *disagreement*: BSE median is +2 to +3 robust-z (no black pixels) while the Inlens median is −2 to −3, with texture nearly unchanged — intensity moved in opposite directions in two detectors recorded simultaneously, which is a detector/preparation signature, not a material one. The three cracked Batch 3 sites show the opposite pattern: intensities are ordinary but ETD texture energy drops to −2 to −3 z while BSE correlation length rises, i.e. texture changed across channels while intensity did not — consistent with a real structural change (large flat crack faces). This is the logic the QC notebook will formalise: intensity-only shifts lower confidence in a material verdict, texture shifts that agree across channels raise it.""")

md("""## 7 · Bright-phase particle size distribution

Area-weighted cumulative distributions of equivalent diameter, pooled per group, plus per-site D50 against number density. The two low-contrast sites are excluded here because their "bright" class is contaminated. Units are pixels; without a scale bar only *relative* shifts are meaningful.""")

code(r'''P["group"] = np.where(P.site.isin(GREY_PORE), "Batch_3 (grey-pore group)", P.batch)
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
for g in groups:
    sub = P[(P.group==g) & ~P.site.isin(LOW_CONTRAST)]
    d = np.sort(sub.equivalent_diameter_area.values); w = sub.area.values[np.argsort(sub.equivalent_diameter_area.values)]
    axes[0].plot(d, np.cumsum(w)/w.sum(), color=COL[g], lw=2, label=f"{g} (n={len(d):,} particles)")
axes[0].set_xscale("log"); axes[0].set_xlabel("equivalent diameter (px)"); axes[0].set_ylabel("cumulative area fraction"); axes[0].set_title("Bright-phase PSD, area-weighted, pooled per group"); axes[0].legend(fontsize=8)
for i, g in enumerate(groups):
    sub = F[(F.group==g) & ~F.bright_low_contrast]
    axes[1].errorbar(sub.bright_d50, sub.bright_count_per_Mpx, fmt="o", color=COL[g], ms=6, label=g)
    for _, r in sub.iterrows(): axes[1].annotate(r.site, (r.bright_d50, r.bright_count_per_Mpx), fontsize=6, alpha=.7, xytext=(3,3), textcoords="offset points")
axes[1].set_xlabel("bright-phase D50 (px)"); axes[1].set_ylabel("bright particles per Mpx"); axes[1].set_title("Per-site: size vs number density"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()''')

md("""## 8 · Through-thickness profiles

Pore and bright-phase fraction in ten equal depth bins from the top of the strip to the bottom. Image orientation relative to the current collector is not recorded, so "top" is image-top; the single site with copper at the bottom (`epqdaau9`) suggests collector-down, but this should be confirmed.""")

code(r'''fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
depth = (np.arange(10)+0.5)/10
for ax, what in zip(axes, ["pore","bright"]):
    cols = [f"profile_{what}_{k}" for k in range(10)]
    for g in groups:
        sub = F[F.group==g]; m, s = sub[cols].mean().values, sub[cols].std().values
        ax.plot(depth, m, color=COL[g], lw=2, label=g); ax.fill_between(depth, m-s, m+s, color=COL[g], alpha=.12)
    ax.set_xlabel("normalised depth (image top → bottom)"); ax.set_ylabel(f"{what} area fraction"); ax.set_title(f"{what} fraction vs depth (mean ± 1 sd across sites)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()''')

md("""## 9 · Spatial texture scale (radial power spectrum)

Mean radially-averaged power spectrum of 512-px BSE patches per group. The slope reflects edge sharpness and fine texture; a shift of the curve left or right indicates coarser or finer microstructure overall — a scale-free cross-check on the segmentation-based sizes.""")

code(r'''fig, ax = plt.subplots(figsize=(9, 4.5))
k = np.arange(1, 256)
for g in groups:
    keys = [f"{r.batch}/{r.site}" for _, r in F[F.group==g].iterrows()]
    S = np.array([SPEC[kk] for kk in keys]); m = S.mean(0)
    ax.plot(512/k, m, color=COL[g], lw=2, label=g)
    ax.fill_between(512/k, np.percentile(S,10,0), np.percentile(S,90,0), color=COL[g], alpha=.1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("spatial period (px)"); ax.set_ylabel("power (a.u.)"); ax.set_title("Radial PSD of BSE texture (mean, 10–90 % band)"); ax.legend(fontsize=8)
plt.show()''')

md("""## 10 · Site-level structure: do batches separate?

Standardise the candidate KPIs, project to two principal components, and cluster. Also compute, per KPI, how much of the variance is *between* batches versus *within* (one-way ANOVA F and η²), using Batch 3 split into its two groups.""")

code(r'''feat_cols = ["pore_frac","pore_d50","pore_elong","pore_max_d","crack_frac","bright_frac","bright_count_per_Mpx","bright_d50","bright_d90","bright_circ","fft_slope","corr_len_px",
             "etd_crack_density_particles","etd_boundary_sharpness","inlens_particle_texture"]
X = F[feat_cols].fillna(F[feat_cols].median()).values; Xs = (X - X.mean(0)) / X.std(0)
U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False); pcs = U[:, :2] * S[:2]; expl = S**2 / (S**2).sum()
fig, axes = plt.subplots(1, 2, figsize=(17, 6))
for g in groups:
    m = (F.group==g).values; axes[0].scatter(pcs[m,0], pcs[m,1], s=50, color=COL[g], label=g, zorder=3)
for i, r in F.iterrows(): axes[0].annotate(r.site, pcs[i], fontsize=7, alpha=.75, xytext=(4,3), textcoords="offset points")
axes[0].set_xlabel(f"PC1 ({expl[0]:.0%})"); axes[0].set_ylabel(f"PC2 ({expl[1]:.0%})"); axes[0].set_title("PCA of standardised site KPIs"); axes[0].legend(fontsize=8)
load_ = pd.DataFrame(Vt[:2].T, index=feat_cols, columns=["PC1","PC2"]).round(2)
axes[1].barh(feat_cols, load_.PC1, color="#2a78d6", height=.38, label="PC1"); axes[1].barh(np.arange(len(feat_cols))+.4, load_.PC2, color="#eb6834", height=.38, label="PC2")
axes[1].set_yticks(np.arange(len(feat_cols))+.2); axes[1].set_yticklabels(feat_cols); axes[1].set_title("PC loadings"); axes[1].legend()
plt.tight_layout(); plt.show()

from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(Xs, "ward")
fig, ax = plt.subplots(figsize=(17, 4))
dn = dendrogram(Z, labels=[f"{r.batch[-1]}·{r.site}" for _, r in F.iterrows()], ax=ax, leaf_font_size=8, color_threshold=0, above_threshold_color="#777")
for lbl in ax.get_xmajorticklabels():
    s = lbl.get_text().split("·")[1]; g = F[F.site==s].group.iloc[0]; lbl.set_color(COL[g])
ax.set_title("Ward clustering of sites (leaf colour = batch / group)"); ax.grid(False); plt.show()''')

code(r'''rows = []
for k in feat_cols:
    samples = [F[F.group==g][k].values for g in groups]
    Fstat, p = stats.f_oneway(*samples)
    grand = F[k].mean(); ssb = sum(len(s)*(s.mean()-grand)**2 for s in samples); sst = ((F[k]-grand)**2).sum()
    # same test with Batch 3 kept whole, to show how much the grey-pore split matters
    s3 = [F[F.batch==b][k].values for b in BATCHES]; F3, p3 = stats.f_oneway(*s3)
    rows.append(dict(KPI=k, eta2_split=ssb/sst, F_split=Fstat, p_split=p, F_batch_only=F3, p_batch_only=p3))
sep = pd.DataFrame(rows).sort_values("eta2_split", ascending=False).round(4); display(sep)
print("η² = share of variance explained by group membership (4 groups: B1, B2, B3-ordinary, B3-grey). 'batch_only' uses the three folder labels as given.")''')

md("""**Reading §10.** The first split in the dendrogram is the two low-contrast Batch 1 sites (`4ih2ggld`, `5n1q8atc`) — an acquisition artefact, not material. The second split is the three Batch 3 sites with large delamination-like cracks (`hzumfsms`, `0grcilhi`, `ufdvpb81`) — the only candidate *material* defect signature in the data. Below those two branches, Batch 1, Batch 2 and Batch 3 sites interleave freely: on these KPIs the three folders are **not** separable as batches. The grey-pore sites sit together next to Batch 2 sites, because both have lower pore fraction and smaller pores.

The η² table says the same thing: `inlens_particle_texture` tops the list (η² ≈ 0.53) but is confounded with Inlens brightness (§6c) and should be read as a session effect until proven otherwise. Among trusted KPIs, once Batch 3 is split, pore-structure KPIs (`pore_frac`, `pore_max_d`, `pore_d50`, `crack_frac`) explain 27–36 % of between-site variance, bright-phase KPIs 18–24 % (and that mostly from the two low-contrast sites), and nothing reaches the level where a 7-site batch could be called different with confidence. Using the folder labels as given, only the bright-phase KPIs reach p < 0.05 — and those are the ones contaminated by the Batch 1 contrast problem.""")

md("""## 11 · Findings

**What the dataset is**
- 31 sites × 3 detectors = 93 stitched BSE / ETD / Inlens cross-sections, ~7000 × 1600–2300 px, 8-bit. Batch 1: 7 sites, Batch 2: 7, Batch 3: 17. Porous graphite-plate coating with a sparse bright (higher-Z) additive phase; chemistry unconfirmed.
- Microscope settings did not survive; the resolution tags give a nominal 25.0 nm/px (unverified). All sizes are reported in pixels with nominal µm in parentheses where helpful (bright-phase D50 ≈ 150 px ≈ 3.8 µm; crack-like void cutoff 500 px ≈ 12.5 µm). Strip height carries no batch signal and partly fingerprints acquisition sessions.

**Acquisition is not uniform, and it matters more than batch**
1. *Contrast stretching* was applied unevenly (~half of images, strongest in Inlens). Raw gray values are not comparable between sites; every structural KPI must use per-image, histogram-anchored thresholds, and intensity statistics belong in an "instrument flags" bucket.
2. *Two Batch 1 sites* (`4ih2ggld`, `5n1q8atc`; both 2316 px tall) were recorded with the bright phase only ~25–35 gray levels above graphite instead of the usual 50–70. No threshold can separate the additive from particle rims and binder there; their bright-phase fraction and particle count come out 2–8× too high. They dominate PC1 and are the first split in the clustering. **Bright-phase KPIs for these two sites are unreliable and must be flagged, not averaged.**
3. *Four Batch 3 sites* (`71vgq3fw`, `kbdh4tri`, `tuy3zymq`, `x7u69zsw`; all 2060 px tall) have no black pixels, a grey plateau where pores should be, the lowest bright-phase separation of the normal-contrast sites (46–47 levels), and darker Inlens. This "grey-pore group" is most likely a different preparation or session (e.g. resin-filled pores). Their pore KPIs rest on a fallback threshold and read systematically low (pore fraction 0.075 vs ~0.09, pore D50 75 vs ~105 px).
4. A few images carry a thin copper or stitching band at one edge (≤ 56 rows); trimmed before measurement.

**Cross-channel measurements (§6c–6d)**
- ETD and Inlens are pixel-aligned with BSE, so BSE masks transfer directly. Intra-particle crack density of the bright phase (ETD dark ridges inside particle interiors, curtaining-corrected) is 0.02–0.5 % everywhere: the additive particles are intact in all three batches. The KPI stays because it is the one that would catch fractured particles in the unseen batch.
- Curtaining fraction, graphite ridge density and boundary sharpness behave as *preparation flags*: the grey-pore group has the most curtaining and ridge relief, Batch 3 overall the softest boundaries.
- Inlens intra-particle texture is the strongest batch-ordered feature in the whole table (B1 > B2 > B3 > grey), but ~75 % of it is explained by Inlens brightness and session fingerprints, and only a weak group effect (p ≈ 0.02) survives after removing them. It is kept as a *candidate* KPI, flagged as confounded, and must not drive a verdict without better normalisation.
- A three-channel agreement check (robust z of per-channel intensity and texture) separates the two kinds of anomaly already visible: the grey-pore group moves BSE and Inlens intensity in opposite directions with texture unchanged (instrument / preparation), the cracked sites move texture across channels with intensity unchanged (material).

**Material structure**
- Phase fractions are stable where acquisition is normal: bright-phase area fraction 0.04–0.08 (median ≈ 0.055) in every group, bright-phase D50 ≈ 140–165 px, pore fraction 0.06–0.14. Formulation loading does not differ between batches on this evidence.
- The one clear *material* anomaly is **large, elongated crack-like voids** in three Batch 3 sites (`hzumfsms`, `0grcilhi`, `ufdvpb81`): crack-like void fraction 0.048–0.062 versus a median of 0.015–0.022 elsewhere, largest pore ⌀ 450–600 px versus ≈ 250–300 px. §6b shows them: delamination-style cracks running along the coating. They form the second split in the clustering and drive PC2.
- Batch 2 has the lowest and tightest pore fraction (0.075 ± 0.008) and the fewest crack-like voids. Batch 3 ordinary sites have a slightly longer texture correlation length (22 vs 18–19 px), i.e. marginally coarser microstructure. Both are small relative to within-batch spread.
- Through-thickness profiles show no systematic gradient in any group; the bright-phase profile of Batch 1 is wide only because of the two low-contrast sites.
- After removing the two acquisition sub-groups and the three cracked sites, the three batches are **not separable** on these KPIs: sites interleave across all three folders in PCA and clustering, and no KPI reaches a between-batch effect that 7 sites could establish with confidence.

**Implications for the QC design**
1. Build on BSE-segmentation KPIs (pore fraction and size, crack-like void fraction, largest void, bright-phase fraction / count / D50 / circularity) plus one scale-free texture descriptor. From ETD take intra-particle crack density and the preparation flags (curtaining, boundary sharpness); from Inlens take intra-particle texture only after local-contrast normalisation, and label it as confounded until validated. Never use any channel's raw intensity as a material KPI.
2. Make acquisition-quality flags first-class outputs: bright-phase separation, black level, contrast stretching, edge bands, detector label, curtaining, boundary sharpness, and the three-channel agreement score. The two biggest structures in this dataset are acquisition effects, and a QC system that cannot say "the microscope changed, not the material" will produce false rejects.
3. Calibrate every threshold against *within-baseline* spread (split-half / leave-one-site-out). With 7 sites per batch the sampling noise is large and has to be displayed, not hidden; several KPIs differ between batches by less than their within-batch standard deviation.
4. Crack-like void fraction and largest void size are the KPIs most likely to carry a real accept / investigate / reject signal; show the detected voids on the image when they drive a verdict.
5. Keep Batch 3 split until the organisers confirm what it is. If it is the baseline, the grey-pore sites and the three cracked sites must be handled explicitly (excluded, or modelled as known sub-populations), otherwise the baseline's own spread will swallow any incoming defect.

**Open questions for the organisers:** which batch is the approved baseline; whether the four grey-pore sites and the two low-contrast sites are intentional; pixel size; image orientation relative to the current collector; whether the cracked Batch 3 sites are known defects.

**Cached outputs** (reused by later notebooks): `analysis_cache/site_features.csv` (one row per site, all KPIs and quality flags), `analysis_cache/etd_inlens_features.csv` (ETD / Inlens KPIs per site), `analysis_cache/bright_particles.csv` (one row per bright-phase particle), `analysis_cache/image_quality.csv` (one row per image).""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = "/Users/santoshkumarsaravanan/Documents/Polaron/notebooks/01_dataset_analysis.ipynb"
nbf.write(nb, out); print("wrote", out, len(cells), "cells")
