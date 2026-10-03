"""Build docs/assumption_register.html: every interpretive assumption we rely on, tied to an annotated
example image, with a review status + notes the reader can set (stored in the browser) and export.

Run from the repo root:  python3 docs/_build_assumption_register.py
Images are embedded as base64 JPEG so the file is self-contained (~5 MB).
"""
import os, io, base64, html, datetime
import numpy as np, pandas as pd, tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage.filters import gaussian, sobel
from skimage.feature import hessian_matrix
from skimage.measure import label, regionprops, regionprops_table
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Dataset"); CACHE = os.path.join(ROOT, "analysis_cache")
OUT = os.path.join(ROOT, "docs", "assumption_register.html")
F = pd.read_csv(os.path.join(CACHE, "site_features.csv")).merge(pd.read_csv(os.path.join(CACHE, "etd_inlens_features.csv")), on=["batch", "site"])
Q = pd.read_csv(os.path.join(CACHE, "image_quality.csv"))
NM = 25.0  # nominal nm/px from TIFF tags (unverified)

# ----------------------------------------------------------------------------- helpers
def load(batch, site, det="BSE"):
    p = os.path.join(DATA, batch, f"img_{site}_{det}.tif")
    if not os.path.exists(p) and det == "ETD": p = os.path.join(DATA, batch, f"img_{site}_SE.tif")
    return tifffile.imread(p)[..., 0]

def row(site): return F[F.site == site].iloc[0]

def fig_to_b64(fig, width_px=1100, quality=82):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.05, facecolor="white"); plt.close(fig)
    im = Image.open(buf).convert("RGB")
    if im.width > width_px: im = im.resize((width_px, int(im.height * width_px / im.width)), Image.LANCZOS)
    out = io.BytesIO(); im.save(out, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()

def new_fig(ncols=1, nrows=1, w=11, h=4.2):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h)); return fig, np.atleast_1d(axes).ravel()

def show(ax, img, title=None, cmap="gray", vmin=0, vmax=255, aspect=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect or "equal"); ax.set_axis_off()
    if title: ax.set_title(title, fontsize=10, loc="left")

def mark(ax, xy, text, color="#ffd166", dxy=(90, -70), fs=9):
    ax.annotate(text, xy=xy, xytext=(xy[0] + dxy[0], xy[1] + dxy[1]), fontsize=fs, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6))

def box(ax, x0, y0, w, h, color="#ffd166", lw=1.6, text=None, fs=9):
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=color, lw=lw))
    if text: ax.text(x0, y0 - 6, text, fontsize=fs, color="black", bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"))

def crop(img, cy, cx, h, w):
    H, W = img.shape; y0 = int(np.clip(cy - h // 2, 0, H - h)); x0 = int(np.clip(cx - w // 2, 0, W - w))
    return img[y0:y0 + h, x0:x0 + w], (y0, x0)

def masks(bse, r):
    sm = gaussian(bse, 1.0, preserve_range=True)
    return sm, ndi.binary_opening(sm < r.th_lo, iterations=1), ndi.binary_opening(sm > r.th_hi, iterations=1)

def largest_component_centroid(mask, min_area=1, solidity_min=0.0):
    lab = label(mask); best = None
    for p in regionprops(lab):
        if p.area >= min_area and p.solidity >= solidity_min and (best is None or p.area > best.area): best = p
    return None if best is None else (best.centroid[1], best.centroid[0], best)

def smoothed_hist(img):
    h = np.bincount(img.ravel()[::3], minlength=256).astype(float); return ndi.gaussian_filter1d(h, 2) / h.sum()

def um(px): return f"{px * NM / 1000:.1f} µm"

CARDS = []  # dicts: id, title, confidence, used_for, figures=[(b64, caption)], think, why, wrong_if, ask
REVIEW = {  # persistent review log, appended by hand when feedback arrives (date, who, what)
    "A3": ["2026-10-03 · Santosh: confirmed.", "2026-10-03 · Santosh: fresh graphite–Si/SiOx electrode confirmed. This confirms the material family, not the chemical identity of every thresholded fragment; exact Si versus SiOx and pixelwise chemical validation are unspecified."],
    "A2": ["2026-10-03 · Santosh: the black regions are not 'vacuum' in any meaningful sense — they are the open porosity that electrolyte fills in the cell. Wording changed. The binder arrow pointed at plain graphite; replaced by full-resolution insets of the binder / conductive-carbon network in BSE and ETD.", "2026-10-03 · Santosh: fresh graphite–Si/SiOx electrode confirmed. Material identity is now known; binder/network identification and each mask label are still unvalidated."],
    "A10": ["2026-10-03 · battery application review: low detected ridge-pixel coverage cannot establish intact particles or electronic connectivity. Wording corrected; fresh-state metadata rules out interpreting visible features as prior cycling damage."],
    "A17": ["2026-10-03 · Santosh, direct reply: 'Fresh graphite–Si/SiOx electrode confirmed'. Fresh/uncycled state and material family confirmed; exact chemistry and formulation not supplied."],
    "A13": ["2026-10-03 · Santosh: confirmed that the files labelled 'SE' are the same secondary-electron detector as 'ETD'. (Pixel alignment of the three channels is our own measurement, not externally confirmed.)"],
    "A15": ["2026-10-03 · Santosh (from the problem providers): the three folders are three supplier batches of the same nominal product. Batch 3 is one batch with more samples, the closest available to a baseline but not a perfect one. There is no clear baseline; the goal is to differentiate. Card rewritten accordingly.",
            "2026-10-03 · challenge provider, verbatim via Santosh: 'batch 3 is the reference dataset (it should have more images). Note that reference doesn't necessarily mean no defects, and it's not just the presence of defects that define the batches — there's a lot of complex morphology features to examine!' → confirms the reference; the three cracked reference sites are consistent with 'reference does not mean no defects'; differentiation should lean on morphology (pore and particle shape, orientation, arrangement, size-distribution shape, through-thickness structure), not only defect KPIs."],
    "A4":  ["2026-10-03 · wording of 'vacuum' changed following the A2 comment.", "2026-10-03 · with A15 confirmed, the four grey-pore sites belong to the same production batch as the other 13 Batch 3 sites, which strengthens the preparation / session interpretation over a material one — but does not prove it."],
}

def add(id_, title, confidence, figures, think, why, used_for, wrong_if, ask):
    CARDS.append(dict(id=id_, title=title, confidence=confidence, figures=figures, think=think, why=why, used_for=used_for, wrong_if=wrong_if, ask=ask))
    print("built", id_, flush=True)

# ----------------------------------------------------------------------------- A1 what the images are
b, s = "Batch_2", "epqdaau9"; bse = load(b, s)
fig = plt.figure(figsize=(12, 6.2)); gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.25)
ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax = [ax0, ax1]
show(ax[0], bse[::4, ::4], f"{b} / {s} / BSE — full stitched strip, 4× downsampled ({bse.shape[1]} × {bse.shape[0]} px ≈ {um(bse.shape[1])} × {um(bse.shape[0])} nominal)", aspect="auto")
H4, W4 = bse.shape[0] // 4, bse.shape[1] // 4
box(ax[0], 2, 2, W4 - 4, 40, "#ffd166"); ax[0].text(12, 30, "top of strip: assumed free coating surface (separator side)", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="#ffd166", ec="none"))
box(ax[0], 2, H4 - 45, W4 - 4, 42, "#ef476f"); ax[0].text(12, H4 - 52, "bottom of strip: thin bright band (see crop below) = current collector (assumed copper) → collector side", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="#ef476f", ec="none"))
show(ax[1], bse[-220:, 1200:3600], "bottom 220 rows at full resolution, x = 1200–3600: continuous bright layer under the coating", aspect="auto")
mark(ax[1], (1200, 212), "bright band ≈ 15 px (≈ 0.4 µm nominal) — foil surface, not its full thickness", "#ef476f", (150, -120))
f1 = fig_to_b64(fig)
add("A1", "Each image is an ion-polished cross-section through the full thickness of one electrode coating, stitched into a wide strip; image top is the free surface and image bottom is the current-collector side.", "medium",
    [(f1, "Full strip of the one site where a bright metallic band is visible at the bottom. On the other 30 sites no collector is in frame, so orientation is inferred from this one.")],
    "The strips are ~7000 px wide and 1600–2300 px tall, polished flat (ETD shows curtaining, see A9), with no mounting resin visible at top or bottom in most sites.",
    "Flat polished faces across all phases; one site shows a continuous bright band at the bottom with the morphology and BSE brightness of a metal foil; strip heights (≈ 40–58 µm nominal) match typical anode coating thickness.",
    "Through-thickness profiles (pore / bright fraction vs depth) are reported top→bottom and interpreted as surface→collector. Strip height is used as a coating-thickness proxy with caveats (see A12).",
    "Strips were cropped out of a larger field at arbitrary positions, or were imaged collector-up, or include part of the current collector / resin that we count as coating.",
    "Were all strips cropped to exactly the coating? Which edge faces the current collector? Is the bright band in Batch_2/epqdaau9 copper?")

# ----------------------------------------------------------------------------- A2 phases
b, s = "Batch_1", "5n1q8atc"; r = row(s); bse = load(b, s); h, w = bse.shape
c, (y0, x0) = crop(bse, h // 2, w // 2, 800, 1600); sm, pore, bright = masks(c, r)
fig, ax = new_fig(1, 1, 12, 6); show(ax[0], c, f"{b} / {s} / BSE — 1600 × 800 px crop (≈ {um(1600)} × {um(800)} nominal)")
pc = largest_component_centroid(bright, 2000, 0.85); mark(ax[0], (pc[0], pc[1]), "bright particle = additive candidate in confirmed graphite–Si/SiOx", "#ffd166", (120, 90))
po = largest_component_centroid(pore, 2000); mark(ax[0], (po[0], po[1]), "black = open porosity (empty in the section; electrolyte-filled in the cell)", "#06d6a0", (120, 60))
graph = ndi.binary_erosion((sm >= r.th_lo) & (sm <= r.th_hi), iterations=25); gc = largest_component_centroid(graph, 5000)
mark(ax[0], (gc[0], gc[1]), "mid-grey plate with striations = graphite flake", "#8ecae6", (-10, 120))
BINDER = [(560, 1400), (130, 860)]   # (y, x) in crop coordinates: granular binder / carbon-black network between plates
for (by, bx) in BINDER: box(ax[0], bx - 90, by - 90, 180, 180, "#ef476f")
mark(ax[0], (BINDER[0][1] - 90, BINDER[0][0] + 90), "red boxes: granular carbon-black / binder network (full-resolution insets below)", "#ef476f", (-620, 110))
f1 = fig_to_b64(fig)
etd_c, _ = crop(load(b, s, "ETD"), h // 2, w // 2, 800, 1600)
fig, axg = plt.subplots(2, 2, figsize=(8, 8)); axg = axg.ravel()
for k, (by, bx) in enumerate(BINDER):
    show(axg[2 * k], c[by - 90:by + 90, bx - 90:bx + 90], f"inset {k + 1} · BSE")
    show(axg[2 * k + 1], etd_c[by - 90:by + 90, bx - 90:bx + 90], f"inset {k + 1} · ETD")
fig.suptitle("Binder / conductive-carbon network at full resolution (each panel 180 × 180 px ≈ 4.5 µm nominal)", fontsize=10, x=0.02, ha="left")
f1b = fig_to_b64(fig, 800)
add("A2", "Four constituents are visible in BSE: mid-grey graphite plates (matrix), a sparse brighter particulate additive, black open porosity (the electrolyte pathways of the finished cell), and a granular binder / conductive-carbon network between the plates.", "medium",
    [(f1, "Labels were placed automatically on the largest bright particle, the largest pore and a large graphite plate; the two red boxes mark binder / carbon-black regions chosen by hand and shown enlarged below."),
     (f1b, "The binder network at full resolution (180 × 180 px ≈ 4.5 µm nominal). In BSE it is a mid-dark granular fill; in ETD its open, foam-like structure is unmistakable. We do not segment it as its own phase today: depending on local density it falls into the pore or the graphite class.")],
    "Santosh confirms fresh graphite–Si/SiOx electrode. BSE contrast supports a bright additive class against graphite/residual carbon-rich solid, but no pixelwise chemical labels, exact Si versus SiOx identity or binder recipe have been supplied. Black section regions are interpreted as visible voids; connected electrolyte pathways in the cell are a separate 3-D question.",
    "Three well-separated modes in the BSE histogram (black, grey, bright); plate-like morphology with basal striations typical of flake graphite; bright particles are blocky, 2–6 µm nominal, consistent with Si/SiOx additive.",
    "The whole KPI catalogue: phase fractions, particle sizes, pore sizes and crack-like voids are defined by these three BSE classes.",
    "The bright particles are not a separate chemistry (e.g. channelling / orientation contrast in graphite, or a conductive coating), or the material is a cathode, or the spongy regions are polishing damage rather than binder.",
    "Is the additive Si, SiOx, or a mixture, and is there EDS or another chemical map? What is the binder/conductive-carbon recipe?")

# ----------------------------------------------------------------------------- A3 bright phase is a distinct composition
b, s = "Batch_3", "9luzk4jm"; r = row(s); bse, etd, il = [load(b, s, d) for d in ("BSE", "ETD", "Inlens")]
sm, pore, bright = masks(bse, r); pc = largest_component_centroid(ndi.binary_opening(bright, iterations=3), 20000, 0.8)
cy, cx = int(pc[1]), int(pc[0]); figs = []
fig, ax = new_fig(3, 1, 13, 4.3)
for a, (img, t) in zip(ax, [(bse, "BSE: uniformly brighter than graphite"), (etd, "ETD: same polished surface, no relief step"), (il, "Inlens: different brightness again")]):
    cc, _ = crop(img, cy, cx, 500, 700); show(a, cc, t)
f1 = fig_to_b64(fig)
fig, ax = new_fig(1, 1, 7, 3.2); hs = smoothed_hist(bse); ax[0].plot(hs, color="#2a78d6", lw=1.6); ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, .1)
ax[0].axvline(r.th_lo, color="#06d6a0", ls="--"); ax[0].axvline(r.th_hi, color="#ffd166", ls="--")
ax[0].set_xlabel("BSE gray level"); ax[0].set_title(f"{b}/{s}: smoothed BSE histogram — three modes (pore ≈ 0, graphite ≈ {int(r.graphite_mode)}, bright ≈ {int(r.graphite_mode + r.bright_sep)}); dashed = thresholds used", fontsize=9, loc="left")
f2 = fig_to_b64(fig, 800)
add("A3", "The bright particles are a distinct composition (higher mean atomic number), not an orientation or charging effect.", "medium-high",
    [(f1, "The same particle in the three detectors. It is bright in BSE, flat and featureless in ETD (so no topographic cause), and does not stand out in Inlens the way charging artefacts do."),
     (f2, "The BSE histogram has a separate bright mode on 29 of 31 sites, which is what a second phase produces; orientation contrast in graphite gives a continuum, not a mode.")],
    "A separate additive composition is confirmed by Santosh and is consistent with cross-channel contrast. Some thresholded bright rims/fragments remain acquisition-dependent and are not thereby chemically confirmed.",
    "Bimodal BSE histogram; particle brightness uniform inside each particle regardless of orientation; no relief in ETD; many particles have the blocky, fractured habit of milled Si/SiOx powder.",
    "Bright-phase area fraction, number density, size distribution and circularity are reported as 'additive' KPIs.",
    "The bright class were graphite in a particular crystallographic orientation (channelling contrast), or a surface coating.",
    "Can you confirm the bright phase is a second component of the formulation, and roughly its mass or volume fraction?")

# ----------------------------------------------------------------------------- A4 black = open pore ; A5 grey-pore group
b1, s1 = "Batch_3", "9luzk4jm"; b2, s2 = "Batch_3", "kbdh4tri"
fig, ax = new_fig(2, 1, 13, 4.6)
for a, (b, s, t) in zip(ax, [(b1, s1, "ordinary site: open porosity is black (gray 0–20)"), (b2, s2, "grey-pore group: same pores are mid-grey (gray 20–45)")]):
    img = load(b, s); h, w = img.shape; cc, _ = crop(img, h // 2, w // 3, 600, 800); show(a, cc, f"{b}/{s} — {t}")
    r = row(s); sm = gaussian(cc, 1, preserve_range=True); pm = ndi.binary_opening(sm < r.th_lo, iterations=1); pc = largest_component_centroid(pm, 1500)
    if pc: mark(a, (pc[0], pc[1]), "pore", "#06d6a0", (90, -60))
f1 = fig_to_b64(fig)
fig, ax = new_fig(1, 1, 8, 3.2)
for s, col, t in [(s1, "#1baf7a", "ordinary"), (s2, "#eda100", "grey-pore group")]:
    ax[0].plot(smoothed_hist(load("Batch_3", s)), color=col, lw=1.6, label=f"{s} ({t})")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, .1); ax[0].legend(frameon=False); ax[0].set_xlabel("BSE gray level"); ax[0].set_title("BSE histograms: the grey-pore sites have no pixels below ≈ 20 and a plateau where the black spike should be", fontsize=9, loc="left")
f2 = fig_to_b64(fig, 800)
add("A4", "Black regions in BSE are open porosity (empty in the sectioned, dried sample). On four Batch 3 sites the same regions are mid-grey, which we interpret as resin-filled pores or a different detector black level from a separate session — not a different material.", "medium",
    [(f1, "Same magnification, same phase layout, but the pore interiors differ: black on the left, grey with faint texture on the right."),
     (f2, "The grey-pore sites are also the only images with no post-acquisition contrast stretch (see A6), which again points to a different session.")],
    "Open porosity in a dried cross-section gives no backscatter signal and appears black; epoxy infiltration or a raised black-level offset both lift it to grey. The four sites (71vgq3fw, kbdh4tri, tuy3zymq, x7u69zsw) share an identical frame height, darker Inlens, more curtaining in ETD, and the lowest bright-phase contrast — a session fingerprint.",
    "Minimum gray value 23–25 vs 0 on every other image; identical heights; no comb histogram; consistent across all three channels.",
    "These four sites are analysed as a separate group. Their pore KPIs come from a fallback threshold and are flagged as not directly comparable.",
    "The grey pore filling were a real phase (e.g. electrolyte residue, SEI, or a different binder), in which case the group would be a material anomaly, not a preparation one.",
    "Were these four Batch 3 sites prepared or imaged differently (resin, session, black level)? Should they be treated as part of Batch 3?")

# ----------------------------------------------------------------------------- A6 low-contrast Batch 1 sites
fig, ax = new_fig(2, 1, 13, 4.6)
for a, (s, t) in zip(ax, [("f1vzngrs", "normal contrast: bright phase ≈ 65 levels above graphite"), ("4ih2ggld", "low contrast: bright phase ≈ 25–35 levels above graphite")]):
    img = load("Batch_1", s); h, w = img.shape; cc, _ = crop(img, h // 2, w // 2, 600, 800); show(a, cc, f"Batch_1/{s} — {t}")
f1 = fig_to_b64(fig)
fig, ax = new_fig(1, 1, 8, 3.2)
for s, col in [("f1vzngrs", "#2a78d6"), ("4ih2ggld", "#e34948"), ("5n1q8atc", "#eb6834")]:
    r = row(s); hs = smoothed_hist(load("Batch_1", s)); ax[0].plot(np.arange(256) - r.graphite_mode, hs, color=col, lw=1.6, label=f"{s}")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, .1); ax[0].set_xlim(-60, 110); ax[0].axvline(0, color="#888", lw=.8); ax[0].legend(frameon=False)
ax[0].set_xlabel("gray level − graphite mode"); ax[0].set_title("Histograms aligned on the graphite peak: two Batch 1 sites have the bright phase as a shoulder, not a separate mode", fontsize=9, loc="left")
f2 = fig_to_b64(fig, 800)
add("A5", "Two Batch 1 sites (4ih2ggld, 5n1q8atc) were recorded with compressed detector contrast; the additive is still there but cannot be segmented reliably. This is an acquisition difference, not a material one.", "medium-high",
    [(f1, "Same phases, but on the right the bright particles are barely lighter than graphite and the binder network is almost as bright as the particles."),
     (f2, "No valley between graphite and bright phase on these two sites, so any threshold also captures particle rims and binder, inflating bright-phase fraction 2–8×.")],
    "Detector gain / contrast was set differently for these two sites (both 2316 px tall, a height no other site has).",
    "Histogram shape; identical frame height; all other KPIs (pore structure, texture) are ordinary for these sites.",
    "Their bright-phase KPIs are flagged `bright_low_contrast` and excluded from pooled particle-size statistics; they still count for pore and crack KPIs.",
    "The additive in these two sites genuinely had lower atomic-number contrast (different composition), which would make them a material anomaly inside Batch 1.",
    "Were Batch_1/4ih2ggld and 5n1q8atc imaged in a different session or with different detector settings?")

# ----------------------------------------------------------------------------- A7 contrast stretching
fig, ax = new_fig(2, 1, 13, 3.4)
for a, (b, s, t) in zip(ax, [("Batch_3", "hzumfsms", "comb histogram: 50 % of gray levels empty → contrast-stretched after acquisition"), ("Batch_3", "71vgq3fw", "continuous histogram: no stretch")]):
    img = load(b, s); h = np.bincount(img.ravel()[::3], minlength=256); a.bar(np.arange(256), h / h.sum(), width=1, color="#2a78d6"); a.set_yscale("log"); a.set_ylim(1e-7, .1); a.set_xlim(0, 160)
    a.set_title(f"{b}/{s} BSE — {t}", fontsize=9, loc="left"); a.set_xlabel("gray level")
f1 = fig_to_b64(fig)
add("A6", "Roughly half of the images were contrast-stretched after acquisition, unevenly across sites and channels. Raw gray values are therefore not comparable between images and are never used as a material KPI.", "high",
    [(f1, "Regularly spaced empty bins are the signature of a linear remap of an 8-bit image. We cannot tell whether it happened in the microscope software or at export.")],
    "Any intensity-based comparison (mean brightness, histogram distance, learned embeddings on raw pixels) would partly measure the operator's display settings.",
    "Empty-bin fractions up to 65 %; stretched and unstretched images coexist within the same batch.",
    "All segmentation thresholds are derived per image from its own histogram; intensity statistics are reported only as acquisition flags.",
    "The stretch were applied identically to all images from one batch and not the others — it would then still be an acquisition effect, but one correlated with batch.",
    "Were any images processed (contrast, brightness, gamma) after acquisition? Are the raw .tif exports available?")

# ----------------------------------------------------------------------------- A8 segmentation thresholds
b, s = "Batch_2", "epqdaau9"; r = row(s); bse = load(b, s); h, w = bse.shape; c, _ = crop(bse, h // 2, w // 2, 700, 1200); sm, pore, bright = masks(c, r)
seg = np.zeros(c.shape + (3,)); seg[pore] = (0.05, 0.05, 0.05); seg[~pore & ~bright] = (0.55, 0.55, 0.55); seg[bright] = (0.92, 0.63, 0.0)
fig, ax = new_fig(2, 1, 13, 4.2); show(ax[0], c, f"{b}/{s} BSE crop"); ax[1].imshow(seg); ax[1].set_axis_off(); ax[1].set_title(f"segmentation: pore (black) < {r.th_lo:.0f} · graphite (grey) · bright phase (orange) > {r.th_hi:.0f}", fontsize=10, loc="left")
f1 = fig_to_b64(fig)
add("A7", "Per-image thresholds anchored on the graphite peak (valley to the bright mode; valley to the dark mode) assign pixels to pore / graphite / bright phase accurately enough for area fractions and particle sizes.", "medium-high",
    [(f1, "Please check: are the orange regions all additive particles, and the black regions all pores? Particle rims and the binder network are the usual failure points.")],
    "Valley thresholds between resolved histogram modes are the least arbitrary choice and adapt to each image's stretch and offset.",
    "Visual agreement on overlays from every batch; fractions stable across ordinary sites (bright 0.04–0.08, pore 0.06–0.14).",
    "Everything downstream of segmentation.",
    "Thin bright rims around graphite plates (edge effect) or binder were systematically counted as additive; sub-resolution pores in the binder network were counted as solid.",
    "Does this phase assignment match what you would draw by hand? Should the binder / carbon-black network be its own class?")

# ----------------------------------------------------------------------------- A9 cracks
b, s = "Batch_3", "hzumfsms"; r = row(s); bse = load(b, s); sm, pore, bright = masks(bse, r)
lab = label(pore); rp = pd.DataFrame(regionprops_table(lab, properties=("label", "major_axis_length", "area", "centroid")))
big = rp[rp.major_axis_length > 500]; bigmask = np.isin(lab, big.label.values)
rgb = np.stack([bse] * 3, -1).astype(float) / 255; rgb[bigmask] = (0.89, 0.29, 0.28)
fig, ax = new_fig(1, 1, 12, 3.4); ax[0].imshow(rgb[::4, ::4], aspect="auto"); ax[0].set_axis_off(); ax[0].set_title(f"{b}/{s} BSE full strip — voids with long axis > 500 px (≈ {um(500)} nominal) painted red: crack-like void fraction {r.crack_frac:.3f}", fontsize=10, loc="left")
f1 = fig_to_b64(fig)
bb = big.sort_values("area", ascending=False).iloc[0]; cy, cx = int(bb["centroid-0"]), int(bb["centroid-1"])
fig, ax = new_fig(2, 1, 13, 4.2)
cc, _ = crop(bse, cy, cx, 600, 900); show(ax[0], cc, "BSE: largest void — runs between/along graphite plates")
cc2, _ = crop(load(b, s, "ETD"), cy, cx, 600, 900); show(ax[1], cc2, "ETD: same region — void walls are sharp, plates intact around it")
f2 = fig_to_b64(fig)
add("A8", "Long, elongated voids running along the coating in three Batch 3 sites (hzumfsms, 0grcilhi, ufdvpb81) are real delamination-style cracks in the electrode, i.e. a material / processing defect rather than polishing pull-out.", "medium",
    [(f1, "Our 'crack-like void' KPI is defined on BSE as pores whose long axis exceeds 500 px; it does not detect fine cracks, only voids large enough to be dark in BSE."),
     (f2, "Sharp walls and undisturbed plates on both sides argue against pull-out during polishing, but we cannot exclude damage during sectioning.")],
    "Interconnected voids 450–600 px in equivalent diameter, 3–4× the largest void on ordinary sites, aligned with the plate orientation — the pattern of drying cracks or calendering delamination.",
    "Crack-like void fraction 0.048–0.062 vs median 0.015–0.022 elsewhere; present in three of seventeen Batch 3 sites and none in Batch 1 or 2.",
    "This is the strongest candidate for a genuine accept / investigate / reject signal; it will be shown on the image whenever it drives a verdict.",
    "These voids were introduced by sample preparation (ion-mill heating, embedding), or they are a normal feature of this electrode design.",
    "Are the three cracked Batch 3 sites known defective material? Could the voids be a sectioning artefact?")

# ----------------------------------------------------------------------------- A10 curtaining ; A11 particles intact
b, s = "Batch_3", "71vgq3fw"; r = row(s); bse, etd, il = [load(b, s, d) for d in ("BSE", "ETD", "Inlens")]
sm, pore, bright = masks(bse, r); pc = largest_component_centroid(ndi.binary_opening(bright, iterations=3), 30000, 0.8); cy, cx = int(pc[1]), int(pc[0])
e = etd.astype(float); scale = np.subtract(*np.percentile(e, [75, 25])); Hrr, Hrc, Hcc = hessian_matrix(e, sigma=2.0, order="rc", use_gaussian_derivatives=False)
tr = Hrr + Hcc; det = Hrr * Hcc - Hrc ** 2; lam1 = tr / 2 + np.sqrt(np.maximum(tr ** 2 / 4 - det, 0)); ridge = np.maximum(lam1, 0) * 4 / scale
theta = np.degrees(0.5 * np.arctan2(2 * Hrc, Hrr - Hcc)); dom = r.etd_ridge_dom_angle; aligned = np.abs((theta - dom + 90) % 180 - 90) < 15; strong = ridge > 0.4
cc_e, (y0, x0) = crop(etd, cy, cx, 600, 900); sl = (slice(y0, y0 + 600), slice(x0, x0 + 900))
rgb = np.stack([cc_e] * 3, -1).astype(float) / 255; rgb[(strong & aligned)[sl]] = (0.2, 0.5, 1.0); rgb[(strong & ~aligned)[sl]] = (1.0, 0.3, 0.2)
fig, ax = new_fig(2, 1, 13, 4.2); show(ax[0], cc_e, f"{b}/{s} ETD: diagonal streaks cross particles and pores alike"); ax[1].imshow(rgb); ax[1].set_axis_off()
ax[1].set_title(f"blue = ridges within ±15° of the dominant direction ({dom:.0f}°) → curtaining · red = other ridges (mostly phase boundaries)", fontsize=10, loc="left")
f1 = fig_to_b64(fig)
add("A9", "The parallel diagonal streaks in ETD are ion-milling curtaining (a preparation artefact), not cracks; they are removed by orientation before counting cracks.", "high",
    [(f1, "Curtaining runs in one direction across the whole image regardless of phase; cracks would follow particle geometry.")],
    "Broad-beam ion polishing leaves parallel grooves downstream of harder features; the direction is set by the beam, which is why it is constant per image.",
    "One dominant ridge orientation per image (≈ ±52–62° on most sites); streaks continue straight across phase boundaries.",
    "ETD crack-density KPIs count only ridges that deviate > 15° from the dominant direction; curtaining fraction is reported as a preparation flag.",
    "The streaks were real planar cracks aligned with the ion beam by coincidence (unlikely), or cracks happen to run parallel to the curtaining and are suppressed.",
    "Were the sections prepared by broad-ion-beam (Ar) cross-section polishing? Same instrument and settings for all batches?")

fig, ax = new_fig(3, 1, 13, 4.3)
for a, (img, t) in zip(ax, [(bse, "BSE"), (etd, "ETD: polished face, no internal cracks"), (il, "Inlens: smooth interior")]):
    c2, _ = crop(img, cy, cx, 500, 700); show(a, c2, f"{b}/{s} {t}")
f1 = fig_to_b64(fig)
add("A10", "Detected ETD ridge coverage inside bright-particle interiors is low (0.02–0.5 %); this does not establish that every additive particle is intact or electrically connected.", "measurement review pending",
    [(f1, "A typical large particle in all three channels. No obvious internal branching lines appear in this example, but unresolved, filled or orientation-filtered cracks can escape detection.")],
    "These are fresh, uncycled electrodes. Visible fracture-like features could originate in powder processing, manufacturing or specimen preparation; cycling damage is not an explanation for their current state. The ridge detector measures pixels, not the fraction of intact particles.",
    "Measured across all 31 sites after curtaining removal; visual inspection of the largest particles per site.",
    "The KPI is kept in the pipeline because it is the measurement most likely to catch a fractured-additive batch in the unseen data.",
    "Cracks were narrower than ~2 px (below the ridge filter scale) or filled and invisible in ETD.",
    "Is particle fracture a manufacturing failure mode expected in incoming batches? Which visible lines are genuine fractures, and what is the validated detection limit?")

# ----------------------------------------------------------------------------- A12 Inlens speckle
fig, ax = new_fig(2, 1, 13, 4.6)
for a, (b, s, t) in zip(ax, [("Batch_1", "f1vzngrs", "speckled interior (texture 0.36)"), ("Batch_3", "x7u69zsw", "smooth interior (texture 0.13)")]):
    r = row(s); bse, il = load(b, s), load(b, s, "Inlens"); sm, pore, bright = masks(bse, r)
    pc = largest_component_centroid(ndi.binary_opening(bright, iterations=3), 20000, 0.8); c2, _ = crop(il, int(pc[1]), int(pc[0]), 450, 650); show(a, c2, f"{b}/{s} Inlens — {t}")
f1 = fig_to_b64(fig)
add("A11", "Inlens shows two kinds of additive-particle interior, speckled and smooth. Whether this is a real difference (porous composite vs dense particle) or a charging / contrast effect is unresolved; the KPI is treated as confounded.", "low",
    [(f1, "The speckled appearance tracks Inlens brightness and session fingerprints (ρ ≈ 0.7–0.8), so we do not trust it as a material KPI yet.")],
    "Inlens is surface-sensitive and charging-dominated; a porous Si–C composite particle would plausibly look speckled, but so would a charging particle at higher detector gain.",
    "Strongest batch ordering of any feature (Batch 1 > 2 > 3) yet ~75 % explained by Inlens acquisition statistics.",
    "Reported as a candidate KPI with a confound warning; never allowed to drive a verdict alone.",
    "The speckle were purely instrumental (then the KPI is dropped) or purely material (then it is our best discriminator and we are under-using it).",
    "Are there two types of additive particle in the formulation (e.g. dense Si and porous Si–C)? Were Inlens settings held constant across batches?")

# ----------------------------------------------------------------------------- A13 scale & height
geo = F[["batch", "site", "H"]].copy()
fig, ax = new_fig(1, 1, 9, 3.2)
cols = {"Batch_1": "#2a78d6", "Batch_2": "#eb6834", "Batch_3": "#1baf7a"}
for i, b in enumerate(["Batch_1", "Batch_2", "Batch_3"]):
    y = geo[geo.batch == b].H.values; ax[0].scatter(np.full(len(y), i) + np.random.default_rng(0).uniform(-.12, .12, len(y)), y, color=cols[b], s=28)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["Batch_1", "Batch_2", "Batch_3"]); ax[0].set_ylabel("strip height (px)")
ax2 = ax[0].secondary_yaxis("right", functions=(lambda p: p * NM / 1000, lambda u: u * 1000 / NM)); ax2.set_ylabel("nominal µm at 25 nm/px")
ax[0].set_title("Strip heights: repeated exact values across batches (2080, 2148, 2156, 2272) and inside sub-groups (2316 ×2, 2060 ×4)", fontsize=9, loc="left")
f1 = fig_to_b64(fig, 800)
add("A12", "The TIFF resolution tags (25.0 nm/px on every file) are the true pixel size, and strip height is at best a rough coating-thickness proxy because identical heights recur across batches like a session fingerprint.", "low-medium",
    [(f1, "At 25 nm/px the coatings are 40–58 µm thick and strips 175 µm wide, which is plausible — but the tag could be a default written at export.")],
    "Microscope metadata was stripped on re-save; only the resolution tag survived. Repeated heights are more plausibly a crop or stage setting than coincidence.",
    "Tag values 24.9992–25.0005 nm/px across all 93 files; height distribution above.",
    "All KPIs stay in pixels; µm are quoted only as 'nominal'. Height is not used as a KPI.",
    "The pixel size were different (all µm statements scale accordingly), or heights really are thickness (then thickness is a KPI we are ignoring).",
    "Is 25 nm/px correct? Was magnification identical for all sites? Were the strips cropped to the coating automatically?")

# ----------------------------------------------------------------------------- A14 channels co-registered
b, s = "Batch_2", "rxax5ozo"; bse, etd = load(b, s), load(b, s, "ETD"); h, w = bse.shape; c1, _ = crop(bse, h // 2, w // 2, 400, 600); c2, _ = crop(etd, h // 2, w // 2, 400, 600)
edge = sobel(gaussian(c2.astype(float), 1.5)); edge = edge > np.percentile(edge, 94)
rgb = np.stack([c1] * 3, -1).astype(float) / 255; rgb[edge] = (1.0, 0.85, 0.2)
fig, ax = new_fig(2, 1, 12, 3.9); show(ax[0], c1, f"{b}/{s} BSE (this site's secondary-electron file is labelled 'SE')"); ax[1].imshow(rgb); ax[1].set_axis_off(); ax[1].set_title("ETD/SE edges (yellow) drawn on the BSE crop: they sit exactly on the BSE phase boundaries", fontsize=10, loc="left")
f1 = fig_to_b64(fig)
add("A13", "The three detector images of a site are the same scan (pixel-aligned), and the files labelled 'SE' on four sites are the same secondary-electron detector as 'ETD'.", "high",
    [(f1, "Phase-correlation shifts ≤ 0.2 px on six sites; edge overlay shown here on an 'SE'-labelled site.")],
    "Simultaneous multi-detector acquisition is standard; the shape, size and content of SE and ETD files are identical in kind.",
    "Sub-pixel registration; identical dimensions; SE files have the same histogram shape as ETD files.",
    "BSE masks are applied directly to ETD and Inlens for all cross-channel KPIs; SE is merged into ETD.",
    "SE were a different detector (e.g. a chamber SE detector with different take-off angle), which would make texture features on those four sites slightly different.",
    "Why do four sites carry the label 'SE' instead of 'ETD'? Same detector?")

# ----------------------------------------------------------------------------- A15 sites independent ; A16 batches & baseline
fig, ax = new_fig(1, 1, 12, 3.2)
for i, (b, s) in enumerate([("Batch_3", "hawkfj64"), ("Batch_3", "0grcilhi"), ("Batch_3", "mgxahqnk")]):
    img = load(b, s)[::8, ::8]; ax[0].imshow(img, cmap="gray", vmin=0, vmax=255, extent=(0, img.shape[1], (i + 1) * 300, i * 300 + 20), aspect="auto")
ax[0].set_ylim(900, 0); ax[0].set_axis_off(); ax[0].set_title("Three Batch 3 sites that share the same frame height (1904 px): different locations, or the same specimen imaged three times?", fontsize=10, loc="left")
f1 = fig_to_b64(fig)
add("A14", "Each site is an independent sample of its batch (a different location or electrode), so site-to-site spread estimates batch variability and n = 7 or 17 is the real sample size.", "low",
    [(f1, "We cannot verify independence from the images alone. If several sites come from one specimen, the effective sample size is smaller and our confidence intervals are too narrow.")],
    "Statistical calibration (split-half, leave-one-site-out, bootstrap) treats sites as exchangeable draws from the batch.",
    "Sites do not visibly overlap; heights cluster into a few values, which could mean several strips per specimen.",
    "All uncertainty statements in the QC notebook.",
    "Several sites were taken from the same electrode piece — then within-batch spread is understated and 'investigate' should be the default more often.",
    "How many physical specimens per batch, and how many strips per specimen? Are site IDs random?")

add("A15", "Batch_1/2/3 are three supplier batches of the same nominal product. Batch 3 is a single batch with more samples and is the closest thing to a baseline, but not a perfect one; there is no clean approved baseline and the task is to differentiate the batches.", "confirmed",
    [],
    "Batch 3 (17 sites) is used as the working reference. Its internal variation — the grey-pore group (A4) and the three cracked sites (A8) — is part of the reference, so the QC logic must show the reference's own heterogeneity rather than assume it is clean, and must report how Batch 1 and Batch 2 differ from Batch 3 and from each other.",
    "Stated by the problem providers (relayed 2026-10-03).",
    "The calibration null is built from Batch 3 with robust statistics (median / MAD, leave-one-site-out) and with its sub-populations shown explicitly; every pairwise batch comparison is reported, not only incoming-vs-reference.",
    "Batch 3's heterogeneity were larger than the between-batch differences we are asked to detect — then the honest output is 'the reference does not constrain this KPI' rather than a verdict.",
    "Is the within-Batch-3 variation (grey-pore sites, cracked sites) representative of what an acceptable batch may contain? Are there any known differences between the three supplier batches we should be able to recover?")

# ----------------------------------------------------------------------------- A16 exploratory morphology geometry
morph_evidence = os.path.join(ROOT, "analysis", "morphology", "output", "geometry_examples.png")
morph_figures = []
if os.path.exists(morph_evidence):
    with Image.open(morph_evidence) as im:
        im = im.convert("RGB")
        if im.width > 1500:
            im = im.resize((1500, int(im.height * 1500 / im.width)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=85, optimize=True)
        morph_figures = [("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                          "E18 inspection extremes in trimmed BSE coordinates: centroids / nearest neighbours in gold and major axes in cyan. Small bright rim fragments are retained by the original 50-pixel component floor; they can dominate spacing measurements. These examples were selected after the exploratory screen.")]
add("A16", "Mask-derived shape, orientation, centroid spacing and local void width describe observed 2-D components, but are not yet validated measures of additive agglomeration or manufacturing harm.", "low-medium; exploratory",
    morph_figures,
    "Area-weighted shape and axial orientation summaries can add morphology evidence to the five primary KPIs. Orientation is relative to image horizontal, only for non-clipped components with aspect ratio at least 1.5. Centroid spacing includes small bright fragments; its comparison changes when the minimum component area changes.",
    "E18: geometry conventions pass synthetic-shape checks; nominal masks reproduce cached measurements; threshold and area-floor sensitivities are reported. The expanded panels do not robustly separate the known batches at these site counts. This does not establish equivalence. E24 adds a deterministic medial-axis width proxy and a hysteresis candidate, with separate expert annotations still pending. Width excludes edge-connected components and is not a 3-D pore throat; hysteresis agreement with an earlier mask is not accuracy.",
    "Exploratory morphology report only. No primary KPI, decision threshold or release verdict changes; thousands of components remain summaries of one site.",
    "Bright rims or threshold fragments were mistaken for additive particles; disconnected void geometry were interpreted as actual cracks; imaging/section orientation changed; or centroids from finite particles were interpreted as a validated random-point clustering test.",
    "Which component sizes correspond to actual additive particles? Are the detected orientation, spacing and local-width patterns physically meaningful? Review the E24 annotation pack, including uncertain or unmeasurable phase boundaries, before treating candidate segmentation as improved.")

# E26G: extend A16 without changing its expert-review state.
graph_card = next(c for c in CARDS if c["id"] == "A16")
graph_card["why"] += " E26G adds a fixed 3-NN bright-centroid graph with four secondary summaries, site-level controls and threshold/object-floor sensitivity. Proximity edges are not physical/electrical contacts or a defect label; finite-frame neighbour bias remains. Expert component and specimen/process validation are pending."
graph_image = os.path.join(ROOT, "analysis", "ml_options", "e_graph", "overlay_Batch_2_3806gxp0.png")
if os.path.exists(graph_image):
    with Image.open(graph_image) as im:
        im = im.convert("RGB")
        if im.width > 1500: im = im.resize((1500, int(im.height*1500/im.width)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf,"JPEG",quality=85,optimize=True)
        graph_card["figures"].append(("data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode(), "E26G prespecified central BSE crop; full-site graphs at 50/100 px² object floors. Source coordinates, exclusions and thresholds are in the graph report. This is unreviewed geometry, not expert truth."))

# ----------------------------------------------------------------------------- A17 confirmed material/state metadata
material_example = next(c for c in CARDS if c["id"] == "A3")["figures"][0][0]
add("A17", "The sections are fresh, uncycled graphite–Si/SiOx electrode material.", "confirmed by Santosh",
    [(material_example, "Illustration of the confirmed material family. Fresh/uncycled state and chemistry are supplied metadata, not properties established from this image.")],
    "Santosh directly confirmed 'Fresh graphite–Si/SiOx electrode confirmed' on 2026-10-03. Exact Si versus SiOx chemistry, recipe fractions and binder identity are unspecified.",
    "Human-provided material/state metadata; see the persistent review log. No cycling history is inferred from SEM appearance.",
    "Focus battery checks on manufacturing geometry and possible susceptibility during formation/later cycling: additive neighbourhoods, void accommodation, graphite arrangement and reviewed interfaces. Do not label existing voids or ridges cycling-induced, SEI or lithium plating.",
    "The provided metadata were later corrected, or some sites had a different processing/cycling history. Material-family confirmation also would not rescue low-contrast segmentation or validate chemical identity at every pixel.",
    "What are the exact additive chemistry, recipe, binder system, coating/calendering history and physical specimen IDs?")

# ----------------------------------------------------------------------------- A18 battery geometry and coverage
battery_figures = []
for filename, caption in [
    ("neighbourhood_overlay_Batch_2_3806gxp0.png", "E25 real BSE example: segmented bright pixels, visible voids and 16 px mask bands. Rings include threshold holes within the silhouette; complete windows remain within-site samples. Source/trim coordinates and excluded objects are displayed."),
    ("void_full_site_overlays.png", "E25 long-void examples: internal and image-edge-clipped cavities are separated. Row location is an image coordinate; collector direction is unconfirmed. Width estimates exclude clipped cavities.")]:
    image_path = os.path.join(ROOT, "analysis", "battery", "output", filename)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if im.width > 1500: im = im.resize((1500, int(im.height * 1500 / im.width)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=85, optimize=True)
        battery_figures.append(("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), caption))
add("A18", "Neighbourhood, local homogeneity and internal long-void geometry are observable mask descriptors; their battery implications remain hypotheses.", "experimental; expert review pending",
    battery_figures,
    "Eight new secondary KPIs quantify distance/ring void fraction, bright dispersion and bright/void association at two fixed window scales, and internal long-void burden/location. Mask bands, components and windows are not chemical labels, electrical contact, physical expansion clearance or additional independent material samples.",
    "All known sites were measured with existing masks and paired threshold perturbations. Synthetic convention tests and nominal-cache regressions pass; independent phase/instance annotations are still absent. Exact raster bright/void adjacency is zero on current usable sites and remains diagnostic. The pore-mode flag is unresolved everywhere; the grey-pore subgroup is a separate data-derived acquisition flag.",
    "Extraction/report secondary section only; excluded from primary tests, classifier inputs and verdicts. Coverage, quality exclusions, missing values, paired threshold ranges and site-bootstrap intervals are disclosed separately. Physics wording now avoids lithiation-time, total-expansion, intact-share and monotonic transport predictions.",
    "Threshold fragments or residual-solid transition bands were mistaken for physical neighbourhoods; clipped-cavity exclusions hid visible area; window variation reflected loading/section placement; image direction was mistaken for collector direction; or correlated sites were treated as independent specimens.",
    "Independently review phase boundaries, ambiguous fragments, graphite plate instances and the collector candidate. Can these observed geometries be reproduced across sections from identified specimens? Application claims need matched recipe/process and contact, adhesion, wetting or electrochemical evidence.")

# E28J extends image-texture observability on A16 without confirming review.
gabor_card = next(c for c in CARDS if c["id"] == "A16")
gabor_card["why"] += " E28J adds three fixed-Gabor sampled BSE appearance summaries and contrast/resolution/FFT-geometry-tensor controls. Four windows cover only a small portion of a frame; wavevectors are normal to stripes, not plate axes. Negative nuisance predictability does not prove invariance. All material/QC use and independent expert/spatial validation remain deferred."
gabor_image = os.path.join(ROOT,"analysis","ml_options","j_gabor","overlay_Batch_2_3806gxp0.png")
if os.path.exists(gabor_image):
    with Image.open(gabor_image) as im:
        im = im.convert("RGB")
        if im.width > 1500: im = im.resize((1500,int(im.height*1500/im.width)),Image.LANCZOS)
        buf = io.BytesIO();im.save(buf,"JPEG",quality=85,optimize=True)
        gabor_card["figures"].append(("data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode(),"E28J measured quarter-frame BSE window and fixed Gabor response maps. Gold bounds the valid interior; maps describe appearance, not particles or defects. Source coordinates and sampled-site coverage are in the J report."))

# E30K extends A16 while keeping expert review separate from computation.
k_card = next(c for c in CARDS if c["id"] == "A16")
k_card["why"] += " E30K adds count-weighted bright shape distributions and two fixed finite-frame x/y phase-correlation contrasts. Clipped-object exclusion, phase-specific quality flags, pair-domain marginals and threshold/floor sensitivity are explicit. Component/pixel counts are coverage, not independent units; image axes are not confirmed collector directions. Geometry/phase and application validity remain unreviewed and all QC use deferred."
k_image = os.path.join(ROOT,"analysis","morphology","k_pilot","overlay_Batch_2_3806gxp0.png")
if os.path.exists(k_image):
    with Image.open(k_image) as im:
        im = im.convert("RGB")
        if im.width > 1500: im = im.resize((1500,int(im.height*1500/im.width)),Image.LANCZOS)
        buf = io.BytesIO();im.save(buf,"JPEG",quality=85,optimize=True)
        k_card["figures"].append(("data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode(),"E30K fixed central BSE reference with eligible bright-object bounds, full-frame count-weighted shape histograms and x/y phase-correlation bars. Masks and axes remain expert-unreviewed; pair counts describe coverage."))

# ----------------------------------------------------------------------------- HTML
def esc(t): return html.escape(t)
css = """
:root{--bg:#fbfaf7;--card:#ffffff;--ink:#141414;--muted:#5b5a56;--line:#e6e3dc;--accent:#2a78d6;--ok:#1baf7a;--warn:#eda100;--bad:#e34948;--chip:#f1efe9}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--card:#1f1f1d;--ink:#f2f1ed;--muted:#b9b7af;--line:#333330;--chip:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 80px}h1{font-size:26px;margin:0 0 6px}h2{font-size:17px;margin:0;line-height:1.35}
.lede{color:var(--muted);max-width:900px}.toolbar{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
.toolbar .counts span{display:inline-block;padding:2px 10px;border-radius:999px;background:var(--chip);margin-right:6px;font-size:13px}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);cursor:pointer}button:hover{border-color:var(--accent)}
.card{scroll-margin-top:64px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:0 0 22px}
.card header{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:start;margin-bottom:10px}
.id{font-weight:700;color:var(--accent);font-size:15px;padding-top:2px}.conf{font-size:12px;color:var(--muted);white-space:nowrap;padding-top:3px}
figure{margin:12px 0}figure img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);display:block}figcaption{font-size:13px;color:var(--muted);margin-top:6px}
dl{display:grid;grid-template-columns:150px 1fr;gap:6px 14px;margin:12px 0 0}dt{font-weight:600;font-size:13px;color:var(--muted)}dd{margin:0}
.ask dd{font-weight:500}.log dd{font-size:13px;color:var(--muted);border-left:3px solid var(--ok);padding-left:10px}.review{margin-top:14px;padding-top:12px;border-top:1px dashed var(--line);display:grid;grid-template-columns:auto 1fr;gap:10px 16px;align-items:start}
.status label{margin-right:12px;font-size:14px;cursor:pointer}.status input{margin-right:4px}
textarea{width:100%;min-height:54px;font:inherit;font-size:14px;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
.card[data-status=confirmed]{border-left:5px solid var(--ok)}.card[data-status=refuted]{border-left:5px solid var(--bad)}.card[data-status=unsure]{border-left:5px solid var(--warn)}
.toc{columns:2;gap:24px;font-size:14px;margin:0 0 18px;padding-left:18px}.toc a{color:var(--ink);text-decoration:none}.toc a:hover{color:var(--accent)}
@media(max-width:700px){dl{grid-template-columns:1fr}.review{grid-template-columns:1fr}.toc{columns:1}.card header{grid-template-columns:auto 1fr}.conf{grid-column:2}}
@media print{.toolbar,textarea,.status{display:none}.card{break-inside:avoid}}
"""
js = """
const KEY='polaron-assumption-register-v1';let state={};try{state=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){state={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(state))}catch(e){}updateCounts()}
function setStatus(id,v){state[id]=Object.assign({},state[id],{status:v});document.getElementById(id).dataset.status=v;save()}
function setNote(id,v){state[id]=Object.assign({},state[id],{note:v});save()}
function updateCounts(){const c={confirmed:0,unsure:0,refuted:0,unreviewed:0};document.querySelectorAll('.card').forEach(el=>{c[el.dataset.status||'unreviewed']++});
document.getElementById('counts').innerHTML=`<span>✔ confirmed ${c.confirmed}</span><span>? unsure ${c.unsure}</span><span>✘ refuted ${c.refuted}</span><span>unreviewed ${c.unreviewed}</span>`}
function exportMd(){let md='# Assumption register review\\n\\n_Exported '+new Date().toISOString().slice(0,16).replace('T',' ')+'_\\n\\n';
document.querySelectorAll('.card').forEach(el=>{const id=el.id,s=state[id]||{};md+=`## ${id} — ${s.status||'unreviewed'}\\n\\n${el.querySelector('h2').textContent}\\n\\n`;if(s.note)md+=`**Notes:** ${s.note}\\n\\n`});
navigator.clipboard&&navigator.clipboard.writeText(md).catch(()=>{});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([md],{type:'text/markdown'}));a.download='assumption_review.md';a.click()}
function resetAll(){if(confirm('Clear all statuses and notes stored in this browser?')){state={};save();location.reload()}}
window.addEventListener('DOMContentLoaded',()=>{document.querySelectorAll('.card').forEach(el=>{const s=state[el.id]||{};if(s.status){el.dataset.status=s.status;const r=el.querySelector(`input[value=${s.status}]`);if(r)r.checked=true}
if(s.note){el.querySelector('textarea').value=s.note}});updateCounts()})
"""
parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Assumption register</title><style>{css}</style></head><body><div class="wrap">
<h1>Polaron SEM QC — assumption register</h1>
<p class="lede">Every interpretive assumption our analysis relies on, each tied to an example image. Built {datetime.date.today().isoformat()} from <code>docs/_build_assumption_register.py</code>.
Mark each card <b>confirmed</b>, <b>unsure</b> or <b>refuted</b> and add notes; your review is kept in this browser and can be exported as markdown to send back to us or to the organisers. Lengths are in pixels; µm are nominal at 25 nm/px (see A12).</p>
<div class="toolbar"><div class="counts" id="counts"></div><button onclick="exportMd()">Export review (.md)</button><button onclick="resetAll()">Reset</button></div>
<ol class="toc">{''.join(f'<li><a href="#{c["id"]}">{c["id"]} · {esc(c["title"][:90])}{"…" if len(c["title"])>90 else ""}</a></li>' for c in CARDS)}</ol>"""]
for c in CARDS:
    figs = "".join(f'<figure><img src="{b64}" alt="{esc(c["id"])} evidence" loading="lazy"><figcaption>{esc(cap)}</figcaption></figure>' for b64, cap in c["figures"])
    parts.append(f"""<section class="card" id="{c['id']}"><header><span class="id">{c['id']}</span><h2>{esc(c['title'])}</h2><span class="conf">our confidence: {esc(c['confidence'])}</span></header>
{figs}
<dl><dt>What we think</dt><dd>{esc(c['think'])}</dd><dt>Why</dt><dd>{esc(c['why'])}</dd><dt>How we use it</dt><dd>{esc(c['used_for'])}</dd><dt>We would be wrong if</dt><dd>{esc(c['wrong_if'])}</dd></dl>
<dl class="ask"><dt>Question for the organisers</dt><dd>{esc(c['ask'])}</dd></dl>
{('<dl class="log"><dt>Review log</dt><dd>' + '<br>'.join(esc(x) for x in REVIEW.get(c['id'], [])) + '</dd></dl>') if REVIEW.get(c['id']) else ''}
<div class="review"><div class="status"><label><input type="radio" name="s-{c['id']}" value="confirmed" onchange="setStatus('{c['id']}','confirmed')">✔ confirmed</label><label><input type="radio" name="s-{c['id']}" value="unsure" onchange="setStatus('{c['id']}','unsure')">? unsure</label><label><input type="radio" name="s-{c['id']}" value="refuted" onchange="setStatus('{c['id']}','refuted')">✘ refuted</label></div>
<textarea placeholder="Notes, corrections, who confirmed it and when…" oninput="setNote('{c['id']}',this.value)"></textarea></div></section>""")
parts.append(f"</div><script>{js}</script></body></html>")
open(OUT, "w").write("".join(parts))
print("wrote", OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB,", len(CARDS), "cards")
