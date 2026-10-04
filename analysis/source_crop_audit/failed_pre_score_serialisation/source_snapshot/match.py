"""Label-blind, translation-only detection of repeated spatial content.

These matches are not reconstructed parent/specimen IDs. Non-overlapping crops
from one parent are invisible to this procedure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal, stats


@dataclass(frozen=True)
class Translation:
    """Coordinates obey ``a[y, x] == b[y-dy, x-dx]`` for an exact overlap."""

    dy: int
    dx: int
    ncc: float
    height: int
    width: int


def _as_float(image):
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("images must be finite two-dimensional arrays")
    return a


def small_image(image, *, stride=8, sigma=2.0, border=4):
    """Fixed blur/lattice; no resized-to-equal-frame geometry or label input."""
    a = _as_float(image)
    if stride < 1 or border < 0 or min(a.shape) <= 2 * border:
        raise ValueError("invalid image geometry or stride")
    body = a[border:a.shape[0]-border, border:a.shape[1]-border] if border else a
    return ndimage.gaussian_filter(body, sigma=sigma, mode="reflect")[::stride, ::stride]


def _integral(a):
    return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _rect_grid(sat, y0, y1, x0, x1):
    return (sat[y1[:, None], x1[None, :]] - sat[y0[:, None], x1[None, :]]
            - sat[y1[:, None], x0[None, :]] + sat[y0[:, None], x0[None, :]])


def bounds(shape_a, shape_b, dy, dx):
    """Intersection in A and B coordinates, with no wrapping/padding."""
    ay0, ax0 = max(0, dy), max(0, dx)
    ay1, ax1 = min(shape_a[0], shape_b[0] + dy), min(shape_a[1], shape_b[1] + dx)
    if ay1 <= ay0 or ax1 <= ax0:
        return None
    return (ay0, ay1, ax0, ax1), (ay0-dy, ay1-dy, ax0-dx, ax1-dx)


def best_translation(image_a, image_b, *, min_height, min_width):
    """Maximum overlap-specific Pearson NCC across all legal translations.

    Every lag uses its own intersecting-pixel means and variances. FFT padding
    cannot contribute artificial matching pixels, and small-border coincidences
    cannot pass the declared minimum intersection size.
    """
    a, b = _as_float(image_a), _as_float(image_b)
    if min_height < 2 or min_width < 2:
        raise ValueError("minimum intersection sides must be at least two")
    if min(a.shape[0], b.shape[0]) < min_height or min(a.shape[1], b.shape[1]) < min_width:
        return None
    if a.std() < 1e-8 or b.std() < 1e-8:
        return None
    # Centre first to reduce cancellation in overlap-specific moments.
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    dy = np.arange(-(b.shape[0]-1), a.shape[0])
    dx = np.arange(-(b.shape[1]-1), a.shape[1])
    ay0, ay1 = np.maximum(0, dy), np.minimum(a.shape[0], b.shape[0] + dy)
    ax0, ax1 = np.maximum(0, dx), np.minimum(a.shape[1], b.shape[1] + dx)
    by0, by1, bx0, bx1 = ay0-dy, ay1-dy, ax0-dx, ax1-dx
    heights, widths = ay1-ay0, ax1-ax0
    n = heights[:, None] * widths[None, :]
    sa = _rect_grid(_integral(a), ay0, ay1, ax0, ax1)
    sb = _rect_grid(_integral(b), by0, by1, bx0, bx1)
    va = _rect_grid(_integral(a*a), ay0, ay1, ax0, ax1) - sa*sa/n
    vb = _rect_grid(_integral(b*b), by0, by1, bx0, bx1) - sb*sb/n
    cross = signal.correlate(a, b, mode="full", method="fft") - sa*sb/n
    legal = ((heights[:, None] >= min_height) & (widths[None, :] >= min_width)
             & (va/n > 1e-8) & (vb/n > 1e-8))
    score = np.full(n.shape, -np.inf)
    denominator = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    np.divide(cross, denominator, out=score, where=legal)
    score[~legal] = -np.inf
    if not np.isfinite(score).any():
        return None
    iy, ix = np.unravel_index(np.argmax(score), score.shape)
    return Translation(int(dy[iy]), int(dx[ix]), float(np.clip(score[iy, ix], -1, 1)),
                       int(heights[iy]), int(widths[ix]))


def _template_scores(search, template):
    """NCC for one fully contained template; result uses valid origins only."""
    t = template - template.mean()
    h, w = t.shape
    n = h*w
    if t.std() < 1e-8 or min(search.shape) < 1:
        return None
    sy, sx = search.shape
    if sy < h or sx < w:
        return None
    y0, x0 = np.arange(sy-h+1), np.arange(sx-w+1)
    sm = _rect_grid(_integral(search), y0, y0+h, x0, x0+w)
    sq = _rect_grid(_integral(search*search), y0, y0+h, x0, x0+w) - sm*sm/n
    corr = signal.correlate(search, t, mode="valid", method="fft")
    den = np.sqrt(np.maximum(sq, 0) * np.sum(t*t))
    out = np.full(corr.shape, -np.inf)
    np.divide(corr, den, out=out, where=den > 1e-8)
    return out


def refine_translation(image_a, image_b, coarse, *, radius=16, patch_size=512):
    """Refine the fixed coarse offset using one central patch, no deformation."""
    a, b = _as_float(image_a), _as_float(image_b)
    box = bounds(a.shape, b.shape, coarse.dy, coarse.dx)
    if box is None:
        return None
    (ay0, ay1, ax0, ax1), _ = box
    if min(ay1-ay0, ax1-ax0) < patch_size:
        return None
    ya, xa = (ay0+ay1-patch_size)//2, (ax0+ax1-patch_size)//2
    template = a[ya:ya+patch_size, xa:xa+patch_size]
    yb, xb = ya-coarse.dy, xa-coarse.dx
    sy0, sx0 = max(0, yb-radius), max(0, xb-radius)
    sy1, sx1 = min(b.shape[0], yb+patch_size+radius), min(b.shape[1], xb+patch_size+radius)
    scores = _template_scores(b[sy0:sy1, sx0:sx1], template)
    if scores is None or not np.isfinite(scores).any():
        return None
    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
    dy, dx = ya-(sy0+iy), xa-(sx0+ix)
    box = bounds(a.shape, b.shape, dy, dx)
    if box is None:
        return None
    (y0, y1, x0, x1), _ = box
    return Translation(dy, dx, float(np.clip(scores[iy, ix], -1, 1)), y1-y0, x1-x0)


def verification_patches(shape_a, shape_b, shift, *, patch_size=256, margin=32):
    box = bounds(shape_a, shape_b, shift.dy, shift.dx)
    if box is None:
        return []
    (y0, y1, x0, x1), _ = box
    if y1-y0 < patch_size+2*margin or x1-x0 < 3*patch_size+2*margin:
        return []
    y = (y0+y1-patch_size)//2
    xx = np.rint(np.linspace(x0+margin, x1-margin-patch_size, 3)).astype(int)
    return [dict(patch=i, a_y=y, a_x=int(x), b_y=y-shift.dy, b_x=int(x)-shift.dx,
                 height=patch_size, width=patch_size) for i, x in enumerate(xx)]


def compare_patch(image_a, image_b, coordinates, *, min_std=1.0):
    """Independent descriptive checks; exact equality is retained separately."""
    c = coordinates
    h, w = c["height"], c["width"]
    a = np.asarray(image_a)[c["a_y"]:c["a_y"]+h, c["a_x"]:c["a_x"]+w]
    b = np.asarray(image_b)[c["b_y"]:c["b_y"]+h, c["b_x"]:c["b_x"]+w]
    if a.shape != (h, w) or b.shape != (h, w):
        return dict(valid=False, reason="incomplete_patch")
    av, bv = a.astype(float).ravel(), b.astype(float).ravel()
    std_a, std_b = av.std(), bv.std()
    if min(std_a, std_b) < min_std:
        return dict(valid=False, reason="patch_std_below_1_gray_level", std_a=float(std_a), std_b=float(std_b))
    ac, bc = av-av.mean(), bv-bv.mean()
    gain = float(np.dot(ac, bc) / np.dot(ac, ac))
    offset = float(bv.mean() - gain*av.mean())
    pearson = float(np.dot(ac, bc) / np.sqrt(np.dot(ac, ac)*np.dot(bc, bc)))
    spearman = float(stats.spearmanr(av, bv).statistic)
    residual = bv-(gain*av+offset)
    residual_norm = float(np.sqrt(np.mean(residual*residual)) / std_b)
    return dict(valid=True, pearson=float(np.clip(pearson, -1, 1)), spearman=spearman,
                exact_pixel_fraction=float(np.mean(a == b)),
                affine_gain_b_from_a=gain, affine_offset_b_from_a=offset,
                affine_residual_target_sd=residual_norm,
                std_a=float(std_a), std_b=float(std_b),
                passed=bool(pearson >= 0.995 and spearman >= 0.995 and residual_norm <= 0.10))


def confirmed(checks):
    """Nine valid passing checks, exactly three patches per named detector."""
    return (len(checks) == 9 and {c.get("detector") for c in checks} == {"BSE", "ETD", "Inlens"}
            and all(sum(c.get("detector") == d for c in checks) == 3 for d in ("BSE", "ETD", "Inlens"))
            and all(c.get("valid") and c.get("passed") for c in checks))


def components(nodes, edges):
    """Nontrivial connected overlap components; isolated sites are not parents."""
    parent = {n: n for n in nodes}

    def find(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    out = {}
    for n in sorted(nodes):
        out.setdefault(find(n), []).append(n)
    return [g for g in out.values() if len(g) > 1]
