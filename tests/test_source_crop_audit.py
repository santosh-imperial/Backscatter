"""Synthetic geometry and falsification checks, not organiser parent-ID truth."""
import numpy as np
import pytest
from scipy import ndimage

from analysis.source_crop_audit.match import (
    Translation, best_translation, compare_patch, components, confirmed,
    refine_translation, small_image, verification_patches,
)


def _cropped_source(dy=131, dx=371):
    rng = np.random.default_rng(418)
    source = ndimage.gaussian_filter(rng.normal(size=(800, 2200)), 3)
    source = 110 + 35 * source/source.std()
    a = source[:600, :1600]
    b = source[dy:dy+580, dx:dx+1600] * 0.65 + 31
    return a, b


@pytest.mark.parametrize("dy,dx", [(131, 371), (132, 372)])
def test_translation_recovered_across_sizes_and_affine_intensity(dy, dx):
    a, b = _cropped_source(dy, dx)
    t = best_translation(small_image(a), small_image(b), min_height=16, min_width=32)
    assert t.ncc >= 0.85  # Recovery must survive the actual coarse candidate gate.
    assert abs(t.dy*8-dy) <= 8
    assert abs(t.dx*8-dx) <= 8
    refined = refine_translation(a, b, Translation(t.dy*8, t.dx*8, t.ncc, t.height*8, t.width*8), patch_size=256)
    assert (refined.dy, refined.dx) == (dy, dx)
    assert isinstance(refined.dy, int) and isinstance(refined.dx, int)
    assert refined.ncc > 0.999999
    coordinates = verification_patches(a.shape, b.shape, refined)
    assert len(coordinates) == 3
    checks = [dict(detector=d, **compare_patch(a, b, c)) for d in ("BSE", "ETD", "Inlens") for c in coordinates]
    assert confirmed(checks)
    assert all(c["affine_residual_target_sd"] < 1e-12 for c in checks)
    assert all(c["exact_pixel_fraction"] < 0.001 for c in checks)


def test_translation_reverses_without_using_site_labels():
    a, b = _cropped_source()
    ab = best_translation(a[::8, ::8], b[::8, ::8], min_height=16, min_width=32)
    ba = best_translation(b[::8, ::8], a[::8, ::8], min_height=16, min_width=32)
    assert (ab.dy, ab.dx) == (-ba.dy, -ba.dx)
    assert np.isclose(ab.ncc, ba.ncc)


def test_constant_and_insufficient_overlaps_cannot_return_a_match():
    a = np.ones((50, 90))
    b = np.random.default_rng(19).normal(size=a.shape)
    assert best_translation(a, b, min_height=20, min_width=20) is None
    assert best_translation(b, b, min_height=51, min_width=20) is None
    assert not compare_patch(a, a, dict(a_y=0, a_x=0, b_y=0, b_x=0, height=30, width=30))["valid"]


def test_unrelated_patches_and_missing_channels_cannot_confirm():
    rng = np.random.default_rng(88)
    a, b = rng.integers(0, 255, (300, 1000)), rng.integers(0, 255, (300, 1000))
    t = best_translation(a[::4, ::4], b[::4, ::4], min_height=32, min_width=64)
    assert t.ncc < 0.15
    c = dict(a_y=0, a_x=0, b_y=0, b_x=0, height=256, width=256)
    assert not compare_patch(a, b, c)["passed"]
    good = dict(valid=True, passed=True)
    assert not confirmed([dict(detector="BSE", **good) for _ in range(9)])
    assert not confirmed([dict(detector=d, **good) for d in ("BSE", "ETD", "Inlens") for _ in range(2)])


def test_overlap_components_do_not_promote_isolates_to_parent_ids():
    assert components(["a", "b", "c", "d"], [("a", "b"), ("b", "c")]) == [["a", "b", "c"]]
    assert components(["a", "b"], []) == []
