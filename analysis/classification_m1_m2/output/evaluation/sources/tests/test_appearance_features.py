"""Numerical/coverage contracts for mask-independent E39S appearance inputs."""
import numpy as np
import pytest

from analysis.classification_m1_m2.appearance import (
    CONTEXT, FEATURE_NAMES, TILE_SIZE, aggregate_tiles, extract_site,
    measure_tile, tile_coordinates, union_area,
)
from analysis.classification_m1_m2.extract import transform


def test_positive_gain_and_offset_cancel_without_clipping():
    rng = np.random.default_rng(123)
    patch = rng.normal(100, 15, size=(TILE_SIZE + 2 * CONTEXT,) * 2)
    baseline, _ = measure_tile(patch)
    shifted, _ = measure_tile(1.35 * patch + 17)
    np.testing.assert_allclose(baseline, shifted, atol=2e-13, rtol=2e-13)
    assert np.isclose(baseline[:3].sum(), 1.0)
    assert np.all((baseline[:3] >= 0) & (baseline[:3] <= 1))


def test_gradient_axis_is_image_gradient_not_an_object_orientation():
    n = TILE_SIZE + 2 * CONTEXT
    x = np.arange(n)
    vertical_bands = np.broadcast_to(np.sin(2 * np.pi * x / 80), (n, n))
    horizontal_bands = vertical_bands.T
    vertical, maps = measure_tile(vertical_bands, return_maps=True)
    horizontal, _ = measure_tile(horizontal_bands)
    assert vertical[3] > .999
    assert horizontal[3] < -.999
    assert abs(vertical[4]) < 1e-12
    assert vertical[5] > .999
    np.testing.assert_allclose(vertical[:3], horizontal[:3], atol=1e-12)
    assert maps["fine_band"].shape == (TILE_SIZE, TILE_SIZE)


def test_constant_and_incomplete_measurements_fail_closed():
    values, maps = measure_tile(np.ones((TILE_SIZE + 2 * CONTEXT,) * 2))
    assert np.isnan(values).all()
    assert maps is None
    tiled = np.ones((9, 6))
    tiled[3, 1] = np.nan
    assert np.isnan(aggregate_tiles(tiled)).all()
    with pytest.raises(ValueError, match="exactly nine"):
        aggregate_tiles(np.ones((8, 6)))


def test_tile_pooling_is_equal_window_not_pixel_count_weighted():
    values = np.tile([.2, .3, .5, .1, -.1, .2], (9, 1))
    values[0, :3] = [.8, .1, .1]
    pooled = aggregate_tiles(values)
    np.testing.assert_allclose(pooled, [.2, .3, .5, 0, 0, .1, -.1, .2])


def test_grid_coverage_counts_overlap_once_and_never_uses_identifiers():
    coordinates = tile_coordinates((800, 800))
    assert len(coordinates) == len(set(coordinates)) == 9
    # The 3x3 overlapping grid fills one contiguous rectangle here.
    assert union_area(coordinates) == (800 - 2 * CONTEXT)**2
    assert union_area([(0, 0), (0, 0)]) == TILE_SIZE**2
    with pytest.raises(ValueError, match="field too small"):
        tile_coordinates((600, 800))
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 24
    assert all("height" not in name and "mean" not in name for name in FEATURE_NAMES)


def test_shared_trim_has_aligned_detectors_and_exposes_no_mask_dependency():
    rng = np.random.default_rng(3)
    # Geometry just large enough for this API, deliberately overlapping windows.
    image = rng.normal(80, 12, (800, 800))
    images = {channel: image.copy() for channel in ("BSE", "ETD", "Inlens")}
    values, diagnostics, _ = extract_site(images, trim=(10, 10))
    assert list(values) == list(FEATURE_NAMES)
    assert diagnostics["available_features"] == 24
    assert diagnostics["tile_origins_raw_yx"][0][0] == CONTEXT + 10
    for suffix in [name.removeprefix("appearance_bse_") for name in FEATURE_NAMES[:8]]:
        assert values[f"appearance_bse_{suffix}"] == values[f"appearance_etd_{suffix}"]
    broken = images.copy()
    broken["ETD"] = image[:799]
    with pytest.raises(ValueError, match="aligned"):
        extract_site(broken, trim=(10, 10))


def test_nuisance_challenges_preserve_geometry_and_do_not_clip_gain():
    image = np.array([[0, 7, 255]], dtype=np.uint8)
    images = {"BSE": image}
    assert transform(images, "gain_offset")["BSE"].max() > 255
    np.testing.assert_array_equal(transform(images, "quantise_step8")["BSE"], [[0, 0, 248]])
    for variant in ("gamma_0p7", "gamma_1p4"):
        changed = transform(images, variant)["BSE"]
        assert changed.shape == image.shape
        assert changed[0, 0] == 0 and changed[0, -1] == 255
    with pytest.raises(ValueError, match="unknown"):
        transform(images, "arbitrary")
