"""Independent synthetic checks for the exploratory geometry measurements."""
import numpy as np
from skimage.draw import ellipse

from analysis.morphology.run_analysis import component_table, orientation_summary, guarded_nn, spacing_summary, profile_slope


def ellipse_mask(vertical):
    mask = np.zeros((300, 300), dtype=bool)
    rr, cc = ellipse(150, 150, 60 if vertical else 15, 15 if vertical else 60)
    mask[rr, cc] = True
    return mask


def test_horizontal_vertical_and_axial_flip_conventions():
    horizontal = ellipse_mask(False)
    for mask, expected in ((horizontal, 1), (ellipse_mask(True), -1), (horizontal[::-1, ::-1], 1)):
        summary = orientation_summary(component_table(mask, 30))
        assert np.isclose(summary['horizontal_alignment'], expected)
        assert np.isclose(summary['alignment_strength'], 1)
        assert summary['n_oriented'] == 1
    # Known diagonal ellipses also check the sign used by the image overlays.
    y, x = np.mgrid[:301, :301] - 150
    for slope in (1, -1):
        major = (x + slope*y) / np.sqrt(2)
        minor = (x - slope*y) / np.sqrt(2)
        mask = (major / 60)**2 + (minor / 15)**2 < 1
        theta = component_table(mask, 30).angle_horizontal_rad.iloc[0]
        assert np.isclose(np.sin(2*theta), slope)


def test_round_and_boundary_clipped_components_have_no_orientation_claim():
    mask = np.zeros((100, 100), dtype=bool)
    rr, cc = ellipse(50, 50, 10, 10)
    mask[rr, cc] = True
    mask[:5, 20:80] = True
    summary = orientation_summary(component_table(mask, 30))
    assert summary['n_oriented'] == 0
    assert np.isnan(summary['horizontal_alignment'])


def test_nearest_neighbour_is_not_self_and_boundary_points_are_censored():
    points = np.array([[50., 50.], [50., 60.], [0., 0.]])
    distances, eligible, _ = guarded_nn(points, (100, 100))
    assert np.allclose(distances, [10, 10])
    assert eligible.tolist() == [True, True, False]


def test_uniform_random_centroids_match_the_same_window_null():
    import pandas as pd
    rng = np.random.default_rng(77)
    points = rng.uniform([0, 0], [999, 3999], (1000, 2))
    table = pd.DataFrame({'centroid-0': points[:, 0], 'centroid-1': points[:, 1]})
    summary = spacing_summary(table, (1000, 4000), seed=88, n_csr=200)
    assert 0.9 < summary['bright_nn_csr_ratio'] < 1.1


def test_profile_slope_follows_image_depth_and_reverses_on_flip():
    mask = np.zeros((100, 100), dtype=bool)
    for i in range(100):
        mask[i, :i] = True
    assert np.isclose(profile_slope(mask), 1)
    assert np.isclose(profile_slope(mask[::-1]), -1)
