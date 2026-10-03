"""Geometry/coverage contracts for secondary battery neighbourhood metrics."""
import numpy as np
import pytest

from polaron_qc.battery_metrics import (
    KPI_NAMES, KPI_DEFINITIONS, measure_neighbourhoods, threshold_sensitivity,
)


def scene():
    b = np.zeros((80, 80), dtype=bool)
    p = b.copy()
    b[24:32, 24:32] = True  # 64 px, 32 exposed faces
    p[24:32, 32] = True
    return p, b


def test_exact_raster_adjacency_and_pixel_distance():
    p, b = scene()
    f, d = measure_neighbourhoods(p, b, return_details=True)
    assert f["bright_void_boundary_frac"] == .25
    assert f["bright_void_distance_d50_px"] == 1
    assert f["neighbourhood_bright_boundary_faces"] == 32
    assert f["neighbourhood_bright_void_faces"] == 8
    assert f["neighbourhood_bright_residual_solid_faces"] == 24
    assert f["neighbourhood_bright_other_bright_faces"] == 0
    c = d["components"].iloc[0]
    assert c.ring_pixels == c.ring_void_pixels + c.ring_other_bright_pixels + c.ring_residual_solid_pixels
    assert f["bright_ring_void_frac_16px"] == c.ring_void_pixels / c.ring_pixels


def test_diagonal_pore_is_near_but_has_no_shared_face():
    p, b = scene()
    p[:] = False; p[32, 32] = True
    f = measure_neighbourhoods(p, b)
    assert f["bright_void_boundary_frac"] == 0
    assert f["bright_void_distance_d50_px"] == pytest.approx(np.sqrt(2))


def test_nearby_bright_is_distinct_from_residual_solid():
    p, b = scene()
    b[42:50, 40:48] = True
    f, d = measure_neighbourhoods(p, b, return_details=True)
    c = d["components"].iloc[0]
    assert c.ring_other_bright_pixels > 0
    assert c.ring_residual_solid_pixels > 0
    assert c.ring_pixels == c.ring_void_pixels + c.ring_other_bright_pixels + c.ring_residual_solid_pixels
    assert f["bright_void_boundary_frac"] == .125  # pooled 8 / 64 faces


def test_clipped_and_small_objects_counted_but_not_used():
    p, b = scene()
    b[0:8, 0:8] = True
    b[60:67, 60:67] = True  # 49 px below fixed floor
    f = measure_neighbourhoods(p, b)
    assert f["neighbourhood_bright_components_all"] == 3
    assert f["neighbourhood_bright_components_small"] == 1
    assert f["neighbourhood_bright_components_retained"] == 2
    assert f["neighbourhood_bright_components_clipped"] == 1
    assert f["neighbourhood_bright_components_used"] == 1
    assert f["bright_void_boundary_frac"] == .25


def test_ring_excludes_incomplete_observation_separately():
    b = np.zeros((80, 80), bool); p = b.copy()
    b[5:13, 5:13] = True; p[13, 5] = True
    f = measure_neighbourhoods(p, b)
    assert f["neighbourhood_bright_components_used"] == 1
    assert f["neighbourhood_ring_components_used"] == 0
    assert f["neighbourhood_ring_components_clipped"] == 1
    assert np.isnan(f["bright_ring_void_frac_16px"])
    assert np.isfinite(f["bright_void_boundary_frac"])


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (32, 32)])
def test_empty_or_absent_bright_is_unmeasurable(shape):
    b = np.zeros(shape, bool)
    f = measure_neighbourhoods(b, b)
    assert all(np.isnan(f[k]) for k in KPI_NAMES)
    assert f["neighbourhood_bright_components_used"] == 0


def test_absent_pore_not_reported_as_zero_contact_or_clearance():
    p, b = scene(); p[:] = False
    f = measure_neighbourhoods(p, b)
    for k in ("bright_void_boundary_frac", "bright_void_distance_d50_px", "bright_ring_void_frac_16px"):
        assert np.isnan(f[k])
    assert f["neighbourhood_pore_pixels"] == 0


@pytest.mark.parametrize("p,b", [
    (np.zeros((3, 4)), np.zeros((4, 3))),
    (np.zeros(5), np.zeros(5)),
    (np.ones((3, 3)), np.ones((3, 3))),
    (np.zeros((3, 3)), np.full((3, 3), 2)),
])
def test_invalid_masks_rejected(p, b):
    with pytest.raises(ValueError):
        measure_neighbourhoods(p, b)


def test_windows_prespecified_grid_complete_only_and_spatial_association():
    b = np.zeros((1025, 2048), bool); p = b.copy()
    # Eight complete 512 windows, two 1024 windows. Fractions vary independently
    # of pixel intensity; all pore rectangles are disjoint from bright rectangles.
    for j in range(8):
        y, x = (j//4)*512, (j%4)*512
        b[y+50:y+50+(j+1)*8, x+50:x+90] = True
        p[y+300:y+300+(j+1)*8, x+50:x+90] = True
    f, d = measure_neighbourhoods(p, b, return_details=True)
    assert f["neighbourhood_windows_512px"] == 8
    assert f["neighbourhood_windows_1024px"] == 2
    assert f["neighbourhood_window_coverage_512px"] == pytest.approx(1024/1025)
    assert f["local_bright_std_512px"] > 0
    assert f["local_bright_pore_spearman_512px"] == pytest.approx(1)
    assert np.isnan(f["local_bright_pore_spearman_1024px"])
    assert len(d["windows"]) == 10


def test_constant_windows_dispersion_zero_but_correlation_undefined():
    b = np.zeros((1024, 1024), bool); p = b.copy()
    for y in (0, 512):
        for x in (0, 512):
            b[y+100:y+108, x+100:x+108] = True
            p[y+200:y+208, x+200:x+208] = True
    f = measure_neighbourhoods(p, b)
    assert f["local_bright_std_512px"] == 0
    assert np.isnan(f["local_bright_pore_spearman_512px"])
    assert np.isnan(f["local_bright_std_1024px"])


def test_bright_only_in_excluded_partial_windows_is_unmeasurable():
    b = np.zeros((1025, 1024), bool); p = b.copy()
    b[1024, :100] = True
    f = measure_neighbourhoods(p, b)
    assert f["neighbourhood_bright_pixels"] == 100
    assert f["neighbourhood_windows_bright_present_512px"] == 0
    assert np.isnan(f["local_bright_std_512px"])


def test_rotation_preserves_component_geometries():
    p, b = scene()
    f = measure_neighbourhoods(p, b)
    r = measure_neighbourhoods(np.rot90(p), np.rot90(b))
    for k in ("bright_void_boundary_frac", "bright_void_distance_d50_px", "bright_ring_void_frac_16px"):
        assert f[k] == r[k]


def test_threshold_envelope_reports_coverage_and_reuses_nominal():
    a = np.full((80, 80), 100., dtype=float)
    a[24:32, 24:32] = 200; a[24:32, 32] = 0
    nominal = measure_neighbourhoods(a < 40, a > 160)
    envelope, variants = threshold_sensitivity(a, 40, 160, nominal=nominal, return_variants=True)
    assert [v["threshold_offset"] for v in variants] == [-5, 0, 5]
    for k in KPI_NAMES:
        finite = [v[k] for v in variants if np.isfinite(v[k])]
        assert envelope[k+"_sensitivity_n"] == len(finite)
        if finite:
            assert envelope[k+"_sensitivity_min"] == min(finite)
            assert envelope[k+"_sensitivity_max"] == max(finite)
    assert variants[1]["bright_void_boundary_frac"] == nominal["bright_void_boundary_frac"]


def test_threshold_input_validation_and_conservative_quality_contract():
    with pytest.raises(ValueError):
        threshold_sensitivity(np.zeros((2, 2)), 90, 80)
    with pytest.raises(ValueError):
        threshold_sensitivity(np.full((2, 2), np.nan), 20, 80)
    assert all(d["required_flags"]["bright_low_contrast"] is False for d in KPI_DEFINITIONS.values())
    assert KPI_DEFINITIONS["bright_void_boundary_frac"]["required_flags"]["grey_pore"] is False
