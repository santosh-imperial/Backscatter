"""Contracts for internal-only long-void geometry; no inference/decision thresholds."""
import numpy as np
import pytest
from skimage.measure import label, regionprops
from polaron_qc.void_metrics import long_void_context, KPI_NAMES, DIAGNOSTIC_NAMES


def test_internal_and_clipped_components_have_explicit_coverage():
    mask=np.zeros((600,1000),bool);mask[250:270,100:850]=1;mask[440:460,0:650]=1
    metrics,details=long_void_context(mask,return_details=True)
    assert metrics['long_void_n_components']==2
    assert metrics['long_void_internal_n_components']==1
    assert metrics['long_void_edge_clipped_n_components']==1
    assert metrics['long_void_internal_area_frac']==pytest.approx(15000/600000)
    assert metrics['long_void_edge_clipped_area_share']==pytest.approx(13000/28000)
    assert metrics['long_void_y_centroid_norm']==pytest.approx(259.5/599)
    assert details['long_mask'].sum()==28000
    assert details['internal_mask'].sum()==15000
    assert details['long_void_depth_area_share'].sum()==pytest.approx(1)
    assert not any(isinstance(v,np.ndarray) for v in long_void_context(mask).values())


def test_only_clipped_long_void_has_valid_zero_and_undefined_location():
    mask=np.zeros((600,1000),bool);mask[200:220,0:800]=1
    metrics=long_void_context(mask)
    assert metrics['long_void_internal_area_frac']==0
    assert np.isnan(metrics['long_void_y_centroid_norm'])
    assert metrics['long_void_edge_clipped_area_share']==1


def test_no_long_void_preserves_undefined_coverage():
    mask=np.zeros((600,1000),bool);mask[100:110,100:110]=1
    metrics=long_void_context(mask)
    assert metrics['long_void_n_components']==0
    assert metrics['long_void_internal_area_frac']==0
    assert np.isnan(metrics['long_void_y_centroid_norm'])
    assert np.isnan(metrics['long_void_internal_area_share'])
    assert set(metrics)==set(KPI_NAMES+DIAGNOSTIC_NAMES)


def test_vertical_flip_changes_location_only():
    mask=np.zeros((600,1000),bool);mask[100:130,100:800]=1
    a=long_void_context(mask);b=long_void_context(mask[::-1])
    assert a['long_void_internal_area_frac']==b['long_void_internal_area_frac']
    assert a['long_void_y_centroid_norm']+b['long_void_y_centroid_norm']==pytest.approx(1)


def test_connectivity_and_strict_axis_threshold_match_existing_definition():
    mask=np.zeros((700,700),bool);ix=np.arange(100,500);mask[ix,ix]=1
    axis=regionprops(label(mask,connectivity=2))[0].major_axis_length
    assert long_void_context(mask,major_axis_cutoff_px=axis)['long_void_n_components']==0
    assert long_void_context(mask,major_axis_cutoff_px=axis-1)['long_void_n_components']==1


def test_invalid_mask_or_parameters_fail_explicitly():
    with pytest.raises(ValueError):long_void_context(np.zeros((4,4,3),bool))
    with pytest.raises(ValueError):long_void_context(np.full((4,4),80))
    with pytest.raises(ValueError):long_void_context(np.zeros((4,4),bool),min_area_px=0)
    with pytest.raises(ValueError):long_void_context(np.zeros((0,0),bool))
    observed=long_void_context(np.zeros((20,20),bool))
    assert observed['long_void_internal_area_frac']==0
    assert np.isnan(observed['long_void_y_centroid_norm'])
