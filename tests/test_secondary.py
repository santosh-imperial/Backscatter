"""Integration guards for E25 secondary geometry, independent of decision tests."""
import numpy as np
import pandas as pd

from polaron_qc import MATERIAL_KPIS, PRIMARY_KPIS
from polaron_qc import secondary


def frame(values, k, **flags):
    return pd.DataFrame({k: values, "bright_low_contrast": False, "grey_pore": False,
                         "pore_mode_resolved": False, **flags})


def test_quality_flags_missing_are_not_reliable_and_unresolved_is_counted():
    joint = "bright_ring_void_frac_16px"
    f = frame([.1, .2, .3], joint, grey_pore=[False, True, False], bright_low_contrast=[False, False, True])
    assert secondary.usable_mask(f, joint).tolist() == [True, False, False]
    assert not secondary.usable_mask(f.drop(columns="grey_pore"), joint).any()
    summary = secondary.summarize(f, f, n_boot=20).set_index("kpi").loc[joint]
    assert summary.n_ref_usable == 1 and summary.n_ref_pore_threshold_unresolved == 1
    assert not summary.interval_available and np.isnan(summary.ci_low)


def test_bright_only_measurement_does_not_require_pore_quality():
    k = "local_bright_std_512px"
    f = frame([.1, .2], k, grey_pore=True).drop(columns="pore_mode_resolved")
    assert secondary.usable_mask(f, k).all()
    assert not secondary.usable_mask(f.drop(columns="bright_low_contrast"), k).any()


def test_raw_unit_intervals_work_at_zero_reference_mad_and_no_primary_promotion():
    k = "local_bright_std_512px"
    a, b = frame([.1]*3, k), frame([.2]*3, k)
    row = secondary.summarize(a, b, n_boot=30).set_index("kpi").loc[k]
    assert row.median_difference == row.ci_low == row.ci_high == .1
    assert row.interval_available and not row.is_primary
    assert set(secondary.KPI_NAMES).isdisjoint(PRIMARY_KPIS + MATERIAL_KPIS)
    assert "bright_void_boundary_frac" not in secondary.KPI_NAMES
    assert "p_perm" not in row.index and "flag_alpha" not in row.index


def test_missing_measurements_and_nan_values_remain_unavailable():
    f = pd.DataFrame({"site": ["a", "b"]})
    out = secondary.summarize(f, f, n_boot=20)
    assert out.n_ref_usable.eq(0).all() and out.ref_median.isna().all()
    assert not out.interval_available.any()
    k = "long_void_y_centroid_norm"
    f = frame([np.nan, np.nan], k)
    assert not secondary.usable_mask(f, k).any()


def test_report_defaults_cannot_make_unobserved_quality_usable():
    from polaron_qc import report
    k = "bright_void_distance_d50_px"
    raw = pd.DataFrame({"batch": ["Batch_X"], "site": ["new_missing_flags"], k: [8.0], "long_void_y_centroid_norm": [0.5]})
    images = pd.DataFrame({"batch": ["Batch_X"], "site": ["new_missing_flags"], "det": ["BSE"], "p1": [np.nan], "p50": [80.0], "std": [40.0], "empty_bin_frac": [0.0], "band_top": [0], "band_bottom": [0]})
    flags = report.derive_flags(raw, images)
    derived = report.apply_derived_flags(raw, flags)
    before = derived.copy(deep=True)
    view = secondary.with_observed_quality(raw, derived, flags)
    assert not secondary.usable_mask(view, k).any()
    assert not secondary.usable_mask(view, "long_void_y_centroid_norm").any()
    pd.testing.assert_frame_equal(derived, before)  # primary flags untouched
    raw["bright_low_contrast"] = False
    flags = report.derive_flags(raw, images)
    derived = report.apply_derived_flags(raw, flags)
    view = secondary.with_observed_quality(raw, derived, flags)
    assert not secondary.usable_mask(view, k).any()  # black level still absent
    flags["bse_p1"] = 0.0
    view = secondary.with_observed_quality(raw, derived, flags)
    assert secondary.usable_mask(view, k).all()
    raw["bright_low_contrast"] = np.nan
    assert not secondary.usable_mask(secondary.with_observed_quality(raw, derived, flags), k).any()
