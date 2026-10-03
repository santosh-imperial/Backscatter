import numpy as np
import pandas as pd
import pytest
from analysis.ml_options.j_gabor.texture import KEYS, ANGLES, kernel, measure_window, summary, windows, usable


def sinusoid(period, vertical=False):
    y, x = np.mgrid[:512, :512]
    return 125 + 80*np.cos(2*np.pi*(y if vertical else x)/period)


def test_kernel_dc_and_energy_norm():
    for period in (16,32,64):
        for theta in ANGLES:
            k = kernel(period, float(theta), 2)
            assert abs(k.sum()) < 1e-12
            assert np.sum(abs(k)**2) == pytest.approx(1.)


def test_affine_control_preserves_features_without_clipping():
    a = sinusoid(32) + sinusoid(64,True)/5
    original, _, reason = measure_window(a)
    remapped, _, _ = measure_window(a, 'affine')
    assert not reason
    np.testing.assert_allclose(original, remapped, rtol=1e-11, atol=1e-11)
    assert summary(original) == pytest.approx(summary(remapped))


def test_rotation_changes_balance_sign_but_not_strength_or_scale():
    e, _, _ = measure_window(sinusoid(32))
    rotated, _, _ = measure_window(sinusoid(32,True))
    a, b = summary(e), summary(rotated)
    assert a[KEYS[2]] > .8 and b[KEYS[2]] < -.8
    assert a[KEYS[0]] == pytest.approx(b[KEYS[0]], abs=1e-9)
    assert a[KEYS[1]] == pytest.approx(b[KEYS[1]], abs=1e-9)


def test_coarse_share_measures_selected_period_not_particle_identity():
    fine, _, _ = measure_window(sinusoid(16))
    coarse, _, _ = measure_window(sinusoid(64))
    assert summary(coarse)[KEYS[0]] > summary(fine)[KEYS[0]] + .8


@pytest.mark.parametrize('a', [np.ones((512,512))*50,
                              np.full((512,512), np.nan), np.ones((30,30))])
def test_unmeasurable_windows_abstain(a):
    energy, _, reason = measure_window(a)
    assert energy is None and reason
    assert all(np.isnan(x) for x in summary(np.zeros((3,4))).values())


def test_window_coverage_and_missing_quality_are_explicit():
    assert windows((800,7000)) == []
    assert len(windows((2048,7000))) == 4
    t = pd.DataFrame({'batch':['Batch_3'],KEYS[0]:[.3], 'gabor_windows_used':[4]})
    assert usable(t,KEYS[0]).tolist() == [True]
    assert usable(t,KEYS[0],'quality_matched').tolist() == [False]
