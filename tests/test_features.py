"""Regression and contract tests for polaron_qc.features.

The regression half re-extracts Dataset/Batch_1 (7 sites, ~1 min) and compares against the notebook-01 caches in
analysis_cache/. Deterministic columns must match to 1e-6 relative; the seeded texture descriptors (fft_slope,
corr_len_px) are expected to match exactly too, but are only required within 2 % as the contract allows.
"""
import os

import numpy as np
import pandas as pd
import pytest

from polaron_qc import GREY_PORE_SITES, LOW_CONTRAST_SITES
from polaron_qc.features import (FEATURE_VERSION, RIDGE_T, IMAGE_COLUMNS, PARTICLE_COLUMNS, PATCH_COLUMNS, extract_batch,
                                 folder_hash, list_sites, load_image, phase_thresholds, segment, smooth_bse, trimmed, ridge_maps)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH1 = os.path.join(ROOT, "Dataset", "Batch_1")
CACHE = os.path.join(ROOT, "analysis_cache")
SAMPLED = ["fft_slope", "corr_len_px"]

pytestmark = pytest.mark.skipif(not os.path.isdir(BATCH1), reason="Dataset/Batch_1 not available")


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    cache = tmp_path_factory.mktemp("features_cache")
    return extract_batch(BATCH1, cache_dir=str(cache), verbose=False), cache


@pytest.fixture(scope="module")
def ref():
    F = pd.read_csv(os.path.join(CACHE, "site_features.csv")); E = pd.read_csv(os.path.join(CACHE, "etd_inlens_features.csv"))
    R = F.merge(E, on=["batch", "site"]); R = R[R.batch == "Batch_1"].sort_values("site").reset_index(drop=True)
    Q = pd.read_csv(os.path.join(CACHE, "image_quality.csv")); Q = Q[Q.batch == "Batch_1"]
    P = pd.read_csv(os.path.join(CACHE, "bright_particles.csv")); P = P[P.batch == "Batch_1"]
    return R, Q, P, list(F.columns), list(E.columns)


def _compare(got, exp, col, rtol):
    g, e = got[col].values, exp[col].values
    if exp[col].dtype == bool or exp[col].dtype == object:
        assert (g == e).all(), f"{col}: {list(zip(e, g))}"
        return
    g = g.astype(float); e = e.astype(float)
    both_nan = np.isnan(g) & np.isnan(e)
    ok = both_nan | np.isclose(g, e, rtol=rtol, atol=1e-12)
    assert ok.all(), f"{col}: expected {e[~ok]} got {g[~ok]}"


# ---------------------------------------------------------------------------------------------------------------
# regression against notebook-01 caches
# ---------------------------------------------------------------------------------------------------------------
def test_sites_columns_superset(tables, ref):
    S = tables[0]["sites"]; _, _, _, fcols, ecols = ref
    missing = (set(fcols) | set(ecols)) - set(S.columns)
    assert not missing, f"missing site columns: {sorted(missing)}"
    for c in ["etd_crack_density_particles", "etd_crack_density_graphite", "etd_curtain_frac", "n_patches", "batch_dir"]:
        assert c in S.columns
    assert len(S) == 7 and sorted(S.site) == list_sites(BATCH1)


def test_sites_deterministic_columns_match(tables, ref):
    S = tables[0]["sites"].sort_values("site").reset_index(drop=True); R = ref[0]
    cols = [c for c in R.columns if c not in SAMPLED]
    for c in cols:
        _compare(S, R, c, rtol=1e-6)


def test_sites_sampled_columns_match(tables, ref):
    S = tables[0]["sites"].sort_values("site").reset_index(drop=True); R = ref[0]
    for c in SAMPLED:
        _compare(S, R, c, rtol=0.02)
    # same seeds and sampling as the notebook: expect bit-for-bit agreement in practice
    assert np.allclose(S.corr_len_px, R.corr_len_px) and np.allclose(S.fft_slope, R.fft_slope, rtol=1e-9)


def test_ridge_t_columns(tables):
    S = tables[0]["sites"]
    assert RIDGE_T == 0.4
    assert np.allclose(S.etd_crack_density_particles, S["crack_p_0.4"], equal_nan=True)
    assert np.allclose(S.etd_crack_density_graphite, S["crack_g_0.4"])
    assert np.allclose(S.etd_curtain_frac, S["curtain_0.4"])
    assert set(S[S.site.isin(LOW_CONTRAST_SITES)].bright_low_contrast) == {True}
    assert not S[~S.site.isin(LOW_CONTRAST_SITES)].bright_low_contrast.any()


def test_images_match(tables, ref):
    I = tables[0]["images"]; Q = ref[1]
    assert list(I.columns) == IMAGE_COLUMNS == list(Q.columns)
    assert len(I) == 21 and set(I.det) == {"BSE", "ETD", "Inlens"}
    key = ["site", "det"]
    I = I.sort_values(key).reset_index(drop=True); Q = Q.sort_values(key).reset_index(drop=True)
    assert (I[key].values == Q[key].values).all()
    for c in IMAGE_COLUMNS[3:]:
        _compare(I, Q, c, rtol=1e-9)


def test_particles_match(tables, ref):
    Pm = tables[0]["particles"]; P = ref[2]
    assert list(Pm.columns) == PARTICLE_COLUMNS == list(P.columns)
    assert len(Pm) == len(P)
    key = ["site", "area", "perimeter", "eccentricity"]
    Pm = Pm.sort_values(key).reset_index(drop=True); P = P.sort_values(key).reset_index(drop=True)
    assert (Pm.site.values == P.site.values).all()
    for c in PARTICLE_COLUMNS[:8]:
        _compare(Pm, P, c, rtol=1e-9)


# ---------------------------------------------------------------------------------------------------------------
# patch table contract
# ---------------------------------------------------------------------------------------------------------------
def test_patches_shape_and_ranges(tables):
    T, S = tables[0]["patches"], tables[0]["sites"]
    assert list(T.columns) == PATCH_COLUMNS
    # shape: full 512-px windows on the trimmed BSE frame, stride 512
    expect = sum((r.H // 512) * (r.W // 512) for r in S.itertuples())
    assert len(T) == expect == S.n_patches.sum()
    for r in S.itertuples():
        t = T[T.site == r.site]
        assert t.patch_id.tolist() == list(range(len(t)))
        assert t.y0.max() + 512 <= r.H and t.x0.max() + 512 <= r.W
        assert (t.bright_low_contrast == r.bright_low_contrast).all() and (t.grey_pore == (r.site in GREY_PORE_SITES)).all()
        # patch means aggregate back to the site fraction on the covered area (same masks, same thresholds)
        assert abs(t.pore_frac.mean() - r.pore_frac) < 0.03 and abs(t.bright_frac.mean() - r.bright_frac) < 0.03
        assert t.pore_max_d.max() <= r.pore_max_d + 1e-9
    for c in ["pore_frac", "bright_frac", "crack_area_frac"]:
        assert T[c].between(0, 1).all()
    assert (T.bright_count >= 0).all() and T.bright_count.dtype.kind == "i"
    assert (T.pore_max_d.dropna() > 0).all()
    assert T.corr_len_px.dropna().between(1, 256).all()
    assert T.graphite_mode_local.between(10, 209).all()
    assert T.pore_frac.std() > 0 and T.corr_len_px.std() > 0  # C01: nothing constant by construction
    assert T.crack_area_frac.max() > 0  # Batch 1 has crack-like voids somewhere


# ---------------------------------------------------------------------------------------------------------------
# helpers and caching
# ---------------------------------------------------------------------------------------------------------------
def test_segment_and_thresholds_consistency(tables):
    S = tables[0]["sites"]; r = S[S.site == "fzrt2k6r"].iloc[0]
    a = trimmed(load_image(BATCH1, "fzrt2k6r", "BSE"))
    th = phase_thresholds(smooth_bse(a))
    assert th["th_lo"] == r.th_lo and th["th_hi"] == r.th_hi and th["graphite_mode"] == r.graphite_mode
    pore, bright, sm = segment(a, r.th_lo, r.th_hi)
    assert pore.shape == a.shape == sm.shape and abs(pore.mean() - r.pore_frac) < 1e-12 and abs(bright.mean() - r.bright_frac) < 1e-12


def test_load_image_se_alias():
    b2 = os.path.join(ROOT, "Dataset", "Batch_2")
    if not os.path.exists(os.path.join(b2, "img_rxax5ozo_SE.tif")):
        pytest.skip("SE-labelled example not present")
    e = load_image(b2, "rxax5ozo", "ETD"); s = load_image(b2, "rxax5ozo", "SE")
    assert e.ndim == 2 and e.dtype == np.uint8 and np.array_equal(e, s)


def test_ridge_maps_shape():
    e = trimmed(load_image(BATCH1, "fzrt2k6r", "ETD"))[:600, :800]
    ridge, theta = ridge_maps(e)
    assert ridge.shape == theta.shape == e.shape and (ridge >= 0).all() and theta.min() >= -90 and theta.max() <= 90


def test_cache_roundtrip(tables):
    out, cache = tables
    h = folder_hash(BATCH1)
    files = sorted(os.listdir(cache))
    assert len(files) == 4 and all(f.startswith(f"Batch_1_{h}.") for f in files)
    again = extract_batch(BATCH1, cache_dir=str(cache), verbose=False)   # cache hit: no recomputation
    for k in out:
        assert list(again[k].columns) == list(out[k].columns) and len(again[k]) == len(out[k])
    pd.testing.assert_frame_equal(again["sites"].drop(columns=["batch_dir"]), out["sites"].drop(columns=["batch_dir"]), check_dtype=False)
    assert isinstance(FEATURE_VERSION, str) and FEATURE_VERSION
