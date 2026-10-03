"""Tests for polaron_qc.stats — calibration under the null, recovery of known effects,
contract shape, and a run on the real site cache. Run: python3 -m pytest tests/test_stats.py -q
"""
import os
import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from polaron_qc import CRACKED_SITES, GREY_PORE_SITES, LOW_CONTRAST_SITES, PRIMARY_KPIS
from polaron_qc import stats as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _null_pair(seed, n_ref=17, n_batch=7, k=None):
    g = np.random.default_rng(seed)
    if k is None:
        return g.normal(size=n_ref), g.normal(size=n_batch)
    return g.normal(size=(n_ref, k)), g.normal(size=(n_batch, k))


# ------------------------------------------------------------------ permutation test
def test_permutation_null_uniform_mean_diff():
    """Exact 7-vs-17 permutation p-values under the null are ~uniform for a statistic with
    a near-continuous null (mean difference). KS p > 0.01 over 300 seeded simulations."""
    ps = np.array([S.permutation_test(*_null_pair(i), statistic="mean_diff", seed=i)["p"] for i in range(300)])
    assert kstest(ps, "uniform").pvalue > 0.01
    assert np.mean(ps <= 0.05) <= 0.05 + 2.5 * np.sqrt(0.05 * 0.95 / 300)


def test_permutation_null_median_diff_is_valid_but_discrete():
    """The plan's difference-of-medians statistic is valid (P(p <= a) <= a) but strongly
    super-uniform at 7 vs 17: only ~56 distinct |T| values over 346,104 allocations, with a
    sizeable mass of null p-values at exactly 1. This test pins that documented property so
    nobody mistakes it for a bug (and nobody expects a uniform KS test from it)."""
    ps = np.array([S.permutation_test(*_null_pair(i), statistic="median_diff", seed=i)["p"] for i in range(150)])
    assert np.mean(ps <= 0.05) <= 0.05 + 2.5 * np.sqrt(0.05 * 0.95 / 150)   # level holds
    assert np.mean(ps >= 1.0) > 0.05                                           # discreteness is real
    r, b = _null_pair(3)
    bi, ri = S._exact_allocations(24, 7)
    pooled = np.concatenate([r, b])
    null = np.median(pooled[bi], 1) - np.median(pooled[ri], 1)
    assert len(np.unique(np.round(np.abs(null), 10))) < 100


def test_permutation_exact_vs_monte_carlo_agree():
    g = np.random.default_rng(7)
    r, b = g.normal(size=17), g.normal(size=7) + 0.8
    ex = S.permutation_test(r, b, statistic="mean_diff")
    mc = S.permutation_test(r, b, statistic="mean_diff", exact_max=0, n_mc=20_000, seed=1)
    assert ex["method"] == "exact" and mc["method"] == "monte_carlo"
    assert abs(ex["p"] - mc["p"]) < 0.02
    assert ex["observed"] == pytest.approx(mc["observed"])


def test_permutation_method_labels():
    assert S.permutation_test(*_null_pair(0, 7, 7))["method"] == "exact"
    assert S.permutation_test(*_null_pair(0, 7, 7))["n_perm"] == 3432
    assert S.permutation_test(*_null_pair(0, 17, 7))["method"] == "exact"
    assert S.permutation_test(*_null_pair(0, 17, 7))["n_perm"] == 346104
    out = S.permutation_test(*_null_pair(0, 40, 20))
    assert out["method"] == "monte_carlo" and out["n_perm"] == 20_000
    assert out["p"] >= 1 / 20_001  # (b+1)/(n+1) never returns 0


def test_permutation_min_p_7v7():
    """Plan §2.0 rule 8: smallest two-sided p at 7 vs 7 is 2/3432 ~ 0.0006."""
    r = np.arange(7, dtype=float)
    b = np.arange(7, dtype=float) + 100
    out = S.permutation_test(r, b, statistic="mean_diff")
    assert out["p"] == pytest.approx(2 / 3432)


def test_permutation_callable_statistic():
    r, b = _null_pair(0, 7, 7)
    out = S.permutation_test(r, b, statistic=lambda R, B: np.mean(B) - np.mean(R))
    ref = S.permutation_test(r, b, statistic="mean_diff")
    assert out["p"] == pytest.approx(ref["p"])


# ------------------------------------------------------------------ robust shift
def test_robust_shift_recovers_known_shift():
    g = np.random.default_rng(11)
    r = g.normal(size=17)
    b = g.normal(size=7) + 2 * S.mad(r)
    out = S.robust_shift(r, b, seed=0)
    assert out["ci_low"] <= 2 <= out["ci_high"]
    assert out["cliffs_delta"] > 0
    assert out["shift_mad"] > 0
    down = S.robust_shift(r, g.normal(size=7) - 2 * S.mad(r), seed=0)
    assert down["cliffs_delta"] < 0 and down["shift_mad"] < 0
    assert out["n_ref"] == 17 and out["n_batch"] == 7
    assert "approximate" in out["ci_method"]


def test_robust_shift_bootstrap_coverage_reasonable():
    """Approximate interval: coverage of a true 2-MAD shift over 40 seeds should be at least
    0.8 (percentile bootstrap on 7 sites is not exact; we only require it is not wildly off)."""
    cov = 0
    for i in range(40):
        g = np.random.default_rng(i)
        r = g.normal(size=17)
        b = g.normal(size=7) + 2 * S.mad(r)
        o = S.robust_shift(r, b, n_boot=1000, seed=i)
        cov += o["ci_low"] <= 2 <= o["ci_high"]
    assert cov / 40 >= 0.8


# ------------------------------------------------------------------ energy distance
def test_energy_distance_null_uniform():
    ps = np.array([S.energy_distance_test(*_null_pair(i, k=5), n_mc=1000, seed=i)["p"] for i in range(100)])
    assert kstest(ps, "uniform").pvalue > 0.01


def test_energy_distance_power():
    rej = 0
    for i in range(30):
        g = np.random.default_rng(100 + i)
        R, B = g.normal(size=(17, 5)), g.normal(size=(7, 5))
        B[:, :2] += 2 * S.mad(R[:, :2], axis=0)
        rej += S.energy_distance_test(R, B, n_mc=1000, seed=i)["p"] <= 0.05
    assert rej / 30 >= 0.8


def test_energy_distance_weights_and_shapes():
    R, B = _null_pair(0, k=3)
    out = S.energy_distance_test(R, B, weights=[2, 1, 0], n_mc=200)
    assert out["n_perm"] == 200 and out["method"] == "monte_carlo" and 0 < out["p"] <= 1
    out0 = S.energy_distance_test(R[:, :2], B[:, :2], weights=[2, 1], n_mc=200)
    assert out["statistic"] == pytest.approx(out0["statistic"])  # zero-weight column is inert
    with pytest.raises(ValueError):
        S.energy_distance_test(R, B[:, :2])


# ------------------------------------------------------------------ per-site drift
def test_per_site_drift_shape_and_percentiles():
    R, B = _null_pair(0, k=3)
    B[0] += 10
    df = S.per_site_drift(pd.DataFrame(R, index=[f"r{i}" for i in range(17)], columns=list("abc")),
                          pd.DataFrame(B, index=[f"b{i}" for i in range(7)], columns=list("abc")))
    assert len(df) == 24 and set(df.group) == {"reference", "batch"}
    assert df.loc[df.site == "b0", "percentile_in_ref"].item() == 100.0
    assert df.loc[df.site == "b0", "distance"].item() > df.loc[df.group == "reference", "distance"].max()
    assert {"z_a", "z_b", "z_c", "top_driver"} <= set(df.columns)


# ------------------------------------------------------------------ MDC
def test_mdc_properties():
    r = np.random.default_rng(1).normal(size=17)
    t0 = time.perf_counter()
    m7 = S.mdc(r, 7, seed=0)
    assert time.perf_counter() - t0 < 60
    pc = m7["power_curve"]
    assert np.all(np.diff(pc["power_monotone"]) >= 0)
    assert np.max(pc["power_monotone"] - pc["power"]) <= 0.05  # raw curve is near-monotone (common random numbers)
    assert np.isfinite(m7["mdc_mad"])
    assert pc["power"].iloc[0] <= 0.05 + 2.5 * np.sqrt(0.05 * 0.95 / 400)  # level holds at zero shift
    assert m7["mean_n_remaining"] == 10
    m4 = S.mdc(r, 4, seed=0)
    assert m4["mdc_mad"] >= m7["mdc_mad"]
    assert "without replacement" in m7["design"]


# ------------------------------------------------------------------ local exceedance
def test_iid_flag_probability():
    assert S.iid_flag_probability(10, 7) == pytest.approx(7 / 17)
    assert S.iid_flag_probability(10, 1) == pytest.approx(1 / 11)


def test_local_exceedance():
    ref = pd.Series(np.arange(10, dtype=float), index=[f"r{i}" for i in range(10)])
    m = S.mad(ref.to_numpy())
    bat = pd.Series([ref.max() + 0.5 * m, ref.max() + 1.5 * m, ref.max() - 1], index=["b0", "b1", "b2"])
    out = S.local_exceedance(ref, bat, margin_mad=1.0)
    t = out["table"].set_index("site")
    assert t.loc["b0", "exceeds"] and not t.loc["b0", "credible_severity"]
    assert t.loc["b0", "margin_in_mad"] == pytest.approx(0.5)
    assert t.loc["b1", "exceeds"] and t.loc["b1", "credible_severity"]
    assert not t.loc["b2", "exceeds"]
    assert out["iid_flag_probability"] == pytest.approx(3 / 13)
    assert out["n_exceed"] == 2 and out["n_credible"] == 1


# ------------------------------------------------------------------ multiplicity
def test_holm_standard_values():
    assert np.allclose(S.holm([0.01, 0.02, 0.03, 0.04, 0.05]), [0.05, 0.08, 0.09, 0.09, 0.09])
    adj = S.holm([0.01, np.nan, 0.03])
    assert np.isnan(adj[1]) and np.allclose(adj[[0, 2]], [0.02, 0.03])


def test_bh_standard_values():
    assert np.allclose(S.bh([0.01, 0.02, 0.03, 0.04, 0.05]), [0.05, 0.05, 0.05, 0.05, 0.05])
    assert np.allclose(S.bh([0.001, 0.5, 0.04]), [0.003, 0.5, 0.06])


# ------------------------------------------------------------------ jackknife / splits
def test_jackknife_and_stability():
    runs = S.jackknife(lambda kept, thr: "high" if np.mean(kept) > thr else "low", [1, 2, 3, 10], thr=3)
    assert [r["left_out"] for r in runs] == [1, 2, 3, 10]
    assert all(len(r["kept"]) == 3 for r in runs)
    assert S.stability_share(runs, "high") == pytest.approx(0.75)
    assert S.stability_share(["a", "a", "b"], "a") == pytest.approx(2 / 3)


def test_reference_split_diagnostics_moves_whole_sites():
    ref = pd.DataFrame({"site": [f"s{i}" for i in range(17)], "k": np.arange(17.0)})
    seen = []

    def pipe(ref_part, pseudo):
        seen.append((set(ref_part.site), set(pseudo.site)))
        return {"alert": bool(pseudo.k.mean() > ref_part.k.mean())}

    df = S.reference_split_diagnostics(ref, ["k"], 7, pipe, n_splits=20, seed=0)
    assert len(df) == 20 and set(df.n_pseudo) == {7} and set(df.n_ref_part) == {10}
    assert all(a.isdisjoint(b) and len(a | b) == 17 for a, b in seen)
    assert "alert" in df.columns
    with pytest.raises(KeyError):
        S.reference_split_diagnostics(ref, ["missing"], 7, pipe, n_splits=1)


# ------------------------------------------------------------------ usable n / real data
@pytest.fixture(scope="module")
def cache():
    a = pd.read_csv(os.path.join(ROOT, "analysis_cache", "site_features.csv"))
    b = pd.read_csv(os.path.join(ROOT, "analysis_cache", "etd_inlens_features.csv"))
    return a.merge(b, on=["batch", "site"])


def test_usable_n_rules(cache):
    b1 = cache[cache.batch == "Batch_1"]
    b3 = cache[cache.batch == "Batch_3"]
    u = S.usable_n(b1, "bright_frac")
    assert u["n_usable"] == 5 and sorted(u["excluded_sites"]) == sorted(LOW_CONTRAST_SITES)
    u = S.usable_n(b3, "pore_frac")
    assert u["n_usable"] == 17 and sorted(u["fallback_sites"]) == sorted(GREY_PORE_SITES)
    u = S.usable_n(b3, "crack_frac")
    assert u["n_usable"] == 17 and u["n_fallback"] == 4
    u = S.usable_n(b3, "fft_slope")
    assert u["n_usable"] == 17 and u["n_excluded"] == 0 and u["n_fallback"] == 0
    # a flag column on an unseen batch is honoured even when the site is not in the list
    b1b = b1.copy()
    b1b.loc[b1b.site == "f1vzngrs", "bright_low_contrast"] = True
    assert S.usable_n(b1b, "bright_d50")["n_usable"] == 4


def test_compare_kpis_real_cache(cache):
    kpis = PRIMARY_KPIS + ["pore_d50", "bright_d90", "etd_boundary_sharpness", "fft_slope"]
    ref = cache[cache.batch == "Batch_3"]
    bat = cache[cache.batch == "Batch_1"]
    t0 = time.perf_counter()
    out = S.compare_kpis(ref, bat, kpis, seed=0)
    assert time.perf_counter() - t0 < 60
    assert list(out.kpi) == kpis and len(out) == len(kpis)
    prim = out[out.is_primary]
    assert len(prim) == 5 and prim.p_holm.notna().all() and out[~out.is_primary].p_holm.isna().all()
    assert np.allclose(prim.p_holm.to_numpy(), S.holm(prim.p_perm.to_numpy()))
    assert (prim.p_holm >= prim.p_perm - 1e-12).all()
    bright = out[out.kpi.str.startswith("bright_")]
    assert (bright.n_batch_usable == 5).all() and (bright.n_ref_usable == 17).all()
    assert (out.loc[out.kpi.str.startswith(("pore_", "crack_")), "n_batch_usable"] == 7).all()
    assert (out.p_method == "exact").all()
    assert set(out.direction) <= {"higher", "lower", "none"}
    is_bright = out.kpi.str.startswith("bright_")
    assert set(out.n_perm[is_bright]) == {S.math.comb(22, 5)} and set(out.n_perm[~is_bright]) == {346104}
    for c in ["shift_mad", "ci_low", "ci_high", "cliffs_delta", "p_perm"]:
        assert out[c].notna().all()
