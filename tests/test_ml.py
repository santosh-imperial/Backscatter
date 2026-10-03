"""Fast tests for polaron_qc.ml. No model download: embeddings are synthetic Gaussians."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

import polaron_qc.ml as ml
from polaron_qc.ml import _site_weights, c2st, novelty


def _grouped_data(rng, n_sites, n_feat, patches=(4, 8), site_sd=1.0, patch_sd=1.0, shift=None, prefix="s"):
    """Sites with a random site effect plus patch noise; optional per-feature shift added to every site."""
    X, g = [], []
    for i in range(n_sites):
        n_p = rng.integers(patches[0], patches[1] + 1)
        mu = rng.normal(0, site_sd, n_feat) + (shift if shift is not None else 0)
        X.append(mu + rng.normal(0, patch_sd, (n_p, n_feat)))
        g += [f"{prefix}{i}"] * n_p
    return np.vstack(X), np.array(g)


def test_c2st_p_uniform_under_null_with_patches():
    rng = np.random.default_rng(1)
    ps = []
    for sim in range(30):
        Xr, gr = _grouped_data(rng, 5, 5, prefix="r")
        Xq, gq = _grouped_data(rng, 5, 5, prefix="q")
        r = c2st(Xr, Xq, gr, gq, [f"f{i}" for i in range(5)], n_perm=30, seed=sim)
        ps.append(r["p"])
    ks = sps.kstest(ps, "uniform")
    assert ks.pvalue > 0.01, f"p-values not uniform: KS p = {ks.pvalue:.4f}, ps = {np.round(ps, 2)}"
    assert np.mean(np.array(ps) <= 0.05) <= 0.2         # no gross anti-conservativeness in 30 sims


def test_c2st_detects_shift_and_selects_features():
    rng = np.random.default_rng(2)
    shift = np.zeros(10); shift[[0, 1]] = 2.0           # 2 SD shift (site SD = patch SD = 1 -> total SD sqrt2)
    Xr, gr = _grouped_data(rng, 8, 10, prefix="r", site_sd=0.5, patch_sd=0.5)
    Xq, gq = _grouped_data(rng, 8, 10, prefix="q", site_sd=0.5, patch_sd=0.5, shift=shift * 0.5 * np.sqrt(2))
    names = [f"f{i}" for i in range(10)]
    r = c2st(Xr, Xq, gr, gq, names, n_perm=60, seed=0)
    assert r["p"] < 0.05, r["p"]
    assert r["auc"] > 0.8
    sel = set(r["coef"][r["coef"].selected].feature)
    assert {"f0", "f1"} <= sel, sel
    top2 = set(r["coef"].sort_values("mean_abs_shap", ascending=False).feature[:2])
    assert top2 == {"f0", "f1"}, top2
    assert (r["coef"].set_index("feature").loc[["f0", "f1"], "sign"] == "+").all()
    assert r["n_perm"] == 60 and "Monte Carlo" in r["p_method"]
    assert len(r["auc_null"]) == 60 and len(r["per_site_scores"]) == 16


def test_site_weights_equalise_sites_and_classes():
    g = np.array(["a"] * 50 + ["b"] * 5 + ["c"] * 5 + ["d"] * 50)
    y = np.array([0] * 55 + [1] * 55)
    w = _site_weights(g, y)
    tot = pd.Series(w).groupby(g).sum()
    assert np.allclose(tot.values, tot.values[0]), tot          # 50-patch site == 5-patch site
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())
    assert np.isclose(w.mean(), 1.0)
    # class balance is by SITE count: class 0 has 2 sites with 60 rows, class 1 has 3 sites with 15 rows
    g2 = np.array(["a"] * 50 + ["b"] * 10 + ["c"] * 5 + ["d"] * 5 + ["e"] * 5)
    y2 = np.array([0] * 60 + [1] * 15)
    w2 = _site_weights(g2, y2)
    assert np.isclose(w2[y2 == 0].sum(), w2[y2 == 1].sum())
    assert np.allclose(pd.Series(w2[y2 == 1]).groupby(g2[y2 == 1]).sum().values, w2[y2 == 1].sum() / 3)


def test_site_weighting_changes_fit_not_dominated_by_big_site():
    """A 50-patch site whose patches point one way must not out-vote five 5-patch sites of the same class.
    With equal row weights the big site would dominate the coefficient sign; with site weights it does not."""
    rng = np.random.default_rng(3)
    # reference: 6 sites at 0; batch: 5 small sites at +1 on f0, one 50-patch site at -1 on f0
    Xr = rng.normal(0, 0.3, (30, 2)); gr = np.repeat([f"r{i}" for i in range(6)], 5)
    small = rng.normal([1, 0], 0.3, (25, 2)); big = rng.normal([-1, 0], 0.3, (50, 2))
    Xq = np.vstack([small, big]); gq = np.r_[np.repeat([f"q{i}" for i in range(5)], 5), ["big"] * 50]
    r = c2st(Xr, Xq, gr, gq, ["f0", "f1"], n_perm=5, seed=0)
    coef = r["coef"].set_index("feature").coef
    assert coef["f0"] > 0, coef                                  # five small sites win over one big site
    # same data with the big site's rows re-weighted equally would flip the sign (sanity on the construction)
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([Xr, Xq]); y = np.r_[np.zeros(30), np.ones(75)]
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.5).fit((X - X.mean(0)) / X.std(0), y)
    assert lr.coef_[0][0] < 0


def test_c2st_rejects_overlapping_site_ids():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    with pytest.raises(ValueError):
        c2st(X[:5], X[5:], ["a"] * 5, ["a"] * 5, list("xyz"), n_perm=2)


def _invariance_inputs(shape, rng):
    """Baseline inputs: patch-level (4–8 rows per site, group ids repeated) or one row per site."""
    if shape == "patches":
        Xr, gr = _grouped_data(rng, 6, 4, prefix="r")
        Xq, gq = _grouped_data(rng, 6, 4, prefix="q")
    else:
        Xr, gr = _grouped_data(rng, 7, 4, patches=(1, 1), prefix="r")
        Xq, gq = _grouped_data(rng, 6, 4, patches=(1, 1), prefix="q")
    return Xr, Xq, gr, gq


def _renamed(gr, gq, rng):
    """Random 8-char ids whose sort order differs from the originals' (asserted)."""
    old = list(np.unique(np.r_[gr, gq]))
    while True:
        new_id = {g: f"{rng.integers(0, 16**8):08x}" for g in old}
        if [new_id[g] for g in old] != sorted(new_id.values()):
            return new_id


def _assert_bit_identical(r1, r2, names, site_map):
    assert r1["auc"] == r2["auc"]
    assert r1["p"] == r2["p"]
    assert np.array_equal(r1["auc_null"], r2["auc_null"])
    assert r1["cv"] == r2["cv"]
    c1, c2 = r1["coef"].set_index("feature"), r2["coef"].set_index("feature")
    for col in ("coef", "selection_stability", "mean_abs_shap"):
        assert np.array_equal(c1.loc[names, col].to_numpy(), c2.loc[names, col].to_numpy()), col
    # per-site scores carry the caller's ids and agree site by site
    s1 = r1["per_site_scores"].set_index("site")
    s2 = r2["per_site_scores"].set_index("site")
    assert set(s1.index) == set(site_map) and set(s2.index) == set(site_map.values())
    old = list(site_map)
    assert np.array_equal(s1.loc[old, "mean_prob"].to_numpy(), s2.loc[[site_map[g] for g in old], "mean_prob"].to_numpy())
    assert np.array_equal(s1.loc[old, "n_rows"].to_numpy(), s2.loc[[site_map[g] for g in old], "n_rows"].to_numpy())


@pytest.mark.parametrize("shape", ["patches", "one_row_per_site"])
def test_c2st_invariant_to_site_id_renaming(shape):
    """(1) E23 / D34 open item: identical content under new site ids (sorting differently) gave AUC 0.51 → 0.41 because
    the grouped folds and the site-label permutation stream followed site-id sort order. Same rows, renamed ids."""
    rng = np.random.default_rng(9)
    Xr, Xq, gr, gq = _invariance_inputs(shape, rng)
    names = list("abcd")
    r1 = c2st(Xr, Xq, gr, gq, names, n_perm=25, seed=0)
    new_id = _renamed(gr, gq, rng)
    r2 = c2st(Xr, Xq, [new_id[g] for g in gr], [new_id[g] for g in gq], names, n_perm=25, seed=0)
    _assert_bit_identical(r1, r2, names, new_id)


@pytest.mark.parametrize("shape", ["patches", "one_row_per_site"])
def test_c2st_invariant_to_row_order(shape):
    """(2) Same ids, rows permuted: within each block, and across the two blocks while keeping ref / batch membership
    (c2st concatenates reference then batch, so 'across' means the rows of each block arrive in an arbitrary order and
    site rows are no longer contiguous)."""
    rng = np.random.default_rng(10)
    Xr, Xq, gr, gq = _invariance_inputs(shape, rng)
    names = list("abcd")
    r1 = c2st(Xr, Xq, gr, gq, names, n_perm=25, seed=0)
    ident = {g: g for g in np.unique(np.r_[gr, gq])}
    # within-block permutation
    pr, pq = rng.permutation(len(Xr)), rng.permutation(len(Xq))
    assert not (np.array_equal(pr, np.arange(len(Xr))) or np.array_equal(pq, np.arange(len(Xq))))
    r2 = c2st(Xr[pr], Xq[pq], gr[pr], gq[pq], names, n_perm=25, seed=0)
    _assert_bit_identical(r1, r2, names, ident)
    # reverse order inside each block (sites become contiguous in the opposite order)
    r3 = c2st(Xr[::-1], Xq[::-1], gr[::-1], gq[::-1], names, n_perm=25, seed=0)
    _assert_bit_identical(r1, r3, names, ident)
    # patches of one site interleaved with other sites' (membership unchanged): sort rows by a random key per row
    if shape == "patches":
        kr, kq = rng.random(len(Xr)), rng.random(len(Xq))
        r4 = c2st(Xr[np.argsort(kr)], Xq[np.argsort(kq)], gr[np.argsort(kr)], gq[np.argsort(kq)], names, n_perm=25, seed=0)
        _assert_bit_identical(r1, r4, names, ident)


@pytest.mark.parametrize("shape", ["patches", "one_row_per_site"])
def test_c2st_invariant_to_renaming_and_row_order_combined(shape):
    """(3) Both at once: renamed ids and permuted rows."""
    rng = np.random.default_rng(11)
    Xr, Xq, gr, gq = _invariance_inputs(shape, rng)
    names = list("abcd")
    r1 = c2st(Xr, Xq, gr, gq, names, n_perm=25, seed=0)
    new_id = _renamed(gr, gq, rng)
    pr, pq = rng.permutation(len(Xr)), rng.permutation(len(Xq))
    r2 = c2st(Xr[pr], Xq[pq], np.array([new_id[g] for g in gr])[pr], np.array([new_id[g] for g in gq])[pq], names, n_perm=25, seed=0)
    _assert_bit_identical(r1, r2, names, new_id)


def test_c2st_is_deterministic_for_fixed_seed_and_changes_with_content():
    """The same call twice is bit-identical (liblinear is seeded); a genuinely different batch is not a no-op."""
    rng = np.random.default_rng(12)
    Xr, Xq, gr, gq = _invariance_inputs("patches", rng)
    names = list("abcd")
    r1 = c2st(Xr, Xq, gr, gq, names, n_perm=20, seed=0)
    r2 = c2st(Xr, Xq, gr, gq, names, n_perm=20, seed=0)
    assert r1["auc"] == r2["auc"] and np.array_equal(r1["auc_null"], r2["auc_null"])
    assert np.array_equal(r1["coef"].coef.to_numpy(), r2["coef"].coef.to_numpy())
    r3 = c2st(Xr, Xq + 0.5, gr, gq, names, n_perm=20, seed=0)
    assert not np.array_equal(r1["auc_null"], r3["auc_null"]) or r1["auc"] != r3["auc"]


def _gauss_sites(rng, n_sites, n_patch, dim, mu, prefix):
    E = np.vstack([mu + rng.normal(0, 1, (n_patch, dim)) for _ in range(n_sites)])
    g = np.repeat([f"{prefix}{i}" for i in range(n_sites)], n_patch)
    return E, g


def test_novelty_higher_for_shifted_cluster_and_calibrated_percentiles():
    rng = np.random.default_rng(4)
    dim = 48
    Er, gr = _gauss_sites(rng, 8, 30, dim, 0, "ref")
    mu = np.zeros(dim); mu[:6] = 3.0
    Es, gs = _gauss_sites(rng, 2, 30, dim, mu, "shift")
    En, gn = _gauss_sites(rng, 2, 30, dim, 0, "null")
    out = novelty(Er, np.vstack([Es, En]), gr, np.r_[gs, gn], n_components=16, k=5, seed=0)
    ps = out["per_site"].set_index("site")
    assert ps.loc[["shift0", "shift1"], "novelty_median"].min() > ps.loc[["null0", "null1"], "novelty_median"].max()
    qp = out["query_percentile"].set_index("site")
    assert (qp.loc[["shift0", "shift1"], "pct_median"] == 100).all()
    assert qp.loc[["shift0", "shift1"], "exceeds_ref_loo_max"].all()
    assert not qp.loc[["null0", "null1"], "exceeds_ref_loo_max"].any()
    # same-distribution query sites look like held-out reference sites (within the ref_loo spread)
    lo, hi = out["ref_loo"].novelty_median.min(), out["ref_loo"].novelty_median.max()
    assert (ps.loc[["null0", "null1"], "novelty_median"] < hi * 1.05).all()
    assert len(out["per_patch"]) == 120 and set(out["per_patch"].columns) == {"site", "patch_id", "novelty"}
    assert len(out["ref_loo"]) == 8 and out["n_components"] == 16


def test_ref_loo_refits_pca_per_fold(monkeypatch):
    rng = np.random.default_rng(5)
    Er, gr = _gauss_sites(rng, 6, 20, 32, 0, "ref")
    Eq, gq = _gauss_sites(rng, 1, 20, 32, 0, "q")
    calls = []
    orig = ml._fit_pca

    def spy(X, n_components, seed):
        calls.append(len(X))
        return orig(X, n_components, seed)

    monkeypatch.setattr(ml, "_fit_pca", spy)
    out = novelty(Er, Eq, gr, gq, n_components=8, k=3, seed=0)
    assert len(calls) == 1 + 6                              # one full-reference fit + one per held-out site
    assert calls[0] == 120 and all(c == 100 for c in calls[1:])  # each fold fits on the remaining 5 sites only
    comps = out["ref_loo_pca_first_component"]
    assert comps.shape == (6, 32)
    # first principal axes differ between folds (allowing for sign flips)
    cos = np.abs(comps @ comps.T)
    off = cos[~np.eye(6, dtype=bool)]
    assert (off < 0.999).all(), off


def test_novelty_correlates_shapes():
    rng = np.random.default_rng(6)
    sites = [f"s{i}" for i in range(12)]
    nov = pd.DataFrame({"site": sites, "novelty_median": rng.normal(size=12)})
    feats = pd.DataFrame({"site": sites, "pore_frac": nov.novelty_median * 2 + rng.normal(0, 0.1, 12),
                          "bright_frac": rng.normal(size=12), "bse_p50": rng.normal(size=12),
                          "constant": 1.0, "flag_grey_pore": [0] * 9 + [1] * 3})
    out = ml.novelty_correlates(nov, feats, acquisition_cols=["bse_p50", "constant", "flag_grey_pore", "missing"])
    assert set(out.columns) == {"variable", "kind", "rho", "p", "n"}
    assert out.iloc[0].variable == "pore_frac" and out.iloc[0].rho > 0.9
    assert "constant" not in set(out.variable) and "missing" not in set(out.variable)
    assert set(out[out.kind == "acquisition"].variable) == {"bse_p50", "flag_grey_pore"}


def test_image_helpers():
    rng = np.random.default_rng(7)
    img = rng.integers(40, 70, (300, 700)).astype(np.uint8)
    img[:10] = 200; img[-6:] = 220                              # bright bands at both edges
    top, bot = ml.bright_bands(img)
    assert top == 10 and bot == 6
    x, meta = ml.normalise_image(img[top:-bot])
    assert x.min() >= 0 and x.max() <= 1 and 0 < meta["iqr"]
    assert ml.tile_coords(284, 700, 128, 128) == [(i * 5 + j, i * 128, j * 128) for i in range(2) for j in range(5)]


def test_novelty_nearest_ref_points_at_reference_patches():
    rng = np.random.default_rng(8)
    Er, gr = _gauss_sites(rng, 4, 10, 16, 0, "ref")
    Eq, gq = _gauss_sites(rng, 1, 5, 16, 0, "q")
    out = novelty(Er, Eq, gr, gq, n_components=4, k=3, seed=0)
    nr = out["nearest_ref"]
    assert len(nr) == 5 * 3 and set(nr.ref_site) <= set(gr)
    assert (nr.groupby(["site", "patch_id"]).size() == 3).all()
