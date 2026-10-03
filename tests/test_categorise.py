"""Fast tests for polaron_qc.categorise: synthetic 3-class site tables, plus one real-cache smoke test guarded by cache
availability. Permutation counts are small (the procedure, not the Monte Carlo precision, is under test)."""
import os

import numpy as np
import pandas as pd
import pytest

import polaron_qc.categorise as cat
from polaron_qc import KPI_TRUST, MATERIAL_KPIS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "Dataset")
FEATURE_CACHE = os.path.join(ROOT, "analysis_cache", "features")
FEATS = [f"f{i}" for i in range(8)]


def _synthetic(rng, n_per=(8, 8, 12), shift=0.0, prefix="s", nan_one=False):
    """Three classes, 8 features; classes 0/1 shifted on f0/f1 by ±shift (class 2 = reference at the origin)."""
    rows = []
    for c, n in enumerate(n_per):
        mu = np.zeros(8)
        if c == 0: mu[[0, 1]] = +shift
        if c == 1: mu[[0, 1]] = -shift
        X = mu + rng.normal(0, 1, (n, 8))
        for i in range(n):
            rows.append(dict(batch=f"B{c}", site=f"{prefix}{c}_{i}", acquisition_group="ordinary", **dict(zip(FEATS, X[i]))))
    df = pd.DataFrame(rows)
    if nan_one:
        df.loc[0, "f3"] = np.nan
    return df


# ---------------------------------------------------------------- feature family hygiene (C21) ----------------------
def test_feature_families_are_disjoint_and_trusted():
    assert set(MATERIAL_KPIS) <= set(cat.MORPH_FEATURES)
    assert all(KPI_TRUST[k] in ("high", "medium") for k in cat.MORPH_FEATURES)
    assert not set(cat.MORPH_FEATURES) & set(cat.ACQ_FEATURES)
    assert not set(cat.MORPH_FEATURES) & set(cat.TEXTURE_FEATURES)
    assert not set(cat.ACQ_FEATURES) & set(cat.TEXTURE_FEATURES)
    assert len(cat.FAMILIES["combined"]) == len(set(cat.FAMILIES["combined"])) == 38
    assert len(cat.FAMILIES["material"]) == 29 and set(cat.FAMILIES["material"]) == set(cat.FAMILIES["morph"]) | set(cat.TEXTURE_FEATURES)
    assert all("H" not in f for f in cat.FAMILIES.values()), "frame height is a session fingerprint and may not enter any family (D52)"
    assert "etd_boundary_sharpness" not in cat.MORPH_FEATURES          # D24: a flag, never material
    assert all("inlens" not in f for f in cat.MORPH_FEATURES)          # confounded texture never in morphology
    assert cat.PRIMARY_FAMILY == "material" and cat.PRIMARY_OOD_VARIANT == "morph"
    assert "not established" in cat.TEXTURE_LABEL and "not established" in cat.FAMILY_LABELS["material"]
    assert "no session statistics" in cat.FAMILY_LABELS["material"] and "comparator" in cat.FAMILY_LABELS["combined"]


# ---------------------------------------------------------------- categoriser ---------------------------------------
def test_loo_detects_separated_classes_and_reports_full_metrics():
    df = _synthetic(np.random.default_rng(0), shift=3.0, nan_one=True)
    r = cat.loo_evaluate(df, features=FEATS, n_perm=25, seed=0, n_jobs=1)
    m = r["metrics"]
    assert m["balanced_accuracy"] > 0.7 and m["accuracy"] > 0.7
    assert r["p"] <= 2 / 26 + 1e-12, (r["p"], r["perm_null"])
    assert m["confusion"].to_numpy().sum() == 28 and set(m["recall"]) == {"B0", "B1", "B2"}
    lo, hi = m["accuracy_ci95"]
    assert 0 <= lo <= m["accuracy"] <= hi <= 1
    assert abs(m["majority_accuracy"] - 12 / 28) < 1e-12 and abs(m["chance_balanced"] - 1 / 3) < 1e-12
    ps = r["per_site"]
    assert np.allclose(ps[[f"mp_{c}" for c in r["classes"]]].sum(axis=1), 1.0)
    assert set(ps["C"]) <= set(cat.C_GRID) and len(r["fold_models"]) == 28
    assert "Monte Carlo" in r["p_method"] and r["n_perm"] == 25


def test_loo_null_data_gives_no_small_p_and_near_chance():
    df = _synthetic(np.random.default_rng(3), shift=0.0)
    r = cat.loo_evaluate(df, features=FEATS, n_perm=25, seed=1, n_jobs=1)
    assert r["metrics"]["balanced_accuracy"] < 0.6
    assert r["p"] > 0.05


def test_loo_is_invariant_to_site_ids_and_row_order():
    rng = np.random.default_rng(5)
    df = _synthetic(rng, shift=1.5)
    a = cat.loo_evaluate(df, features=FEATS, n_perm=10, seed=0, n_jobs=1)
    df2 = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    df2["site"] = ["z" + str(rng.integers(1e9)) for _ in range(len(df2))]
    b = cat.loo_evaluate(df2, features=FEATS, n_perm=10, seed=0, n_jobs=1)
    pa = a["per_site"].sort_values([f"mp_{c}" for c in a["classes"]]).reset_index(drop=True)
    pb = b["per_site"].sort_values([f"mp_{c}" for c in b["classes"]]).reset_index(drop=True)
    assert np.allclose(pa[[f"mp_{c}" for c in a["classes"]]].to_numpy(), pb[[f"mp_{c}" for c in b["classes"]]].to_numpy())
    assert a["metrics"]["balanced_accuracy"] == b["metrics"]["balanced_accuracy"]
    assert np.array_equal(a["perm_null"], b["perm_null"])          # same canonical order -> same permutation stream


def test_select_C_is_fold_local_and_in_grid():
    df = _synthetic(np.random.default_rng(8), shift=1.0)
    X = df[FEATS].to_numpy(); classes, y = cat._encode(df)
    C = cat._select_C(X, y, 3, cat.C_GRID, 0, cat.N_INNER)
    assert C in cat.C_GRID
    # degenerate: a class with one member -> middle of the grid, no crash
    y2 = y.copy(); y2[y2 == 0] = 2; y2[0] = 0
    assert cat._select_C(X, y2, 3, cat.C_GRID, 0, cat.N_INNER) == cat.C_GRID[1]


def test_calibration_scores():
    y = np.array([0, 1, 2, 2])
    P_perfect = np.eye(3)[y]
    c = cat.calibration(y, P_perfect)
    assert c["brier"] == 0 and c["ece"] == 0 and c["top_accuracy"] == 1
    P_flat = np.full((4, 3), 1 / 3)
    c2 = cat.calibration(y, P_flat)
    assert abs(c2["brier"] - 2 / 3) < 1e-12
    prior = np.array([0.25, 0.25, 0.5]); exp_prior = np.mean(((prior[None, :] - np.eye(3)[y]) ** 2).sum(axis=1))
    assert abs(c2["brier_prior"] - exp_prior) < 1e-12
    assert list(c2["reliability"]["n"]) == [4, 0, 0]


def test_fit_predict_and_contributions():
    df = _synthetic(np.random.default_rng(1), shift=2.5)
    m = cat.fit_categoriser(df, features=FEATS, ref_batch="B2")
    assert m.C in cat.C_GRID and m.classes == ["B0", "B1", "B2"] and m.n_train == 28
    q = pd.DataFrame([dict(batch="?", site="new", **{f: v for f, v in zip(FEATS, [3, 3, 0, 0, 0, 0, 0, 0])})])
    pr = cat.predict_proba(m, q)
    assert pr["argmax"].iloc[0] == "B0" and abs(pr[["mp_B0", "mp_B1", "mp_B2"]].sum(axis=1).iloc[0] - 1) < 1e-9
    cs = cat.categoriser_contributions(m.pipeline, q[FEATS].to_numpy()[0], FEATS, 0, m.ref_median, m.ref_mad, top=3)
    assert cs and {c["feature"] for c in cs} & {"f0", "f1"}
    assert all(c["sign"] in "+-" and np.isfinite(c["ref_median"]) for c in cs)
    assert all(c["sign"] == "+" for c in cs if c["feature"] in ("f0", "f1"))   # pushed towards B0


# ---------------------------------------------------------------- OOD -----------------------------------------------
def test_ood_reference_loo_and_query_percentiles():
    rng = np.random.default_rng(2)
    ref = _synthetic(rng, n_per=(0, 0, 17), shift=0.0)
    ref["batch"] = "B2"
    om = cat.fit_ood(ref, features=FEATS, k=3)
    assert om.n_ref == 17 and len(om.ref_loo) == 17 and len(om.loo_fits) == 17 and not om.dropped
    assert (om.ref_loo["score_knn"] > 0).all()                      # no self-match in the reference LOO
    far = pd.DataFrame([dict(batch="Q", site="far", **{f: 8.0 for f in FEATS})])
    near = ref.iloc[[0]].copy(); near["site"] = "near"; near["batch"] = "Q"
    s = cat.ood_score(om, pd.concat([far, near], ignore_index=True))
    assert s.loc[0, "ood_pct_knn"] == 100 and s.loc[0, "ood_exceeds_knn"] and abs(s.loc[0, "ood_rank_p_knn"] - 1 / 18) < 1e-12
    assert s.loc[1, "ood_pct_knn"] < 50 and not s.loc[1, "ood_exceeds_knn"]
    assert s.loc[1, "ood_score_knn"] >= 0 and s.loc[0, "ood_exceeds_rms"] and s.loc[0, "ood_exceeds_maha"]
    assert s.loc[0, "ood_nearest_ref"].count("|") == 2 and "(ordinary)" in s.loc[0, "ood_nearest_ref"]
    assert s.loc[0, "ood_top_features"].count("|") == 2 and "+z=" in s.loc[0, "ood_top_features"]
    assert s.loc[0, "ood_pct_knn_matched_median"] == 100
    # rank p can never be below the floor 1/(n_ref + 1)
    assert (s["ood_rank_p_knn"] >= 1 / 18 - 1e-12).all()


def test_ood_reference_loo_percentiles_roughly_uniform_under_exchangeability():
    rng = np.random.default_rng(11)
    pcts = []
    for sim in range(20):
        ref = _synthetic(rng, n_per=(0, 0, 17), shift=0.0, prefix=f"r{sim}"); ref["batch"] = "B2"
        om = cat.fit_ood(ref, features=FEATS, k=3)
        q = _synthetic(rng, n_per=(0, 0, 1), shift=0.0, prefix=f"q{sim}")
        pcts.append(cat.ood_score(om, q)["ood_pct_knn"].iloc[0])
    pcts = np.array(pcts)
    assert 0.1 < np.mean(pcts > 50) < 0.9 and np.mean(pcts == 100) < 0.5


def test_ood_zero_mad_fallback_and_drop():
    ref = _synthetic(np.random.default_rng(4), n_per=(0, 0, 17)); ref["batch"] = "B2"
    ref["f7"] = 0.0                                              # constant -> dropped
    ref["f6"] = [1.0] * 9 + [2, 3, 4, 5, 6, 7, 8, 9]              # 9/17 at the median -> MAD 0; q25 = 1, q75 = 5 -> IQR fallback
    om = cat.fit_ood(ref, features=FEATS, k=3)
    assert om.dropped == ["f7"] and om.keep.sum() == 7
    q = ref.iloc[[0]].copy(); q["site"] = "q"
    assert np.isfinite(cat.ood_score(om, q)["ood_score_knn"].iloc[0])


# ---------------------------------------------------------------- hand-off ------------------------------------------
def test_categorise_site_and_sites_tables():
    rng = np.random.default_rng(6)
    df = _synthetic(rng, shift=2.0)
    for c in cat.FLAG_COLS:
        if c not in df.columns:
            df[c] = False
    fams = {"a": FEATS[:4], "b": FEATS}
    models = {k: cat.fit_categoriser(df, features=v, ref_batch="B2") for k, v in fams.items()}
    oods = {"v1": cat.fit_ood(df[df.batch == "B2"], features=FEATS[:4]), "v2": cat.fit_ood(df[df.batch == "B2"], features=FEATS)}
    one = cat.categorise_site(df.iloc[0], models["b"], oods)
    assert {"mp_B0", "mp_B1", "mp_B2", "argmax", "cat_top_features", "qc_pointer", "probability_note"} <= set(one)
    assert "ood_pct_knn__v1" in one and "ood_pct_knn__v2" in one and "model probabilities" in one["probability_note"]
    loo = {"a": cat.loo_evaluate(df, features=FEATS[:4], n_perm=0, n_jobs=1)}
    tab = cat.categorise_sites(df, models, oods, role="known", loo_results=loo)
    assert len(tab) == len(df) and (tab["source__a"] == "loo").all() and (tab["source__b"] == "full_fit").all()
    ref_rows = tab[tab.batch == "B2"]
    assert (~ref_rows["ood_exceeds_knn__v1"].astype(bool)).all() and ref_rows["ood_nearest_ref__v1"].str.contains("leave-one-out").all()
    assert ref_rows["ood_top_features__v1"].str.len().gt(0).all()
    new = df.iloc[[1]].copy(); new["site"] = "new"; new["batch"] = "X"
    tab2 = cat.categorise_sites(new, models, oods, role="scored")
    assert tab2["role"].iloc[0] == "scored" and (tab2["source__a"] == "full_fit").all()
    # summaries and the markdown / html renderers run on a synthetic result
    summ = cat.summary_tables({"a": loo["a"]})
    assert summ.loc[0, "n_sites"] == 28 and np.isnan(summ.loc[0, "perm_p"])
    kw = cat.kruskal_context(df, FEATS)
    assert len(kw) == 8 and kw["p"].min() < 0.05
    oods = cat.ood_batch_summary(tab, variants=("v1", "v2"), ref_batch="B2")
    assert len(oods) == 4 and set(oods.batch) == {"B0", "B1"}
    md = cat.render_markdown({"a": loo["a"]}, summ, oods, tab, "B2")
    html = cat.render_html({"a": loo["a"]}, summ, oods, tab, "B2")
    assert "model probabilities" in md.lower() and "not posteriors" in html and cat.QC_POINTER in md


# ---------------------------------------------------------------- real cache (guarded) ------------------------------
@pytest.mark.skipif(not (os.path.isdir(os.path.join(DATASET, "Batch_3")) and os.path.isdir(FEATURE_CACHE)
                         and any(f.startswith("Batch_3_") and f.endswith(".sites.parquet") for f in os.listdir(FEATURE_CACHE))),
                    reason="no dataset / feature cache")
def test_real_cache_loo_smoke():
    known = cat.load_known([os.path.join(DATASET, b) for b in ("Batch_3", "Batch_1", "Batch_2")], FEATURE_CACHE)
    assert len(known) == 31 and known.groupby("batch").size().to_dict() == {"Batch_1": 7, "Batch_2": 7, "Batch_3": 17}
    assert set(cat.FAMILIES["combined"]) <= set(known.columns) and "acquisition_group" in known.columns
    assert sorted(known.loc[known.acquisition_group == "grey_pore", "site"]) == ["71vgq3fw", "kbdh4tri", "tuy3zymq", "x7u69zsw"]
    r = cat.loo_evaluate(known, "morph", n_perm=0, n_jobs=1)
    assert r["metrics"]["confusion"].to_numpy().sum() == 31 and len(r["features"]) == 19
    ref = known[known.batch == "Batch_3"]
    om = cat.fit_ood(ref, "morph")
    assert om.n_ref == 17 and (om.ref_loo["score_knn"] > 0).all()
    s = cat.ood_score(om, known[known.batch == "Batch_1"])
    assert len(s) == 7 and (s["ood_rank_p_knn"] >= 1 / 18 - 1e-12).all()
