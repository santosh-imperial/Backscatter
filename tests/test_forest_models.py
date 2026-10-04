"""E41S regression contracts: same folds, rejected inputs, exact explanations."""
import numpy as np
import pandas as pd
import pytest

from analysis.forest_geometry import models as m, geometry as g
from analysis.classification_m1_m2 import models as old
from analysis.quality_policy_audit import policy as qp


def frame():
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.normal(size=(18, 37)), columns=qp.FEATURES + g.FEATURES)
    df["batch"] = np.repeat(m.CLASSES, 6)
    df["site"] = [str(i) for i in range(18)]
    df["bright_low_contrast"] = False
    df["bse_p1"] = 0
    return df


def test_rejected_geometry_and_old_inputs_cannot_influence_measurement_model():
    df = frame()
    df.loc[0, "bright_low_contrast"] = True
    df.loc[1, "bse_p1"] = 30
    original, names = m.matrix(df, "geometry")
    changed = df.copy()
    changed.loc[0, qp.BRIGHT + qp.JOINT + qp.PARTICLE_INLENS + g.BRIGHT + g.JOINT] = 1e12
    changed.loc[1, qp.PORE + qp.JOINT + g.PORE + g.JOINT] = -1e12
    second, _ = m.matrix(changed, "geometry")
    np.testing.assert_equal(original, second)
    assert len(names) == 37
    availability, availability_names = m.matrix(df, "availability")
    np.testing.assert_equal(availability, np.isfinite(original).astype(float))
    assert all(n.startswith("available__") for n in availability_names)
    flags, _ = m.matrix(df, "quality")
    assert flags.tolist()[:2] == [[1., 0.], [0., 1.]]


def test_forest_fold_imputation_and_empty_positions():
    X = np.array([[1., np.nan], [2., np.nan], [3., np.nan], [4., np.nan], [5., np.nan], [6., np.nan]])
    fitted = m.pipeline("rf").fit(X, np.array([0, 1, 2, 0, 1, 2]))
    assert fitted["impute"].statistics_.tolist() == [3.5, 0.]
    assert fitted["classifier"].n_features_in_ == 2
    assert fitted["classifier"].get_params()["max_depth"] == 3
    assert fitted["classifier"].get_params()["min_samples_leaf"] == 4
    np.testing.assert_allclose(m.probabilities(fitted, [[999., np.nan]]).sum(axis=1), 1.)


def test_exact_forest_margin_marks_imputed_terms_and_float32_paths():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 4))
    X[::4, 2] = np.nan
    model = m.pipeline("rf").fit(X, np.tile([0, 1, 2], 10))
    query = X[0:1].copy()
    for a, b in [(0, 2), (1, 0), (2, 1)]:
        rows, root, margin = m.forest_contributions(model, query, list("abcd"), a, b)
        assert np.isclose(sum(r["contribution"] for r in rows) + root, margin, atol=1e-12)
        assert not next(r for r in rows if r["feature"] == "c")["observed"]
    with pytest.raises(ValueError):
        m.forest_contributions(model, X[:2], list("abcd"), 0, 2)


def test_nominal_unavailability_and_replacement_are_preserved():
    X = np.array([[1., np.nan, 5.], [2., 3., np.nan]])
    perturbed = m.preserve_nominal_availability(X, np.full_like(X, 100.))
    np.testing.assert_equal(np.isfinite(perturbed), np.isfinite(X))
    train = np.array([[1., np.nan, 5.], [3., np.nan, 7.], [4., np.nan, 9.],
                      [2., np.nan, 3.], [5., np.nan, 8.], [6., np.nan, 10.]])
    model = m.pipeline("rf").fit(train, np.tile([0, 1, 2], 2))
    out, n = m.replace_observed_group(model, X[:1], list("abc"), list("abc"))
    assert n == 2 and np.isnan(out[0, 1])
    assert out[0, 0] == 3.5 and out[0, 2] == 7.5


def test_matched_l1_selection_and_content_order_ignore_site_ids():
    df = frame()
    X, _ = m.matrix(df, "base")
    y = df.batch.map(dict(zip(m.CLASSES, range(3)))).to_numpy(int)
    assert m.select_C(X, y, "l1") == old.select_C(X, y, "l1")
    assert m.select_C(X, y, "rf") is None
    renamed = df.sample(frac=1, random_state=9).reset_index(drop=True)
    renamed["site"] = ["renamed" + str(i) for i in range(len(df))]
    canonical = df.iloc[qp.shared_order(df)][qp.ANCHORS].to_numpy()
    np.testing.assert_equal(canonical, renamed.iloc[qp.shared_order(renamed)][qp.ANCHORS].to_numpy())
    with pytest.raises(ValueError):
        m.select_C(X[y != 2], y[y != 2], "rf")


def test_groups_partition_fixed37():
    groups = sum(m.GROUPS.values(), [])
    assert len(groups) == 37 and len(set(groups)) == 37
    assert set(groups) == set(qp.FEATURES + g.FEATURES)


def test_join_rejects_duplicates_missing_sites_labels_and_flag_disagreement():
    from analysis.forest_geometry.run import join_geometry
    base = frame().drop(columns=g.FEATURES)
    extra = frame()[["batch", "site", "bse_p1", "bright_low_contrast"] + g.FEATURES]
    joined = join_geometry(base, extra.sample(frac=1, random_state=3))
    np.testing.assert_equal(joined[g.FEATURES].to_numpy(), frame()[g.FEATURES].to_numpy())
    assert joined[qp.ANCHORS].equals(base[qp.ANCHORS])
    with pytest.raises(ValueError):
        join_geometry(base, pd.concat([extra, extra.iloc[:1]]))
    with pytest.raises(ValueError):
        join_geometry(base, extra.iloc[:-1])
    different = extra.copy()
    different.loc[0, "batch"] = "Batch_3"
    with pytest.raises(ValueError):
        join_geometry(base, different)
    different = extra.copy()
    different.loc[0, "bse_p1"] = 20
    with pytest.raises(ValueError):
        join_geometry(base, different)
