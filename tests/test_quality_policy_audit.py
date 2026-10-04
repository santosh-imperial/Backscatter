"""Regression checks for the experimental quality gate, independent of raw images."""
import numpy as np
import pandas as pd
import pytest

from analysis.quality_policy_audit import policy as qp


def frame(n=15):
    rng = np.random.default_rng(13)
    rows = pd.DataFrame(rng.normal(size=(n, len(qp.FEATURES))), columns=qp.FEATURES)
    rows["site"] = [f"site{i}" for i in range(n)]
    rows["batch"] = [f"Batch_{1 + i % 3}" for i in range(n)]
    rows["bright_low_contrast"] = False
    rows["bse_p1"] = 0.
    return rows


def test_bright_failure_propagates_to_all_cross_detector_dependencies():
    df = frame(3); df.loc[0, "bright_low_contrast"] = True
    X, names, audit, _ = qp.prepare(df)
    bad = set(qp.BRIGHT + qp.PARTICLE_INLENS + qp.JOINT)
    assert len(bad) == 19
    assert set(np.asarray(names)[np.isnan(X[0])]) == bad
    assert np.isfinite(X[0, [names.index(k) for k in qp.PORE + qp.ANCHORS]]).all()
    assert (audit.query("site == 'site0' and not observed_model_input").reason == "bright_low_contrast").all()
    assert df.loc[0, qp.FEATURES].notna().all(), "Preserve raw measurements"


def test_pore_failure_propagates_through_joint_orientation_but_retains_bright_texture():
    df = frame(3); df.loc[0, "bse_p1"] = 23
    X, names, _, _ = qp.prepare(df)
    assert set(np.asarray(names)[np.isnan(X[0])]) == set(qp.PORE + qp.JOINT)
    assert np.isfinite(X[0, [names.index(k) for k in qp.BRIGHT + qp.PARTICLE_INLENS]]).all()


def test_unknown_quality_is_not_good_and_false_strings_are_not_truthy():
    df = frame(3); df["bright_low_contrast"] = ["False", "True", None]
    X, _, _, q = qp.prepare(df)
    assert q.bright_bad.tolist() == [False, True, True]
    assert np.isfinite(X[0]).all()
    missing = df.drop(columns=["bright_low_contrast", "bse_p1"])
    X, names, _, q = qp.prepare(missing)
    assert np.isfinite(X).sum(axis=1).tolist() == [2, 2, 2]
    assert set(np.asarray(names)[np.isfinite(X[0])]) == set(qp.ANCHORS)
    assert q.pore_reason.eq("pore_quality_unknown").all()


def test_flags_are_label_and_site_independent_and_no_acquisition_numbers_enter_matrix():
    df = frame(3); df.loc[0, "bse_p1"] = 23
    df["grey_pore"] = False; df["H"] = [2080, 2080, 2080]
    a, names, _, _ = qp.prepare(df)
    copy = df.copy(); copy["batch"] = "unknown"; copy["site"] = ["x", "y", "z"]
    copy["grey_pore"] = True; copy["H"] = 99999
    b, other, _, _ = qp.prepare(copy)
    np.testing.assert_array_equal(a, b)
    assert names == other == qp.FEATURES
    assert not set(names) & {"H", "bse_p1", "bright_low_contrast", "grey_pore"}


def test_rejected_value_perturbations_cannot_affect_fits_scores_or_order():
    df = frame(); df.loc[0:2, "bright_low_contrast"] = True
    changed = df.copy()
    bad = qp.BRIGHT + qp.PARTICLE_INLENS + qp.JOINT
    changed.loc[0:2, bad] = 1e9
    np.testing.assert_array_equal(qp.shared_order(df), qp.shared_order(changed))
    a = qp.fit_model(df, "dependency_gate"); b = qp.fit_model(changed, "dependency_gate")
    np.testing.assert_array_equal(a.pipeline["lr"].coef_, b.pipeline["lr"].coef_)
    np.testing.assert_array_equal(a.predict(df)[0], b.predict(changed)[0])


def test_held_out_extremes_do_not_enter_training_imputer():
    df = frame(); df.loc[0, "bright_low_contrast"] = True
    X, _, _, _ = qp.prepare(df)
    train = X[1:]
    pipe = qp.pipeline(.5).fit(train, np.array([i % 3 for i in range(14)]))
    medians = np.nanmedian(train, axis=0)
    np.testing.assert_allclose(pipe["impute"].statistics_, medians)
    old = pipe["impute"].statistics_.copy()
    heldout = X[:1].copy(); heldout[:, qp.FEATURES.index("pore_frac")] = 1e9
    qp.proba(pipe, heldout)
    np.testing.assert_array_equal(pipe["impute"].statistics_, old)


def test_all_empty_training_features_keep_positions_and_have_no_learned_weight():
    df = frame(); df["bright_low_contrast"] = True
    model = qp.fit_model(df, "dependency_gate")
    assert model.pipeline["lr"].coef_.shape[1] == 29
    for name in qp.BRIGHT + qp.PARTICLE_INLENS + qp.JOINT:
        j = qp.FEATURES.index(name)
        assert model.train_observed_counts[j] == 0
        assert not model.pipeline["lr"].coef_[:, j].any()
    assert np.isfinite(model.predict(df)[0]).all()


def test_row_and_site_renaming_invariance():
    df = frame(); df.loc[0, "bright_low_contrast"] = True
    a = qp.fit_model(df, "dependency_gate")
    changed = df.iloc[::-1].reset_index(drop=True); changed["site"] = [f"new{i}" for i in range(len(df))]
    b = qp.fit_model(changed, "dependency_gate")
    np.testing.assert_array_equal(a.predict(df)[0], b.predict(df)[0])


def test_invalid_flags_and_duplicate_order_anchors_fail_explicitly():
    df = frame(3); df.loc[0, "bright_low_contrast"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported quality"):
        qp.prepare(df)
    df = frame(3); df.loc[1, qp.ANCHORS] = df.loc[0, qp.ANCHORS].values
    with pytest.raises(ValueError, match="unique anchor"):
        qp.shared_order(df)


def test_sensitivity_exposes_indirect_mask_dependence_and_nonfinite_values_stay_missing():
    df = frame(3); df.loc[0, "bright_low_contrast"] = True
    direct, names, _, _ = qp.prepare(df, "direct_mask_gate")
    assert np.isfinite(direct[0, names.index("inlens_grad_energy")])
    assert np.isnan(direct[0, names.index("etd_crack_density_particles")])
    df.loc[1, "pore_d50"] = np.inf
    X, _, audit, _ = qp.prepare(df)
    assert np.isnan(X[1, names.index("pore_d50")])
    assert audit.query("site == 'site1' and feature == 'pore_d50'").reason.iloc[0] == "measurement_unavailable"
