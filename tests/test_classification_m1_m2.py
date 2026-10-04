import numpy as np
import pandas as pd
import pytest

from analysis.classification_m1_m2 import models as m
from analysis.quality_policy_audit import policy as qp


def fixture():
    rng = np.random.default_rng(390)
    n = 12
    frame = pd.DataFrame(rng.normal(size=(n,len(qp.FEATURES))), columns=qp.FEATURES)
    frame["site"] = [f"s{i}" for i in range(n)]
    frame["batch"] = np.repeat(m.CLASSES, 4)
    frame["bright_low_contrast"] = [True]+[False]*11
    frame["bse_p1"] = [0]*11+[20]
    frame["appearance_test"] = rng.normal(size=n)
    return frame


@pytest.mark.parametrize("name", ["gated_l1","gated_l2","gated_lda","augmented_l2"])
def test_rejected_values_cannot_change_model(name):
    frame = fixture(); changed = frame.copy()
    changed.loc[0, qp.BRIGHT+qp.PARTICLE_INLENS+qp.JOINT] = 1e12
    changed.loc[11, qp.PORE+qp.JOINT] = -1e12
    a = m.fit(frame,name,["appearance_test"])
    b = m.fit(changed,name,["appearance_test"])
    np.testing.assert_array_equal(a.predict(frame),b.predict(changed))


def test_row_and_site_name_invariance():
    frame=fixture(); changed=frame.sample(frac=1,random_state=8).copy()
    changed["site"] = "renamed"
    a=m.fit(frame,"gated_l2"); b=m.fit(changed,"gated_l2")
    np.testing.assert_array_equal(a.predict(frame),b.predict(frame))
    for f in (frame,changed):
        f.sort_values(qp.ANCHORS, inplace=True)
    X,_=m.matrix(frame,"gated"); Y,_=m.matrix(changed,"gated")
    y=frame.batch.map({c:i for i,c in enumerate(m.CLASSES)}).to_numpy()
    np.testing.assert_array_equal(m.nested_loo(X,y,"l2")[0],m.nested_loo(Y,y,"l2")[0])


def test_held_out_value_not_in_training_statistics_or_selection():
    X=np.arange(24,dtype=float).reshape(12,2); X[1,0]=np.nan
    y=np.repeat(np.arange(3),4)
    changed=X.copy(); changed[0]=1e8
    _,C,A=m.nested_loo(X,y,"l2")
    _,D,B=m.nested_loo(changed,y,"l2")
    assert C[0]==D[0]
    np.testing.assert_array_equal(A[0]["impute"].statistics_,B[0]["impute"].statistics_)
    np.testing.assert_array_equal(A[0]["scale"].mean_,B[0]["scale"].mean_)


@pytest.mark.parametrize("kind",["l1","l2","lda"])
def test_missing_columns_and_explicit_class_scores(kind):
    X=np.column_stack([np.arange(12),np.full(12,np.nan),np.ones(12)])
    y=np.tile([2,0,1],4)
    model=m.pipeline(kind).fit(X,y)
    P=m.probabilities(model,X)
    assert P.shape==(12,3)
    assert model["scale"].n_features_in_==3
    np.testing.assert_allclose(P.sum(axis=1),1)
    np.testing.assert_array_equal(P,model.predict_proba(X))


def test_margin_sum_and_missing_class_fail():
    frame=fixture(); model=m.fit(frame,"gated_l2")
    X,names=m.matrix(frame,"gated")
    rows,intercept,margin=m.margin_contributions(model.pipeline,X[:1],names,0,2)
    assert np.isclose(sum(r["contribution"] for r in rows)+intercept,margin)
    with pytest.raises(ValueError, match="lacks"):
        m.select_C(X[:4],np.zeros(4,dtype=int),"l2")


def test_no_metadata_in_new_descriptor_panel():
    frame=fixture()
    with pytest.raises(ValueError,match="metadata"):
        m.matrix(frame,"appearance",["H"])
