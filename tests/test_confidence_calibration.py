import numpy as np
import pytest
from analysis.confidence_calibration.calibration import transform, fit_scalar, promotion, LOSS_NAMES


def test_temperature_identity_preserves_normalized_ovr_not_logit_softmax():
    # OvR normalization is not softmax of the classifier's three raw logits.
    logits = np.array([[2., 1., -1.]])
    p = 1/(1+np.exp(-logits)); p /= p.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(transform(p, 1.), p)
    softmax_logits = np.exp(logits)/np.exp(logits).sum(axis=1,keepdims=True)
    assert not np.allclose(p, softmax_logits)


def test_positive_temperature_preserves_ranking_and_normalization():
    rng = np.random.default_rng(43)
    p = rng.dirichlet(np.ones(3), size=100)
    for alpha in [.05, .5, 1, 2, 20]:
        q = transform(p, alpha)
        np.testing.assert_array_equal(np.argsort(p,axis=1), np.argsort(q,axis=1))
        np.testing.assert_allclose(q.sum(axis=1), 1)
    assert np.all(transform(p, .5).max(axis=1) <= p.max(axis=1))


def test_scalar_fit_uses_only_supplied_training_targets():
    p = np.tile([.8,.1,.1], (30,1)); y = np.tile([0,1,2], 10)
    fitted = fit_scalar(y,p)
    assert fitted['alpha'] == .05
    assert fitted['training_balanced_nll'] < fitted['identity_training_balanced_nll']
    # When all scores are identical uniform vectors, identity wins the flat tie.
    assert fit_scalar(y,np.full((30,3),1/3))['alpha'] == 1.
    with pytest.raises(ValueError): fit_scalar(np.zeros(30,dtype=int),p)


def test_zero_probability_keeps_identity_and_forced_bet():
    p=np.array([[1.,0,0],[.7,.3,0]])
    for alpha in [.05,.5,1,2,20]:
        q=transform(p,alpha)
        np.testing.assert_array_equal(p.argmax(1),q.argmax(1))
        np.testing.assert_allclose(q.sum(1),1)
        assert np.all(q[:,2]==0)
    np.testing.assert_array_equal(transform(p,1),p)


@pytest.mark.parametrize('alpha',[0,-1,np.nan,np.inf])
def test_invalid_scalar_rejected(alpha):
    with pytest.raises(ValueError): transform([[.5,.3,.2]],alpha)


def test_promotion_requires_both_confidence_targets_and_no_other_loss_regression():
    original = {m:1. for m in LOSS_NAMES}; improved = {m:.98 for m in LOSS_NAMES}
    assert promotion(original,improved)['loss_gate_passed']
    improved['bet_nll'] = 1.01
    assert not promotion(original,improved)['loss_gate_passed']
