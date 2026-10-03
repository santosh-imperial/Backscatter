import numpy as np
import pandas as pd
import pytest
from analysis.morphology.k_pilot.descriptors import (SHAPE_KEYS,KEYS,shape_summary,pair_counts,phase_summary,shifted_pairs,usable)

def test_pair_formula_matches_pearson_and_crops_without_wrap():
    rng=np.random.default_rng(21);a=rng.random((200,250))<.3;b=rng.random((200,250))<.4
    for axis in (0,1):
        for lag in (-64,64):
            r=pair_counts(a,b,lag,axis);x,y=shifted_pairs(a,b,lag,axis)
            assert r['pair_pixels']==(a.shape[axis]-64)*a.shape[1-axis]
            np.testing.assert_allclose(r['rho'],np.corrcoef(x.ravel().astype(float),y.ravel().astype(float))[0,1],atol=1e-14)

def test_axis_transpose_reverses_both_directional_contrasts():
    rng=np.random.default_rng(8);a=rng.random((600,700))<.3;b=rng.random((600,700))<.25
    m,_=phase_summary(a,b);t,_=phase_summary(a.T,b.T)
    for k in KEYS[3:]:np.testing.assert_allclose(m[k],-t[k],atol=1e-14)

def test_cross_symmetry_under_phase_exchange_and_reflection():
    rng=np.random.default_rng(6);a=rng.random((600,700))<.3;b=rng.random((600,700))<.15
    m,_=phase_summary(a,b);t,_=phase_summary(b,a);f,_=phase_summary(a[:,::-1],b[:,::-1])
    for k in KEYS[5:]:
        np.testing.assert_allclose(m[k],t[k],atol=1e-14)
        np.testing.assert_allclose(m[k],f[k],atol=1e-14)

def test_horizontal_stripes_have_positive_auto_xy_contrast():
    a=np.indices((700,700))[0]%200<70;p=~a
    m,_=phase_summary(a,p)
    assert m[KEYS[3]]>0

def test_empty_constant_and_small_domains_abstain():
    for a in (np.zeros((500,500),bool),np.ones((500,500),bool),np.zeros((20,20),bool)):
        m,_=phase_summary(a,~a)
        assert all(np.isnan(m[k]) for k in KEYS[3:])

def test_shape_count_weighting_linear_quantiles_and_exclusions():
    x=np.arange(1,22,dtype=float)
    t=pd.DataFrame(dict(area=np.ones(21)*100,major_axis_length=x,minor_axis_length=np.ones(21),
                        solidity=x/22,perimeter=np.ones(21)*40,touches_edge=[False]*20+[True]))
    m,eligible=shape_summary(t);assert len(eligible)==20
    assert m[SHAPE_KEYS[0]]==9.5
    np.testing.assert_allclose(m[SHAPE_KEYS[1]],np.quantile(x[:20]/22,.1))
    assert m[SHAPE_KEYS[2]]==0
    # Count-weighted quantiles do not turn a large component into many objects.
    t.loc[0,'area']=100000
    m2,_=shape_summary(t)
    assert m2[SHAPE_KEYS[0]]==m[SHAPE_KEYS[0]] and m2[SHAPE_KEYS[1]]==m[SHAPE_KEYS[1]]
    assert np.isnan(shape_summary(t.iloc[:19])[0][SHAPE_KEYS[0]])

def test_quality_contract_is_phase_specific_and_missing_abstains():
    t=pd.DataFrame({'batch':['Batch_1','Batch_3'],'bright_low_contrast':[False,False],
                    'grey_pore':[True,False],'cracked_known':[False,True],**{k:[.1,.2] for k in KEYS}})
    assert usable(t,SHAPE_KEYS[0]).tolist()==[True,True]
    assert usable(t,KEYS[5]).tolist()==[False,True]
    assert usable(t,SHAPE_KEYS[0],'ordinary_reference').tolist()==[False,False]
    assert not usable(t.drop(columns='bright_low_contrast'),SHAPE_KEYS[0]).any()
    assert not usable(t.drop(columns='grey_pore'),KEYS[5]).any()

@pytest.mark.parametrize('kind',['float','mismatched','invalid_lag','unknown_clip'])
def test_invalid_inputs_fail_explicitly(kind):
    a=np.zeros((200,200),bool)
    with pytest.raises(ValueError):
        if kind=='float':shifted_pairs(a.astype(float),a,64,1)
        elif kind=='mismatched':shifted_pairs(a,a[:100],64,1)
        elif kind=='invalid_lag':shifted_pairs(a,a,0,1)
        else:shape_summary(pd.DataFrame(dict(area=[100],major_axis_length=[3],minor_axis_length=[2],solidity=[.8],perimeter=[20],touches_edge=[None])))
