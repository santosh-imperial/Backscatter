import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from analysis.morphology.l_audit import methods as m
from analysis.morphology.k_pilot.descriptors import phase_summary
from analysis.morphology.l_audit.measurement import component_table,shape

def test_energy_manual_allocation_refits():
    z=np.array([[0.,1.],[1.,2.],[2.,3.],[9.,0.],[11.,7.]])
    orders=np.array([[0,1,2,3,4],[0,3,4,1,2]])
    actual=m.statistics(z,3,orders,{'x':[0,1]})['x']
    expected=[]
    for o in orders:
        a=z[o];med,scale,_=m.scale_fit(a[:3]);s=(a-med)/scale/np.sqrt(2);d=cdist(s,s)
        expected.append(2*d[:3,3:].mean()-d[:3,:3].mean()-d[3:,3:].mean())
    np.testing.assert_allclose(actual,expected)
    assert actual[0]!=actual[1]

def test_energy_affine_and_row_invariance():
    rng=np.random.default_rng(42);r=rng.normal(size=(6,8));q=rng.normal(size=(5,8))
    a=m.energy_tests(r,q,panels={'x':list(range(8))})[0][0]
    b=m.energy_tests(r[::-1]*3+17,q[::-1]*3+17,panels={'x':list(range(8))})[0][0]
    assert a['p']==b['p'];assert a['statistic']==pytest.approx(b['statistic'])

def test_seeded_mc_ignores_site_order():
    rng=np.random.default_rng(4);r=rng.normal(size=(8,8));q=rng.normal(size=(5,8))
    a=m.energy_tests(r,q,n_mc=99,exact_limit=1)[1]
    b=m.energy_tests(r[[3,7,1,2,0,6,4,5]],q[::-1],n_mc=99,exact_limit=1)[1]
    assert a==b

def test_exact_includes_observed_ties_and_mc_plus_one():
    orders,method,n=m.allocation_orders(7,4)
    assert len(orders)==35 and n==35 and method=='exact'
    assert np.unique(orders[:,4:],axis=0).shape[0]==35
    assert m.tail_p(np.array([1,2,2,3]),2,'exact')==.75
    assert m.tail_p(np.array([0,0,0]),1,'monte_carlo')==.25
    np.testing.assert_allclose(m.exact_ranks([1.,2.,2.,3.]),[1.,.75,.75,.25])

def test_holm_union_controls_two_opportunities():
    np.testing.assert_allclose(m.holm([.04,.04]),[.08,.08])
    p={'primary5':.04,'morphology3':.04,'extended8':.6}
    assert m.policies(p)['naive_or'] and not m.policies(p)['holm_union']
    assert m.policies(dict(p,primary5=.02))['holm_union']

def test_quality_missing_is_not_good():
    d=pd.DataFrame({'bright_low_contrast':[False,None,False,False],
                    'grey_pore':[False,False,True,None],m.CAND[0]:[2]*4,
                    'bright_n_interior':[30]*4,'bright_n_oriented':[30]*4})
    assert m.eligible(d,[m.CAND[0]]).tolist()==[True,False,True,True]
    assert m.eligible(d,[m.CAND[0]],shared=True).tolist()==[True,False,False,False]
    d.loc[0,m.CAND[0]]=np.inf;assert not m.eligible(d,[m.CAND[0]])[0]

def test_subminimum_fold_has_no_alert_status():
    rng=np.random.default_rng(1)
    rows,_,_=m.energy_tests(rng.normal(size=(6,8)),rng.normal(size=(4,8)),panels={'x':[0]})
    assert np.isfinite(rows[0]['p']) and rows[0]['alert'] is None and not rows[0]['alert_available']

def test_fallback_disclosed_and_missing_rejected():
    _,s,f=m.scale_fit(np.array([[0,2],[0,2],[0,2],[1,2]],float))
    assert f.tolist()==['sd','unit'];assert s[1]==1
    with pytest.raises(ValueError):m.canonical([[1,np.nan]])

def test_nearest_feature_contributions_are_distances_not_probabilities():
    r=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
    score,near,c,_=m.neighbour_score(r,np.array([[0.,0.],[8.,8.]]))
    assert near.tolist()==[0,3] and score[1]>score[0]
    med,s,_=m.scale_fit(r)
    np.testing.assert_allclose(c.sum(axis=1),np.sum(((np.array([[0,0],[8,8]])-r[near])/s)**2,axis=1)/2)

def test_binary_phase_axis_interchange_negates():
    rng=np.random.default_rng(3);a=rng.random((600,700))<.2;b=np.roll(a,256,axis=1)
    k=m.CAND[2];v,_=phase_summary(a,b);t,_=phase_summary(a.T.copy(),b.T.copy())
    assert t[k]==pytest.approx(-v[k],abs=1e-14)

def test_shape_strength_transpose_invariant_and_clips_excluded():
    a=np.zeros((190,230),bool)
    for y in range(10,170,20):
        for x in range(10,210,25):a[y:y+5,x:x+16]=1
    a[:5,:20]=1
    s=shape(component_table(a,50));t=shape(component_table(a.T.copy(),50))
    assert s['bright_n_interior']==64
    assert s['bright_aspect_aw']==pytest.approx(t['bright_aspect_aw'])
    assert s['bright_alignment_strength']==pytest.approx(t['bright_alignment_strength'])
