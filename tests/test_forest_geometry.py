"""E41S geometry contract checks: quality rejection, coverage and cardinality."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from analysis.forest_geometry import geometry as g
from analysis.forest_geometry.prepare import _one, known_table, require_known_stage


def frame(bright=False, p1=0):
    return pd.DataFrame([dict(site='synthetic', batch='Batch_1', bright_low_contrast=bright,
        bse_p1=p1, **{k:float(i+1) for i,k in enumerate(g.FEATURES)},
        **{field:float(minimum) for req in g.COVERAGE.values() for field,minimum in req})])


def test_rejected_values_do_not_enter_fixed_geometry_inputs():
    a=frame(bright=True,p1=0)
    b=a.copy()
    b[g.BRIGHT+g.JOINT]=1e100
    x,names,audit,q=g.prepare(a)
    xx,*_=g.prepare(b)
    np.testing.assert_allclose(x,xx,equal_nan=True)
    assert names==g.FEATURES
    assert set(audit.loc[~audit.observed_model_input,'feature'])==set(g.BRIGHT+g.JOINT)
    assert np.isfinite(x[0,[names.index(k) for k in g.PORE]]).all()
    assert all('bright_low_contrast' in r for r in audit.loc[~audit.observed_model_input,'reason'])


def test_pore_rejection_covers_widths_and_joint_not_bright_only():
    a=frame(bright=False,p1=11)
    b=a.copy();b[g.PORE+g.JOINT]=-1e99
    x,names,audit,q=g.prepare(a);xx,*_=g.prepare(b)
    np.testing.assert_allclose(x,xx,equal_nan=True)
    assert np.isfinite(x[0,[names.index(k) for k in g.BRIGHT]]).all()
    assert np.isnan(x[0,[names.index(k) for k in g.PORE+g.JOINT]]).all()
    assert all('raised_black_level' in r for r in audit.loc[~audit.observed_model_input,'reason'])


def test_unknown_quality_fails_closed_and_malformed_flags_raise():
    a=frame().drop(columns=['bright_low_contrast','bse_p1'])
    x,names,audit,q=g.prepare(a)
    assert np.isnan(x).all()
    assert audit.policy_eligible.eq(False).all()
    bad=frame(bright='maybe')
    with pytest.raises(ValueError,match='Unsupported quality flag'):g.prepare(bad)
    # Exact string bool parsing, not truthiness of the nonempty string 'False'.
    x,*_=g.prepare(frame(bright='False'))
    assert np.isfinite(x).all()


def test_measurement_coverage_unavailable_is_nan_and_explicit():
    a=frame()
    k=g.FEATURES[0];a[k]=np.nan;a[k+'_unavailable_reason']='fewer_than_20_eligible_components'
    x,names,audit,q=g.prepare(a)
    assert np.isnan(x[0,names.index(k)])
    r=audit.loc[audit.feature.eq(k)].iloc[0]
    assert r.policy_eligible and not r.observed_model_input
    assert r.reason=='fewer_than_20_eligible_components'
    a[g.FEATURES[1]]=np.inf
    assert np.isnan(g.prepare(a)[0][0,1])


def test_coverage_gate_is_explicit_and_does_not_accept_finite_unsupported_values():
    a=frame();a['shape_objects_eligible']=19
    x,names,audit,q=g.prepare(a)
    assert np.isnan(x[0,:3]).all()
    assert audit.loc[audit.feature.isin(g.FEATURES[:3]),'coverage_eligible'].eq(False).all()
    a=frame().drop(columns=['crosscorr_min_target_complement_pixels'])
    x,names,audit,q=g.prepare(a)
    assert np.isnan(x[0,-1]) and 'coverage_unknown' in audit.iloc[-1].reason
    a=frame();a['crosscorr_min_source_phase_pixels']=99
    assert np.isnan(g.prepare(a)[0][0,-1])


def test_site_and_variant_joins_reject_ambiguity():
    a=frame();a['variant']='nominal'
    b=a.copy();b['variant']='floor100'
    v=pd.concat([a,b],ignore_index=True)
    assert len(g.nominal(v))==1
    with pytest.raises(ValueError,match='Duplicate'):g.prepare(v)
    with pytest.raises(ValueError,match='Duplicate'):g.nominal(pd.concat([v,a],ignore_index=True))
    with pytest.raises(ValueError,match='exactly one row'):_one(v,'test',site='synthetic')
    with pytest.raises(ValueError,match='exactly one row'):_one(v,'test',site='absent')


def test_true_zero_window_spread_vs_absent_or_partial_coverage():
    mask=np.zeros((1024,1024),bool)
    val,d,_=g.local_bright_spread(mask)
    assert np.isnan(val) and d['local_bright_std_512px_unavailable_reason']=='no_bright_pixels_in_complete_windows'
    # Same observed bright fraction in all four full windows is a true zero.
    mask[0:64,:]=True;mask[512:576,:]=True
    val,d,windows=g.local_bright_spread(mask)
    assert val==0 and d['local_bright_std_512px_unavailable_reason']==''
    assert windows.shape==(2,2) and np.all(windows==.125)
    val,d,_=g.local_bright_spread(np.ones((512,512),bool))
    assert np.isnan(val) and d['neighbourhood_windows_512px']==1


def test_drop_stage_gate_runs_before_any_data_read(tmp_path):
    with pytest.raises(ValueError,match='explicit'):require_known_stage(None)
    p=tmp_path/'stage.json';p.write_text(json.dumps({'stage':'incomplete'}))
    with pytest.raises(ValueError,match='not been saved'):require_known_stage(p)
    p.write_text(json.dumps({'stage':'known_only_complete'}))
    result=require_known_stage(p)
    assert result['stage']=='known_only_complete' and len(result['sha256'])==64


def test_raw_api_rejects_changed_estimands_and_non_uint8():
    with pytest.raises(ValueError,match='uint8'):g.extract_site(np.zeros((64,64),float))
    with pytest.raises(ValueError,match='preregistered'):g.extract_site(np.zeros((64,64),np.uint8),area_floor=75)
    with pytest.raises(ValueError,match='preregistered'):g.extract_site(np.zeros((64,64),np.uint8),threshold_offset=10)


def test_bound_known_output_cardinality_floor_conventions_and_parity():
    out=Path(__file__).resolve().parents[1]/'analysis/forest_geometry/output'
    if not (out/'geometry_known.csv').exists():pytest.skip('Known geometry extraction not completed yet')
    t=pd.read_csv(out/'geometry_known_variants.csv')
    assert len(t)==124 and t[['batch','site','variant']].duplicated().sum()==0
    assert len(g.nominal(t))==31
    a=t.loc[t.variant.eq('nominal')].set_index(['batch','site'])
    b=t.loc[t.variant.eq('floor100')].set_index(['batch','site'])
    unchanged=g.PORE+[g.FEATURES[6],g.FEATURES[7]]
    np.testing.assert_allclose(a[unchanged],b.loc[a.index,unchanged],equal_nan=True)
    if (out/'geometry_nominal_replay.csv').exists():
        replay=pd.read_csv(out/'geometry_nominal_replay.csv')
        assert len(replay)==32 and replay.match.all()
