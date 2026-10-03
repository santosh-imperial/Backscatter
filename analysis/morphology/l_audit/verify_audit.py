"""Verify saved E31L tables against sources and independent conventions."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from . import methods as m

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    receipt=json.loads((OUT/'manifest.json').read_text());reg=json.loads((OUT/'protocol_registration.json').read_text())
    assert receipt['inputs_unchanged'] and not receipt['production_activated'] and not receipt['unseen_read']
    assert reg['protocol_sha256']==sha(OUT/'protocol.json')
    assert reg['registered_utc']<receipt['completed_utc']
    for p,h in receipt['inputs'].items():assert sha(ROOT/p)==h,('input',p)
    for p,h in receipt['outputs'].items():assert sha(OUT/p)==h,('output',p)
    repair=json.loads((OUT/'execution_repairs.json').read_text())
    old=(OUT/'source_snapshot/run_audit_before_column_lookup_repair.py').read_text()
    new=(OUT/'run_audit.py').read_text()
    assert sha(OUT/'run_audit.py')==repair['new_run_sha256']
    assert sha(OUT/'source_snapshot/run_audit_before_column_lookup_repair.py')==repair['old_run_sha256']
    assert new==old.replace(repair['exact_replacement']['old'],repair['exact_replacement']['new'])
    before=json.loads((OUT/'simulation_receipt_before_column_lookup_repair.json').read_text())
    after=json.loads((OUT/'simulation_receipt.json').read_text())
    assert before['outputs']==after['outputs']
    def read(name):return pd.read_csv(OUT/(name+'.csv'))
    v=read('variants');t=read('sites');meta=pd.read_csv(ROOT/'analysis/morphology/output/morphology_sites.csv')
    assert len(v)==124 and len(t)==31 and t.groupby('batch').size().to_dict()=={'Batch_1':7,'Batch_2':7,'Batch_3':17}
    assert not v.duplicated(['batch','site','mode']).any()
    join=t.merge(meta,on=['batch','site'],suffixes=('_fresh','_saved'),validate='one_to_one')
    for k in m.BASE+m.CAND[:2]:np.testing.assert_allclose(join[k+'_fresh'],join[k+'_saved'],rtol=1e-10,atol=1e-10,equal_nan=True)
    shared=t[m.eligible(t,m.KEYS,shared=True)]
    assert shared.groupby('batch').size().to_dict()=={'Batch_1':5,'Batch_2':7,'Batch_3':13}
    assert set(t.loc[t.cracked_known & t.batch.eq('Batch_3'),'site'])<=set(shared.site)
    nominal=t.set_index(['batch','site']);floor=v[v['mode'].eq('floor100')].set_index(['batch','site'])
    for k in m.BASE+[m.CAND[2]]:np.testing.assert_array_equal(nominal[k].sort_index(),floor[k].sort_index())
    pairs=read('pair_counts')
    for row in pairs.itertuples():
        if row.unavailable_reason==row.unavailable_reason:continue
        px=row.source_phase_pixels/row.pair_pixels;py=row.target_phase_pixels/row.pair_pixels
        expected=(row.joint_phase_pixels/row.pair_pixels-px*py)/np.sqrt(px*(1-px)*py*(1-py))
        assert np.isclose(row.rho,expected,atol=1e-13)
    pr=read('phase_replay');assert len(pr)==12
    np.testing.assert_allclose(pr.cached,pr.replayed,atol=1e-13)
    axis=read('axis_raw');assert len(axis)==12 and axis.absolute_error.max()<1e-10
    assert len(read('radiometric'))==12
    ref=shared[shared.batch.eq('Batch_3')]
    bt=read('batch_tests');bp=read('batch_policies')
    for b in ['Batch_1','Batch_2']:
        query=shared[shared.batch.eq(b)];rows,p,_=m.energy_tests(ref[m.KEYS],query[m.KEYS])
        saved=bt[bt['mode'].eq('nominal')&bt.view.eq('shared')&bt.batch.eq(b)].set_index('panel')
        for row in rows:
            assert np.isclose(saved.loc[row['panel'],'p'],row['p'],atol=1e-14)
            assert np.isclose(saved.loc[row['panel'],'statistic'],row['statistic'],atol=1e-13)
        actual=m.policies(p);savedp=bp[bp['mode'].eq('nominal')&bp.batch.eq(b)].iloc[0]
        for k,value in actual.items():assert savedp[k]==value
    deletion=read('deletions');small=deletion.n_batch.lt(5)|deletion.n_ref.lt(5)
    assert small.any() and deletion.loc[small,'alert'].isna().all() and deletion.loc[small,'holm_union'].isna().all()
    allocations=read('reference_allocations');internal=read('reference_diagnostics')
    for row in internal.itertuples():
        sub=allocations[allocations.query_n.eq(row.query_n)]
        assert len(sub)==row.n_allocations and row.n_alerts==sub[row.policy].sum()
    assert len(allocations)==3003
    iid=read('iid_trials');summary=read('iid_summary')
    assert len(iid)==400
    for row in summary.itertuples():
        sub=iid[iid.query_n.eq(row.query_n)];hits=int(sub[row.policy].sum());lo,hi=m.wilson(hits,len(sub))
        assert hits==row.n_alerts
        np.testing.assert_allclose([row.ci_low,row.ci_high],[lo,hi],atol=1e-14)
    power=read('power_trials');gains=read('paired_gains')
    assert len(power)==960 and power.groupby('scenario').size().eq(120).all()
    for row in gains.itertuples():
        sub=power[power.scenario.eq(row.scenario)]
        delta=sub[row.policy].astype(int)-sub.primary5.astype(int)
        actual=m.paired_interval(delta.to_numpy())
        np.testing.assert_allclose([row.mean_gain,row.ci_low,row.ci_high],actual,atol=1e-14)
    scores=read('site_scores');contributions=read('site_contributions')
    assert len(scores)==75 and scores.groupby(['batch','site']).size().eq(3).all()
    for row in scores.itertuples():
        keys=[m.KEYS[i] for i in m.PANELS[row.panel]]
        rr=ref[ref.site.ne(row.site)] if row.batch=='Batch_3' else ref
        q=t[t.site.eq(row.site)][keys].to_numpy()
        score,near,c,_=m.neighbour_score(rr[keys].to_numpy(),q)
        assert np.isclose(row.score_3nn,score[0],atol=1e-13)
        cc=contributions[contributions.site.eq(row.site)&contributions.panel.eq(row.panel)].set_index('kpi').loc[keys]
        np.testing.assert_allclose(cc.squared_distance_contribution,c[0],atol=1e-13)
    # In isolation compare the audit base; on main use the actual integration
    # snapshot. Unrelated Claude additions since 87ecee3 are not L edits.
    integrated=OUT/'integration_started.json'
    if integrated.exists():
        captured=json.loads(integrated.read_text())['protected_tracked_files_before']
        final=OUT/'integration_receipt.json'
        if final.exists():
            saved=json.loads(final.read_text())
            assert saved['science_code_tests_and_original_caches_unchanged']
            for p,h in captured.items():
                if not p.startswith(('notebooks/','reports/')):assert saved['protected_tracked_files_after'][p]==h
        else:
            # Claude owns concurrent notebook/report regeneration. These are
            # delivery artifacts, not input dependencies of this standalone audit.
            for p,h in captured.items():
                if not p.startswith(('notebooks/','reports/')):
                    assert sha(ROOT/p)==h,('Scientific protected input changed during L integration',p)
    else:
        changed=subprocess.check_output(['git','diff','87ecee3','--name-only'],cwd=ROOT,text=True).splitlines()
        protected=('polaron_qc/','tests/','notebooks/','reports/','analysis_cache/')
        assert not any(p.startswith(protected) for p in changed),changed
    report=(OUT/'report.html').read_text()
    assert report.count('src="data:image/png;base64,')==4 and 'not a calibrated membership probability' in report
    import re
    for link in re.findall('href="([^"]+)"',report):assert (OUT/link).exists(),('broken report link',link)
    print('PASS: protocol/source/raw hashes; all 31 nominal replays; 124 modes; 12 phase replays; 12 raw axes; 12 radiometric copies; shared coverage/long-void retention; known tests; subminimum abstention; 3003 exact allocations; 400 IID and 960 power trials; gains/scores/contributions; report; science code/tests/original caches preserved by L (concurrent notebook/report delivery excluded on main).')

if __name__=='__main__':main()
