"""Run the registered E31L.1 audit from repository root.

python -m analysis.morphology.l_audit.run_audit
Writes only its own experiment directory. No held-back data, production inputs,
original measurements or verdicts are readjusted. Restart checkpoints are hash-bound.
"""
from __future__ import annotations
import hashlib,json,os,shutil,time
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/polaron-l-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from polaron_qc import features
from analysis.morphology.k_pilot.descriptors import phase_summary
from . import methods as m
from .measurement import summaries,shape,masks,adaptive_copy

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
EXAMPLES=[('Batch_2','3806gxp0'),('Batch_1','4ih2ggld'),('Batch_3','71vgq3fw'),('Batch_3','hzumfsms')]
MODES={-5:'threshold_minus5',0:'nominal',5:'threshold_plus5'}
ACQ=['bright_sep','bse_p1','bse_std','bse_empty_bin_frac','etd_boundary_sharpness','H']
GEO=m.BASE+['bright_d90','bright_circ','bright_count_per_Mpx']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def save(name,rows):
    t=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    t.to_csv(OUT/name,index=False)
    return t

def source_manifest():
    km=json.loads((ROOT/'analysis/morphology/k_pilot/manifest.json').read_text())
    for p in ['analysis/morphology/output/morphology_sites.csv','analysis/ml_options/e_graph/nodes.csv.gz']:
        assert sha(ROOT/p)==km['inputs'][p],('K source mismatch',p)
    files=['analysis/morphology/output/morphology_sites.csv','analysis/morphology/k_pilot/variants.csv',
           'analysis/morphology/k_pilot/manifest.json','polaron_qc/features.py','polaron_qc/stats.py',
           'analysis/morphology/k_pilot/descriptors.py']
    sources=['methods.py','measurement.py','run_audit.py','test_methods.py','protocol.json','protocol_registration.json']
    files += ['analysis/morphology/l_audit/'+p for p in sources]
    files += [str(p.relative_to(ROOT)) for b in ['Batch_1','Batch_2','Batch_3'] for p in sorted((ROOT/'Dataset'/b).glob('*_BSE.tif'))]
    inputs={p:sha(ROOT/p) for p in files}
    for p,h in inputs.items():
        if p.startswith('Dataset/'):
            assert h==km['inputs'][p],('raw K hash mismatch',p)
    snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in files:
        if p.endswith('.py'):
            dst=snap/p.replace('/','__');shutil.copyfile(ROOT/p,dst)
    return inputs

def extraction(meta,inputs):
    checkpoint=OUT/'measurement_receipt.json'
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text())
        if old['inputs']==inputs and all(sha(OUT/p)==h for p,h in old['outputs'].items()):
            print('Hash-bound raw replay checkpoint reused',flush=True)
            return pd.read_csv(OUT/'variants.csv'),pd.read_csv(OUT/'radiometric.csv'),pd.read_csv(OUT/'axis_raw.csv')
        raise RuntimeError('Existing measurement checkpoint differs; preserve run and register a new version')
    kv=pd.read_csv(ROOT/'analysis/morphology/k_pilot/variants.csv').set_index(['batch','site','mode'])
    rows=[];controls=[];axes=[];pairs=[];replay=[]
    fig,grid=plt.subplots(4,3,figsize=(14,13))
    for i,r in enumerate(meta.itertuples(index=False)):
        raw=features.load_image(str(ROOT/'Dataset'/r.batch),r.site)
        trimmed=features.trimmed(raw);sm=features.smooth_bse(trimmed)
        th=features.phase_thresholds(sm)
        assert trimmed.shape==(r.H,r.W),(r.site,trimmed.shape,r.H,r.W)
        assert np.allclose([th['th_lo'],th['th_hi']],[r.th_lo,r.th_hi]),r.site
        for offset,mode in MODES.items():
            pore,bright=masks(sm,th['th_lo']+offset,th['th_hi']+offset)
            vals,bt=summaries(pore,bright)
            krow=kv.loc[(r.batch,r.site,mode)]
            vals[m.CAND[2]]=krow[m.CAND[2]]
            row=dict(batch=r.batch,site=r.site,mode=mode,threshold_offset=offset,area_floor_px2=50,**vals)
            rows.append(row)
            if offset==0:
                for k in m.BASE+m.CAND[:2]:
                    expected=getattr(r,k)
                    assert np.isclose(vals[k],expected,atol=1e-10,rtol=1e-10,equal_nan=True),(r.site,k,vals[k],expected)
                floor=dict(row,mode='floor100',area_floor_px2=100,**shape(bt,100))
                floor['bright_interior_area_share']/=max(float(bt.area.sum()),1.)
                rows.append(floor)
            if (r.batch,r.site) in EXAMPLES:
                ph,pr=phase_summary(bright,pore)
                assert np.isclose(ph[m.CAND[2]],krow[m.CAND[2]],atol=1e-13,equal_nan=True)
                replay.append(dict(batch=r.batch,site=r.site,offset=offset,cached=krow[m.CAND[2]],replayed=ph[m.CAND[2]]))
                pairs.extend(pr.assign(batch=r.batch,site=r.site,mode=mode).to_dict('records'))
                if offset==0:
                    # Transpose already-trimmed ROI: same physical pixels and thresholds.
                    # Re-trimming an untrimmed rotated frame would change the ROI/measurand.
                    tp,tb=masks(features.smooth_bse(trimmed.T.copy()),th['th_lo'],th['th_hi'])
                    tv,_=summaries(tp,tb);tphase,_=phase_summary(tb,tp)
                    for k in m.CAND:
                        observed=tphase[k] if k==m.CAND[2] else tv[k]
                        expected=-vals[k] if k==m.CAND[2] else vals[k]
                        assert np.isclose(observed,expected,atol=1e-10,equal_nan=True),(r.site,k)
                        axes.append(dict(batch=r.batch,site=r.site,kpi=k,nominal=vals[k],transposed=observed,expected=expected,
                                         absolute_error=abs(observed-expected),scope='same trimmed ROI, swapped image axes'))
                    j=EXAMPLES.index((r.batch,r.site))
                    step=max(1,int(np.ceil(trimmed.shape[1]/700)))
                    a=trimmed[::step,::step]
                    grid[j,0].imshow(a,cmap='gray',vmin=0,vmax=255)
                    grid[j,0].set_title(f'{r.batch} / {r.site}: trimmed BSE')
                    rgb=np.repeat(a[:,:,None]/255,3,axis=2)
                    rgb[bright[::step,::step]]=[1,.7,.12];rgb[pore[::step,::step]]=[.12,.5,1.]
                    grid[j,1].imshow(rgb);grid[j,1].set_title('Bright mask (gold); void mask (blue)')
                    grid[j,2].imshow(a,cmap='gray',vmin=0,vmax=255)
                    for _,obj in bt[(~bt.touches_edge)&bt.aspect.ge(1.5)].iterrows():
                        y=(obj['bbox-0']+obj['bbox-2'])/2/step;x=(obj['bbox-1']+obj['bbox-3'])/2/step
                        half=obj.major_axis_length/2/step
                        grid[j,2].plot([x-half*np.cos(obj.theta),x+half*np.cos(obj.theta)],
                                       [y-half*np.sin(obj.theta),y+half*np.sin(obj.theta)],color='#fd6',lw=.7)
                    grid[j,2].set_title(f'Eligible elongated bright axes (≥1.5)\n{int(vals["bright_n_oriented"])} components; one site')
        if (r.batch,r.site) in EXAMPLES:
            copies={'gamma_0.75':np.rint(255*(raw.astype(float)/255)**.75).astype('uint8'),
                    'gamma_1.25':np.rint(255*(raw.astype(float)/255)**1.25).astype('uint8'),
                    'affine_0.8I_plus20':np.rint(.8*raw.astype(float)+20).astype('uint8')}
            for name,copy in copies.items():
                cv,_,_,_,_,cp=adaptive_copy(copy)
                controls.append(dict(batch=r.batch,site=r.site,transform=name,**cv))
                pairs.extend(cp.assign(batch=r.batch,site=r.site,mode=name).to_dict('records'))
        print(f'Raw replay {i+1}/{len(meta)} {r.batch}/{r.site}',flush=True)
    for ax in grid.ravel():ax.axis('off')
    fig.tight_layout();(OUT/'assets').mkdir(exist_ok=True);fig.savefig(OUT/'assets/examples.png',dpi=135);plt.close(fig)
    values=save('raw_summaries.csv',rows)
    carry=[c for c in meta if c not in values or c in ('batch','site')]
    joined=values.merge(meta[carry],on=['batch','site'],validate='many_to_one')
    # Every warm association mode must be measured and coverage-bound; floor is identical raster data.
    assert len(joined)==124 and not joined.duplicated(['batch','site','mode']).any()
    radio=save('radiometric.csv',controls);axis=save('axis_raw.csv',axes)
    save('phase_replay.csv',replay);save('pair_counts.csv',pairs);save('variants.csv',joined)
    outputfiles=['raw_summaries.csv','variants.csv','radiometric.csv','axis_raw.csv','phase_replay.csv','pair_counts.csv','assets/examples.png']
    dump('measurement_receipt.json',dict(inputs=inputs,outputs={p:sha(OUT/p) for p in outputfiles},
                                       completed_utc=datetime.now(timezone.utc).isoformat(),fresh_sites=31,
                                       warm_reuse='K signed raster associations, independently replayed at fixed 4 sites × 3 thresholds'))
    return joined,radio,axis

def known_tests(variants):
    tests=[];policies=[];uni=[];deletions=[];cover=[];nulls={}
    for mode,t in variants.groupby('mode',sort=False):
        for view in ['shared','native']:
            for name,cols in m.PANELS.items():
                keys=[m.KEYS[c] for c in cols];ok=m.eligible(t,keys,shared=view=='shared')
                for b in ['Batch_1','Batch_2','Batch_3']:
                    mask=t.batch.eq(b)
                    cover.append(dict(mode=mode,view=view,panel=name,batch=b,n_folder=int(mask.sum()),n_usable=int((mask&ok).sum()),
                                      n_abstained=int((mask&~ok).sum())))
        shared=t[m.eligible(t,m.KEYS,shared=True)]
        ref=shared[shared.batch.eq('Batch_3')]
        for batch in ['Batch_1','Batch_2']:
            query=shared[shared.batch.eq(batch)]
            rows,p,null=m.energy_tests(ref[m.KEYS].to_numpy(),query[m.KEYS].to_numpy())
            for r in rows:tests.append(dict(mode=mode,view='shared',batch=batch,**r))
            policy=m.policies(p);adjusted=m.holm([p['primary5'],p['morphology3']])
            policies.append(dict(mode=mode,batch=batch,n_ref=len(ref),n_batch=len(query),
                                 p_primary5=p['primary5'],p_morphology3=p['morphology3'],p_extended8=p['extended8'],
                                 holm_primary=adjusted[0],holm_morphology=adjusted[1],**policy))
            nulls.update({f'{mode}__{batch}__{panel}':array for panel,array in null.items()})
            if mode=='nominal':
                # Eight explanatory location tests: HL, not coarse median differences.
                pooled=np.vstack([m.canonical(ref[m.KEYS]),m.canonical(query[m.KEYS])])
                orders,method,_=m.allocation_orders(len(pooled),len(ref));ps=[];u=[]
                for c,k in enumerate(m.KEYS):
                    null_hl=[]
                    for start in range(0,len(orders),128):
                        z=pooled[orders[start:start+128],c]
                        dif=z[:,len(ref):,None]-z[:,None,:len(ref)]
                        null_hl.extend(np.median(dif,axis=(1,2)))
                    observed=float(np.median(query[k].to_numpy()[:,None]-ref[k].to_numpy()))
                    p1=m.tail_p(np.abs(null_hl),abs(observed),method);ps.append(p1)
                    diff,lo,hi=m.median_interval(ref[k],query[k])
                    u.append(dict(batch=batch,kpi=k,n_ref=len(ref),n_batch=len(query),hl_difference=observed,
                                  median_difference=diff,ci_low=lo,ci_high=hi,p=p1,n_boot=4000,method=method))
                for r,adj in zip(u,m.holm(ps)):uni.append(dict(r,p_holm8=adj))
                for role,frame in [('reference',ref),('incoming',query)]:
                    for idx,row in frame.iterrows():
                        rr=ref.drop(index=idx) if role=='reference' else ref
                        qq=query.drop(index=idx) if role=='incoming' else query
                        folds,pfold,_=m.energy_tests(rr[m.KEYS],qq[m.KEYS])
                        pol=m.policies(pfold);available=min(len(rr),len(qq))>=5
                        for r in folds:deletions.append(dict(batch=batch,deleted_role=role,deleted_site=row.site,**r,
                                                           holm_union=pol['holm_union'] if available else None))
                for name,cols in m.PANELS.items():
                    keys=[m.KEYS[c] for c in cols]
                    eligible=t[m.eligible(t,keys)]
                    rr=eligible[eligible.batch.eq('Batch_3')];qq=eligible[eligible.batch.eq(batch)]
                    row,pnat,_=m.energy_tests(rr[keys],qq[keys],panels={name:list(range(len(cols)))})
                    tests.append(dict(mode=mode,view='native',batch=batch,**row[0]))
    np.savez_compressed(OUT/'known_nulls.npz',**nulls)
    return save('batch_tests.csv',tests),save('batch_policies.csv',policies),save('univariate.csv',uni),save('deletions.csv',deletions),save('coverage.csv',cover)

def reference_diagnostics(nominal):
    ref=nominal[nominal.batch.eq('Batch_3')&m.eligible(nominal,m.KEYS,shared=True)]
    z=m.canonical(ref[m.KEYS]);results=[];detail=[]
    panels={k:m.PANELS[k] for k in ['primary5','morphology3','extended8']}
    for nb in [5,7]:
        nr=len(ref)-nb;orders,method,total=m.allocation_orders(len(ref),nr)
        assert method=='exact'
        values=m.statistics(z,nr,orders,panels);ps={p:m.exact_ranks(v) for p,v in values.items()}
        for i in range(len(orders)):
            p={k:ps[k][i] for k in ps};pol=m.policies(p)
            detail.append(dict(query_n=nb,allocation=i,n_ref=nr,n_batch=nb,**{f'p_{k}':v for k,v in p.items()},**pol))
        d=pd.DataFrame(detail);d=d[d.query_n.eq(nb)]
        for policy in ['primary5','extended8','holm_union','naive_or']:
            results.append(dict(query_n=nb,n_ref=nr,policy=policy,n_allocations=total,n_alerts=int(d[policy].sum()),
                                alert_fraction=float(d[policy].mean()),n_abstained=0,
                                interpretation='overlapping exact internal ranks, not independent external false-alert trials'))
    save('reference_allocations.csv',detail)
    return save('reference_diagnostics.csv',results)

def simulation():
    receipt=OUT/'simulation_receipt.json';codehash={p:sha(OUT/p) for p in ['methods.py','run_audit.py','protocol.json']}
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        assert saved['codehash']==codehash
        assert all(sha(OUT/p)==h for p,h in saved['outputs'].items())
        print('Hash-bound simulation checkpoint reused',flush=True)
        return pd.read_csv(OUT/'iid_summary.csv'),pd.read_csv(OUT/'power_summary.csv'),pd.read_csv(OUT/'paired_gains.csv')
    rng=np.random.default_rng(m.SEED);null=[];power=[];gains=[]
    for nb in [5,7]:
        for trial in range(200):
            z=m.gaussian_draw(rng,13+nb)
            _,p,_=m.energy_tests(z[:13],z[13:],n_mc=499,exact_limit=0,seed=m.SEED+trial)
            null.append(dict(query_n=nb,trial=trial,**{f'p_{k}':p[k] for k in ['primary5','morphology3','extended8']},**m.policies(p)))
        print(f'IID null: 200 independent trials at 13/{nb}',flush=True)
    scenarios=[(f'{k}__{sign:+d}',c+5,sign*2.) for c,k in enumerate(m.CAND) for sign in [-1,1]]
    scenarios += [('candidate_spread_x2',None,0),('primary5_shift_plus2',None,2.)]
    # A shared draw within a trial pairs alternatives and zero-shift baseline, never increases independent n.
    for trial in range(120):
        z=m.gaussian_draw(rng,20)
        for name,col,delta in scenarios:
            v=z.copy()
            if col is not None:v[13:,col]+=delta
            elif name=='candidate_spread_x2':v[13:,5:]*=2
            else:v[13:,:5]+=delta
            _,p,_=m.energy_tests(v[:13],v[13:],n_mc=499,exact_limit=0,seed=m.SEED+500+trial)
            pol=m.policies(p)
            power.append(dict(scenario=name,trial=trial,**{f'p_{k}':val for k,val in p.items()},**pol,
                              **{f'alert_{k}':val <= .05 for k,val in p.items() if k.startswith('primary5_plus_')}))
        if (trial+1)%20==0:print(f'Feature-space controls: {trial+1}/120 paired independent draws',flush=True)
    nd=save('iid_trials.csv',null);pd1=save('power_trials.csv',power)
    nsummary=[];psummary=[]
    policies=['primary5','extended8','holm_union','naive_or']
    for nb,g in nd.groupby('query_n'):
        for policy in policies:
            hits=int(g[policy].sum());lo,hi=m.wilson(hits,len(g))
            nsummary.append(dict(query_n=nb,policy=policy,n_trials=len(g),n_alerts=hits,rate=hits/len(g),ci_low=lo,ci_high=hi))
    for name,g in pd1.groupby('scenario',sort=False):
        for policy in policies+[f'alert_primary5_plus_{k}' for k in m.CAND]:
            hits=int(g[policy].sum());lo,hi=m.wilson(hits,len(g))
            psummary.append(dict(scenario=name,policy=policy,n_trials=len(g),n_alerts=hits,rate=hits/len(g),ci_low=lo,ci_high=hi))
            if policy!='primary5':
                v=g[policy].astype(int).to_numpy()-g.primary5.astype(int).to_numpy()
                mean,low,high=m.paired_interval(v)
                gains.append(dict(scenario=name,policy=policy,n_trials=len(g),mean_gain=mean,ci_low=low,ci_high=high))
    ns=save('iid_summary.csv',nsummary);ps=save('power_summary.csv',psummary);gs=save('paired_gains.csv',gains)
    files=['iid_trials.csv','power_trials.csv','iid_summary.csv','power_summary.csv','paired_gains.csv']
    dump('simulation_receipt.json',dict(codehash=codehash,outputs={p:sha(OUT/p) for p in files},
                                      completed_utc=datetime.now(timezone.utc).isoformat(),n_mc=499))
    return ns,ps,gs

def robustness(nominal,variants,radio):
    rows=[];summary=[];radiomoves=[];axistests=[]
    for k in m.CAND:
        eligible=m.eligible(nominal,[k]);ref=nominal[nominal.batch.eq('Batch_3')&eligible]
        mad=float(1.4826*np.median(np.abs(ref[k]-np.median(ref[k]))))
        assert mad>0,('Degenerate reference MAD makes gate unavailable',k)
        source=nominal.loc[eligible,['batch','site',k]].set_index(['batch','site'])
        for mode in ['threshold_minus5','threshold_plus5','floor100']:
            v=variants[variants['mode'].eq(mode)].copy();ok=m.eligible(v,[k]);v=v.set_index(['batch','site'])
            for ix,r in source.iterrows():
                delta=float(v.loc[ix,k]-r[k]);valid=bool(pd.Series(ok,index=v.index).loc[ix])
                rows.append(dict(batch=ix[0],site=ix[1],kpi=k,mode=mode,reference_mad=mad,delta=delta,
                                 abs_delta_mad=abs(delta)/mad,paired_available=valid,
                                 raster_floor_duplicate=(k==m.CAND[2] and mode=='floor100')))
            vals=[r['abs_delta_mad'] for r in rows if r['kpi']==k and r['mode']==mode and r['paired_available']]
            summary.append(dict(kpi=k,mode=mode,n_source=len(source),n_pairs=len(vals),reference_mad=mad,
                                median_movement_mad=float(np.median(vals)),max_movement_mad=float(max(vals)),
                                gate_pass=bool(len(vals)==len(source) and np.median(vals)<=.5 and max(vals)<=2)))
        for r in radio.itertuples(index=False):
            src=nominal[nominal.batch.eq(r.batch)&nominal.site.eq(r.site)]
            sourceok=bool(m.eligible(src,[k])[0]);trans=radio[(radio.batch.eq(r.batch))&(radio.site.eq(r.site))&radio['transform'].eq(r.transform)]
            valid=sourceok and bool(m.eligible(trans,[k])[0])
            delta=float(getattr(r,k)-src.iloc[0][k])
            radiomoves.append(dict(batch=r.batch,site=r.site,kpi=k,transform=r.transform,source_available=sourceok,
                                   transformed_available=bool(m.eligible(trans,[k])[0]),paired_available=valid,
                                   delta=delta,abs_delta_mad=abs(delta)/mad,reference_mad=mad,
                                   source_low_contrast=bool(src.iloc[0].bright_low_contrast),source_grey_pore=bool(src.iloc[0].grey_pore),
                                   transformed_low_contrast=bool(r.bright_low_contrast),transformed_grey_pore=bool(r.grey_pore)))
    shared=nominal[m.eligible(nominal,m.KEYS,shared=True)]
    ref=shared[shared.batch.eq('Batch_3')][m.KEYS].to_numpy()
    for b in ['Batch_1','Batch_2']:
        query=shared[shared.batch.eq(b)][m.KEYS].to_numpy()
        sign=np.array([1]*7+[-1])
        _,p,_=m.energy_tests(ref,query)
        _,globalp,_=m.energy_tests(ref*sign,query*sign)
        _,queryp,_=m.energy_tests(ref,query*sign)
        for panel in ['primary5','morphology3','extended8']:
            assert p[panel]==globalp[panel],('Global axis invariance',b,panel)
            axistests.append(dict(batch=b,panel=panel,nominal_p=p[panel],global_axis_swap_p=globalp[panel],query_only_swap_p=queryp[panel],
                                   query_only_interpretation='unsupported section-orientation shift, not a material control'))
    return save('movement_pairs.csv',rows),save('movement_summary.csv',summary),save('radiometric_movement.csv',radiomoves),save('axis_tests.csv',axistests)

def controls(nominal):
    scores=[];predictions=[];correlations=[]
    for k in m.CAND:
        t=nominal[m.eligible(nominal,[k])]
        for family,cov in [('acquisition',ACQ),('geometry_loading',GEO)]:
            x=t[cov].to_numpy(float);y=t[k].to_numpy(float);pred=[]
            for i in range(len(t)):
                train=np.arange(len(t))!=i
                model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=1.))
                model.fit(x[train],y[train]);pred.append(float(model.predict(x[i:i+1])[0]))
            scores.append(dict(kpi=k,family=family,n_sites=len(t),loo_r2=float(r2_score(y,pred)),alpha=1.,n_covariates=len(cov),
                               interpretation='association screen; not causal attribution or independence proof'))
            predictions.extend(dict(batch=r.batch,site=r.site,kpi=k,family=family,observed=yy,predicted=pp) for r,yy,pp in zip(t.itertuples(),y,pred))
            for b,g in t.groupby('batch'):
                for c in cov:
                    valid=np.isfinite(g[c].to_numpy(float))&np.isfinite(g[k].to_numpy(float))
                    rho,p=(np.nan,np.nan)
                    if valid.sum()>=3 and g.loc[valid,c].nunique()>1 and g.loc[valid,k].nunique()>1:
                        rho,p=spearmanr(g.loc[valid,c],g.loc[valid,k])
                    correlations.append(dict(batch=b,kpi=k,family=family,covariate=c,n_sites=int(valid.sum()),rho=rho,p=p,
                                             status='descriptive_multiple_correlations_not_adjusted'))
    return save('control_scores.csv',scores),save('control_predictions.csv',predictions),save('within_batch_correlations.csv',correlations)

def site_scores(nominal):
    t=nominal[m.eligible(nominal,m.KEYS,shared=True)];ref=t[t.batch.eq('Batch_3')];rows=[];contr=[]
    for panel in ['primary5','morphology3','extended8']:
        cols=m.PANELS[panel];keys=[m.KEYS[c] for c in cols];r=ref[keys].to_numpy()
        ref_scores=[]
        for i in range(len(ref)):
            train=np.arange(len(ref))!=i
            value,_,_,_=m.neighbour_score(r[train],r[i:i+1]);ref_scores.append(value[0])
        for _,row in t.iterrows():
            query=row[keys].to_numpy(float)[None]
            isref=row.batch=='Batch_3';keep=ref.site.ne(row.site).to_numpy() if isref else np.ones(len(ref),bool)
            score,near,c,fallback=m.neighbour_score(r[keep],query)
            nearest=ref.loc[keep].iloc[near[0]]
            loo=[m.neighbour_score(np.delete(r,i,axis=0),query)[0][0] for i in range(len(ref))] if not isref else [score[0]]
            # Reference score rank excludes itself. Query compares to all reference LOO scores.
            ranks=np.asarray([s for sid,s in zip(ref.site,ref_scores) if not isref or sid!=row.site])
            rows.append(dict(batch=row.batch,site=row.site,panel=panel,n_fit=len(r[keep]),score_3nn=score[0],
                             reference_loo_rank=float((ranks<=score[0]).mean()),n_rank_reference=len(ranks),
                             matched_count_score_median=float(np.median(loo)),matched_count_score_min=float(min(loo)),
                             matched_count_score_max=float(max(loo)),nearest_reference=nearest.site,
                             fallback_sd=int((fallback=='sd').sum()),fallback_unit=int((fallback=='unit').sum()),
                             status='descriptive rank only, not probability/conformal p-value/forced label'))
            for k,cc in zip(keys,c[0]):contr.append(dict(batch=row.batch,site=row.site,panel=panel,kpi=k,
                                                        nearest_reference=nearest.site,squared_distance_contribution=float(cc)))
    return save('site_scores.csv',rows),save('site_contributions.csv',contr)

def recommendation(nominal,movement,radio,control,gains,iid):
    union=iid[iid.policy.eq('holm_union')]
    model_pass=bool(((union.ci_low<=.05)&(union.ci_high>=.05)&(union.ci_high<=.10)).all())
    rec=[]
    for k in m.CAND:
        cov=nominal[m.eligible(nominal,[k])].groupby('batch').size()
        threshold=bool(movement[movement.kpi.eq(k)].gate_pass.all())
        rad=radio[radio.kpi.eq(k)&radio.source_available]
        validrad=rad[rad.paired_available]
        radiometric_covered=bool(len(rad)>0 and len(validrad)==len(rad))
        radio_pass=bool(len(validrad)>0 and validrad.abs_delta_mad.max()<=2 and radiometric_covered)
        acq=float(control[(control.kpi.eq(k))&control.family.eq('acquisition')].loo_r2.iloc[0])
        geometry=float(control[(control.kpi.eq(k))&control.family.eq('geometry_loading')].loo_r2.iloc[0])
        power=gains[gains.scenario.str.startswith(k+'__')&gains.policy.eq('holm_union')]
        # Both registered shift directions must supply added detection; do not choose the better sign.
        gainpass=bool(len(power)==2 and ((power.mean_gain>=.10)&power.ci_low.gt(0)).all())
        axisp=bool(k!=m.CAND[2]) # no specimen/section direction supplied for the directional candidate.
        countpass=bool(len(cov)==3 and cov.min()>=5)
        passed=all([countpass,threshold,radio_pass,acq<=.25,gainpass,axisp])
        failures=[]
        for gate,ok in [('coverage',countpass),('threshold_floor',threshold),('radiometric',radio_pass),
                        ('acquisition',acq<=.25),('added_detection',gainpass),('axis_metadata',axisp)]:
            if not ok:failures.append(gate)
        rec.append(dict(kpi=k,n_Batch_1=int(cov.get('Batch_1',0)),n_Batch_2=int(cov.get('Batch_2',0)),n_Batch_3=int(cov.get('Batch_3',0)),
                        coverage_pass=countpass,threshold_floor_pass=threshold,radiometric_pass=radio_pass,
                        radiometric_source_cases=len(rad),radiometric_valid_cases=len(validrad),
                        acquisition_loo_r2=acq,acquisition_pass=acq<=.25,geometry_loading_loo_r2=geometry,
                        redundancy_warning=geometry>.75,added_detection_pass=gainpass,axis_gate_pass=axisp,
                        candidate_gate_pass=passed,joint_model_gate_pass=model_pass,
                        recommendation='versioned shadow OOD development' if passed else 'defer automatic OOD input',
                        unresolved_gates=';'.join(failures),expert_masks='unreviewed',unseen_validation='not run'))
    return save('recommendations.csv',rec)

def main():
    started=time.time();reg=json.loads((OUT/'protocol_registration.json').read_text())
    assert sha(OUT/'protocol.json')==reg['protocol_sha256']
    assert reg['new_results_exist'] is False
    inputs=source_manifest()
    dump('run_started.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,experiment='E31L',version='E31L.1'))
    meta=pd.read_csv(ROOT/'analysis/morphology/output/morphology_sites.csv')
    variants,radio,axis=extraction(meta,inputs)
    nominal=variants[variants['mode'].eq('nominal')]
    save('sites.csv',nominal)
    batch,policy,uni,deletion,coverage=known_tests(variants);print('Known-batch and deletion tests saved',flush=True)
    internal=reference_diagnostics(nominal);print('Exact reference allocation diagnostics saved',flush=True)
    iid,power,gains=simulation()
    movement,ms,rm,at=robustness(nominal,variants,radio)
    cs,cp,corr=controls(nominal);ss,contr=site_scores(nominal)
    rec=recommendation(nominal,ms,rm,cs,gains,iid)
    assert all(sha(ROOT/p)==h for p,h in inputs.items()),'Input mutated during run'
    outputs=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.csv','.npz','.png']]
    dump('manifest.json',dict(experiment='E31L',version='E31L.1',base_commit='87ecee3',
                             registered_utc=reg['registered_utc'],completed_utc=datetime.now(timezone.utc).isoformat(),
                             inputs=inputs,inputs_unchanged=True,outputs={str(p.relative_to(OUT)):sha(p) for p in outputs},
                             elapsed_seconds=round(time.time()-started,3),n_sites=31,n_variants=124,
                             raw_replay='31 sites at -5/0/+5; 4 signed phase examples ×3; 4 same-ROI transposes; 12 radiometric copies',
                             production_activated=False,unseen_read=False,scope='development-known-site audit plus independent Gaussian controls'))
    print(rec[['kpi','recommendation','unresolved_gates']].to_string(index=False),flush=True)
    from .report import build
    build()

if __name__=='__main__':main()
