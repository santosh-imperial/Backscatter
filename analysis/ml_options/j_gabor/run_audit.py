"""Task J / E28J: one fixed known-site Gabor audit, outside production QC."""
from __future__ import annotations
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/polaron-j-gabor-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from polaron_qc import features, PRIMARY_KPIS, MATERIAL_KPIS, BATCH_COLORS, UNSEEN_COLOR
from analysis.ml_options.j_gabor.texture import (KEYS, PERIODS, ANGLES, MODES,
    DEFINITIONS, WINDOW, GUARD_RAW, measure_window, summary, windows, usable)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED = 20261003
N_BOOT = 4000
VIEWS = ('all_known', 'quality_matched', 'ordinary_reference')
ACQ = ('bright_sep','bse_p1','bse_std','bse_empty_bin_frac','etd_boundary_sharpness','H')
BASE = ('fft_slope','corr_len_px','pore_frac','bright_frac','bright_d50',
        'local_bright_std_512px','local_bright_std_1024px',
        'bse_tensor_strength_3px','bse_tensor_strength_12px',
        'bse_tensor_balance_3px','bse_tensor_balance_12px')
EXAMPLES = (('Batch_2','3806gxp0'),('Batch_1','4ih2ggld'),
            ('Batch_3','71vgq3fw'),('Batch_3','hzumfsms'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def compare(variants):
    rows = []
    for mode in MODES:
        t = variants[variants['mode'].eq(mode)]
        for view in VIEWS:
            for key in KEYS:
                good = t.loc[usable(t, key, view)]
                for reference, batch in (('Batch_3','Batch_1'),('Batch_3','Batch_2'),('Batch_1','Batch_2')):
                    a = good[good.batch.eq(reference)][key].to_numpy(float)
                    b = good[good.batch.eq(batch)][key].to_numpy(float)
                    lo = hi = np.nan
                    deleted = []
                    if min(len(a),len(b)) >= 2:
                        rng = np.random.default_rng(SEED+len(rows))
                        boot = np.median(b[rng.integers(len(b),size=(N_BOOT,len(b)))],axis=1) - np.median(a[rng.integers(len(a),size=(N_BOOT,len(a)))],axis=1)
                        lo, hi = np.quantile(boot,[.025,.975])
                        deleted = [np.median(b)-np.median(np.delete(a,i)) for i in range(len(a))]
                        deleted += [np.median(np.delete(b,i))-np.median(a) for i in range(len(b))]
                    rows.append(dict(mode=mode,view=view,kpi=key,reference=reference,batch=batch,
                        n_reference=len(a),n_batch=len(b),median_reference=np.median(a) if len(a) else np.nan,
                        median_batch=np.median(b) if len(b) else np.nan,
                        median_difference=np.median(b)-np.median(a) if min(len(a),len(b)) else np.nan,
                        ci_low=lo,ci_high=hi,deletion_min=min(deleted) if deleted else np.nan,
                        deletion_max=max(deleted) if deleted else np.nan))
    return pd.DataFrame(rows)


def nuisance_controls(sites):
    corr, predictions = [], []
    for key in KEYS:
        for view in ('all_known','quality_matched'):
            t = sites.loc[usable(sites,key,view)]
            groups = [('pooled',t)] + [(b,t[t.batch.eq(b)]) for b in sorted(t.batch.unique())]
            if view == 'quality_matched':
                groups += [('Batch_3_ordinary',sites.loc[usable(sites,key,'ordinary_reference') & sites.batch.eq('Batch_3')])]
            for control, covs in (('acquisition',ACQ),('fft_geometry_tensor',BASE)):
                for group,g in groups:
                    for cov in covs:
                        pair = g[[key,cov]].replace([np.inf,-np.inf],np.nan).dropna()
                        rho = float(spearmanr(pair[key],pair[cov]).statistic) if len(pair)>=4 and min(pair[key].nunique(),pair[cov].nunique())>1 else np.nan
                        corr.append(dict(view=view,kpi=key,group=group,control=control,covariate=cov,n_sites=len(pair),rho=rho))
                x = t[list(covs)].replace([np.inf,-np.inf],np.nan).to_numpy(float)
                y = t[key].to_numpy(float)
                pred = np.full(len(y),np.nan)
                if len(y)>=5 and np.ptp(y)>0:
                    for i in range(len(y)):
                        train = np.arange(len(y)) != i
                        model = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=1))
                        model.fit(x[train],y[train]); pred[i]=model.predict(x[i:i+1])[0]
                valid = np.isfinite(pred)
                r2 = 1-np.sum((y[valid]-pred[valid])**2)/np.sum((y[valid]-y[valid].mean())**2) if valid.sum()>=5 and np.ptp(y[valid]) else np.nan
                for i,row in enumerate(t.itertuples()):
                    predictions.append(dict(view=view,kpi=key,control=control,batch=row.batch,site=row.site,
                        observed=y[i],prediction=pred[i],loo_r2=r2,n_sites=len(y)))
    return pd.DataFrame(corr),pd.DataFrame(predictions)


def sensitivity(sites,variants):
    rows = []
    for key in KEYS:
        for view in ('all_known','quality_matched'):
            a = sites.loc[usable(sites,key,view),['batch','site',key]]
            iqr = np.subtract(*np.quantile(a[key],[.75,.25]))
            for mode in MODES:
                b = variants[variants['mode'].eq(mode)][['batch','site',key]]
                t = a.merge(b,on=['batch','site'],suffixes=('_nominal','_mode')).dropna()
                delta = (t[key+'_mode']-t[key+'_nominal']).abs()
                rho = float(spearmanr(t[key+'_nominal'],t[key+'_mode']).statistic) if len(t)>=4 else np.nan
                rows.append(dict(view=view,kpi=key,mode=mode,n_sites=len(t),nominal_site_iqr=iqr,
                    median_abs_delta=delta.median(),max_abs_delta=delta.max(),
                    median_delta_over_site_iqr=delta.median()/iqr if iqr>0 else np.nan,
                    spearman_nominal_mode=rho))
    return pd.DataFrame(rows)


def illustration(batch,site,raw,top,origins,energy,maps):
    y,x = origins[0]
    crop = raw[y:y+WINDOW,x:x+WINDOW]
    coarse = np.mean([maps[(64,i)] for i in range(4)],axis=0)
    direction = (maps[(32,0)]-maps[(32,2)])/(maps[(32,0)]+maps[(32,2)]+1e-12)
    fig,axes = plt.subplots(1,3,figsize=(12.6,4.4))
    axes[0].imshow(crop,cmap='gray',vmin=0,vmax=255)
    axes[0].add_patch(Rectangle((GUARD_RAW,GUARD_RAW),WINDOW-2*GUARD_RAW,WINDOW-2*GUARD_RAW,fill=False,color='#ffc65b',lw=1.5))
    axes[0].set_title('Fixed quarter-frame BSE window')
    im = axes[1].imshow(np.log1p(coarse),cmap='magma',extent=(96,416,416,96))
    axes[1].set(xlim=(0,512),ylim=(512,0),title='64px response energy · log(1+power)')
    fig.colorbar(im,ax=axes[1],shrink=.7)
    im = axes[2].imshow(direction,cmap='coolwarm',vmin=-1,vmax=1,extent=(96,416,416,96))
    axes[2].set(xlim=(0,512),ylim=(512,0),title='32px wavevector balance (x versus y)')
    fig.colorbar(im,ax=axes[2],shrink=.7)
    for ax in axes: ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'{batch}/{site} · raw x={x}:{x+512}, y={top+y}:{top+y+512}',fontsize=11)
    fig.text(.015,.02,'Gold square: common valid convolution interior. Blank border excluded from energies.\nWavevectors describe image modulation, normal to stripes; they are not graphite plate axes or defect masks.',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.92))
    name=f'overlay_{batch}_{site}.png'; fig.savefig(OUT/name,dpi=140);plt.close(fig)
    np.savez_compressed(OUT/f'maps_{batch}_{site}.npz',coarse_power=coarse.astype('float32'),direction_balance=direction.astype('float32'))
    return dict(batch=batch,site=site,image=name,x=x,y_trimmed=y,y_raw=top+y,width=512,height=512,window_id=0)


def report(sites,comparisons,controls,sens,manifest):
    fig,axes = plt.subplots(1,3,figsize=(12.5,4.2))
    for ax,key in zip(axes,KEYS):
        for i,batch in enumerate(sorted(sites.batch.unique())):
            t=sites[sites.batch.eq(batch)];good=usable(t,key,'quality_matched')
            ax.scatter(np.full(len(t),i),t[key],color=BATCH_COLORS.get(batch,UNSEEN_COLOR),label=batch)
            ax.scatter(np.full((~good).sum(),i),t.loc[~good,key],marker='x',color='#e34948',s=60)
            ax.plot([i-.14,i+.14],[t[key].median()]*2,color='black')
        ax.set_xticks(range(3),['B1','B2','B3']);ax.set_title(key.replace('gabor_','').replace('_',' '));ax.legend(fontsize=7)
    fig.suptitle('Sampled-site summaries · red × quality-matched exclusion · black median (all known)')
    fig.tight_layout();fig.savefig(OUT/'site_descriptors.png',dpi=150);plt.close(fig)
    nominal=comparisons[comparisons['mode'].eq('nominal')]
    lines=['# Task J / E28J: fixed Gabor texture pilot','',
        'Known-site development, fixed before new outputs. This is deterministic image filtering, with no neural training, material labels, defect accuracy or unseen-batch validation. Three summaries stay outside primary/ML/verdict inputs.','',
        'Four non-overlapping 512px BSE windows are centred at quarter-frame coordinates. The nominal 2×2 area mean precedes per-window mean-zero/unit-SD normalisation. Twelve DC-removed, unit-L2 complex kernels cover raw periods16/32/64px and wavevector angles0/45/90/135°. Sigma is half the period; support is ±3 sigma with fixed square support. A common96px raw border is discarded. Each window contributes equally; four windows are coverage, never n.','',
        'Scale/direction refer to appearance of the BSE image, not phase identity, graphite plates, pore transport or material defect labels. Filter energy share is not a particle-size fraction. The full-frame sampling mask and valid-area fractions are saved per site. Spatial representativeness is unvalidated.','',
        'All-known view includes acquisition subgroups because no phase mask is used. Quality-matched excludes known low-bright-contrast/grey-pore flags; ordinary-reference additionally excludes known long-void B3 examples without calling that subset clean. Unknown quality does not certify a matched site.','',
        'Sampling intervals use4000 percentile site-bootstrap draws, conditional on unknown specimen independence. Whole-site deletion ranges, resolution and gamma/affine sensitivity are separate from sampling uncertainty. No p-values, C2ST or equivalence test.','',
        '## Nominal site comparisons','',nominal.to_markdown(index=False,floatfmt='.5g'),'',
        '## Fixed nuisance / redundancy controls','',
        'Ridge alpha1 predicts each summary from six acquisition covariates, separately from eleven existing FFT/phase/loading/image-tensor summaries. Median imputation/scaling refit inside every site-LOO fold. Both all-known and quality-matched controls accompany within-batch and ordinary-reference correlations. Prediction is not causal attribution; poor prediction does not prove invariance or independent information.','',
        controls.groupby(['view','kpi','control']).first()[['loo_r2','n_sites']].reset_index().to_markdown(index=False,floatfmt='.4g'),'',
        '## Fixed sensitivity','',sens.to_markdown(index=False,floatfmt='.5g'),'',
        'Non-clipping affine copies are a numerical control. Gamma0.75/1.25 changes contrast shape; full resolution changes sampling while retaining raw filter periods. Originals remain read-only. Synthetic controls do not validate real preparation or lost contrast.','',
        '## Scope of retain/defer finding','',
        'Retain response maps as exploratory appearance references. Defer material/QC use pending expert review, spatial repeatability and independent specimen/process evidence; examine redundancy and acquisition sensitivity before any expanded work. No favourable separation is used to choose a scale, direction, quality subset or a new primary KPI. Numerical and source contracts do not validate a battery mechanism.','',
        'Review applied: C01–C07/C09/C11/C13/C14/C16/C17/C21–C23/C28–C30.','']
    (OUT/'findings.md').write_text('\n'.join(lines))
    def image(name):
        data=base64.b64encode((OUT/name).read_bytes()).decode()
        return f'<img alt="{html.escape(name)}" src="data:image/png;base64,{data}">'
    diagrams=''.join(f'<details open><summary>{c["batch"]}/{c["site"]}</summary>{image(c["image"])}</details>' for c in manifest['crops'])
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fixed Gabor texture audit</title><style>body{font:16px/1.5 system-ui;max-width:1250px;margin:25px auto;padding:15px;color:#173044;background:#f5f7fa}img{width:100%;height:auto}details{background:white;padding:12px;margin:15px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace}.notice{padding:15px;background:#fff1d5;border:1px solid #c8ae78}</style></head><body>'
    page+='<h1>Fixed Gabor texture pilot · task J</h1><p class="notice">Exploratory BSE appearance, sampled on four windows per site. Expert validity and spatial representativeness remain pending; excluded from QC verdicts.</p>'
    page+=image('site_descriptors.png')+diagrams+'<h2>Exact methods, measurements and qualification</h2><pre>'+html.escape('\n'.join(lines))+'</pre></body></html>'
    (OUT/'report.html').write_text(page)


def main():
    start=time.perf_counter(); protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['keys']==list(KEYS) and protocol['modes']==list(MODES)
    metadata_path=ROOT/'analysis/battery/output/neighbourhood_sites.csv'
    tensor_path=ROOT/'analysis/battery/output/orientation_sites.csv'
    metadata=pd.read_csv(metadata_path).sort_values(['batch','site'])
    tensors=pd.read_csv(tensor_path)
    for sigma in (3,12):
        t=tensors[tensors.detector.eq('BSE') & tensors.sigma_px.eq(sigma) & tensors.threshold_offset.eq(0)]
        assert not t.duplicated(['batch','site']).any()
        t=t[['batch','site','texture_alignment_strength','texture_horizontal_alignment']].rename(columns={
            'texture_alignment_strength':f'bse_tensor_strength_{sigma}px',
            'texture_horizontal_alignment':f'bse_tensor_balance_{sigma}px'})
        metadata=metadata.merge(t,on=['batch','site'],how='left',validate='one_to_one')
    source_paths=[metadata_path,tensor_path,OUT/'protocol.json',OUT/'texture.py',Path(__file__),ROOT/'polaron_qc/features.py']
    source_paths += [ROOT/'Dataset'/r.batch/f'img_{r.site}_BSE.tif' for r in metadata.itertuples()]
    before={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    records, energies, crops=[] ,[],[]
    for r in metadata.itertuples():
        raw=features.load_image(ROOT/'Dataset'/r.batch,r.site,'BSE')
        top,bottom=features.bright_bands(raw);raw=raw[top:len(raw)-bottom if bottom else None]
        origins=windows(raw.shape)
        assert raw.shape==(r.H,r.W)
        for mode in MODES:
            bank=[];reasons=[]
            for wi,(y,x) in enumerate(origins):
                keep=mode=='nominal' and (r.batch,r.site) in EXAMPLES and wi==0
                e,m,reason=measure_window(raw[y:y+512,x:x+512],mode,keep)
                if reason: reasons.append(reason);continue
                bank.append(e)
                for si,period in enumerate(PERIODS):
                    for oi,angle in enumerate(ANGLES):
                        energies.append(dict(batch=r.batch,site=r.site,mode=mode,window_id=wi,
                            x=x,y_trimmed=y,y_raw=top+y,period_px=period,angle_deg=int(round(np.degrees(angle))),mean_power=e[si,oi]))
                if keep:crops.append(illustration(r.batch,r.site,raw,top,origins,e,m))
            scalars=summary(np.mean(bank,axis=0)) if len(bank)==4 else {k:np.nan for k in KEYS}
            records.append(r._asdict() | scalars | dict(mode=mode,trim_top=top,trim_bottom=bottom,
                gabor_windows_used=len(bank),gabor_unavailable_reason=';'.join(reasons) if reasons else ('' if len(bank)==4 else 'insufficient_window_coverage'),
                gabor_window_area_fraction=len(origins)*512**2/raw.size,
                gabor_valid_area_fraction=len(bank)*(512-2*96)**2/raw.size))
        print(f'{r.batch}/{r.site}: five fixed modes',flush=True)
    variants=pd.DataFrame(records);sites=variants[variants['mode'].eq('nominal')].copy()
    comp=compare(variants);corr,controls=nuisance_controls(sites);sens=sensitivity(sites,variants)
    for name,t in (('sites',sites),('variants',variants),('window_energies',pd.DataFrame(energies)),
                   ('comparisons',comp),('correlations',corr),('control_predictions',controls),('sensitivity_summary',sens)):
        t.to_csv(OUT/f'{name}.csv',index=False)
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    if changed:raise RuntimeError(f'Inputs changed during run: {changed}')
    manifest=dict(experiment='E28J',task='J',version='E28J.1',date='2026-10-03',inputs=before,inputs_unchanged=True,
        n_sites=len(sites),n_variants=len(variants),n_boot=N_BOOT,seed=SEED,crops=crops,
        primary_kpis=list(PRIMARY_KPIS),material_kpis=list(MATERIAL_KPIS),base_commit='76cbe53',
        elapsed_seconds=time.perf_counter()-start,raw_data_root=str((ROOT/'Dataset').resolve()),
        scope='Known-site development; no unseen, independent expert or outcome validation.')
    (OUT/'provenance').mkdir(exist_ok=True)
    for p in (OUT/'texture.py',Path(__file__),ROOT/'polaron_qc/features.py'):
        (OUT/'provenance'/p.name).write_bytes(p.read_bytes())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report(sites,comp,controls,sens,manifest)
    print(f'Complete: {len(sites)} sites; {len(variants)} modes; {len(comp)} descriptive comparisons.',flush=True)


if __name__=='__main__':main()
