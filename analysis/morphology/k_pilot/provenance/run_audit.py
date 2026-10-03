"""E30K: one fixed known-site audit; no production input changes."""
from pathlib import Path
import base64
import hashlib
import html
import json
import os
import time
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/polaron-k-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from polaron_qc import features,PRIMARY_KPIS,MATERIAL_KPIS,BATCH_COLORS,UNSEEN_COLOR
from analysis.morphology.k_pilot.descriptors import *

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SEED=20261003
N_BOOT=4000
VIEWS=('phase_usable','quality_matched','ordinary_reference')
ACQ=('bright_sep','bse_p1','bse_std','bse_empty_bin_frac','etd_boundary_sharpness','H')
BASE=('bright_frac','pore_frac','bright_d50','bright_circ','bright_solidity','bright_aspect_aw',
      'bright_nn_mean_px','local_bright_std_512px','graph_edge_axial_strength','graph_node_density_per_mpx')
EXAMPLES=(('Batch_2','3806gxp0'),('Batch_1','4ih2ggld'),('Batch_3','71vgq3fw'),('Batch_3','hzumfsms'))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
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
        for view in ('phase_usable','quality_matched'):
            t = sites.loc[usable(sites,key,view)]
            groups = [('pooled',t)] + [(b,t[t.batch.eq(b)]) for b in sorted(t.batch.unique())]
            if view == 'quality_matched':
                groups += [('Batch_3_ordinary',sites.loc[usable(sites,key,'ordinary_reference') & sites.batch.eq('Batch_3')])]
            for control, covs in (('acquisition',ACQ),('existing_geometry_loading',BASE)):
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
        for view in ('phase_usable','quality_matched'):
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




def overlay(raw,bright,pore,top,row,nodes,pairs):
    h,w=raw.shape;height,width=min(h,768),min(w,1280);y,x=(h-height)//2,(w-width)//2
    fig,axes=plt.subplots(2,3,figsize=(15,8.2))
    for ax in axes[0,:2]:
        ax.imshow(raw[y:y+height,x:x+width],cmap='gray',vmin=0,vmax=255)
        ax.set_axis_off()
    axes[0,0].set_title('Fixed central BSE crop · full-frame statistics')
    rgba=np.zeros((height,width,4))
    for mask,color in ((bright,(.1,.8,.9)),(pore,(1,.25,.4))):
        use=mask[y:y+height,x:x+width];rgba[use,:3]=color;rgba[use,3]=.45
    axes[0,1].imshow(rgba)
    for _,r in nodes.iterrows():
        if y<=r['centroid-0']<y+height and x<=r['centroid-1']<x+width:
            axes[0,1].add_patch(Rectangle((r['bbox-1']-x,r['bbox-0']-y),r['bbox-3']-r['bbox-1'],r['bbox-2']-r['bbox-0'],fill=False,edgecolor='#ffca58',lw=.7))
    for lag,color in ((64,'#ffca58'),(256,'white')):
        axes[0,1].annotate('',xy=(80+lag,height-60),xytext=(80,height-60),arrowprops={'arrowstyle':'->','color':color})
        axes[0,1].annotate('',xy=(80,height-60-lag),xytext=(80,height-60),arrowprops={'arrowstyle':'->','color':color})
    axes[0,1].set_title('Bright cyan / void red; eligible objects gold')
    labels=[];xx=[];yy=[]
    for kind in ('auto','cross'):
        for lag in LAGS:
            t=pairs[(pairs.kind==kind)&pairs.lag_px.abs().eq(lag)]
            labels.append(f'{kind} {lag}px');xx.append(t.loc[t.axis=='x','rho'].mean());yy.append(t.loc[t.axis=='y','rho'].mean())
    pos=np.arange(4)
    axes[0,2].bar(pos-.18,xx,.36,label='x',color='#2a78d6');axes[0,2].bar(pos+.18,yy,.36,label='y',color='#eb6834')
    axes[0,2].set(xticks=pos,xticklabels=labels,ylabel='Finite-domain binary correlation',ylim=(-1,1),title='Full-frame phase arrangement; KPI = x − y')
    axes[0,2].axhline(0,color='grey',lw=.5);axes[0,2].legend()
    shape,eligible=shape_summary(nodes)
    for ax,col,title,quantiles in zip(axes[1],('shape_aspect','solidity','shape_circularity'),
                    ('Aspect-ratio spread','Solidity lower tail','Circularity spread'),((.25,.75),(.1,),(.25,.75))):
        vals=eligible[col].to_numpy(float);ax.hist(vals,bins=24,color='#7c98b7')
        for q in quantiles:
            if len(vals):ax.axvline(np.quantile(vals,q),color='#ce4343',ls='--',label=f'Q{int(q*100)}')
        ax.set(xlabel=col,ylabel='Eligible components (coverage)',title=title)
        ax.legend(fontsize=8)
    fig.suptitle(f'{row.batch}/{row.site} · bright_low_contrast={row.bright_low_contrast}; grey_pore={row.grey_pore}',fontsize=12)
    fig.text(.02,.012,'Gold boxes include only unclipped bright objects ≥50 px²; arrows show fixed x/y lags. All values use the full trimmed frame. Components/pixel pairs are coverage, not n. Masks and image axes remain expert-unreviewed.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.97));name=f'overlay_{row.batch}_{row.site}.png'
    fig.savefig(OUT/name,dpi=130);plt.close(fig)
    return dict(batch=row.batch,site=row.site,image=name,x=x,y_trimmed=y,y_raw=y+top,width=width,height=height)

def report(sites,comp,controls,sens,manifest):
    fig,axes=plt.subplots(3,3,figsize=(15,10));rng=np.random.default_rng(SEED)
    for ax,key in zip(axes.flat,KEYS):
        for i,b in enumerate(('Batch_1','Batch_2','Batch_3')):
            t=sites[sites.batch.eq(b)];good=usable(t,key)
            ax.scatter(i+rng.uniform(-.13,.13,len(t)),t[key],s=18,color=BATCH_COLORS.get(b,UNSEEN_COLOR))
            if good.any():ax.hlines(t.loc[good,key].median(),i-.2,i+.2,color='black')
        ax.set(xticks=range(3),xticklabels=['B1','B2','B3'],title=key, ylabel=DEFINITIONS[key][0]);ax.tick_params(labelsize=8)
    for ax in axes.flat[len(KEYS):]:ax.set_visible(False)
    fig.suptitle('Known sites · dots include flagged observations; bars use phase-usable sites',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'site_descriptors.png',dpi=120);plt.close(fig)
    nominal=comp[comp['mode'].eq('nominal')]
    matched=nominal[nominal.view.eq('quality_matched')]
    finite=matched.dropna(subset=['ci_low','ci_high'])
    beyond=finite[(finite.ci_low>0)|(finite.ci_high<0)]
    lines=['# Task K / E30K — shape variability and directional phase association','',
      'Retain the measured geometry and image explanations for review. All seven material/QC uses remain deferred; phase truth, section representativeness and specimen independence are unresolved. No feature is selected or promoted from the strongest known-batch separation.','',
      f'One fixed run: {len(sites)} sites, {len(comp)} descriptive comparison rows. {len(beyond)}/{len(finite)} nominal quality-matched intervals exclude zero; these are correlated exploratory intervals without a multiplicity-adjusted test. A null interval is not equivalence.','',
      '## Nominal quality-matched comparisons','',
      '| KPI | incoming − reference | usable n (reference/incoming) | median difference | site-bootstrap 95% interval | deletion range |',
      '|---|---|---|---|---|---|']
    for r in matched.itertuples():lines.append(f'| `{r.kpi}` | {r.batch} − {r.reference} | {r.n_reference}/{r.n_batch} | {r.median_difference:.6g} | [{r.ci_low:.6g}, {r.ci_high:.6g}] | [{r.deletion_min:.6g}, {r.deletion_max:.6g}] |')
    lines+=['','## Paired method sensitivity','',
      'Shape descriptors vary under ±5 intensity-level mask thresholds and a 100 px² object floor. Raster phase descriptors use the full mask, so their floor100 copies are identical by design; this is not an independent robustness result. No joint floor/threshold interaction was tested.','',
      'Pixel-pair marginals are re-estimated in each signed overlapping domain. There is no circular wrap or FFT periodic boundary. Correlations describe finite 2-D arrangement, not contact, chemistry, transport or confirmed collector/through-thickness direction.','',
      '## Control qualification','',
      'Two fixed ridge controls use fold-local median imputation/scaling and whole-site LOO. Negative R² means these linear controls predict poorly at this n; it does not prove novel material information or acquisition invariance. Count-weighted shape quantiles describe a different population from existing area-weighted means.','',
      '| KPI | view | acquisition LOO R² | existing-geometry/loading LOO R² |','|---|---|---|---|']
    for key in KEYS:
        for view in ('phase_usable','quality_matched'):
            t=controls[controls.kpi.eq(key)&controls.view.eq(view)]
            vals=[t.loc[t.control.eq(c),'loo_r2'].iloc[0] if len(t[t.control.eq(c)]) else np.nan for c in ('acquisition','existing_geometry_loading')]
            lines.append(f'| `{key}` | {view} | {vals[0]:.4g} | {vals[1]:.4g} |')
    lines+=['','## Provenance and validation limits','',
      'Shape geometry is a declared warm reuse of E26G verified full-frame component tables at the selected settings; mask phase-pair counts are freshly reconstructed from raw BSE. Original E26G raw/source hashes were checked before reuse. No originals or historical caches were overwritten. Raw replay of prespecified nominal examples checks geometry binding, not chemical identity or expert accuracy.','',
      'At least 20 eligible bright components are required for count quantiles. Finite pair domains require at least 10000 pixels and at least 100 pixels of each phase and complement. These are computational coverage gates, not statistical validation. Missing quality flags abstain; grey-pore exclusion is required for cross-phase descriptors.','',
      'Nonlinear depth structure is deferred: no confirmed collector direction or independently reviewed phase/section labels. Independent annotation and practical tolerances remain the next gates. Known batches were explored previously; no unseen batch was used.','',
      'Review: C01–C07/C09/C11/C13/C14/C16/C17/C21–C23/C28–C30.','']
    (OUT/'findings.md').write_text('\n'.join(lines).rstrip()+'\n')
    image=lambda name:'<img style="width:100%;height:auto" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'
    defs=''.join(f'<li><strong>{html.escape(k)}</strong> ({u}): {html.escape(d)}</li>' for k,(u,d) in DEFINITIONS.items())
    page='<!doctype html><html><head><meta charset="utf-8"><title>Task K morphology audit</title><style>body{font:16px system-ui;line-height:1.55;margin:30px auto;max-width:1300px;padding:0 20px;color:#183044;background:#f3f6fa}table{font-size:12px;border-collapse:collapse}td,th{padding:6px;border:1px solid #ccd6df}details{padding:15px;background:white;margin:20px 0}.scroll{overflow:auto}</style></head><body><h1>Shape variability and directional phase association</h1><p>Exploratory known-site audit · seven measured descriptors · all material/QC use deferred · expert review pending.</p><ul>'+defs+'</ul>'
    page+='<h2>Measured site distributions</h2>'+image('site_descriptors.png')
    page+='<h2>Image reference and full-site distributions</h2>'+''.join('<details open><summary>'+c['batch']+'/'+c['site']+'</summary>'+image(c['image'])+'</details>' for c in manifest['crops'])
    page+='<h2>Paired threshold/floor sensitivity</h2><div class="scroll">'+sens.to_html(index=False,float_format=lambda x:f'{x:.4g}')+'</div>'
    page+='<h2>Findings and honest limits</h2><pre style="white-space:pre-wrap">'+html.escape('\n'.join(lines))+'</pre></body></html>'
    (OUT/'report.html').write_text(page)

def main():
    start=time.perf_counter();protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['keys']==list(KEYS) and protocol['lags_px']==list(LAGS)
    metadata_path=ROOT/'analysis/ml_options/e_graph/sites.csv';nodes_path=ROOT/'analysis/ml_options/e_graph/nodes.csv.gz'
    morph_path=ROOT/'analysis/morphology/output/morphology_sites.csv'
    metadata=pd.read_csv(metadata_path).sort_values(['batch','site'])
    morph=pd.read_csv(morph_path);metadata=metadata.merge(morph[['batch','site','bright_aspect_aw','bright_nn_mean_px']],on=['batch','site'],validate='one_to_one')
    assert len(metadata)==31 and set(metadata.batch)=={'Batch_1','Batch_2','Batch_3'}
    graph_manifest=json.loads((ROOT/'analysis/ml_options/e_graph/manifest.json').read_text())
    for p,h in graph_manifest['inputs'].items():assert sha(ROOT/p)==h,p
    nodes=pd.read_csv(nodes_path)
    sources=[metadata_path,nodes_path,morph_path,ROOT/'analysis/ml_options/e_graph/manifest.json',
             ROOT/'polaron_qc/features.py',OUT/'descriptors.py',Path(__file__),OUT/'protocol.json',OUT/'protocol_registration.json']
    sources += [ROOT/'Dataset'/r.batch/f'img_{r.site}_BSE.tif' for r in metadata.itertuples()]
    before={str(p.relative_to(ROOT)):sha(p) for p in sources}
    records=[];all_pairs=[];crops=[]
    for row in metadata.itertuples():
        raw=features.load_image(ROOT/'Dataset'/row.batch,row.site,'BSE');top,bottom=features.bright_bands(raw)
        raw=raw[top:len(raw)-bottom if bottom else None];assert raw.shape==(row.H,row.W)
        sm=features.smooth_bse(raw);phase_cache={}
        for offset in (-5,0,5):
            bright=ndi.binary_opening(sm>row.th_hi+offset,iterations=1)
            pore=ndi.binary_opening(sm<row.th_lo+offset,iterations=1)
            values,pairs=phase_summary(bright,pore);phase_cache[offset]=values
            all_pairs.append(pairs.assign(batch=row.batch,site=row.site,threshold_offset=offset))
            if offset==0:
                assert np.isclose(bright.mean(),row.bright_frac,atol=1e-10)
                assert np.isclose(pore.mean(),row.pore_frac,atol=1e-10)
                if (row.batch,row.site) in EXAMPLES:
                    nt=nodes[nodes.batch.eq(row.batch)&nodes.site.eq(row.site)&nodes.threshold_offset.eq(0)&nodes.area_floor_px2.eq(50)]
                    crops.append(overlay(raw,bright,pore,top,row,nt,pairs))
        for mode,(offset,floor) in SETTINGS.items():
            nt=nodes[nodes.batch.eq(row.batch)&nodes.site.eq(row.site)&nodes.threshold_offset.eq(offset)&nodes.area_floor_px2.eq(floor)]
            shape,_=shape_summary(nt)
            records.append(row._asdict()|dict(mode=mode,threshold_offset=offset,area_floor_px2=floor,trim_top=top,trim_bottom=bottom)|shape|phase_cache[offset])
        print(f'{row.batch}/{row.site}: fixed shape reuse and three raw-mask phase variants',flush=True)
    variants=pd.DataFrame(records);sites=variants[variants['mode'].eq('nominal')].copy()
    comparisons=compare(variants);correlations,predictions=nuisance_controls(sites);sens=sensitivity(sites,variants)
    for name,t in (('sites',sites),('variants',variants),('pair_counts',pd.concat(all_pairs,ignore_index=True)),
                   ('comparisons',comparisons),('correlations',correlations),('control_predictions',predictions),('sensitivity_summary',sens)):
        t.to_csv(OUT/f'{name}.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in before.items()),'Inputs changed during extraction'
    manifest=dict(experiment='E30K',task='K',version=protocol['version'],date='2026-10-03',
                  base_commit=protocol['base_commit'],inputs=before,inputs_unchanged=True,
                  shape_geometry_reuse='Verified E26G cached unclipped components; no fresh full-site shape extraction',
                  raster_measurement='Fresh raw BSE full-trimmed-frame signed-lag pair counts',
                  n_sites=len(sites),n_variants=len(variants),n_comparisons=len(comparisons),n_boot=N_BOOT,seed=SEED,
                  primary_kpis=list(PRIMARY_KPIS),material_kpis=list(MATERIAL_KPIS),
                  elapsed_seconds=time.perf_counter()-start,crops=crops,raw_data_root=str((ROOT/'Dataset').resolve()),
                  scope='Known-site development; expert, specimen, phase and unseen validity unresolved; excluded from all production inputs')
    (OUT/'provenance').mkdir(exist_ok=True)
    for source in (OUT/'descriptors.py',Path(__file__),ROOT/'polaron_qc/features.py'):
        (OUT/'provenance'/source.name).write_bytes(source.read_bytes())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report(sites,comparisons,predictions,sens,manifest)
    print(f'Complete: {len(sites)} sites, {len(variants)} modes, {len(comparisons)} comparisons.',flush=True)

if __name__=='__main__':main()
