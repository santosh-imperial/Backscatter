"""E25 B02/B05: full-site long-void context and retained widths by image depth.

Read-only raw TIFFs/caches. New outputs use void_* exclusively. Geometry runs
all 31 sites at nominal and ±5 thresholds. Existing E24 global width values are
reused; new depth profiles run nominal on all sites and ±5 on an explicitly
stratified 11-site sensitivity set. No collector-interface KPI is asserted.
"""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/polaron-void-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from polaron_qc import features
from polaron_qc.void_metrics import long_void_context,KPI_NAMES
from analysis.morphology.benchmark_methods import local_width

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'output'
SEED=20261003
N_BOOT=4000
CRACKED={'hzumfsms','0grcilhi','ufdvpb81'}
GREY={'71vgq3fw','kbdh4tri','tuy3zymq','x7u69zsw'}
# Predeclared coverage: all known crack/grey sites, one ordinary per batch, foil candidate.
WIDTH_SENSITIVITY=CRACKED|GREY|{'ffwubibz','3806gxp0','xgj4xftb','epqdaau9'}
EXAMPLES=['hzumfsms','0grcilhi','ufdvpb81','xgj4xftb','71vgq3fw','epqdaau9']
COLORS={'Batch_1':'#2a78d6','Batch_2':'#eb6834','Batch_3':'#1baf7a','grey':'#eda100','cracked':'#8851b2'}
ANALYSED=list(KPI_NAMES)+['long_void_edge_clipped_area_share','crack_frac','void_local_width_d50_px','void_local_width_d90_px','crack_local_width_d50_px','void_width_depth_slope_px','crack_width_depth_slope_px']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,(np.integer,np.floating)):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def width_depth(mask):
    global_values,maps=local_width(mask,return_maps=True)
    h=len(mask);edges=np.linspace(0,h,11).astype(int);rows=[]
    for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        sk=maps['skeleton'][lo:hi];cs=maps['crack_skeleton'][lo:hi];dist=maps['distance'][lo:hi]
        vs=2*dist[sk];vc=2*dist[cs]
        rows.append(dict(depth_bin=k,depth_mid_norm=(k+.5)/10,
                         void_width_d50_px=float(np.median(vs)) if len(vs) else np.nan,
                         void_width_d90_px=float(np.quantile(vs,.9)) if len(vs) else np.nan,
                         crack_width_d50_px=float(np.median(vc)) if len(vc) else np.nan,
                         crack_width_d90_px=float(np.quantile(vc,.9)) if len(vc) else np.nan,
                         n_width_samples=len(vs),n_crack_width_samples=len(vc)))
    return global_values,rows,maps


def slope(rows,key):
    x=np.array([r['depth_mid_norm'] for r in rows]);y=np.array([r[key] for r in rows]);use=np.isfinite(y)
    return float(np.polyfit(x[use],y[use],1)[0]) if use.sum()>=3 else np.nan


def group(row):
    if row.grey_pore:return 'grey'
    if row.site in CRACKED:return 'cracked'
    return row.batch


def overlay(ax,mask,color,alpha=.45):
    rgba=np.zeros((*mask.shape,4));rgba[:,:,:3]=matplotlib.colors.to_rgb(color);rgba[:,:,3]=mask*alpha;ax.imshow(rgba)


def save(fig,name):
    fig.savefig(OUT/name,dpi=145,bbox_inches='tight',facecolor='white');plt.close(fig)


def figures(nominal,profiles,examples):
    panels=[('long_void_internal_area_frac','Internal long-void area / complete frame'),
            ('long_void_y_centroid_norm','Internal long-void image-row centroid'),
            ('long_void_edge_clipped_area_share','Long-void area in clipped components'),
            ('crack_frac','Original long-void fraction (all components)'),
            ('crack_local_width_d50_px','Retained long-void medial width D50 (px)'),
            ('void_width_depth_slope_px','Retained-void width/depth slope (px / height fraction)')]
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    labels=['Batch_1','Batch_2','Batch_3']
    for ax,(key,title) in zip(axes.ravel(),panels):
        for j,b in enumerate(labels):
            selected=nominal[nominal.batch==b].sort_values('site')
            for x,(_,r) in zip(j+np.linspace(-.15,.15,len(selected)),selected.iterrows()):
                if np.isfinite(r[key]):ax.scatter(x,r[key],c=COLORS[group(r)],marker='^' if r.site in CRACKED else 'o',s=30,
                                                 facecolors='none' if r.grey_pore else COLORS[group(r)])
            ordinary=selected[~selected.grey_pore & ~selected.cracked_known]
            ax.plot([j-.19,j+.19],[ordinary[key].median()]*2,color=COLORS[b],lw=2.5)
        ax.set_xticks(range(3),['Batch 1','Batch 2','Batch 3']);ax.set_title(title,loc='left',fontsize=10);ax.grid(axis='y',alpha=.16)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker='^',color=COLORS['cracked'],linestyle='none',label='Known long-void reference sites (n=3)'),
                        Line2D([],[],marker='o',markerfacecolor='none',color=COLORS['grey'],linestyle='none',label='Grey-pore flags (n=4)'),
                        Line2D([],[],color='grey',lw=2,label='Ordinary-site median')],loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('E25: each point is a site; internal area and retained width omit edge-clipped components\nImage-row depth is not collector depth. No material acceptance or adhesion inference.',fontsize=12)
    fig.tight_layout(rect=(0,.05,1,.93));save(fig,'void_site_descriptors.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4.7))
    pp=profiles[profiles.threshold_offset==0]
    groups=[('Batch 1',nominal[nominal.batch=='Batch_1'],'#2a78d6','-'),('Batch 2',nominal[nominal.batch=='Batch_2'],'#eb6834','-'),
            ('Batch 3 ordinary',nominal[(nominal.batch=='Batch_3') & ~nominal.grey_pore & ~nominal.cracked_known],'#1baf7a','-'),
            ('Batch 3 known long voids',nominal[nominal.cracked_known],'#8851b2','--'),('Batch 3 grey flags',nominal[nominal.grey_pore],'#eda100',':')]
    profile_summaries=[]
    for ax,key in zip(axes,['void_width_d50_px','crack_width_d50_px']):
        for name,sub,color,style in groups:
            matrix=pp[pp.site.isin(sub.site)].pivot(index='site',columns='depth_bin',values=key).reindex(columns=range(10)).to_numpy()
            rng=np.random.default_rng(SEED);boot=np.nanmedian(matrix[rng.integers(0,len(matrix),(N_BOOT,len(matrix)))],axis=1)
            med=np.nanmedian(matrix,axis=0);lo,hi=np.nanquantile(boot,[.025,.975],axis=0);depth=np.arange(10)/10+.05
            ax.plot(depth,med,color=color,ls=style,label=f'{name} (n={len(sub)})');ax.fill_between(depth,lo,hi,color=color,alpha=.07)
            for k in range(10):profile_summaries.append(dict(group=name,kpi=key,depth_bin=k,n_sites=len(sub),median=med[k],ci_low=lo[k],ci_high=hi[k]))
        ax.set(xlabel='Normalised image row (top → bottom)',ylabel='Medial-axis width D50 (px)',title='All retained voids' if key.startswith('void') else 'Retained long voids only');ax.grid(alpha=.15)
    axes[1].legend(fontsize=8,loc='best');fig.suptitle('Site medians by image depth; shading is an approximate 95% site-bootstrap interval\nCentreline pixels determine a site measurement and never increase statistical n.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.86));save(fig,'void_width_depth.png')
    pd.DataFrame(profile_summaries).to_csv(OUT/'void_width_depth_summary.csv',index=False)
    fig,axes=plt.subplots(6,2,figsize=(14,13))
    for (site,d),axs in zip(examples.items(),axes):
        for ax in axs:ax.set_axis_off()
        axs[0].imshow(d['raw'],cmap='gray',vmin=0,vmax=255)
        axs[0].set_title(f'{d["batch"]}/{site}: raw full BSE, h={len(d["raw"])} px; removed top/bottom={d["top"]}/{d["bottom"]}',fontsize=9,loc='left')
        axs[1].imshow(d['bse'],cmap='gray',vmin=0,vmax=255)
        clipped=d['details']['long_mask'] & ~d['details']['internal_mask']
        overlay(axs[1],clipped,'#f96791',.52);overlay(axs[1],d['details']['internal_mask'],'#54d7e4',.52)
        cm=d['details']['components']
        for _,r in cm.iterrows():axs[1].plot(r['centroid-1'],r['centroid-0'],'.',color='white',ms=2)
        val=d['metrics'];axs[1].set_title(f'All long voids: pink clipped / cyan internal; internal fraction={val["long_void_internal_area_frac"]:.4f}\nInternal count={val["long_void_internal_n_components"]}; clipped-area share={val["long_void_edge_clipped_area_share"]:.3f}',fontsize=9,loc='left')
    fig.suptitle('Genuine full frames: >500 px major-axis voids are measured before crop selection\nFull/raw versus trimmed coordinates remain explicit. Pink components are absent from width and internal-context KPIs.',fontsize=12)
    fig.subplots_adjust(left=.015,right=.99,bottom=.03,top=.92,hspace=.45,wspace=.06);save(fig,'void_full_site_overlays.png')
    fig,axes=plt.subplots(2,3,figsize=(14,8));records=[]
    for ax,(site,d) in zip(axes.ravel(),examples.items()):
        maps=d['width_maps'];sk=maps['crack_skeleton'];dist=maps['distance']
        if not sk.any():sk=maps['skeleton']
        yy,xx=np.unravel_index(np.argmax(dist*sk),dist.shape);h,w=d['bse'].shape;ch,cw=min(500,h),min(900,w)
        y0=int(np.clip(yy-ch/2,0,h-ch));x0=int(np.clip(xx-cw/2,0,w-cw));win=(slice(y0,y0+ch),slice(x0,x0+cw))
        ax.imshow(d['bse'][win],cmap='gray',vmin=0,vmax=255)
        overlay(ax,(d['details']['long_mask'] & ~d['details']['internal_mask'])[win],'#f96791',.15)
        if maps['retained'][win].any():ax.contour(maps['retained'][win],[.5],colors=['#54d7e4'],linewidths=.7)
        skdisplay=ndi.binary_dilation(sk[win],iterations=1);display=np.ma.masked_where(~skdisplay,ndi.maximum_filter(2*dist[win]*sk[win],size=3))
        ax.imshow(display,cmap='turbo',vmin=0,vmax=max(150,float(2*dist[sk].max()) if sk.any() else 150))
        if sk.any():
            rad=float(dist[yy,xx]);cx,cy=xx-x0,yy-y0;ax.add_patch(Circle((cx,cy),max(rad-1,.5),fill=False,ec='white',lw=.8))
        ax.set_title(f'{d["batch"]}/{site}: raw crop x={x0}, y={y0+d["top"]}\n{cw}×{ch} px; trimmed y={y0}; retained centreline sample',fontsize=9,loc='left');ax.set_axis_off()
        records.append(dict(batch=d['batch'],site=site,x0=x0,y0_trimmed=y0,y0_raw=y0+d['top'],width=cw,height=ch,
                            selected_retained_sample_y_raw=int(yy+d['top']),selected_retained_sample_x=int(xx)))
    fig.suptitle('Width examples: only non-edge-clipped components ≥30 px² are sampled\nCyan = retained boundary; pink = omitted clipped long-void context; coloured centreline thickened for display.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'void_width_examples.png');pd.DataFrame(records).to_csv(OUT/'void_crop_provenance.csv',index=False)
    d=examples['epqdaau9'];y0=len(d['raw'])-220;x0=2500;cw=1500;fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,det in zip(axes,['BSE','ETD','Inlens']):
        raw=features.load_image(ROOT/'Dataset'/d['batch'],d['site'],det);ax.imshow(raw[y0:,x0:x0+cw],cmap='gray',vmin=0,vmax=255)
        if d['bottom']:ax.axhline(220-d['bottom'],color='#f96791',lw=1)
        ax.set_title(f'{det}: raw x={x0}, y={y0}, {cw}×220 px',fontsize=9,loc='left');ax.set_axis_off()
    fig.suptitle('epqdaau9: candidate visible lower collector/stitch band, reviewed as raw aligned channels\nPink is the BSE trim boundary. Identity, collector direction and adjacent-gap geometry are unconfirmed; no interface KPI computed.',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.78));save(fig,'void_collector_candidate.png')


def comparisons(nominal,variants):
    output=[]
    comparisons=[('Batch_3','Batch_1'),('Batch_3','Batch_2'),('Batch_1','Batch_2')]
    for view in ['all_sites','exclude_grey','ordinary_reference']:
        for ref,batch in comparisons:
            r=nominal[nominal.batch==ref];b=nominal[nominal.batch==batch]
            if view!='all_sites':r=r[~r.grey_pore];b=b[~b.grey_pore]
            if view=='ordinary_reference' and ref=='Batch_3':r=r[~r.cracked_known]
            for k in ANALYSED:
                rv=r.dropna(subset=[k]);bv=b.dropna(subset=[k]);x=rv[k].to_numpy();y=bv[k].to_numpy()
                if not len(x) or not len(y):continue
                rng=np.random.default_rng(SEED);d=np.median(y[rng.integers(0,len(y),(N_BOOT,len(y)))],axis=1)-np.median(x[rng.integers(0,len(x),(N_BOOT,len(x)))],axis=1);lo,hi=np.quantile(d,[.025,.975])
                bands=variants[variants.site.isin(set(rv.site)|set(bv.site))].groupby('site')[k].agg(['min','max','count'])
                complete=set(bands[bands['count']==3].index)
                envelope=(np.nan,np.nan)
                if set(rv.site).issubset(complete) and set(bv.site).issubset(complete):
                    envelope=(float(bands.loc[bv.site,'min'].median()-bands.loc[rv.site,'max'].median()),float(bands.loc[bv.site,'max'].median()-bands.loc[rv.site,'min'].median()))
                output.append(dict(view=view,reference=ref,batch=batch,kpi=k,n_ref=len(x),n_batch=len(y),reference_median=np.median(x),batch_median=np.median(y),median_delta=np.median(y)-np.median(x),delta_ci_low=lo,delta_ci_high=hi,threshold_delta_low=envelope[0],threshold_delta_high=envelope[1],n_boot=N_BOOT))
    # Known morphological subgroup contrast is descriptive, not a clean independent validation.
    for name,chosen in [('known_long_voids',nominal.cracked_known),('grey_pore_flags',nominal.grey_pore)]:
        r=nominal[(nominal.batch=='Batch_3') & ~nominal.grey_pore & ~nominal.cracked_known];b=nominal[chosen]
        for k in ANALYSED:
            x=r[k].dropna().to_numpy();y=b[k].dropna().to_numpy()
            if not len(x) or not len(y):continue
            rng=np.random.default_rng(SEED);d=np.median(y[rng.integers(0,len(y),(N_BOOT,len(y)))],axis=1)-np.median(x[rng.integers(0,len(x),(N_BOOT,len(x)))],axis=1);lo,hi=np.quantile(d,[.025,.975])
            output.append(dict(view='reference_subgroups',reference='Batch_3_ordinary',batch=name,kpi=k,n_ref=len(x),n_batch=len(y),reference_median=np.median(x),batch_median=np.median(y),median_delta=np.median(y)-np.median(x),delta_ci_low=lo,delta_ci_high=hi,threshold_delta_low=np.nan,threshold_delta_high=np.nan,n_boot=N_BOOT))
    table=pd.DataFrame(output);table.to_csv(OUT/'void_comparisons.csv',index=False);return table


def acquisition_checks(nominal):
    variables=['H_trim','bse_p1','bse_empty_bin_frac','etd_boundary_sharpness']
    rows=[];predict=[]
    for name,subset in [('all',nominal),('exclude_grey',nominal[~nominal.grey_pore]),('ordinary_reference',nominal[~nominal.grey_pore & ~nominal.cracked_known])]:
        for subgroup,frame in [('pooled',subset)]+[(b,subset[subset.batch==b]) for b in sorted(subset.batch.unique())]:
            for k in ANALYSED:
                for v in variables:
                    pair=frame[[k,v]].dropna()
                    if len(pair)>=4 and pair[k].nunique()>1 and pair[v].nunique()>1:
                        rho,p=spearmanr(pair[k],pair[v]);rows.append(dict(view=name,group=subgroup,kpi=k,acquisition_variable=v,n_sites=len(pair),spearman_rho=rho,p_descriptive=p))
        if name=='exclude_grey':
            for k in KPI_NAMES:
                frame=subset[[k]+variables].dropna();x=frame[variables].to_numpy();y=frame[k].to_numpy();pred=np.full(len(y),np.nan)
                for i in range(len(y)):
                    keep=np.arange(len(y))!=i;model=make_pipeline(StandardScaler(),Ridge(alpha=1.));model.fit(x[keep],y[keep]);pred[i]=model.predict(x[i:i+1])[0]
                r2=1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2) if np.var(y)>0 else np.nan
                predict.append(dict(kpi=k,n_sites=len(y),covariates=';'.join(variables),ridge_alpha=1.,acquisition_loo_r2=r2))
    pd.DataFrame(rows).to_csv(OUT/'void_acquisition_correlations.csv',index=False);pd.DataFrame(predict).to_csv(OUT/'void_acquisition_predictability.csv',index=False)


def write_findings(nominal,comparison,manifest):
    chosen=nominal[nominal.site.isin(EXAMPLES)][['batch','site','crack_frac','long_void_internal_area_frac','long_void_y_centroid_norm','long_void_internal_n_components','long_void_edge_clipped_area_share','crack_local_width_d50_px','band_top','band_bottom']]
    subgroups=comparison[comparison.view=='reference_subgroups']
    shortlist=comparison[(comparison.view=='exclude_grey') & comparison.kpi.isin(list(KPI_NAMES)+['crack_local_width_d50_px'])]
    summary={'n_sites':len(nominal),'n_threshold_variants':len(nominal)*3,'n_width_sensitivity_sites':len(WIDTH_SENSITIVITY),
             'n_pore_mode_resolved':int(nominal.pore_mode_resolved.sum()),'n_grey_flags':int(nominal.grey_pore.sum()),
             'internal_centroid_available_sites':int(nominal.long_void_y_centroid_norm.notna().sum()),
             'collector_interface_confirmed_sites':0,'example_sites':chosen.to_dict(orient='records'),
             'quality_comparisons':shortlist.to_dict(orient='records'),'subgroup_comparisons':subgroups.to_dict(orient='records')}
    (OUT/'void_summary.json').write_text(json.dumps(safe(summary),indent=2)+'\n')
    md=f'''# E25: B02/B05 long-void spatial context and image-depth widths

Measured from all 31 full/raw BSE sites, with the current band trimming and per-site pore thresholds. Structural context is repeated at threshold −5, nominal and +5. The existing E24 global widths are reused and checked against new nominal medial-axis maps. New width-by-depth profiles cover every site at nominal thresholds; ±5 profiles cover the predeclared 11-site set (all three known long-void sites, all four grey flags, one ordinary control per batch and epqdaau9). Components, centreline pixels and depth bins never become independent statistical n.

## Measurement scope and coverage

- Existing `crack_frac` includes every component ≥30 px² with major axis >500 px, including edge-clipped ones. It is reproduced before any exclusion.
- `long_void_internal_area_frac` measures only components touching no frame edge, divided by the whole trimmed image area. Observed absence is valid zero; it is not acceptance.
- `long_void_y_centroid_norm` measures only those fully observable components, with image top=0 and bottom=1; no internal component gives NaN. Available on {summary['internal_centroid_available_sites']}/31 sites.
- `long_void_edge_clipped_area_share` is an observability diagnostic, not a material KPI. Retained local widths omit these same edge-clipped components; they do not characterise all visible long voids.
- All 31 cached `pore_mode_resolved` entries are False. This invariant cannot select a reliable subset. The data-derived grey-pore flag identifies four preparation/contrast cases, excluded in the quality view but displayed separately. Every segmentation remains an unreviewed prediction.
- Image-row coordinates are not collector depth or calibrated coating thickness. Only epqdaau9 is shown as a candidate visible lower foil/stitch band; its identity and orientation remain unconfirmed. **No collector-interface gap fraction or width is computed.** Raw views are retained for that review.

## Actual selected-site measurements

{chosen.to_markdown(index=False,floatfmt='.4f')}

## Quality-view batch comparisons

All differences are incoming minus reference site medians. Intervals are approximate 95% site-bootstrap intervals ({N_BOOT} draws). Threshold envelopes are separate measurement-sensitivity ranges, not confidence intervals. No hypothesis test or new alarm threshold is adopted.

{shortlist[['reference','batch','kpi','n_ref','n_batch','median_delta','delta_ci_low','delta_ci_high','threshold_delta_low','threshold_delta_high']].to_markdown(index=False,floatfmt='.4f')}

## Existing reference subgroups (descriptive)

These sites were selected as known morphological examples during prior exploratory analysis; this comparison is not independent validation of a defect label or a new batch classifier.

{subgroups[subgroups.kpi.isin(list(KPI_NAMES)+['crack_frac','crack_local_width_d50_px'])][['batch','kpi','n_ref','n_batch','median_delta','delta_ci_low','delta_ci_high']].to_markdown(index=False,floatfmt='.4f')}

## API and next use

`polaron_qc.void_metrics.long_void_context(pore_mask, *, major_axis_cutoff_px=500, min_area_px=30, n_depth_bins=10, return_details=False)` returns two secondary geometry KPIs plus separate observability diagnostics. `return_details=True` additionally returns component records and masks for audit. The default has no large maps. `KPI_NAMES` and `KPI_DEFINITIONS` define units and interpretation. Width/skeleton profiling remains in this standalone analysis; it is not added to the core extractor.

Use the internal burden and image-row centroid as descriptive context, retaining original crack fraction and clipped-area coverage. Expert phase/void review is still needed; a broad or narrow 2-D section does not establish a 3-D pore throat, wetting rate, adhesion failure or electrical disconnection. Fresh graphite–Si/SiOx is confirmed; no cycling-induced damage is inferred.

## Outputs and verification

See `void_sites.csv`, `void_components.csv.gz`, `void_depth_profiles.csv`, `void_width_depth_profiles.csv`, `void_width_depth_summary.csv`, `void_comparisons.csv`, `void_acquisition_correlations.csv`, `void_acquisition_predictability.csv`, `void_crop_provenance.csv`, `void_summary.json` and `void_manifest.json`. Figures: `void_full_site_overlays.png`, `void_width_examples.png`, `void_width_depth.png`, `void_site_descriptors.png`, `void_collector_candidate.png`.

Input/cache/raw hashes and parameters are recorded. Original crack fractions and E24 global widths agree with the saved values; depth shares sum to one where defined; raw/trimmed offsets and crop bounds are checked. Six core geometry/edge/absence/flip/connectivity tests are supplied. Acquisition correlations and a held-out-site fixed ridge model are exploratory confound screens; poor predictability cannot prove an acquisition-free feature. No QC verdict or primary threshold changes in this analysis.

Checklist applied: C01–C06, C09, C13–C18, C21–C24 and C28. Independent expert annotation, collector-interface review, specimen grouping and physical calibration remain unresolved.
'''
    (OUT/'void_findings.md').write_text(md)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'analysis/morphology/output/morphology_sites.csv',ROOT/'analysis_cache/site_features.csv',ROOT/'analysis/morphology/benchmark/local_width_sites.csv',Path(__file__),ROOT/'polaron_qc/void_metrics.py',ROOT/'analysis/morphology/benchmark_methods.py']
    sources=pd.read_csv(paths[0]);widths=pd.read_csv(paths[2]);input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    raw_hashes={};site_rows=[];components=[];profiles=[];width_profiles=[];examples={}
    for i,r in enumerate(sources.itertuples(index=False)):
        path=Path(features._image_path(ROOT/'Dataset'/r.batch,r.site,'BSE'));raw_hashes[str(path.relative_to(ROOT))]=sha(path)
        raw=features.load_image(ROOT/'Dataset'/r.batch,r.site);top,bottom=features.bright_bands(raw);bse=raw[top:len(raw)-bottom if bottom else None];sm=features.smooth_bse(bse)
        nominal_maps=None
        for offset in [-5,0,5]:
            mask=ndi.binary_opening(sm<r.th_lo+offset,iterations=1)
            measured,details=long_void_context(mask,return_details=True)
            cm=details['components'];allfrac=float(cm.area.sum()/mask.size)
            if offset==0:assert np.isclose(allfrac,r.crack_frac,atol=1e-12),r.site
            cached=widths[(widths.site==r.site)&(widths.threshold_offset==offset)].iloc[0]
            row=dict(batch=r.batch,site=r.site,threshold_offset=offset,H_raw=len(raw),H_trim=len(bse),W=bse.shape[1],band_top=top,band_bottom=bottom,
                     grey_pore=bool(r.grey_pore),cracked_known=r.site in CRACKED,bright_low_contrast=bool(r.bright_low_contrast),pore_mode_resolved=bool(r.pore_mode_resolved),
                     bse_p1=r.bse_p1,bse_empty_bin_frac=r.bse_empty_bin_frac,etd_boundary_sharpness=r.etd_boundary_sharpness,
                     pore_frac=float(mask.mean()),crack_frac=allfrac,collector_interface_confirmed=False,
                     collector_candidate=(r.site=='epqdaau9'),collector_adjacent_gap_fraction=np.nan,collector_adjacent_gap_width_px=np.nan,**measured)
            for k in ['void_local_width_d50_px','void_local_width_d90_px','crack_local_width_d50_px','n_retained_voids','n_crack_width_voids','retained_void_area_frac','excluded_void_area_frac']:row[k]=cached[k]
            if offset==0 or r.site in WIDTH_SENSITIVITY:
                globals_new,wrows,maps=width_depth(mask)
                for k in ['void_local_width_d50_px','void_local_width_d90_px','crack_local_width_d50_px','retained_void_area_frac','excluded_void_area_frac']:
                    assert np.isclose(globals_new[k],cached[k],equal_nan=True,atol=1e-10),(r.site,offset,k)
                row['void_width_depth_slope_px']=slope(wrows,'void_width_d50_px');row['crack_width_depth_slope_px']=slope(wrows,'crack_width_d50_px')
                width_profiles += [dict(batch=r.batch,site=r.site,threshold_offset=offset,grey_pore=bool(r.grey_pore),cracked_known=r.site in CRACKED,**x) for x in wrows]
                if offset==0 and r.site in EXAMPLES:
                    nominal_maps=maps
            else:
                row['void_width_depth_slope_px']=np.nan;row['crack_width_depth_slope_px']=np.nan
            site_rows.append(row)
            cm=cm.assign(batch=r.batch,site=r.site,threshold_offset=offset,band_top=top,band_bottom=bottom)
            cm['centroid_y_raw']=cm['centroid-0']+top;cm['bbox_y0_raw']=cm['bbox-0']+top;cm['bbox_y1_raw']=cm['bbox-2']+top
            cm['candidate_lower_band_distance_px']=cm['edge_bottom_px'] if r.site=='epqdaau9' and bottom else np.nan
            components.append(cm)
            for k,(lo,hi) in enumerate(zip(details['depth_edges'][:-1],details['depth_edges'][1:])):
                profiles.append(dict(batch=r.batch,site=r.site,threshold_offset=offset,depth_bin=k,y0_trimmed=int(lo),y1_trimmed=int(hi),y0_raw=int(lo+top),y1_raw=int(hi+top),
                                     long_void_area_share=details['long_void_depth_area_share'][k],internal_long_void_area_share=details['internal_long_void_depth_area_share'][k],
                                     pore_area_frac=float(mask[lo:hi].mean())))
            if offset==0 and r.site in EXAMPLES:
                examples[r.site]=dict(batch=r.batch,raw=raw,bse=bse,top=top,bottom=bottom,details=details,metrics=measured,width_maps=nominal_maps)
        print(f'E25 full-site context {i+1}/{len(sources)} {r.batch}/{r.site}',flush=True)
    variants=pd.DataFrame(site_rows);nominal=variants[variants.threshold_offset==0].copy();profile=pd.DataFrame(profiles);wprofile=pd.DataFrame(width_profiles)
    variants.to_csv(OUT/'void_sites.csv',index=False);pd.concat(components,ignore_index=True).to_csv(OUT/'void_components.csv.gz',index=False,compression='gzip')
    profile.to_csv(OUT/'void_depth_profiles.csv',index=False);wprofile.to_csv(OUT/'void_width_depth_profiles.csv',index=False)
    comparison=comparisons(nominal,variants);acquisition_checks(nominal)
    examples={s:examples[s] for s in EXAMPLES};figures(nominal,wprofile,examples)
    assert len(variants)==93 and len(profile)==930 and len(wprofile)==530
    for _,sub in profile.groupby(['site','threshold_offset']):
        if sub.long_void_area_share.notna().any():assert np.isclose(sub.long_void_area_share.sum(),1)
        if sub.internal_long_void_area_share.notna().any():assert np.isclose(sub.internal_long_void_area_share.sum(),1)
    assert variants.collector_adjacent_gap_width_px.isna().all() and not variants.collector_interface_confirmed.any()
    for path,h in input_hashes.items():assert sha(ROOT/path)==h,'Input changed during measurement: '+path
    for path,h in raw_hashes.items():assert sha(ROOT/path)==h,'Raw input changed: '+path
    manifest=dict(experiment='E25',seed=SEED,n_sites=31,n_threshold_variants=93,n_bootstrap=N_BOOT,
                  width_depth_nominal_sites=31,width_depth_sensitivity_sites=sorted(WIDTH_SENSITIVITY),n_width_depth_site_variants=53,
                  input_sha256=input_hashes,raw_BSE_sha256=raw_hashes,inputs_unchanged=True,
                  coordinates='x unchanged; trimmed y=raw y-band_top; bbox upper bounds exclusive; no four-pixel crop',
                  assumptions=['fresh uncycled graphite–Si/SiOx confirmed','pore segmentation unreviewed','image row is not collector depth','pixel scale unverified','site/specimen independence unresolved','grey flag sensitivity; pore_mode_resolved is False on every cached site'],
                  collector_interface_confirmed_sites=0,primary_KPIs_changed=False,geometry_tests=6)
    (OUT/'void_manifest.json').write_text(json.dumps(safe(manifest),indent=2)+'\n');write_findings(nominal,comparison,manifest)
    print('E25 output contracts passed; files saved under '+str(OUT),flush=True)

if __name__=='__main__':main()
