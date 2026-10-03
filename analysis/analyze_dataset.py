"""Reproducible, label-free SEM audit. Requires numpy, pandas and Pillow.
Run from workspace root using the bundled Python runtime. No source images altered.
"""
from pathlib import Path
import hashlib, json, math, html
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis'
ASSETS = OUT / 'assets'
ASSETS.mkdir(exist_ok=True)
CHANNELS = ['BSE', 'ETD', 'SE', 'Inlens']
rows, histograms, smalls = [], {}, {}

def otsu(h):
    p = h / h.sum(); w = np.cumsum(p); m = np.cumsum(p * np.arange(256))
    v = (m[-1]*w-m)**2 / np.maximum(w*(1-w), 1e-12)
    return int(np.argmax(v[:-1]))

for path in sorted((ROOT/'Dataset').glob('*/*.tif')):
    batch = path.parent.name; field, channel = path.stem.rsplit('_',1)
    with Image.open(path) as im:
        tags = im.tag_v2
        arr = np.asarray(im)
        rgb_delta = int(np.max(np.abs(arr[:,:,0].astype('int16')-arr[:,:,1]))) if arr.ndim==3 else 0
        if arr.ndim==3:
            rgb_delta=max(rgb_delta,int(np.max(np.abs(arr[:,:,0].astype('int16')-arr[:,:,2]))))
        chromatic=(arr[:,:,0]!=arr[:,:,1]) | (arr[:,:,0]!=arr[:,:,2]) if arr.ndim==3 else np.zeros(arr.shape,dtype=bool)
        chromatic_fraction=float(chromatic.mean())
        interior_chromatic_fraction=float(chromatic[4:-4,4:-4].mean())
        # Exclude a fixed four-pixel edge, including observed colored export borders.
        gray = im.crop((4,4,im.width-4,im.height-4)).convert('L')
        h = np.bincount(np.asarray(gray).ravel(), minlength=256)
        thumb = gray.copy(); thumb.thumbnail((1400,1400),Image.Resampling.LANCZOS)
        a = np.asarray(thumb).astype(float)/255
        histograms[path.name] = h; smalls[(batch,field,channel)] = a
        thumb.convert('RGB').save(ASSETS/(path.stem+'.jpg'),quality=85)
        probs = h/h.sum(); q = np.searchsorted(np.cumsum(probs),[.01,.05,.25,.5,.75,.95,.99])
        gx=np.diff(a,axis=1); gy=np.diff(a,axis=0)
        gradient=float((np.mean(gx**2)+np.mean(gy**2))**.5)
        tiles=[a[y:y+a.shape[0]//3,x:x+a.shape[1]//5].mean() for y in range(0,a.shape[0]-a.shape[0]//3+1,a.shape[0]//3) for x in range(0,a.shape[1]-a.shape[1]//5+1,a.shape[1]//5)]
        xres=float(tags.get(282,0)); yres=float(tags.get(283,0)); unit=int(tags.get(296,1))
        nm=25.4e6/xres if unit==2 and xres else (1e7/xres if unit==3 and xres else None)
        rows.append(dict(batch=batch,field=field,channel=channel,file=str(path.relative_to(ROOT)),width=im.width,height=im.height,mode=im.mode,bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pixel_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),rgb_max_difference=rgb_delta,chromatic_fraction=chromatic_fraction,interior_chromatic_fraction=interior_chromatic_fraction,analysis_width=gray.width,analysis_height=gray.height,
            x_resolution=xres,y_resolution=yres,resolution_unit=unit,nominal_nm_per_pixel=nm,
            mean_gray=float(np.dot(probs,np.arange(256))),std_gray=float(np.sqrt(np.dot(probs,(np.arange(256)-np.dot(probs,np.arange(256)))**2))),
            p01=q[0],p05=q[1],p25=q[2],median_gray=q[3],p75=q[4],p95=q[5],p99=q[6],
            black_fraction=float(probs[0]),white_fraction=float(probs[255]),entropy_bits=float(-np.sum(probs[probs>0]*np.log2(probs[probs>0]))),
            gradient_rms=gradient,gradient_anisotropy=float(np.mean(gx**2)/(np.mean(gy**2)+1e-12)),tile_mean_sd=float(np.std(tiles)),
            thumb_width=thumb.width,thumb_height=thumb.height,description=str(tags.get(270,'')),software=str(tags.get(305,''))))
    print('Audited',path.relative_to(ROOT),flush=True)

df=pd.DataFrame(rows)
thresholds={c:otsu(sum((histograms[Path(r.file).name] for r in df[df.channel==c].itertuples()),np.zeros(256))) for c in CHANNELS if (df.channel==c).any()}
for i,r in df.iterrows():
    h=histograms[Path(r.file).name]; t=thresholds[r.channel]
    for delta,label in [(-10,'low'),(0,'central'),(10,'high')]:
        df.loc[i,'dark_fraction_'+label]=h[:max(0,min(255,t+delta))+1].sum()/h.sum()
    a=smalls[(r.batch,r.field,r.channel)]
    mask=a*255<=t
    # Mean length of horizontal/vertical dark runs: a 2D intensity proxy, not pore size.
    for axis,label in [(1,'x'),(0,'y')]:
        starts=np.sum(np.diff(mask.astype('int8'),axis=axis)==1)+np.sum(mask[:,0] if axis==1 else mask[0,:])
        df.loc[i,'dark_run_'+label+'_thumb_px']=float(mask.sum()/max(1,starts))

df.to_csv(OUT/'image_inventory.csv',index=False)
metrics=['mean_gray','std_gray','entropy_bits','gradient_rms','gradient_anisotropy','tile_mean_sd','dark_fraction_central','dark_run_x_thumb_px','dark_run_y_thumb_px']
rng=np.random.default_rng(20261003)
summary=[]
for (batch,channel),g in df.groupby(['batch','channel']):
    for metric in metrics:
        v=g[metric].to_numpy(); boot=np.mean(rng.choice(v,(5000,len(v)),replace=True),axis=1)
        summary.append(dict(batch=batch,channel=channel,metric=metric,n_fields=len(v),mean=v.mean(),median=np.median(v),sd=float(np.std(v,ddof=1)) if len(v)>1 else None,minimum=v.min(),maximum=v.max(),bootstrap_mean_low=np.quantile(boot,.025) if len(v)>1 else None,bootstrap_mean_high=np.quantile(boot,.975) if len(v)>1 else None))
summary=pd.DataFrame(summary); summary.to_csv(OUT/'batch_channel_summary.csv',index=False)
pairs=[]
for (batch,field),g in df.groupby(['batch','field']):
    ch=list(g.channel)
    for i,c1 in enumerate(ch):
        for c2 in ch[i+1:]:
            a=smalls[(batch,field,c1)]; b=smalls[(batch,field,c2)]
            same=a.shape==b.shape
            pairs.append(dict(batch=batch,field=field,channel_1=c1,channel_2=c2,same_dimensions=same,zero_shift_correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if same else None))
pd.DataFrame(pairs).to_csv(OUT/'channel_pair_checks.csv',index=False)
with open(OUT/'audit.json','w') as f:
    json.dump(dict(image_count=len(df),field_count=len(df.groupby(['batch','field'])),pooled_otsu_thresholds=thresholds,exact_file_duplicate_groups=int((df.groupby('sha256').size()>1).sum()),exact_pixel_duplicate_groups=int((df.groupby('pixel_sha256').size()>1).sum()),total_bytes=int(df.bytes.sum()),seed=20261003),f,indent=2)

# Every BSE field plus a matched detector contact sheet.
for batch,g in df[df.channel=='BSE'].groupby('batch'):
    canvas=Image.new('RGB',(1000,180*math.ceil(len(g)/2)),'white'); draw=ImageDraw.Draw(canvas)
    for k,r in enumerate(g.itertuples()):
        im=Image.open(ASSETS/(Path(r.file).stem+'.jpg')); im.thumbnail((490,145))
        x=(k%2)*500;y=(k//2)*180; canvas.paste(im,(x,y+25));draw.text((x+5,y+5),r.field,fill='black')
    canvas.save(ASSETS/(batch+'_BSE_montage.jpg'),quality=90)

colors={'Batch_1':'#2563eb','Batch_2':'#d97706','Batch_3':'#059669'}
def plot(metric,channel):
    sub=df[df.channel==channel]; values=sub[metric].to_numpy(); lo=float(values.min()); hi=float(values.max()); margin=(hi-lo)*.12 or .01;lo-=margin;hi+=margin
    w,h=650,240
    svg=[f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{metric} by batch"><rect width="650" height="240" fill="white"/>']
    for tick in np.linspace(lo,hi,5):
        y=190-(tick-lo)/(hi-lo)*150
        svg.append(f'<path d="M70 {y}H620" stroke="#e2e8f0"/><text x="5" y="{y+4}" font-size="12">{tick:.3g}</text>')
    for j,(batch,g) in enumerate(sub.groupby('batch')):
        x=160+j*180
        for k,v in enumerate(g[metric]):
            jitter=((k*7)%11-5)*4;y=190-(v-lo)/(hi-lo)*150
            svg.append(f'<circle cx="{x+jitter}" cy="{y}" r="4" fill="{colors[batch]}" opacity=".75"/>')
        s=summary[(summary.batch==batch)&(summary.channel==channel)&(summary.metric==metric)].iloc[0]
        if pd.notna(s.bootstrap_mean_low):
            yl=190-(s.bootstrap_mean_low-lo)/(hi-lo)*150;yh=190-(s.bootstrap_mean_high-lo)/(hi-lo)*150
            svg.append(f'<path d="M{x} {yl}V{yh}" stroke="#111827" stroke-width="3"/>')
        ym=190-(s['mean']-lo)/(hi-lo)*150
        svg.append(f'<path d="M{x-12} {ym}H{x+12}" stroke="#111827" stroke-width="3"/><text x="{x-40}" y="220" font-size="13">{batch} (n={len(g)})</text>')
    svg.append('</svg>');return ''.join(svg)

def table(frame):return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}',escape=True)
sections=[]
for channel in CHANNELS:
    if not (df.channel==channel).any():continue
    sections.append(f'<h2>{channel}: exploratory measurements</h2><p>Pooled histogram Otsu threshold: {thresholds[channel]}/255. Each dot is one field. Black marks show means and 95% field-bootstrap intervals, conditional on these sampled fields; intervals do not include calibration or segmentation uncertainty.</p>')
    for metric in metrics:
        sections.append(f'<div class="chart"><h3>{metric}</h3>{plot(metric,channel)}</div>')
    ss=summary[(summary.channel==channel)&summary.metric.isin(['mean_gray','dark_fraction_central','gradient_anisotropy','dark_run_x_thumb_px'])]
    sections.append(table(ss[['batch','metric','n_fields','mean','bootstrap_mean_low','bootstrap_mean_high']]))

report='''<!doctype html><html><head><meta charset="utf-8"><title>SEM dataset exploratory audit</title><style>body{font:16px system-ui;color:#172033;background:#f5f7fb;margin:0}main{max-width:1100px;margin:auto;padding:32px}h1,h2{color:#102a43}h2{margin-top:42px}p,li{line-height:1.6}.card{background:white;padding:20px;border-radius:12px;margin:18px 0}.chart{display:inline-block;width:49%;vertical-align:top}svg{width:100%}table{border-collapse:collapse;width:100%;font-size:13px;background:white}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.gallery a{font-size:12px}a{color:#2563eb}.note{border-left:5px solid #d97706;padding:16px;background:#fff7ed}@media(max-width:700px){.chart{width:100%}.gallery{grid-template-columns:1fr}}</style></head><body><main><h1>Incoming electrode microscopy: detailed exploratory audit</h1><p>All 93 TIFF images · 31 field IDs · three batches · 3 October 2026</p><div class="note">No approved baseline, defect labels, material identity, sampling design or verified calibration supplied. This report describes the images; it does not issue accept/reject verdicts or infer manufacturing performance.</div>'''
report+='<h2>Dataset structure</h2>'+table(pd.crosstab(df.batch,df.channel).reset_index())
report+='<p>Channels sharing a field ID are paired observations, not independent replicates. Field IDs themselves may share specimens or overlap spatially; independence is unverified. Batch_3 has more sampled fields. ETD and SE are kept separate.</p>'
report+='<h2>Image and metadata audit</h2>'+table(df.groupby(['batch','channel']).agg(images=('file','count'),width_min=('width','min'),width_max=('width','max'),height_min=('height','min'),height_max=('height','max'),nominal_nm_min=('nominal_nm_per_pixel','min'),nominal_nm_max=('nominal_nm_per_pixel','max'),max_rgb_difference=('rgb_max_difference','max')).reset_index())
report+='<p>Nominal nm/pixel is converted from TIFF resolution tags (inch or centimetre units). Treat it as unverified export metadata: confirm against acquisition records or a known scale before reporting physical dimensions. The varying image heights imply different sampled areas. Means below give each field equal weight. All measurements exclude four pixels on each edge to remove observed colored export borders; original dimensions and hashes are retained.</p>'
report+='<h2>What these measurements mean</h2><ul><li>Mean, spread, entropy and clipping describe image intensity and quality; detector contrast is not a chemical assay.</li><li>Gradient RMS measures fine-scale image variation, combining microstructure, focus, noise and detector effects. Anisotropy is horizontal divided by vertical gradient energy; it is not directly particle orientation.</li><li>Tile mean SD measures coarse spatial contrast heterogeneity using a 3 by 5 grid.</li><li>Dark fraction is the proportion below a common channel-specific threshold, pooled across all supplied images and weighted by pixel count. Dark regions can include voids, shadowing or low-signal material. This is not validated porosity.</li><li>Dark run lengths describe thresholded texture on thumbnails. Their units are thumbnail pixels; they are not segmented particle or pore sizes.</li></ul>'
report+='<p>Full-resolution histograms provide intensity statistics and dark fractions. Texture uses aspect-preserving thumbnails with maximum dimension 1400 pixels. It is exploratory and should be recomputed on a calibrated common grid for final QC. Threshold sensitivity at ±10 gray levels is recorded in the inventory. Pooled thresholds use incoming batches, so they must be replaced by frozen baseline-derived thresholds for the unseen-batch test.</p>'
report+=''.join(sections)
report+='<h2>Clipping and threshold sensitivity</h2>'+table(df.groupby(['batch','channel'])[['black_fraction','white_fraction','dark_fraction_low','dark_fraction_central','dark_fraction_high']].mean().reset_index())
report+='<h2>Paired detector consistency</h2>'+table(pd.DataFrame(pairs).groupby(['batch','channel_1','channel_2']).agg(fields=('field','count'),same_dimensions=('same_dimensions','all'),median_zero_shift_correlation=('zero_shift_correlation','median'),minimum_correlation=('zero_shift_correlation','min')).reset_index())
report+='<p>Zero-shift correlation is a screening check only. Low values can reflect detector physics or misalignment; high values do not establish accurate registration. Do not fuse pixel features before reviewing registration.</p>'
report+='<h2>All BSE fields</h2>'
for batch in sorted(df.batch.unique()):report+=f'<h3>{batch}</h3><a href="assets/{batch}_BSE_montage.jpg"><img src="assets/{batch}_BSE_montage.jpg"></a>'
report+='<h2>Matched channel gallery</h2><p>Click any thumbnail to view the analysis preview. Originals remain unchanged in Dataset.</p>'
for (batch,field),g in df.groupby(['batch','field']):
    report+=f'<h3>{batch} / {field}</h3><div class="gallery">'
    for r in g.itertuples():
        name=Path(r.file).stem+'.jpg';report+=f'<a href="assets/{name}">{r.channel}<img loading="lazy" src="assets/{name}"></a>'
    report+='</div>'
report+='<h2>Validation and next steps</h2><ol><li>Confirm baseline, labels, material identity, scale, specimen grouping and acquisition settings.</li><li>Review representative fields plus intensity/texture outliers with a materials expert.</li><li>Annotate a small sample of phases and particles; evaluate segmentation boundaries and threshold sensitivity.</li><li>Use field/specimen-level holdouts. Avoid treating correlated pixels, patches or channels as replicates.</li><li>Define practical equivalence tolerances and freeze the baseline reference before the unseen batch arrives.</li></ol><p>Downloads: <a href="image_inventory.csv">image inventory</a> · <a href="batch_channel_summary.csv">batch summaries</a> · <a href="channel_pair_checks.csv">channel checks</a> · <a href="audit.json">audit metadata</a> · <a href="analyze_dataset.py">reproducible script</a></p></main></body></html>'
(OUT/'report.html').write_text(report)
print('Saved',OUT/'report.html')
