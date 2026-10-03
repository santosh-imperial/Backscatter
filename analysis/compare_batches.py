"""Exploratory pairwise comparisons; no assumed approved baseline or labels."""
from pathlib import Path
from itertools import combinations
import re
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
df=pd.read_csv(OUT/'image_inventory.csv')
rng=np.random.default_rng(42)
metrics=['mean_gray','std_gray','entropy_bits','gradient_rms','gradient_anisotropy','tile_mean_sd','dark_fraction_central','dark_run_x_thumb_px','dark_run_y_thumb_px']
rows=[]
for channel in ['BSE','ETD','Inlens']:
    for b1,b2 in combinations(sorted(df.batch.unique()),2):
        for metric in metrics:
            a=df[(df.batch==b1)&(df.channel==channel)][metric].to_numpy()
            b=df[(df.batch==b2)&(df.channel==channel)][metric].to_numpy()
            draws=rng.choice(b,(10000,len(b)),replace=True).mean(1)-rng.choice(a,(10000,len(a)),replace=True).mean(1)
            pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
            rows.append(dict(channel=channel,batch_from=b1,batch_to=b2,metric=metric,n_from=len(a),n_to=len(b),mean_from=a.mean(),mean_to=b.mean(),delta=b.mean()-a.mean(),delta_ci_low=np.quantile(draws,.025),delta_ci_high=np.quantile(draws,.975),standardized_difference=(b.mean()-a.mean())/pooled if pooled else np.nan))
cmp=pd.DataFrame(rows);cmp.to_csv(OUT/'exploratory_batch_comparisons.csv',index=False)
lines=['# Detailed dataset analysis','',
    'This is an exploratory audit of all supplied images, not a manufacturing release decision. No approved baseline, defect labels, material identity, specimen grouping, acquisition settings or verified physical calibration were supplied.','',
    '## Inventory','',df.groupby('batch').agg(images=('file','count'),field_ids=('field','nunique'),width_min=('width','min'),width_max=('width','max'),height_min=('height','min'),height_max=('height','max')).to_string(),'',
    'Field IDs have three detector images each. Treat channels as paired views and patches as subsamples. Field independence and spatial overlap remain unverified. Batch_3 has 17 fields versus seven each in Batch_1 and Batch_2.','',
    '## Integrity and calibration','',
    f"- All {len(df)} TIFFs decoded successfully.",
    f"- Exact duplicate pixel groups: {(df.groupby('pixel_sha256').size()>1).sum()}.",
    f"- {int((df.chromatic_fraction>0).sum())} images contain colored pixels. Maximum original colored-pixel fraction: {df.chromatic_fraction.max():.4%}; maximum after excluding a four-pixel edge: {df.interior_chromatic_fraction.max():.4%}. All measurements use this edge exclusion; original files remain unchanged.",
    f"- Nominal TIFF scale range: {df.nominal_nm_per_pixel.min():.6g}–{df.nominal_nm_per_pixel.max():.6g} nm/pixel. This is export metadata, not verified calibration.",
    f"- Maximum fraction exactly black: {df.black_fraction.max():.2%}; exactly white: {df.white_fraction.max():.2%}. Endpoint fractions alone cannot establish clipping or image quality.",'',
    '## Imaging confounders to review','',
    'Fields with the greatest endpoint-white fractions (potential detector saturation):','',
    df.sort_values('white_fraction',ascending=False)[['batch','field','channel','white_fraction','p01','p99']].head(6).to_string(index=False),'',
    'BSE fields with the highest 1st-percentile intensities (raised dark levels can affect thresholded dark fractions):','',
    df[df.channel=='BSE'].sort_values('p01',ascending=False)[['batch','field','p01','p99','std_gray','black_fraction']].head(6).to_string(index=False),'',
    '## BSE batch summary','',
    df[df.channel=='BSE'].groupby('batch')[['mean_gray','std_gray','dark_fraction_low','dark_fraction_central','dark_fraction_high','gradient_rms','gradient_anisotropy','tile_mean_sd','dark_run_x_thumb_px','dark_run_y_thumb_px']].mean().round(4).to_string(),'',
    'Dark fraction is an intensity-threshold proxy, not measured porosity. Threshold ±10 sensitivity is shown above. Dark run lengths use thumbnail pixels and are not segmented particle sizes. Gradient anisotropy is image texture directionality, not a direct particle orientation estimate.','',
    '## Sensitivity to raised dark levels','',
    'Four Batch_3 BSE fields have raised 1st-percentile intensities (above 10/255). This post-hoc grouping is an imaging-confounder check, not a defect classifier. Compare these fields separately before interpreting a batch-level shift as material change.','',
    df[df.channel=='BSE'].assign(dark_level_group=lambda x: np.where(x.p01>10,'raised (>10)','lower (<=10)')).groupby(['batch','dark_level_group']).agg(n_fields=('field','count'),mean_gray=('mean_gray','mean'),std_gray=('std_gray','mean'),gradient_rms=('gradient_rms','mean'),dark_fraction=('dark_fraction_central','mean')).round(4).to_string(),'',
    '## Exploratory pairwise differences','',
    'Comparisons use field-level resampling (10,000 draws, seed 42). Intervals assume independent sampled fields and describe these images only. They omit segmentation, acquisition, calibration and between-production-batch uncertainty. Intervals are not adjusted for multiple comparisons; selecting the largest effects is exploratory. Standardized differences use pooled within-group standard deviation, without small-sample correction. Batch_1 is a reference for display only, not assumed approved.','']
for b1,b2 in combinations(sorted(df.batch.unique()),2):
    sub=cmp[(cmp.channel=='BSE')&(cmp.batch_from==b1)&(cmp.batch_to==b2)].copy()
    sub['abs_effect']=sub.standardized_difference.abs()
    lines += [f'### {b2} minus {b1}, BSE','',sub.sort_values('abs_effect',ascending=False)[['metric','delta','delta_ci_low','delta_ci_high','standardized_difference']].round(4).to_string(index=False),'']
lines+=['## Fields to inspect','', 'The following BSE fields have the highest and lowest image-intensity dark fractions. These are review candidates, not known defects.','',df[df.channel=='BSE'].sort_values('dark_fraction_central')[['batch','field','dark_fraction_central','mean_gray','gradient_anisotropy']].iloc[list(range(3))+list(range(28,31))].to_string(index=False),'',
    '## Next measurements','',
    '1. Confirm which batch is approved and what each batch label means; obtain acquisition settings, physical calibration and specimen IDs.',
    '2. Review dark regions and particle boundaries with a materials expert. Annotate representative fields from every detector and batch.',
    '3. Validate particle/phase segmentation before reporting particle dimensions, void fraction, crack widths or morphology.',
    '4. Quantify segmentation sensitivity and field/specimen sampling uncertainty separately.',
    '5. Freeze baseline-derived preprocessing and practical tolerance limits before the unseen batch arrives.',
    '', '## Reproduction','',
    'Run `analyze_dataset.py` and then `compare_batches.py` with Python providing numpy, pandas and Pillow. Inputs are read-only. All measurements exclude a four-pixel border. Assets are compressed analysis previews; histograms use full-resolution interiors, texture uses maximum-dimension-1400 thumbnails. See report.html for all field images and charts.']
(OUT/'findings.md').write_text('\n'.join('```text\n'+s+'\n```' if '\n' in s else s for s in lines)+'\n')
report=OUT/'report.html'
s=report.read_text()
s=re.sub(r'<h2>Main findings to review</h2>.*?<h2>Dataset structure</h2>', '<h2>Dataset structure</h2>',s,flags=re.S)
s=re.sub(r'<h2>Pairwise comparisons and written findings</h2>.*?</main>', '</main>',s,flags=re.S)
s=s.replace('<h2>Dataset structure</h2>', '<h2>Main findings to review</h2><ul><li>All 93 images decoded. No exact pixel duplicates. Every field has three dimension-matched detector views.</li><li>Batch_2 has higher BSE thresholded dark fractions on average, with wide sampling uncertainty. This is not porosity or proof of a defect.</li><li>Batch_3 has lower BSE gradient texture and contrast on average. Four fields have raised dark intensity levels, a possible imaging confounder.</li><li>Threshold ±10 sensitivity is large. Inlens endpoint-white fractions reach about 6.75%, requiring review for saturation.</li><li>Colored export borders are excluded with a fixed four-pixel edge crop. Calibration remains unverified.</li></ul><h2>Dataset structure</h2>')
s=s.replace('</main>', '<h2>Pairwise comparisons and written findings</h2><p><a href="findings.md">Written findings</a> · <a href="exploratory_batch_comparisons.csv">All pairwise comparisons</a></p>'+cmp[(cmp.channel=='BSE')&(cmp.metric.isin(['dark_fraction_central','gradient_anisotropy','dark_run_x_thumb_px']))][['batch_from','batch_to','metric','delta','delta_ci_low','delta_ci_high']].to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}')+'<p>Difference intervals are exploratory, conditional on field independence, and not adjusted for multiple comparisons. No batch is assumed approved.</p></main>')
report.write_text(s)
print((OUT/'findings.md').read_text())
