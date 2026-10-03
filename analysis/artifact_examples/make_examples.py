"""Evidence examples from original TIFF pixel values, excluding four-pixel borders."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
font=ImageFont.truetype(font_path,18)
title=ImageFont.truetype(font_path,22)
collector=Image.new('RGB',(1400,700),'white');draw=ImageDraw.Draw(collector)
for j,ch in enumerate(['BSE','ETD']):
    p=ROOT/'Dataset'/'Batch_2'/f'img_epqdaau9_{ch}.tif'
    with Image.open(p) as im:
        # Center-left region contains the continuous bright bottom layer.
        box=(800,im.height-450,2800,im.height)
        crop=im.crop(box);crop.thumbnail((1380,310))
        collector.paste(crop,(10,j*350+40))
        draw.text((10,j*350+8),f'Batch_2 / img_epqdaau9 / {ch} — bottom crop, original x=800:2800',font=title,fill='black')
        # Upward pointer from a lower-contrast area; exact collector boundary remains unsegmented.
collector.save(OUT/'collector_bottom_examples.png')

examples=[('Batch_3','img_ufdvpb81'),('Batch_2','img_i9jiqjwl'),('Batch_3','img_71vgq3fw')]
canvas=Image.new('RGB',(1100,990),'white');dr=ImageDraw.Draw(canvas)
lines=['# Artifact examples','',
       'Collector candidate: Batch_2/img_epqdaau9 shows a continuous bright bottom layer in BSE and the matched ETD view. Morphology and BSE brightness are consistent with a current collector; copper identity requires acquisition/sample information or compositional analysis. This region should be excluded from electrode-only KPIs.','',
       'Histogram evidence below is calculated from full-resolution original TIFF intensities, after excluding a four-pixel colored export border. Counts use 256 bins, one per 8-bit gray level. Missing levels refer to the inclusive 1st–99th percentile interval.','']
for j,(batch,field) in enumerate(examples):
    with Image.open(ROOT/'Dataset'/batch/f'{field}_BSE.tif') as im:
        a=np.asarray(im.convert('L'))[4:-4,4:-4]
    h=np.bincount(a.ravel(),minlength=256);q=np.searchsorted(np.cumsum(h)/h.sum(),[.01,.99]);missing=np.flatnonzero(h[q[0]:q[1]+1]==0)+q[0]
    label='Comb-like example' if j<2 else 'Continuous-bin comparison (not assumed unprocessed)'
    top=j*330
    dr.text((20,top+8),f'{batch} / {field}_BSE — {label}',font=title,fill='black')
    dr.text((20,top+38),f'P01–P99: {q[0]}–{q[1]}; {len(missing)} empty bins out of {q[1]-q[0]+1}',font=font,fill='#334155')
    x0=80;y0=top+265;width=980;height=180
    ymax=h[1:141].max()/h.sum()*100
    for t in [0,.5,1]:
        yy=y0-t*height;dr.line((x0,yy,x0+width,yy),fill='#dddddd')
        dr.text((3,yy-10),f'{t*ymax:.2f}%',font=font,fill='black')
    for k in range(141):
        x=x0+k*width/141;v=h[k]/h.sum()*100
        hh=min(height,v/ymax*height)
        dr.rectangle((x,y0-hh,x+width/141-1,y0),fill='#2563eb' if j<2 else '#059669')
        if q[0]<=k<=q[1] and h[k]==0:
            dr.line((x,y0+3,x,y0+10),fill='#dc2626',width=2)
    for t in range(0,141,20):
        x=x0+t*width/141;dr.text((x-10,y0+15),str(t),font=font,fill='black')
    dr.text((460,top+310),'Original 8-bit gray level',font=font,fill='black')
    lines += [f'- {batch}/{field}_BSE: {len(missing)} empty levels within {q[0]}–{q[1]} (out of {q[1]-q[0]+1} levels). Missing examples: {missing[:20].tolist()}.']
canvas.save(OUT/'histogram_comb_examples.png')
lines+=['','Regular skipped intensity levels support quantized intensity remapping, consistent with contrast stretching of a finite-bit-depth image. They cannot establish when or where the remapping occurred; acquisition electronics, export software and post-acquisition processing need to be distinguished using raw data and provenance. A continuous histogram does not prove absence of processing.','',
        'The previous report did not explicitly identify either collector inclusion or histogram combing. Its intensity, entropy, threshold and texture metrics remain exploratory and may be affected by these artifacts.']
(OUT/'examples.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
