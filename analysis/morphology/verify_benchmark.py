"""Verify E24 output/provenance contracts, not material or segmentation accuracy."""
import json

import numpy as np
import pandas as pd
from PIL import Image

from analysis.morphology.build_benchmark import ROOT, OUT, WIDTH_KEYS, digest
from analysis.morphology.metric_register import REGISTER,validate


def main():
    sites=pd.read_csv(ROOT/'analysis_cache/site_features.csv')
    manifest=json.loads((OUT/'roi_manifest.json').read_text())
    widths=pd.read_csv(OUT/'local_width_sites.csv')
    band=pd.read_csv(OUT/'local_width_sensitivity.csv')
    hysteresis=pd.read_csv(OUT/'hysteresis_sites.csv')
    connectivity=pd.read_csv(OUT/'connectivity_sites.csv')
    assert manifest['inputs_unchanged_during_run']
    assert manifest['input_hashes']=={p:digest(ROOT/p) for p in manifest['input_hashes']}
    for current,snapshot in manifest.get('extraction_source_snapshots',{}).items():
        assert (ROOT/snapshot).read_text().replace('E23','E24')==(ROOT/current).read_text(), 'Only the experiment label may differ from measured source'
    assert len(widths)==93 and len(band)==93 and len(hysteresis)==31 and len(connectivity)==62
    assert set(widths.site)==set(sites.site)
    assert widths.groupby('site').size().eq(3).all()
    assert set(widths.threshold_offset)=={-5,0,5}
    assert widths['void_local_width_d50_px'].gt(0).all()
    assert widths['void_local_width_d90_px'].ge(widths['void_local_width_d50_px']).all()
    assert (widths.n_crack_width_samples<=widths.n_width_samples).all()
    assert widths.loc[widths.n_crack_width_samples==0,'crack_local_width_d50_px'].isna().all()
    defined=band[band.nominal.notna()]
    assert (defined.method_band_lo<=defined.nominal).all()
    assert (defined.nominal<=defined.method_band_hi).all()
    h=hysteresis.merge(sites[['site','bright_frac','bright_d50','bright_low_contrast']],on='site',validate='one_to_one',suffixes=('','_cache'))
    assert np.allclose(h.baseline_frac,h.bright_frac)
    assert np.allclose(h.baseline_d50,h.bright_d50,equal_nan=True)
    assert (h.bright_low_contrast==h.bright_low_contrast_cache).all()
    assert h.agreement_iou.between(0,1).all()
    assert h.changed_image_fraction.between(0,1).all()
    rois=manifest['rois']
    assert len(rois)==len({r['site'] for r in rois})==11
    assert sum(r['split']=='development' for r in rois)==6
    assert sum(r['split']=='held_out_site' for r in rois)==5
    assert all(r['review_status']=='unreviewed' for r in rois)
    for r in rois:
        y0,x0,y1,x1=r['box_trimmed_yxyx']
        raw_box=r['box_raw_yxyx']
        assert raw_box==[y0+r['band_top'],x0,y1+r['band_top'],x1]
        for k,p in r['paths'].items():
            im=Image.open(OUT/p)
            if k!='context':assert im.size==(r['W'],r['H'])
        p=np.load(OUT/r['predictions'])
        assert set(np.unique(p['baseline']))<=set([0,1,2])
        assert set(np.unique(p['hysteresis']))<=set([0,1,2])
        assert p['baseline'].shape==(r['H'],r['W'])
        assert (p['local_width_px']>=0).all()
    template=json.loads((OUT/'annotations_template.json').read_text())
    assert template['manifest_id']==manifest['manifest_id']
    assert all(a['review_status']=='unreviewed' and not a['polygons'] and a['background']=='unlabelled' for a in template['rois'])
    html=(OUT/'review.html').read_text()
    assert '__BENCHMARK_DATA__' not in html and 'data:image/png;base64,' in html
    assert 'not reference labels' in html and 'uncertain' in html
    assert (OUT/'review.html').stat().st_size<20_000_000
    validate(json.loads(REGISTER.read_text()))
    print('E24: site counts, phase masks, width bands, flags, crop coordinates, annotation separation, provenance and inventory contracts passed.')


if __name__=='__main__':main()
