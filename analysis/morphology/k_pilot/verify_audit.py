"""Verify saved K math, warm geometry binding and prespecified raw examples."""
from pathlib import Path
import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from polaron_qc import features,PRIMARY_KPIS,MATERIAL_KPIS
from analysis.morphology.run_analysis import component_table
from analysis.morphology.k_pilot.descriptors import KEYS,SHAPE_KEYS,AUTO_KEYS,CROSS_KEYS,SETTINGS,shape_summary,phase_summary,usable

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['inputs_unchanged']
    for p,h in manifest['inputs'].items():assert sha(ROOT/p)==h,p
    assert list(PRIMARY_KPIS)==manifest['primary_kpis'] and list(MATERIAL_KPIS)==manifest['material_kpis']
    assert not set(KEYS)&(set(PRIMARY_KPIS)|set(MATERIAL_KPIS))
    for name in ('descriptors.py','run_audit.py'):
        assert (OUT/name).read_bytes()==(OUT/'provenance'/name).read_bytes()
    original_protocol=json.loads((OUT/'protocol.json').read_text())
    assert original_protocol['settings']=={m:dict(threshold_offset=o,shape_area_floor_px2=f) for m,(o,f) in SETTINGS.items()}
    sites=pd.read_csv(OUT/'sites.csv');variants=pd.read_csv(OUT/'variants.csv');pairs=pd.read_csv(OUT/'pair_counts.csv')
    assert len(sites)==31 and len(variants)==124 and len(pairs)==1116
    assert variants.groupby(['batch','site']).size().eq(4).all()
    cached=pd.read_csv(ROOT/'analysis/ml_options/e_graph/nodes.csv.gz')
    for r in variants.itertuples():
        nt=cached[cached.batch.eq(r.batch)&cached.site.eq(r.site)&cached.threshold_offset.eq(r.threshold_offset)&cached.area_floor_px2.eq(r.area_floor_px2)]
        shape,_=shape_summary(nt)
        for k in SHAPE_KEYS:np.testing.assert_allclose(shape[k],getattr(r,k),rtol=1e-12,atol=1e-14,equal_nan=True)
        t=pairs[pairs.batch.eq(r.batch)&pairs.site.eq(r.site)&pairs.threshold_offset.eq(r.threshold_offset)]
        assert len(t)==12
        for p in t.itertuples():
            n=p.pair_pixels
            expected=np.nan
            if isinstance(p.unavailable_reason,float) and np.isnan(p.unavailable_reason):
                a,b=p.source_phase_pixels/n,p.target_phase_pixels/n
                expected=(p.joint_phase_pixels/n-a*b)/np.sqrt(a*(1-a)*b*(1-b))
            np.testing.assert_allclose(expected,p.rho,atol=1e-14,rtol=1e-12,equal_nan=True)
        for k in AUTO_KEYS+CROSS_KEYS:
            lag=int(re.search(r'(\d+)px$',k).group(1));kind='cross' if k in CROSS_KEYS else 'auto'
            sub=t[(t.kind==kind)&t.lag_px.abs().eq(lag)]
            # Require every signed direction; do not average away a missing one.
            by={axis:np.mean(sub.loc[sub.axis==axis,'rho'].to_numpy(float)) for axis in ('x','y')}
            np.testing.assert_allclose(by['x']-by['y'],getattr(r,k),atol=1e-14,rtol=1e-12,equal_nan=True)
    from analysis.morphology.k_pilot.run_audit import compare
    expected=compare(variants);saved=pd.read_csv(OUT/'comparisons.csv')
    assert len(saved)==252
    nums=['n_reference','n_batch','median_reference','median_batch','median_difference','ci_low','ci_high','deletion_min','deletion_max']
    np.testing.assert_allclose(expected[nums].to_numpy(float),saved[nums].to_numpy(float),rtol=1e-12,atol=1e-14,equal_nan=True)
    replays=0
    for crop in manifest['crops']:
        b,s=crop['batch'],crop['site'];r=sites[(sites.batch==b)&(sites.site==s)].iloc[0]
        raw=features.load_image(ROOT/'Dataset'/b,s,'BSE');top,bottom=features.bright_bands(raw)
        a=raw[top:len(raw)-bottom if bottom else None];sm=features.smooth_bse(a)
        bright=ndi.binary_opening(sm>r.th_hi,iterations=1);pore=ndi.binary_opening(sm<r.th_lo,iterations=1)
        vals,t=phase_summary(bright,pore)
        for k in AUTO_KEYS+CROSS_KEYS:np.testing.assert_allclose(vals[k],r[k],atol=1e-14,rtol=1e-12,equal_nan=True)
        ct=component_table(bright,50);ct=ct.loc[ct.touches_edge.eq(False)]
        fresh,_=shape_summary(ct)
        for k in SHAPE_KEYS:np.testing.assert_allclose(fresh[k],r[k],rtol=1e-12,atol=1e-14,equal_nan=True)
        assert top==r.trim_top and a.shape==(r.H,r.W)
        assert crop['y_raw']==crop['y_trimmed']+top
        assert crop['x']+crop['width']<=a.shape[1] and crop['y_trimmed']+crop['height']<=a.shape[0]
        replays+=1
    reg=json.loads((ROOT/'analysis/morphology/metric_register.json').read_text())
    for m in reg['metrics']:
        if m['id'] in KEYS:assert m['review_status']=='unreviewed' and m['qc_role']=='secondary_exploratory' and m['implementation_status']=='computed'
    original_reg=json.loads(subprocess.check_output(['git','show',manifest['base_commit']+':analysis/morphology/metric_register.json'],cwd=ROOT))
    assert reg['metrics'][:81]==original_reg['metrics'] and reg['methods'][:11]==original_reg['methods']
    presentation=json.loads((OUT/'presentation_manifest.json').read_text())
    assert sha(OUT/presentation['source'])==presentation['sha256']
    assert (OUT/'qualify_report.py').read_bytes()==(OUT/'provenance/qualify_report.py').read_bytes()
    report=(OUT/'report.html').read_text()
    assert report.count('<h2>Measured qualification</h2>')==1
    encoded=re.findall(r'src="data:image/png;base64,([^"]+)"',report);assert len(encoded)==5
    for b64 in encoded:Image.open(io.BytesIO(base64.b64decode(b64))).verify()
    base=manifest['base_commit']
    history=subprocess.check_output(['git','show',base+':experiments/registry.csv'],cwd=ROOT)
    after=(ROOT/'experiments/registry.csv').read_bytes();assert after.startswith(history)
    fields=next(csv.reader(io.StringIO(history.decode())))
    new=list(csv.DictReader(io.StringIO(after[len(history):].decode()),fieldnames=fields))
    assert new and all(r['experiment']=='E30K' for r in new)
    subprocess.run(['git','diff','--exit-code',base,'--','polaron_qc','tests','notebooks','analysis_cache','reports','analysis/ml_options'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    files=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert not any(p.startswith('Dataset') or p.lower().endswith(('.tif','.tiff')) for p in files)
    assert all((ROOT/p).stat().st_size<20_000_000 for p in files)
    receipt=dict(experiment='E30K',n_sites_verified=31,saved_site_modes_verified=124,
        saved_pair_count_rows_recomputed=1116,bootstrap_and_deletion_comparison_rows_reproduced=252,
        prespecified_raw_nominal_sites_replayed=replays,report_images_decoded=len(encoded),
        appended_registry_rows=len(new),original_registry_prefix_preserved=True,
        prior_metrics_methods_preserved=True,primary_classifier_decision_notebook_cache_report_sources_unchanged=True,
        current_source_input_hashes_match=True,shape_cache_scope='Declared warm reuse of E26G; four raw nominal binding checks',
        phase_measurement='Fresh whole-frame raster pair counts',expert_validation='pending',
        specimen_independence='unknown',unseen_validation='not_run',qc_role='excluded_from_all_verdict_inputs')
    (OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
