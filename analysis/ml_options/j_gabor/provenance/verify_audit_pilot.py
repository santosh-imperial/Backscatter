"""Verify fixed Gabor evidence tables, source coordinates and measured sources."""
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
from polaron_qc import PRIMARY_KPIS, MATERIAL_KPIS, features
from analysis.ml_options.j_gabor.texture import KEYS, PERIODS, ANGLES, MODES, summary, windows, measure_window, usable

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['inputs_unchanged']
    for group in ('inputs','presentation_sources'):
        for p,h in manifest[group].items():assert sha(ROOT/p)==h,p
    for name in ('texture.py','run_audit.py','finalize_report.py'):
        assert (OUT/name).read_bytes()==(OUT/'provenance'/name).read_bytes()
    assert list(PRIMARY_KPIS)==manifest['primary_kpis']
    assert list(MATERIAL_KPIS)==manifest['material_kpis']
    assert not set(KEYS)&(set(PRIMARY_KPIS)|set(MATERIAL_KPIS))
    variants=pd.read_csv(OUT/'variants.csv');sites=pd.read_csv(OUT/'sites.csv')
    energies=pd.read_csv(OUT/'window_energies.csv');comp=pd.read_csv(OUT/'comparisons.csv')
    assert len(variants)==155 and len(sites)==31 and len(comp)==135 and len(energies)==7440
    assert variants.groupby(['batch','site']).size().eq(5).all()
    grouped={key:t for key,t in energies.groupby(['batch','site','mode'])}
    for row in variants.itertuples():
        t=grouped[(row.batch,row.site,row.mode)]
        assert len(t)==48 and t.groupby('window_id').size().eq(12).all()
        origins=windows((row.H,row.W))
        for wi,(y,x) in enumerate(origins):
            part=t[t.window_id.eq(wi)]
            assert part.x.eq(x).all() and part.y_trimmed.eq(y).all() and part.y_raw.eq(y+row.trim_top).all()
        bank=t.groupby(['period_px','angle_deg']).mean_power.mean().unstack().loc[list(PERIODS),[0,45,90,135]].to_numpy()
        values=summary(bank)
        for key in KEYS:np.testing.assert_allclose(values[key],getattr(row,key),rtol=1e-12,atol=1e-14)
        assert row.gabor_windows_used==4
        np.testing.assert_allclose(row.gabor_valid_area_fraction,4*320**2/(row.H*row.W))
    for row in comp.itertuples():
        t=variants[variants['mode'].eq(row.mode)]
        t=t.loc[usable(t,row.kpi,row.view)]
        a,b=t[t.batch.eq(row.reference)][row.kpi],t[t.batch.eq(row.batch)][row.kpi]
        assert len(a)==row.n_reference and len(b)==row.n_batch
        np.testing.assert_allclose(b.median()-a.median(),row.median_difference,rtol=1e-12,atol=1e-14)
    # Independent raw-input replay on two prespecified windows checks coordinates
    # and saved response-map binding, without claiming expert image labels.
    raw_windows=0
    for batch,site in (('Batch_2','3806gxp0'),('Batch_3','71vgq3fw')):
        raw=features.load_image(ROOT/'Dataset'/batch,site,'BSE')
        top,bottom=features.bright_bands(raw);a=raw[top:len(raw)-bottom if bottom else None]
        y,x=windows(a.shape)[0]
        bank,maps,reason=measure_window(a[y:y+512,x:x+512],keep_maps=True)
        assert not reason
        t=grouped[(batch,site,'nominal')]
        e=t[t.window_id.eq(0)].pivot(index='period_px',columns='angle_deg',values='mean_power').loc[list(PERIODS),[0,45,90,135]]
        np.testing.assert_allclose(bank,e.to_numpy(),rtol=1e-12)
        saved=np.load(OUT/f'maps_{batch}_{site}.npz')
        np.testing.assert_allclose(saved['coarse_power'],np.mean([maps[(64,i)] for i in range(4)],axis=0),rtol=1e-6)
        balance=(maps[(32,0)]-maps[(32,2)])/(maps[(32,0)]+maps[(32,2)]+1e-12)
        np.testing.assert_allclose(saved['direction_balance'],balance,rtol=1e-6,atol=1e-7)
        raw_windows+=1
    nominal=comp[comp['mode'].eq('nominal')&comp.view.eq('quality_matched')]
    assert len(nominal)==9 and ((nominal.ci_low<=0)&(nominal.ci_high>=0)).all()
    reg=json.loads((ROOT/'analysis/morphology/metric_register.json').read_text())
    for metric in reg['metrics']:
        if metric['id'] in KEYS:
            assert metric['implementation_status']=='computed' and metric['review_status']=='unreviewed'
            assert metric['qc_role']=='secondary_exploratory'
    report=(OUT/'report.html').read_text()
    assert report.count('<h2>Measured qualification</h2>')==1
    assert report.count('<body>')==1 and report.count('</body>')==1
    data=re.findall(r'src="data:image/png;base64,([^"]+)"',report)
    assert len(data)==5
    for b64 in data:Image.open(io.BytesIO(base64.b64decode(b64))).verify()
    base=manifest['base_commit']
    original=subprocess.check_output(['git','show',f'{base}:experiments/registry.csv'],cwd=ROOT)
    after=(ROOT/'experiments/registry.csv').read_bytes();assert after.startswith(original)
    fields=list(csv.DictReader(io.StringIO(original.decode())).fieldnames)
    rows=list(csv.DictReader(io.StringIO(after[len(original):].decode()),fieldnames=fields))
    assert len(rows)==1710 and all(r['experiment']=='E28J' for r in rows)
    subprocess.run(['git','diff','--exit-code',base,'--','polaron_qc','tests','notebooks','analysis_cache','reports'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    new=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert not any(p.startswith('Dataset') or p.lower().endswith(('.tif','.tiff')) for p in new)
    assert all((ROOT/p).stat().st_size<20_000_000 for p in new)
    receipt=dict(experiment='E28J',measured_source_hashes_current=True,
        saved_variant_summaries_recomputed=155,saved_window_energies=7440,
        pairwise_rows_verified=135,raw_window_replays=raw_windows,
        report_images_decoded=5,report_qualification_once=True,
        registry_history_prefix_preserved=True,appended_registry_rows=1710,
        original_primary_classifier_decision_notebook_cache_files_unchanged=True,
        raw_tiffs_excluded=True,new_files_under_size_limit=True,
        expert_material_validity='pending',spatial_representativeness='unvalidated',unseen_evaluation='not_run')
    (OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
