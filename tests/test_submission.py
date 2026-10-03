"""Regression checks for version enforcement, model-specific evidence and preservation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from analysis.submission_test import compose_submission as sub
from analysis.submission_v2 import score_folder as scoring, runtime


@pytest.fixture
def run(tmp_path):
    folder=tmp_path/'scored';folder.mkdir()
    frozen=tmp_path/'freeze';frozen.mkdir()
    (frozen/'models.joblib').write_bytes(b'test-only model hash target')
    scores=dict(model_score_Batch_1=.7,model_score_Batch_2=.2,model_score_Batch_3=.1)
    sites=pd.DataFrame([dict(sample_id='new',predicted_batch='Batch_1',**scores,top_model_score=.7,runner_up='Batch_2',margin=.5,
                            deletion_same_assignment=25,deletion_fit_count=31,morphology_only_bet='Batch_2',acquisition_only_bet='Batch_3',
                            baseline_morphology_percentile=50,acquisition_flags='bright_low_contrast',explanation='pore_count(+1)')])
    sites.to_csv(folder/'submission_sites.csv',index=False)
    images=pd.concat([sites.assign(filename=f'img_new_{d}.tif',detector=d) for d in ['BSE','ETD','Inlens']],ignore_index=True)
    images.to_csv(folder/'submission_images.csv',index=False)
    # The comparator deliberately disagrees with the primary.
    table=pd.DataFrame([dict(site='new',role='scored',mp_Batch_1__material=.7,mp_Batch_2__material=.2,mp_Batch_3__material=.1,
                             argmax__material='Batch_1',cat_top_features__material='pore_count(+1)',
                             mp_Batch_1__combined=.05,mp_Batch_2__combined=.05,mp_Batch_3__combined=.9,argmax__combined='Batch_3')])
    table.to_csv(folder/'site_table.csv',index=False)
    pd.DataFrame([dict(family='material',primary=True,n_sites=31,balanced_accuracy=.6,recall_Batch_1=.7,recall_Batch_2=.3,
                       recall_Batch_3=.8,perm_p=.01)]).to_csv(folder/'categoriser_summary.csv',index=False)
    pd.DataFrame([dict(bin='[0.33, 1]',n=31,mean_top_prob=.65,observed_top_accuracy=.6)]).to_csv(folder/'reliability_material.csv',index=False)
    (folder/'driver_contrasts.csv').write_text('sample_id,feature,contribution\nnew,pore_count,1\n')
    eval_names=['categoriser_summary.csv','reliability_material.csv']
    snapshot=dict(model_version=runtime.MODEL_VERSION,primary_family='material',models_sha256=sub.sha(frozen/'models.joblib'),
                  evaluation_sha256={n:sub.sha(folder/n) for n in eval_names})
    (frozen/'snapshot.json').write_text(json.dumps(snapshot))
    receipt=dict(model_version=runtime.MODEL_VERSION,primary_family='material',snapshot_path=str(frozen/'snapshot.json'),
                 source_snapshot_sha256=sub.sha(frozen/'snapshot.json'),model_sha256=snapshot['models_sha256'],output_sha256={})
    (folder/'prediction_receipt.json').write_text(json.dumps(receipt))
    refresh(folder)
    return folder


def refresh(folder):
    p=folder/'prediction_receipt.json';r=json.loads(p.read_text())
    r['output_sha256']={f.name:sub.sha(f) for f in folder.iterdir() if f.is_file() and f.name!=p.name}
    p.write_text(json.dumps(r))


def test_declared_primary_wins_and_matching_evidence_is_used(run,tmp_path):
    out=tmp_path/'submission';result=sub.compose(run,out,runtime.MODEL_VERSION)
    assert result.predicted_batch.tolist()==['Batch_1']
    text=(out/'submission.md').read_text()
    assert '0.650' in text and '0.600' in text  # the matching fixture's reliability evidence
    assert 'E31 top bin' not in text and 'moderate' not in text
    assert runtime.MODEL_VERSION in text and 'unresolved' in text
    assert len(pd.read_csv(out/'predictions_images.csv'))==3
    receipt=json.loads((out/'submission_receipt.json').read_text())
    assert receipt['primary_family']=='material' and receipt['model_version']==runtime.MODEL_VERSION


def test_missing_primary_is_rejected_instead_of_fallback(run,tmp_path):
    p=run/'site_table.csv';d=pd.read_csv(p);d.drop(columns=[c for c in d if '__material' in c]).to_csv(p,index=False);refresh(run)
    with pytest.raises(ValueError,match='Missing declared primary'):sub.compose(run,tmp_path/'out',runtime.MODEL_VERSION)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind',['wrong_version','wrong_family','changed_model','changed_scored_file','changed_reliability'])
def test_mixed_version_or_changed_artifacts_are_rejected(run,tmp_path,kind):
    p=run/'prediction_receipt.json';r=json.loads(p.read_text())
    if kind=='wrong_version':r['model_version']='legacy-v1'
    if kind=='wrong_family':r['primary_family']='combined'
    p.write_text(json.dumps(r))
    if kind=='changed_model':Path(r['snapshot_path']).with_name('models.joblib').write_bytes(b'changed')
    if kind=='changed_scored_file':(run/'site_table.csv').write_text('changed')
    if kind=='changed_reliability':
        d=pd.read_csv(run/'reliability_material.csv');d.mean_top_prob=.99;d.to_csv(run/'reliability_material.csv',index=False);refresh(run)
    with pytest.raises(ValueError):sub.compose(run,tmp_path/'out',runtime.MODEL_VERSION)


@pytest.mark.parametrize('kind',['nan','not_normalised','argmax','image_score','duplicate_site'])
def test_inconsistent_predictions_are_rejected(run,tmp_path,kind):
    p=run/'site_table.csv';d=pd.read_csv(p)
    if kind=='nan':d.mp_Batch_1__material=np.nan
    if kind=='not_normalised':d.mp_Batch_1__material=.9
    if kind=='argmax':d.argmax__material='Batch_3'
    if kind=='duplicate_site':d=pd.concat([d,d])
    d.to_csv(p,index=False)
    if kind=='image_score':
        p=run/'submission_images.csv';d=pd.read_csv(p);d.loc[0,'model_score_Batch_1']=.99;d.to_csv(p,index=False)
    refresh(run)
    with pytest.raises(ValueError):sub.compose(run,tmp_path/'out',runtime.MODEL_VERSION)


def test_prior_submission_is_not_overwritten(run,tmp_path):
    out=tmp_path/'out';out.mkdir();p=out/'predictions.csv';p.write_text('frozen')
    with pytest.raises(FileExistsError):sub.compose(run,out,runtime.MODEL_VERSION)
    assert p.read_text()=='frozen'


@pytest.mark.parametrize('kind',['missing','geometry','dtype','shape','filename'])
def test_input_preflight_rejects_bad_detector_sets(tmp_path,kind):
    for det in ['BSE','ETD','Inlens']:
        shape=(16,16,4) if kind=='shape' else (16,16)
        dtype=np.float32 if kind=='dtype' else np.uint8
        if kind=='missing' and det=='ETD':continue
        if kind=='geometry' and det=='ETD':shape=(17,16)
        tifffile.imwrite(tmp_path/f'img_new_{det}.tif',np.zeros(shape,dtype))
    if kind=='filename':tifffile.imwrite(tmp_path/'extra.tif',np.zeros((16,16),np.uint8))
    with pytest.raises(ValueError):scoring.inspect_input(tmp_path)


def test_preflight_accepts_se_alias_and_rgb(tmp_path):
    for det in ['BSE','SE','Inlens']:tifffile.imwrite(tmp_path/f'img_new_{det}.tif',np.zeros((16,16,3),np.uint8))
    assert {r['detector'] for r in scoring.inspect_input(tmp_path)}=={'BSE','ETD','Inlens'}


def test_runtime_rejects_wrong_snapshot_version_before_import(tmp_path):
    (tmp_path/'snapshot.json').write_text(json.dumps(dict(model_version='legacy-v1',primary_family='combined')))
    with pytest.raises(ValueError,match='declared D52'):runtime.load_runtime(tmp_path)
