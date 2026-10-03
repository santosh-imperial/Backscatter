import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from analysis.morphology.benchmark_methods import (
    retained_voids, local_width, bright_hysteresis, semantic_errors, semantic_mask,
)
from analysis.morphology.build_benchmark import manual_mask, crop_box
from analysis.morphology import build_benchmark


def test_width_recovers_rectangle_width_and_long_void_category():
    mask=np.zeros((70,680),bool); mask[20:40,40:640]=True
    result,maps=local_width(mask,return_maps=True)
    assert result['n_retained_voids']==1
    assert result['n_crack_width_voids']==1
    assert abs(result['void_local_width_d50_px']-20)<=2
    assert abs(result['crack_local_width_d50_px']-20)<=2
    assert np.all(maps['skeleton']<=maps['retained'])
    assert np.isclose(result['void_local_width_d50_px'],local_width(mask)['void_local_width_d50_px'])


def test_width_excludes_edge_clipped_and_small_components():
    mask=np.zeros((80,100),bool)
    mask[:20,10:30]=True;mask[40:55,40:75]=True;mask[60:62,80:82]=True
    retained,_,keep=retained_voids(mask)
    assert keep.sum()==1
    assert retained.sum()==15*35
    result=local_width(mask)
    assert result['n_retained_voids']==1
    assert result['excluded_void_area_frac']>0


def test_empty_width_is_unavailable_and_not_zero():
    result=local_width(np.zeros((20,20),bool))
    assert result['n_width_samples']==0
    assert np.isnan(result['void_local_width_d50_px'])
    assert np.isnan(result['crack_local_width_d50_px'])


def test_connectivity_changes_only_diagonal_merging():
    mask=np.zeros((30,30),bool);mask[5:10,5:10]=True;mask[10:15,10:15]=True
    assert retained_voids(mask,min_area=1,connectivity=1)[2].sum()==2
    assert retained_voids(mask,min_area=1,connectivity=2)[2].sum()==1


def test_hysteresis_requires_seed_but_grows_connected_weak_region():
    image=np.zeros((35,40),float)
    image[5:16,5:16]=99;image[9:12,9:12]=120
    image[20:30,25:35]=99  # weak isolated object: no confident seed
    grown=bright_hysteresis(image,100,5)
    assert grown[6,6]
    assert not grown[25,30]
    assert np.all(grown<=image.__gt__(95))


def test_zero_window_matches_existing_opening_and_rejects_negative():
    rng=np.random.default_rng(0);image=rng.normal(100,20,(40,40))
    assert np.array_equal(bright_hysteresis(image,100,0),ndi.binary_opening(image>100))
    with pytest.raises(ValueError):bright_hysteresis(image,100,-1)


def test_uncertain_pixels_are_ignored_and_absent_class_is_unavailable():
    truth=np.full((10,10),255,np.uint8);truth[2:5,2:5]=0
    pred=np.ones((10,10),np.uint8);pred[2:5,2:5]=0
    result=semantic_errors(pred,truth)
    assert result['n_labelled_pixels']==9
    assert result['void_iou']==1
    assert np.isnan(result['bright_iou'])
    assert result['void_frac_error_pp']==0


def test_manual_polygon_background_is_unlabelled_by_default_and_bounds_checked():
    roi={'W':30,'H':20}
    a={'polygons':[{'class_id':0,'points':[[5,5],[10,5],[10,10],[5,10]]}]}
    mask=manual_mask(roi,a)
    assert mask[0,0]==255 and mask[7,7]==0
    a['background']='solid';assert manual_mask(roi,a)[0,0]==1
    a['polygons'][0]['points'][0]=[-1,5]
    with pytest.raises(ValueError):manual_mask(roi,a)


def test_crop_box_stays_inside_image_and_semantic_classes_cannot_overlap():
    mask=np.zeros((900,1200),bool);mask[850:890,1120:1190]=True
    y0,x0,y1,x1=crop_box(mask)
    assert 0<=y0<y1<=900 and 0<=x0<x1<=1200
    assert (y1-y0,x1-x0)==(600,800)
    with pytest.raises(ValueError):semantic_mask(mask,mask)


def test_annotation_evaluation_never_scores_unreviewed_predictions(tmp_path,monkeypatch):
    monkeypatch.setattr(build_benchmark,'OUT',tmp_path)
    manifest={'manifest_id':'unit-fixture','rois':[{'id':'test','site':'test','split':'held_out_site','W':20,'H':20}]}
    (tmp_path/'roi_manifest.json').write_text(json.dumps(manifest))
    annotations={'manifest_id':'unit-fixture','rois':[{'id':'test','review_status':'unreviewed'}]}
    path=tmp_path/'manual.json';path.write_text(json.dumps(annotations))
    build_benchmark.evaluate_annotations(path)
    summary=json.loads((tmp_path/'expert_evaluation_summary.json').read_text())
    assert summary['reviewed_measurable_rois']==0 and summary['held_out_sites_scored']==[]
    annotations['manifest_id']='different-inputs';path.write_text(json.dumps(annotations))
    with pytest.raises(ValueError,match='different'):build_benchmark.evaluate_annotations(path)


def test_partial_annotations_cannot_produce_object_or_width_accuracy(tmp_path,monkeypatch):
    import pandas as pd
    monkeypatch.setattr(build_benchmark,'OUT',tmp_path)
    roi={'id':'test','site':'test','split':'held_out_site','W':20,'H':20,'predictions':'p.npz'}
    (tmp_path/'roi_manifest.json').write_text(json.dumps({'manifest_id':'unit-fixture','rois':[roi]}))
    prediction=np.ones((20,20),np.uint8)
    np.savez(tmp_path/'p.npz',baseline=prediction,hysteresis=prediction)
    annotations={'manifest_id':'unit-fixture','rois':[{'id':'test','review_status':'reviewed','measurable':True,'reviewer':'synthetic unit test','background':'unlabelled','polygons':[{'class_id':0,'points':[[5,5],[10,5],[10,10],[5,10]]}]}]}
    path=tmp_path/'manual.json';path.write_text(json.dumps(annotations))
    build_benchmark.evaluate_annotations(path)
    rows=pd.read_csv(tmp_path/'expert_evaluation.csv')
    assert len(rows)==2 and rows.void_iou.eq(0).all()
    assert 'bright_d50_error_px' not in rows and 'void_local_width_d50_px_error_px' not in rows
    summary=json.loads((tmp_path/'expert_evaluation_summary.json').read_text())
    assert summary['development_sites_scored']==[] and summary['held_out_sites_scored']==['test']
