"""Independent saved-fit/preprocessing and sklearn loss replay for E43S."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from analysis.submission_v2.runtime import ROOT, load_runtime, sha, write_json
from .run import OUT, FREEZE, CLASSES, COLS, check
from .calibration import fit_scalar, promotion


def main():
    receipt=json.loads((OUT/'receipt.json').read_text())
    for key in ['sources_sha256','known_outputs_sha256','protected_artifacts_sha256','raw_sha256','feedback_inputs_sha256','output_sha256']:
        check(receipt[key])
    snapshot,cat=load_runtime(FREEZE)
    saved=joblib.load(FREEZE/'models.joblib')
    frame=saved['known'].set_index('site').loc[saved['deleted_sites']].reset_index()
    model=saved['models']['material'];X=cat._matrix(frame,model.features)
    pack=np.load(OUT/'inner_predictions.npz')
    y=frame.batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
    np.testing.assert_array_equal(y,pack['labels'])
    np.testing.assert_array_equal(frame.site,pack['sites'])
    pairs=json.loads((OUT/'pair_fits.json').read_text());assert len(pairs)==465
    seen=set();score_error=0.;stat_error=0.
    for metadata in pairs:
        i,j=metadata['excluded_indices'];assert i<j and (i,j) not in seen;seen.add((i,j))
        train=np.asarray(metadata['train_indices'])
        assert len(train)==29 and set(train)==set(range(31))-{i,j}
        assert sha(ROOT/metadata['path'])==metadata['sha256']
        pipe=joblib.load(ROOT/metadata['path'])
        assert pipe['lr'].C==metadata['C'] and pipe['lr'].class_weight=='balanced'
        assert pipe['lr'].solver=='liblinear' and pipe['lr'].penalty=='l1'
        assert pipe['lr'].random_state==0 and pipe['lr'].n_features_in_==29
        expected=np.nanmedian(X[train],axis=0)
        stat_error=max(stat_error,float(np.abs(pipe['impute'].statistics_-expected).max()))
        imputed=np.where(np.isnan(X[train]),expected,X[train])
        stat_error=max(stat_error,float(np.abs(pipe['scale'].mean_-imputed.mean(axis=0)).max()))
        actual=cat._proba_full(pipe,X[[j,i]],3)
        score_error=max(score_error,float(np.abs(actual-np.array([pack['scores'][i,j],pack['scores'][j,i]])).max()))
    assert score_error<1e-12 and stat_error<1e-10
    temperatures=pd.read_csv(OUT/'temperatures.csv');alpha_error=0.
    for i in range(31):
        train=np.flatnonzero(np.arange(31)!=i)
        alpha=fit_scalar(y[train],pack['scores'][i,train])['alpha']
        alpha_error=max(alpha_error,abs(alpha-temperatures.iloc[i].alpha))
    full=fit_scalar(y,pack['original'])['alpha']
    assert abs(full-temperatures.iloc[-1].alpha)<1e-10 and alpha_error<1e-10
    summary=pd.read_csv(OUT/'summary.csv');loss_error=0.
    for cohort,file in [('known_oof','known_predictions.csv'),('feedback_development','feedback_predictions.csv')]:
        data=pd.read_csv(OUT/file)
        for candidate,part in data.groupby('candidate'):
            P=part[COLS].to_numpy(float)
            yy=part.true_batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
            chosen=part.predicted_batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
            original=data[data.candidate=='original'].set_index('site').loc[part.site,COLS].to_numpy(float)
            if candidate!='uniform_control':
                # Independently compute power-normalization without the implementation helper.
                powered=original**part.alpha.to_numpy(float)[:,None]
                expected=powered/powered.sum(axis=1,keepdims=True)
                np.testing.assert_allclose(P,expected,atol=1e-12,rtol=0)
                np.testing.assert_array_equal(P.argmax(1),chosen)
            z=(chosen==yy).astype(int);q=P[np.arange(len(P)),chosen]
            row=summary[(summary.candidate==candidate)&(summary.cohort==cohort)].iloc[0]
            for balanced in [False,True]:
                weights=np.ones(len(yy)) if not balanced else 1/np.bincount(yy,minlength=3)[yy]
                prefix='balanced_' if balanced else ''
                expected={'multiclass_logloss':log_loss(yy,P,labels=[0,1,2],sample_weight=weights),
                    'multiclass_brier':np.average(((P-np.eye(3)[yy])**2).sum(1),weights=weights),
                    'bet_brier':np.average((q-z)**2,weights=weights),
                    'bet_nll':log_loss(z,np.column_stack([1-q,q]),labels=[0,1],sample_weight=weights)}
                for name,value in expected.items():loss_error=max(loss_error,abs(value-row[prefix+name]))
            for _,threshold in pd.read_csv(OUT/'threshold_counts.csv').query('cohort==@cohort and candidate==@candidate').iterrows():
                high=q>=threshold.threshold
                assert high.sum()==threshold.n_high and (high&(z==0)).sum()==threshold.high_wrong
            bins=pd.read_csv(OUT/'reliability_bins.csv').query('cohort==@cohort and candidate==@candidate')
            assert bins.n_sites.sum()==len(part) and bins.n_correct.sum()==z.sum()
    assert loss_error<1e-10
    ss=summary[summary.cohort=='known_oof'].set_index('candidate')
    expected=promotion(ss.loc['original'],ss.loc['calibrated'])
    actual=json.loads((OUT/'promotion.json').read_text())
    assert expected['loss_gate_passed']==actual['loss_gate_passed']
    result=dict(experiment='E43S',paired_models_replayed=len(pairs),inner_score_vectors_replayed=930,
        nested_scalars_replayed=31,deployment_scalar_replayed=True,preprocessing_max_error=stat_error,
        score_replay_max_error=score_error,scalar_replay_max_error=alpha_error,sklearn_loss_max_error=loss_error,
        protected_artifacts_verified=len(receipt['protected_artifacts_sha256']),raw_files_verified=len(receipt['raw_sha256']),
        validator_sha256=sha(Path(__file__)),all_bets_unchanged=True,promotion_gate_passed=actual['loss_gate_passed'])
    write_json(OUT/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
