"""E43S paired-deletion internal LOO; frozen v2 classifier unchanged."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.submission_v2.runtime import ROOT, load_runtime, sha, write_json
from .calibration import transform, fit_scalar, metrics, promotion

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'
FREEZE = ROOT / 'analysis/submission_v2/freeze'
CLASSES = ['Batch_1','Batch_2','Batch_3']
COLS = ['score_'+c for c in CLASSES]


def check(hashes):
    for rel,digest in hashes.items():
        if sha(ROOT/rel) != digest:
            raise AssertionError('Changed protected file: '+rel)


def protection():
    old = json.loads((ROOT/'analysis/confidence_audit/output/probe_receipt.json').read_text())
    hashes = old['protected_artifacts_sha256'].copy()
    for p in (ROOT/'analysis/confidence_audit').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            hashes[str(p.relative_to(ROOT))] = sha(p)
    return hashes, old['raw_sha256']


def rows(candidate,sites,y,P,chosen,alphas):
    return [dict(candidate=candidate,site=str(sites[i]),true_batch=CLASSES[y[i]],
        predicted_batch=CLASSES[chosen[i]],correct=bool(chosen[i]==y[i]),
        alpha=float(alphas[i]),bet_score=float(P[i,chosen[i]]),
        **{c:float(P[i,j]) for j,c in enumerate(COLS)}) for i in range(len(y))]


def evaluate(sites,y,P,alpha,cohort):
    chosen = P.argmax(axis=1)
    variants = {'original':P,'calibrated':np.vstack([transform(P[i:i+1],a)[0] for i,a in enumerate(alpha)]),
                'fixed_softening':transform(P,.5),'uniform_control':np.full_like(P,1/3)}
    records, summaries, thresholds, bins = [],[],[],[]
    for candidate,Q in variants.items():
        aa = alpha if candidate == 'calibrated' else np.full(len(y),1 if candidate=='original' else .5 if candidate=='fixed_softening' else np.nan)
        records += rows(candidate,sites,y,Q,chosen,aa)
        summaries.append(dict(candidate=candidate,cohort=cohort,**metrics(Q,y,chosen)))
        q,z = Q[np.arange(len(y)),chosen],chosen==y
        for t in [.5,.6,.7,.8,.9]:
            high = q>=t
            thresholds.append(dict(candidate=candidate,cohort=cohort,threshold=t,n_high=int(high.sum()),
                high_correct=int((high&z).sum()),high_wrong=int((high&~z).sum()),coverage=float(high.mean())))
        edges=[1/3,.5,.6,.7,.8,1]
        for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=(q>=lo-(1e-12 if k==0 else 0)) & ((q<=hi) if k==4 else (q<hi))
            bins.append(dict(candidate=candidate,cohort=cohort,bin_low=lo,bin_high=hi,n_sites=int(mask.sum()),
                n_correct=int(z[mask].sum()),mean_bet_score=float(q[mask].mean()) if mask.any() else np.nan,
                empirical_accuracy=float(z[mask].mean()) if mask.any() else np.nan,
                **{'n_'+c:int((mask&(y==j)).sum()) for j,c in enumerate(CLASSES)}))
    return pd.DataFrame(records),pd.DataFrame(summaries),pd.DataFrame(thresholds),pd.DataFrame(bins)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Use a fresh output directory; do not overwrite E43S')
    snapshot, cat = load_runtime(FREEZE) # MUST precede any current polaron_qc import.
    saved = joblib.load(FREEZE/'models.joblib')
    model = saved['models']['material']
    if model.features != snapshot['primary_features'] or model.classes != CLASSES:
        raise AssertionError('Unexpected frozen model schema')
    frame = saved['known'].set_index('site').loc[saved['deleted_sites']].reset_index()
    sites = np.asarray(frame.site.astype(str),dtype=str)
    X = cat._matrix(frame,model.features)
    y = frame.batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
    original = np.vstack([cat._proba_full(pipe,X[i:i+1],3)[0] for i,pipe in enumerate(saved['primary_deletion_models'])])
    frozen_table=pd.read_csv(FREEZE/'evaluation/site_table.csv').set_index('site').loc[sites]
    baseline=frozen_table[['mp_'+c+'__material' for c in CLASSES]].to_numpy(float)
    replay_error = float(np.abs(original-baseline).max())
    if replay_error > 1e-12:
        raise AssertionError('Frozen v2 does not match its own recorded baseline')
    previous=pd.read_csv(ROOT/'analysis/confidence_audit/output/predictions.csv')
    old=previous[(previous.cohort=='known_oof')&(previous.candidate=='legacy_l1')].set_index('site').loc[sites]
    old_delta=np.abs(original-old[COLS].to_numpy(float)).max(axis=1)
    historical_comparison=dict(max_score_difference=float(old_delta.max()),
        sites_different_above_1e12=sites[old_delta>1e-12].tolist(),
        changed_bets=int((original.argmax(1)!=old[COLS].to_numpy(float).argmax(1)).sum()),
        interpretation='Earlier matched comparator uses anchor ordering; actual frozen v2 selects C=2 at epqdaau9, comparator C=0.5. Preserve both.')
    protected,raw = protection(); check(protected);check(raw)
    source_paths = [HERE/'run.py',HERE/'calibration.py',HERE/'protocol.md',HERE/'protocol_initial_threefold.md',ROOT/'tests/test_confidence_calibration.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'fits').mkdir()
    for rel in sources:
        target=OUT/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    frame.to_csv(OUT/'known_inputs.csv',index=False)
    write_json(OUT/'historical_comparator_difference.json',historical_comparison)
    # pair_scores[i,j] predicts j from training excluding i AND j.
    pair_scores=np.full((31,31,3),np.nan); fits=[]
    with threadpool_limits(limits=1):
        for i in range(31):
            for j in range(i+1,31):
                train=np.flatnonzero((np.arange(31)!=i)&(np.arange(31)!=j))
                order=cat._canonical(np.nan_to_num(X[train],nan=-1e30),y[train],sites[train])
                train=train[order]
                C=cat._select_C(X[train],y[train],3,snapshot['C_grid'],snapshot['seed'],snapshot['n_inner'])
                pipe=cat._pipeline(C,snapshot['seed']).fit(X[train],y[train])
                pair_scores[i,j]=cat._proba_full(pipe,X[j:j+1],3)[0]
                pair_scores[j,i]=cat._proba_full(pipe,X[i:i+1],3)[0]
                path=OUT/'fits'/f'pair_{i:02d}_{j:02d}.joblib';joblib.dump(pipe,path,compress=3)
                fits.append(dict(excluded_indices=[i,j],excluded_sites=[str(sites[i]),str(sites[j])],
                    train_indices=train.tolist(),C=C,path=str(path.relative_to(ROOT)),sha256=sha(path)))
            if i%5==0:print(f'paired fits {len(fits)}/465',flush=True)
    np.savez_compressed(OUT/'inner_predictions.npz',scores=pair_scores,original=original,labels=y,sites=sites)
    write_json(OUT/'pair_fits.json',fits)
    temperatures=[]
    for i in range(31):
        train=np.flatnonzero(np.arange(31)!=i)
        fitted=fit_scalar(y[train],pair_scores[i,train])
        temperatures.append(dict(fold=i,excluded_site=str(sites[i]),n_calibration=30,**fitted))
    full=fit_scalar(y,original) # deployment ONLY; not evaluated on these same labels.
    temperatures.append(dict(fold='full',excluded_site='',n_calibration=31,**full))
    pd.DataFrame(temperatures).to_csv(OUT/'temperatures.csv',index=False)
    alpha=np.array([t['alpha'] for t in temperatures[:-1]])
    predictions,summary,thresholds,bins=evaluate(sites,y,original,alpha,'known_oof')
    predictions.to_csv(OUT/'known_predictions.csv',index=False)
    summary.to_csv(OUT/'summary_known.csv',index=False)
    thresholds.to_csv(OUT/'thresholds_known.csv',index=False);bins.to_csv(OUT/'bins_known.csv',index=False)
    indexed=summary.set_index('candidate')
    gate=promotion(indexed.loc['original'],indexed.loc['calibrated'])
    gate.update(all_bets_unchanged=bool(np.array_equal(original.argmax(axis=1),transform(original,full['alpha']).argmax(axis=1))),
        eligible_for_versioned_replacement=bool(gate['loss_gate_passed']),promoted=False,
        interpretation='Practical crop-held-out gate; not source-independent calibration evidence. No drop labels used.')
    write_json(OUT/'promotion.json',gate)
    write_json(OUT/'candidate.json',dict(experiment='E43S',version='categoriser-v2-temperature-E43S-candidate',
        base_model_version=snapshot['model_version'],base_snapshot_sha256=sha(FREEZE/'snapshot.json'),
        base_models_sha256=sha(FREEZE/'models.joblib'),features=model.features,classes=CLASSES,
        **full,score_semantics='Fitted score correction; correctness probability not established.',
        selected_from_feedback=False))
    check(sources);check(protected);check(raw)
    known_outputs={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    write_json(OUT/'known_receipt.json',dict(experiment='E43S',stage='known_complete_before_feedback',
        completed_utc=datetime.now(timezone.utc).isoformat(),sources_sha256=sources,
        known_outputs_sha256=known_outputs,protected_artifacts_sha256=protected,raw_sha256=raw,
        base_freeze_snapshot_sha256=sha(FREEZE/'snapshot.json'),base_models_sha256=sha(FREEZE/'models.joblib'),
        frozen_score_replay_max_error=replay_error,paired_model_count=len(fits),
        historical_comparator_difference=historical_comparison,
        outer_heldout_labels_used_in_calibration=False,source_parent_ids_available=False,
        registry_prefix_bytes=(ROOT/'experiments/registry.csv').stat().st_size,registry_prefix_sha256=sha(ROOT/'experiments/registry.csv')))
    print('Known evaluation and promotion gate sealed; revealed feedback diagnostic now follows.',flush=True)
    # Deliberately load the three already revealed rows only AFTER known decision/receipt.
    truth_path=ROOT/'analysis/feedback_drop_01/truth.json'
    truth=json.loads(truth_path.read_text())['labels']
    feedback=pd.read_csv(ROOT/'analysis/submission_v2/first_run/scored/features.csv').sort_values('site').reset_index(drop=True)
    yf=feedback.site.map(truth).map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
    Pf=cat._proba_full(model.pipeline,cat._matrix(feedback,model.features),3)
    pf,sf,tf,bf=evaluate(feedback.site.to_numpy(),yf,Pf,np.full(3,full['alpha']),'feedback_development')
    pf.to_csv(OUT/'feedback_predictions.csv',index=False)
    pd.concat([summary,sf],ignore_index=True).to_csv(OUT/'summary.csv',index=False)
    pd.concat([thresholds,tf],ignore_index=True).to_csv(OUT/'threshold_counts.csv',index=False)
    pd.concat([bins,bf],ignore_index=True).to_csv(OUT/'reliability_bins.csv',index=False)
    check(known_outputs);check(sources);check(protected);check(raw)
    write_json(OUT/'receipt.json',dict(experiment='E43S',stage='complete',sources_sha256=sources,
        known_outputs_sha256=known_outputs,protected_artifacts_sha256=protected,raw_sha256=raw,
        feedback_inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [truth_path,ROOT/'analysis/submission_v2/first_run/scored/features.csv']},
        output_sha256={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*') if p.is_file()},
        calibration_training_sources='Known31 only; paired nested crop-LOO; no confirmed parent groups',
        no_feedback_fit_or_selection=True,eligible_for_replacement=gate['eligible_for_versioned_replacement']))
    print(summary[['candidate','balanced_multiclass_logloss','balanced_bet_brier','mean_wrong_bet_score']].to_string(index=False),flush=True)
    print('full alpha',full['alpha'],'gate',gate['loss_gate_passed'],flush=True)


if __name__=='__main__':main()
