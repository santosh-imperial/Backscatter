"""Execute E39S's fixed six candidates; preserve outputs and original submissions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.classification_m1_m2 import models as m
from analysis.classification_m1_m2.appearance import FEATURE_NAMES
from analysis.quality_policy_audit import policy as qp

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    def clean(value):
        if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, np.generic): return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    Path(path).write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+"\n")


def join_appearance(frame, appearance):
    if not frame.site.is_unique or not appearance.site.is_unique:
        raise ValueError("Duplicate sites in measurement join")
    subset = appearance.set_index("site").reindex(frame.site)
    if subset[list(FEATURE_NAMES)].isna().any().any():
        raise ValueError("Fixed appearance panel missing a required site/measurement")
    out = frame.copy()
    for name in FEATURE_NAMES:
        out[name] = subset[name].to_numpy(float)
    return out


def prediction_rows(name, frame, P, X, choices=None):
    rows=[]
    for i,row in frame.iterrows():
        pred=m.CLASSES[int(P[i].argmax())]
        rows.append(dict(candidate=name,site=row.site,true_batch=row.get("batch","unknown"),
                         predicted_batch=pred,correct=bool(pred==row.get("batch")),
                         n_observed_inputs=int(np.isfinite(X[i]).sum()),C=None if choices is None else choices[i],
                         **{f"score_{c}":float(P[i,j]) for j,c in enumerate(m.CLASSES)}))
    return rows


def drivers(name, model, frame, X, names, P):
    rows=[]
    for i,row in frame.iterrows():
        chosen=int(P[i].argmax())
        # Explain departure from baseline; a baseline bet is compared with runner-up.
        other=2 if chosen!=2 else int(np.argsort(P[i])[-2])
        contributions,intercept,margin=m.margin_contributions(model,X[i:i+1],names,chosen,other)
        for rank,item in enumerate(contributions,1):
            rows.append(dict(candidate=name,site=row.site,predicted_batch=m.CLASSES[chosen],
                             comparison_batch=m.CLASSES[other],rank=rank,**item,
                             intercept_margin=intercept,total_linear_margin=margin,
                             meaning="linear model evidence; imputed terms unobserved; not causal/material proof"))
    return rows


def execute(out, extraction):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory; do not overwrite experiment results")
    protected=json.loads((ROOT/"analysis/feedback_drop_01/output/receipt.json").read_text())["protected_artifacts_sha256"]
    for rel,digest in protected.items():
        if sha(ROOT/rel)!=digest: raise ValueError(f"Protected artifact changed: {rel}")
    known_path=ROOT/"analysis/submission_v2/freeze/known_sites.csv"
    appearance_path=extraction/"appearance_features.csv"
    app=pd.read_csv(appearance_path)
    known=join_appearance(pd.read_csv(known_path),app)
    if known.groupby("batch").size().to_dict()!={"Batch_1":7,"Batch_2":7,"Batch_3":17}:
        raise ValueError("Known label/sample contract changed")
    known=known.iloc[qp.shared_order(known)].reset_index(drop=True)
    y=known.batch.map({c:i for i,c in enumerate(m.CLASSES)}).to_numpy(int)
    out.mkdir(parents=True,exist_ok=True)
    sources=sorted(HERE.glob("*.py"))+[HERE/"protocol.md",ROOT/"analysis/quality_policy_audit/policy.py",
        ROOT/"tests/test_classification_m1_m2.py",ROOT/"tests/test_appearance_features.py"]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for p in sources:
        target=out/"sources"/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,target)
    known.to_csv(out/"known_measurements.csv",index=False)
    summaries=[]; predictions=[]; attr=[]; fitted={}; outer={}
    for name,(view,kind) in m.CANDIDATES.items():
        X,names=m.matrix(known,view,FEATURE_NAMES)
        P,C,folds=m.nested_loo(X,y,kind)
        summary=m.metrics(y,P); confusion=summary.pop("confusion")
        summary.update(candidate=name,n_features=len(names),C_values="|".join(str(v) for v in sorted(set(C),key=str)))
        summaries.append(summary); outer[name]=(P,C,folds)
        predictions.extend(prediction_rows(name,known,P,X,C))
        pd.DataFrame(confusion,index=m.CLASSES,columns=m.CLASSES).to_csv(out/f"confusion_{name}.csv")
        for i in range(len(known)):
            attr.extend(drivers(name,folds[i],known.iloc[i:i+1].reset_index(drop=True),X[i:i+1],names,P[i:i+1]))
        fitted[name]=m.fit(known,name,FEATURE_NAMES)
        print(f"{name}: BA={summary['balanced_accuracy']:.4f}; recalls="+
              "/".join(f"{summary['recall_'+c]:.3f}" for c in m.CLASSES)+f"; balanced logloss={summary['balanced_logloss']:.4f}",flush=True)
    pd.DataFrame(summaries).to_csv(out/"known_summary.csv",index=False)
    pd.DataFrame(predictions).to_csv(out/"known_oof_predictions.csv",index=False)
    pd.DataFrame(attr).to_csv(out/"known_oof_drivers.csv",index=False)
    joblib.dump(fitted,out/"models_known31.joblib")
    joblib.dump({name:entry[2] for name,entry in outer.items()},out/"outer_models.joblib")
    # Explicit paired crop diagnostics: neither IID interval nor external accuracy.
    changes=[]
    comparisons=[("gated_l2","gated_l1"),("gated_lda","gated_l1"),
                 ("augmented_l2","gated_l2"),("appearance_l2","gated_l2")]
    for candidate,base in comparisons:
        a=outer[candidate][0].argmax(axis=1); b=outer[base][0].argmax(axis=1)
        changes.append(dict(candidate=candidate,reference=base,changed=int((a!=b).sum()),
                            newly_correct=int(((a==y)&(b!=y)).sum()),newly_wrong=int(((a!=y)&(b==y)).sum()),
                            balanced_accuracy_delta=m.metrics(y,outer[candidate][0])["balanced_accuracy"]-
                            m.metrics(y,outer[base][0])["balanced_accuracy"]))
    pd.DataFrame(changes).to_csv(out/"paired_crop_changes.csv",index=False)
    # Reproduce the previously fixed E37S L1 scores by site, before diagnostics.
    old=pd.read_csv(ROOT/"analysis/quality_policy_audit/output/known_oof_predictions.csv")
    now=pd.DataFrame(predictions)
    reproduction={}
    for name,old_name in [("legacy_l1","legacy_matched"),("gated_l1","dependency_gate")]:
        cols=[f"score_{c}" for c in m.CLASSES]
        a=now[now.candidate==name].set_index("site")[cols].sort_index().to_numpy()
        b=old[old.variant==old_name].set_index("site")[cols].sort_index().to_numpy()
        reproduction[name]=dict(max_score_difference=float(np.abs(a-b).max()),matches=bool(np.allclose(a,b,atol=1e-12,rtol=0)))
        if not reproduction[name]["matches"]: raise AssertionError("Matched legacy/gated L1 replay differs")
    write_json(out/"known_stage_receipt.json",dict(experiment="E39S",sources_sha256=hashes,
        known_sha256=sha(known_path),appearance_sha256=sha(appearance_path),n_crops=31,
        preserved_comparator_reproduction=reproduction,first_drop_used_for_fitting=False,
        independent_parent_count=None,validation="crop-LOO development diagnostic; unknown shared-parent leakage",
        permutation_p_values_reported=False,intervals_reported=False))
    sensitivity_path=extraction/"appearance_sensitivity.csv"
    sensitivity=pd.read_csv(sensitivity_path)
    sensitivity_rows=[]
    for variant,part in sensitivity.groupby("variant",sort=False):
        part=part.set_index("site").reindex(known.site)
        if len(part)!=31 or part[list(FEATURE_NAMES)].isna().any().any(): raise ValueError("Perturbation lacks sites")
        X=part[list(FEATURE_NAMES)].to_numpy(float)
        base=outer["appearance_l2"][0]; changed_scores=np.zeros_like(base)
        for i,model in enumerate(outer["appearance_l2"][2]):
            changed_scores[i]=m.probabilities(model,X[i:i+1])[0]
        for i,site in enumerate(known.site):
            sensitivity_rows.append(dict(variant=variant,site=site,true_batch=known.iloc[i].batch,
                original_bet=m.CLASSES[int(base[i].argmax())],perturbed_bet=m.CLASSES[int(changed_scores[i].argmax())],
                bet_changed=bool(base[i].argmax()!=changed_scores[i].argmax()),
                max_absolute_score_change=float(np.abs(base[i]-changed_scores[i]).max()),
                **{f"score_{c}":float(changed_scores[i,j]) for j,c in enumerate(m.CLASSES)}))
    pd.DataFrame(sensitivity_rows).to_csv(out/"appearance_oof_sensitivity.csv",index=False)
    # Only now read revealed failure data; these are not unseen validation.
    drop_path=ROOT/"analysis/submission_v2/first_run/scored/features.csv"
    drop=join_appearance(pd.read_csv(drop_path),app)
    truth=json.loads((ROOT/"analysis/feedback_drop_01/truth.json").read_text())["labels"]
    if set(drop.site)!=set(truth): raise ValueError("First-drop IDs differ from revealed feedback")
    drop["batch"]=drop.site.map(truth)
    drop=drop.sort_values("site").reset_index(drop=True)
    failure=[]; attr=[]
    for name,(view,_) in m.CANDIDATES.items():
        X,names=m.matrix(drop,view,FEATURE_NAMES)
        P=fitted[name].predict(drop)
        failure.extend(prediction_rows(name,drop,P,X))
        attr.extend(drivers(name,fitted[name].pipeline,drop,X,names,P))
    pd.DataFrame(failure).to_csv(out/"feedback_development_predictions.csv",index=False)
    pd.DataFrame(attr).to_csv(out/"feedback_development_drivers.csv",index=False)
    for rel,digest in protected.items():
        if sha(ROOT/rel)!=digest: raise ValueError(f"Protected artifact changed: {rel}")
    write_json(out/"receipt.json",dict(experiment="E39S",version=m.VERSION,completed=datetime.now(timezone.utc).isoformat(),
        candidate_order=list(m.CANDIDATES),classes=m.CLASSES,known_fit_n=31,feedback_diagnostic_n=3,
        m1_primary="gated_l2",m2_primary="augmented_l2",submission_promotion=False,
        no_human_annotations_used=True,source_parent_ids_available=False,sources_sha256=hashes,
        protected_artifacts_sha256=protected,protected_artifacts_unchanged=len(protected),
        extraction_inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(extraction.glob("appearance*")) if p.is_file()},
        score_interpretation="uncalibrated balanced-target model scores; crop-fold dependence unknown",
        feedback_scope="already-revealed development failures; no tuning/selection based on these outcomes"))
    print("Saved fixed candidates, crop diagnostics and feedback failure analysis; original submission unchanged.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"output/evaluation")
    parser.add_argument("--extraction",type=Path,default=HERE/"output")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        execute(args.out,args.extraction)


if __name__=="__main__": main()
