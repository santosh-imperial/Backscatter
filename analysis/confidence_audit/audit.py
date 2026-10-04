"""E42S descriptive probability/bet-confidence audit of fixed OOF outputs."""
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

from analysis.classification_m1_m2.run import sha, write_json
from analysis.confidence_audit.probe import HERE, OUT, ROOT
from analysis.forest_geometry.run import check_hashes, CLASSES, SCORE_COLS

ORDER = ["legacy_l1", "base_l1", "gated_l2", "gated_lda", "appearance_l2", "augmented_l2",
         "base_rf", "geometry_l1", "geometry_rf", "geometry_l2"]
BINS = [1/3, .5, .6, .7, .8, 1.]
THRESHOLDS = [.5, .6, .7, .8, .9]


def weighted_mean(values, y, balanced=False):
    return float(np.mean([values[y == c].mean() for c in np.unique(y)])) if balanced else float(values.mean())


def metrics(P, y, chosen=None):
    if P.shape != (len(y), 3) or not np.isfinite(P).all() or not np.allclose(P.sum(axis=1), 1.) or (P < 0).any():
        raise ValueError("Require a valid three-class score vector per crop")
    chosen = P.argmax(axis=1) if chosen is None else np.asarray(chosen)
    q = P[np.arange(len(y)), chosen]
    z = (chosen == y).astype(float)
    true = P[np.arange(len(y)), y]
    losses = {
        "multiclass_logloss": -np.log(np.clip(true, 1e-12, 1)),
        "multiclass_brier": ((P - np.eye(3)[y])**2).sum(axis=1),
        "bet_brier": (q-z)**2,
        "bet_nll": -(z*np.log(np.clip(q,1e-12,1)) + (1-z)*np.log(np.clip(1-q,1e-12,1))),
    }
    result = dict(n_sites=len(y), n_correct=int(z.sum()), n_wrong=int((1-z).sum()),
        accuracy=float(z.mean()), balanced_accuracy=weighted_mean(z,y,True),
        mean_wrong_bet_score=float(q[z == 0].mean()) if (z == 0).any() else np.nan,
        mean_correct_bet_score=float(q[z == 1].mean()) if (z == 1).any() else np.nan,
        mean_true_class_score=float(true.mean()), mean_bet_score=float(q.mean()))
    for name, values in losses.items():
        result[name] = weighted_mean(values,y)
        result["balanced_" + name] = weighted_mean(values,y,True)
    return result


def load_predictions():
    data, inputs, alias = [], {}, {}
    for cohort, file, probe_file in [("known_oof", "known_oof_predictions.csv", "probe_known_predictions.csv"),
                                   ("feedback_development", "feedback_development_predictions.csv", "probe_feedback_predictions.csv")]:
        e39_path = ROOT / "analysis/classification_m1_m2/output/evaluation" / file
        e41_path = ROOT / "analysis/forest_geometry/output/evaluation" / file
        probe_path = OUT / probe_file
        a, b, c = [pd.read_csv(p) for p in [e39_path,e41_path,probe_path]]
        for p in [e39_path,e41_path,probe_path]:
            inputs[str(p.relative_to(ROOT))] = sha(p)
        for old, new in [("legacy_l1", "legacy_l1"), ("gated_l1", "base_l1")]:
            x = a[a.candidate == old].set_index("site")[SCORE_COLS].sort_index()
            z = b[b.candidate == new].set_index("site")[SCORE_COLS].sort_index()
            if not x.index.equals(z.index) or not np.allclose(x,z,atol=1e-12,rtol=0):
                raise ValueError("Previously identical alias outputs disagree")
            alias[cohort + ":" + old + "=" + new] = float(np.abs(x.to_numpy()-z.to_numpy()).max())
        a = a[a.candidate.isin(["gated_l2","gated_lda","appearance_l2","augmented_l2"])].copy()
        b = b[b.candidate.isin(["legacy_l1","base_l1","base_rf","geometry_l1","geometry_rf"])].copy()
        for frame, experiment in [(a,"E39S"),(b,"E41S"),(c,"E42S")]:
            frame["cohort"] = cohort
            frame["source_experiment"] = experiment
            data.append(frame)
    frame = pd.concat(data,ignore_index=True)
    if frame.duplicated(["candidate","cohort","site"]).any() or set(frame.candidate) != set(ORDER):
        raise ValueError("Duplicate/missing fixed candidates")
    for cohort, group in frame.groupby("cohort"):
        expected = 31 if cohort == "known_oof" else 3
        ref = group[group.candidate == "legacy_l1"].set_index("site").true_batch.sort_index()
        for name in ORDER:
            part = group[group.candidate == name].set_index("site")
            if len(part) != expected or not part.true_batch.sort_index().equals(ref):
                raise ValueError("Candidates must share exactly the same crop labels")
            labels = part.true_batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
            P = part[SCORE_COLS].to_numpy(float)
            if not np.array_equal(P.argmax(axis=1), part.predicted_batch.map({c:i for i,c in enumerate(CLASSES)})):
                raise ValueError("Bet/class mapping mismatch")
            if not np.array_equal(P.argmax(axis=1) == labels, part.correct):
                raise ValueError("Correctness mismatch")
    frame["bet_score"] = frame[SCORE_COLS].max(axis=1)
    frame["true_class_score"] = [row["score_" + row.true_batch] for _,row in frame.iterrows()]
    return frame, inputs, alias


def execute():
    if (OUT / "confidence_summary.csv").exists():
        raise FileExistsError("Never overwrite the confidence trial")
    probe = json.loads((OUT / "probe_receipt.json").read_text())
    check_hashes(probe["sources_sha256"]); check_hashes(probe["known_outputs_sha256"])
    check_hashes(probe["protected_artifacts_sha256"]); check_hashes(probe["raw_sha256"])
    frame, inputs, aliases = load_predictions()
    sources = {str(p.relative_to(ROOT)):sha(p) for p in [HERE / "audit.py", HERE / "protocol.md"]}
    target = OUT / "sources" / "analysis/confidence_audit/audit.py"
    shutil.copyfile(HERE / "audit.py",target)
    frame.to_csv(OUT / "predictions.csv",index=False)
    summary, bins, thresholds, controls, paired = [], [], [], [], []
    for cohort in ["known_oof","feedback_development"]:
        subset = frame[frame.cohort == cohort]
        for name in ORDER:
            part = subset[subset.candidate == name].sort_values("site").reset_index(drop=True)
            y = part.true_batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
            P = part[SCORE_COLS].to_numpy(float)
            summary.append(dict(candidate=name,cohort=cohort,**metrics(P,y)))
            q, z = P.max(axis=1), (P.argmax(axis=1) == y)
            for j,(lo,hi) in enumerate(zip(BINS[:-1],BINS[1:])):
                selected = (q >= lo-1e-12 if j == 0 else q >= lo) & (q <= hi if j == len(BINS)-2 else q < hi)
                bins.append(dict(candidate=name,cohort=cohort,bin_low=lo,bin_high=hi,
                    n_sites=int(selected.sum()),n_correct=int(z[selected].sum()),
                    mean_bet_score=float(q[selected].mean()) if selected.any() else np.nan,
                    empirical_accuracy=float(z[selected].mean()) if selected.any() else np.nan,
                    **{f"n_{c}":int((y[selected] == i).sum()) for i,c in enumerate(CLASSES)}))
            for threshold in THRESHOLDS:
                high = q >= threshold
                thresholds.append(dict(candidate=name,cohort=cohort,threshold=threshold,n_high=int(high.sum()),
                    high_correct=int((high & z).sum()),high_wrong=int((high & ~z).sum()),coverage=float(high.mean())))
            uniform = np.full_like(P,1/3)
            flattened = metrics(uniform,y,chosen=P.argmax(axis=1))
            controls.append(dict(control="uniform_confidence_same_bets",candidate=name,cohort=cohort,**flattened))
        part = subset[subset.candidate == "legacy_l1"].sort_values("site").reset_index(drop=True)
        y = part.true_batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
        counts = np.array([7,7,17])
        prior = np.vstack([(counts-np.eye(3,dtype=int)[c])/30 for c in y]) if cohort == "known_oof" else np.tile(counts/31,(3,1))
        controls.append(dict(control="train_prior_vector",candidate="train_prior_vector",cohort=cohort,**metrics(prior,y)))
        controls.append(dict(control="uniform_vector",candidate="uniform_vector",cohort=cohort,
            **metrics(np.full((len(y),3),1/3),y)))
        for reference in ["legacy_l1","base_l1","gated_l2"]:
            ref = subset[subset.candidate == reference].set_index("site").sort_index()
            for name in ORDER:
                if name == reference:
                    continue
                now = subset[subset.candidate == name].set_index("site").reindex(ref.index)
                same = now.predicted_batch.to_numpy() == ref.predicted_batch.to_numpy()
                a, b = now.correct.to_numpy(bool), ref.correct.to_numpy(bool)
                delta = now.bet_score.to_numpy()-ref.bet_score.to_numpy()
                samewrong, sameright = same & ~b, same & b
                y = ref.true_batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
                losses_now = -np.log(np.clip(now.true_class_score.to_numpy(),1e-12,1))
                losses_ref = -np.log(np.clip(ref.true_class_score.to_numpy(),1e-12,1))
                paired.append(dict(candidate=name,reference=reference,cohort=cohort,
                    changed_bets=int((~same).sum()),newly_correct=int((a & ~b).sum()),newly_wrong=int((~a & b).sum()),
                    same_wrong_n=int(samewrong.sum()),same_wrong_mean_bet_score_delta=float(delta[samewrong].mean()) if samewrong.any() else np.nan,
                    same_correct_n=int(sameright.sum()),same_correct_mean_bet_score_delta=float(delta[sameright].mean()) if sameright.any() else np.nan,
                    mean_true_class_score_delta=float((now.true_class_score-ref.true_class_score).mean()),
                    multiclass_logloss_delta=weighted_mean(losses_now-losses_ref,y),
                    balanced_multiclass_logloss_delta=weighted_mean(losses_now-losses_ref,y,True)))
    for file, rows in [("confidence_summary.csv",summary),("reliability_bins.csv",bins),
                       ("threshold_counts.csv",thresholds),("controls_summary.csv",controls),("paired_changes.csv",paired)]:
        pd.DataFrame(rows).to_csv(OUT / file,index=False)
    check_hashes(inputs);check_hashes(sources);check_hashes(probe["protected_artifacts_sha256"])
    check_hashes(probe["raw_sha256"])
    outputs = {str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.glob("*.csv"))}
    write_json(OUT / "audit_receipt.json",dict(experiment="E42S",candidate_order=ORDER,aliases_deduplicated=aliases,
        inputs_sha256=inputs,sources_sha256=sources,outputs_sha256=outputs,no_calibration_fit=True,
        exact_organiser_rule_available=False,utility_reported=False,threshold_selection=False,
        source_parent_ids_available=False,external_accuracy_claim=False,submission_promotion=False))
    print(pd.DataFrame(summary)[["candidate","cohort","balanced_accuracy","balanced_multiclass_logloss",
          "balanced_multiclass_brier","balanced_bet_brier","balanced_bet_nll"]].to_string(index=False))


if __name__ == "__main__":
    execute()
