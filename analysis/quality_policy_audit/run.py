"""Run the prespecified E37S known-only policy comparison and failure diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from analysis.submission_v2.runtime import load_runtime
from . import policy as qp

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CLASSES = ["Batch_1", "Batch_2", "Batch_3"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def null_one(matrices, labels):
    return {name:qp.balanced_accuracy(labels, qp.nested_loo(X, labels)[0])
            for name, X in matrices.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "output")
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Preserve audit outputs; use a new directory")
    if args.n_perm < 0 or args.workers < 1:
        raise ValueError("Invalid permutation/worker count")
    started = datetime.now(timezone.utc).isoformat()
    protected = json.loads((ROOT / "analysis/feedback_drop_01/output/receipt.json").read_text())["protected_artifacts_sha256"]
    for rel, expected in protected.items():
        assert sha(ROOT / rel) == expected, rel
    snapshot, cat = load_runtime(ROOT / "analysis/submission_v2/freeze")
    assert qp.FEATURES == snapshot["families"]["material"] == cat.FAMILIES["material"]
    assert set(qp.BRIGHT + qp.PORE + qp.PARTICLE_INLENS + qp.JOINT + qp.ANCHORS) == set(qp.FEATURES)
    assert len(qp.BRIGHT + qp.PORE + qp.PARTICLE_INLENS + qp.JOINT + qp.ANCHORS) == 29
    known_path = ROOT / "analysis/submission_v2/freeze/known_sites.csv"
    assert sha(known_path) == snapshot["known_sites_sha256"]
    known = pd.read_csv(known_path)
    assert known.site.is_unique and known.groupby("batch").size().to_dict() == {"Batch_1":7, "Batch_2":7, "Batch_3":17}
    flags = qp.quality(known)
    assert np.array_equal(flags.pore_bad.to_numpy(), known.grey_pore.to_numpy(bool))
    order = qp.shared_order(known)
    known = known.iloc[order].reset_index(drop=True)
    labels = known.batch.map({name:i for i,name in enumerate(CLASSES)}).to_numpy(int)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = sorted(HERE.glob("*.py")) + [HERE / "protocol.md", ROOT / "tests/test_quality_policy_audit.py"]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    for path in sources:
        target = args.out / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    known.to_csv(args.out / "known_sites.csv", index=False)
    catalogue = []
    for feature in qp.FEATURES:
        group = ("bright mask" if feature in qp.BRIGHT else "pore mask" if feature in qp.PORE
                 else "bright interior" if feature in qp.PARTICLE_INLENS else "joint phase interiors/orientation" if feature in qp.JOINT
                 else "sampled raw BSE image")
        catalogue.append(dict(feature=feature, extraction_dependency=group,
                              invalid_when_bright_bad=feature in qp.BRIGHT + qp.PARTICLE_INLENS + qp.JOINT,
                              invalid_when_pore_bad=feature in qp.PORE + qp.JOINT,
                              source="polaron_qc/features.py:bse_site_features" if feature in qp.BRIGHT + qp.PORE + qp.ANCHORS
                              else "polaron_qc/features.py:multichannel_features",
                              qualification="dependency policy; no expert mask validation; appearance remains acquisition-sensitive"))
    pd.DataFrame(catalogue).to_csv(args.out / "feature_dependencies.csv", index=False)

    matrices = {}; results = {}; fits = {}; audits = []; rows = []
    print("Known-only stage: all 31 sites, shared folds and nested C selection", flush=True)
    for variant in qp.VARIANTS:
        X, names, audit, q = qp.prepare(known, variant)
        matrices[variant] = X
        if not audit.empty: audits.append(audit)
        P, choices, folds = qp.nested_loo(X, labels, keep_models=True)
        met = cat.metrics(labels, P, CLASSES); cal = cat.calibration(labels, P)
        result = dict(variant=variant, n_sites=31, n_features=len(names), n_correct=met["n_correct"],
                      accuracy=met["accuracy"], balanced_accuracy=met["balanced_accuracy"],
                      **{f"recall_{name}":met["recall"][name] for name in CLASSES},
                      brier=cal["brier"], brier_prior=cal["brier_prior"], ece_descriptive=cal["ece"],
                      C_values="|".join(str(v) for v in sorted(set(choices))))
        results[variant] = dict(P=P, choices=choices, folds=folds, summary=result)
        full = qp.fit_model(known, variant); fits[variant] = full
        met["confusion"].to_csv(args.out / f"confusion_{variant}.csv")
        cal["reliability"].to_csv(args.out / f"reliability_{variant}.csv", index=False)
        for i, row in known.iterrows():
            rows.append(dict(variant=variant, site=row.site, true_batch=row.batch,
                             predicted_batch=CLASSES[int(P[i].argmax())], C=choices[i],
                             correct=bool(P[i].argmax() == labels[i]),
                             n_observed_inputs=int(np.isfinite(X[i]).sum()),
                             bright_bad=bool(q.iloc[i].bright_bad), pore_bad=bool(q.iloc[i].pore_bad),
                             **{f"score_{name}":float(P[i,j]) for j,name in enumerate(CLASSES)}))
        print(f"  {variant}: balanced accuracy {met['balanced_accuracy']:.3f}, recall {list(met['recall'].values())}", flush=True)
    known_predictions = pd.DataFrame(rows)
    known_predictions.to_csv(args.out / "known_oof_predictions.csv", index=False)
    audit_table = pd.concat(audits, ignore_index=True)
    audit_table.to_csv(args.out / "known_input_eligibility.csv", index=False)
    availability = audit_table.groupby(["variant", "feature"], sort=False).agg(
        n_sites=("site", "size"), n_observed=("observed_model_input", "sum"), n_quality_eligible=("policy_eligible", "sum")).reset_index()
    availability.to_csv(args.out / "feature_availability.csv", index=False)
    subgroup = known_predictions.assign(group=np.select(
        [known_predictions.bright_bad & known_predictions.pore_bad, known_predictions.bright_bad, known_predictions.pore_bad],
        ["both", "bright_bad", "pore_bad"], default="ordinary"))
    subgroup.groupby(["variant", "group"], sort=False).agg(n_sites=("site", "size"), n_correct=("correct", "sum"),
        n_observed_inputs_min=("n_observed_inputs", "min")).reset_index().to_csv(args.out / "quality_subgroups.csv", index=False)

    null_rows = []
    rng = np.random.default_rng(0)
    permutations = [rng.permutation(labels) for _ in range(args.n_perm)]
    for start in range(0, args.n_perm, 25):
        batch = Parallel(n_jobs=args.workers, prefer="threads")(
            delayed(null_one)(matrices, perm) for perm in permutations[start:start + 25])
        null_rows.extend(batch)
        print(f"  complete nested site-label permutations: {len(null_rows)}/{args.n_perm}", flush=True)
    null = pd.DataFrame(null_rows, columns=qp.VARIANTS)
    null.to_csv(args.out / "permutation_null.csv", index=False)
    for variant in qp.VARIANTS:
        observed = results[variant]["summary"]["balanced_accuracy"]
        results[variant]["summary"]["n_perm"] = args.n_perm
        results[variant]["summary"]["perm_p"] = float((1 + (null[variant] >= observed - 1e-12).sum()) / (1 + args.n_perm)) if args.n_perm else None
    legacy_pred = results["legacy_matched"]["P"].argmax(axis=1)
    primary_pred = results["dependency_gate"]["P"].argmax(axis=1)
    delta = qp.balanced_accuracy(labels, results["dependency_gate"]["P"]) - qp.balanced_accuracy(labels, results["legacy_matched"]["P"])
    intervals = []
    boot_rng = np.random.default_rng(37)
    for _ in range(5000):
        idx = np.concatenate([boot_rng.choice(np.flatnonzero(labels == c), int((labels == c).sum()), replace=True) for c in range(3)])
        intervals.append(float(np.mean([(primary_pred[idx][labels[idx] == c] == c).mean() -
                                       (legacy_pred[idx][labels[idx] == c] == c).mean() for c in range(3)])))
    difference = dict(balanced_accuracy_delta=delta,
                      descriptive_paired_oof_site_interval=np.percentile(intervals, [2.5, 97.5]).tolist(),
                      newly_correct=int(((primary_pred == labels) & (legacy_pred != labels)).sum()),
                      newly_wrong=int(((primary_pred != labels) & (legacy_pred == labels)).sum()),
                      changed_assignments=int((primary_pred != legacy_pred).sum()),
                      advantage_perm_p=float((1 + ((null.dependency_gate - null.legacy_matched) >= delta - 1e-12).sum()) / (1 + args.n_perm)) if args.n_perm else None,
                      note="paired OOF resampling is descriptive; overlapping folds and unknown specimen dependence; feedback-informed policy")
    json_write(args.out / "paired_comparison.json", difference)
    summary = pd.DataFrame([results[v]["summary"] for v in qp.VARIANTS])
    summary.to_csv(args.out / "known_summary.csv", index=False)
    model_path = args.out / "audit_models.joblib"
    joblib.dump(dict(models=fits, folds={v:results[v]["folds"] for v in qp.VARIANTS}, known_site_order=known.site.tolist()), model_path, compress=3)
    json_write(args.out / "known_stage_receipt.json", dict(experiment="E37S", version=qp.VERSION,
               completed_utc=datetime.now(timezone.utc).isoformat(), sources_sha256=hashes,
               known_source_sha256=sha(known_path), counts=known.batch.value_counts().to_dict(),
               models_sha256=sha(model_path), first_drop_features_read=False, no_drop_labels_used=True,
               shared_order="SHA256 of finite unique raw-BSE corr_len_px and fft_slope; no site or class label",
               summary=summary.to_dict("records"), paired_comparison=difference,
               pipeline="fold-local median imputation without indicators; keep empty slots; scaling; balanced L1 logistic; nested C .1/.5/2; seed0; inner3",
               site_permutations=args.n_perm, quality_data_rule_matches_known_grey_flags=True))
    print("Known-only models and evaluation saved. Now reading labelled failure-example features; no refit.", flush=True)

    # These are development diagnostics only. No drop data enter the stage above.
    failure_path = ROOT / "analysis/submission_v2/first_run/scored/features.csv"
    failure = pd.read_csv(failure_path)
    feedback_path = ROOT / "analysis/feedback_drop_01/truth.csv"
    truth = pd.read_csv(feedback_path).set_index("sample_id").true_batch.to_dict()
    failure_rows = []; failure_audits = []; contrasts = []; forbidden_legacy = []
    for variant, model in fits.items():
        P, audit, q = model.predict(failure)
        X, names, _, _ = qp.prepare(failure, variant)
        if not audit.empty: failure_audits.append(audit)
        for i, row in failure.iterrows():
            order_prob = np.argsort(-P[i], kind="stable")
            a, b = int(order_prob[0]), int(order_prob[1])
            votes = np.array([qp.proba(pipe, X[i:i + 1])[0] for pipe in results[variant]["folds"]])
            failure_rows.append(dict(variant=variant, site=row.site, true_batch=truth[row.site],
                                     predicted_batch=CLASSES[a], correct=bool(CLASSES[a] == truth[row.site]),
                                     top_model_score=float(P[i,a]), true_batch_model_score=float(P[i,CLASSES.index(truth[row.site])]),
                                     runner_up=CLASSES[b], margin=float(P[i,a] - P[i,b]),
                                     deletion_same_assignment=int((votes.argmax(axis=1) == a).sum()), deletion_fit_count=len(votes),
                                     n_observed_inputs=int(np.isfinite(X[i]).sum()), n_declared_inputs=len(names),
                                     bright_bad=bool(q.iloc[i].bright_bad), pore_bad=bool(q.iloc[i].pore_bad),
                                     qualification="feedback-informed development example; scores uncalibrated; no model selection or accuracy validation"))
            z = model.pipeline["scale"].transform(model.pipeline["impute"].transform(X[i:i + 1]))[0]
            lr = model.pipeline["lr"]; terms = (lr.coef_[a] - lr.coef_[b]) * z
            intercept = float(lr.intercept_[a] - lr.intercept_[b])
            logits = model.pipeline.decision_function(X[i:i + 1])[0]
            np.testing.assert_allclose(intercept + terms.sum(), logits[a] - logits[b], atol=1e-10)
            for j, feature in enumerate(names):
                contrasts.append(dict(variant=variant, site=row.site, assigned_batch=CLASSES[a], comparison_batch=CLASSES[b],
                                      feature=feature, contribution=float(terms[j]), intercept_difference=intercept,
                                      observed=bool(np.isfinite(X[i,j])), input_value=X[i,j],
                                      train_observed_count=model.train_observed_counts[j],
                                      evidence_type="observed input (validity not expert-confirmed)" if np.isfinite(X[i,j])
                                      else "imputed baseline term; not a measured driver"))
                if variant == "legacy_matched":
                    primary_X = qp.prepare(failure, "dependency_gate")[0]
                    if not np.isfinite(primary_X[i,j]) and np.isfinite(X[i,j]):
                        forbidden_legacy.append(dict(site=row.site, feature=feature, legacy_pairwise_contribution=float(terms[j]),
                                                     selected_in_legacy=bool(lr.coef_[:,j].any())))
    pd.DataFrame(failure_rows).to_csv(args.out / "failure_examples.csv", index=False)
    pd.concat(failure_audits, ignore_index=True).to_csv(args.out / "failure_input_eligibility.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.out / "failure_logit_contrasts.csv", index=False)
    pd.DataFrame(forbidden_legacy).to_csv(args.out / "legacy_invalid_input_influence.csv", index=False)
    for variant, model in fits.items():
        if variant == "quality_flags_only": continue
        for j, feature in enumerate(model.feature_names):
            selected = model.pipeline["lr"].coef_[:,j].tolist()
            rows = dict(variant=variant, feature=feature, train_observed_count=model.train_observed_counts[j],
                        **{f"coef_{name}":selected[k] for k,name in enumerate(CLASSES)})
            # Append small model tables without modifying measurements.
            if j == 0: coefficient_rows = []
            coefficient_rows.append(rows)
        pd.DataFrame(coefficient_rows).to_csv(args.out / f"coefficients_{variant}.csv", index=False)
    for rel, expected in hashes.items(): assert sha(ROOT / rel) == expected, rel
    for rel, expected in protected.items(): assert sha(ROOT / rel) == expected, rel
    json_write(args.out / "receipt.json", dict(experiment="E37S", decision="D56S", version=qp.VERSION,
               started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
               sources_sha256=hashes, known_source_sha256=sha(known_path), failure_features_sha256=sha(failure_path),
               feedback_sha256=sha(feedback_path), protected_artifacts_sha256=protected,
               artifact_sha256={str(p.relative_to(args.out)):sha(p) for p in sorted(args.out.rglob("*")) if p.is_file()},
               no_drop_data_in_training_or_selection=True, post_feedback_development=True,
               frozen_submission_unchanged=True, no_live_package_or_QC_change=True,
               dependency_candidate_auto_promoted=False, known_stage_saved_before_drop_read=True))
    print(summary[["variant", "balanced_accuracy", "accuracy", "perm_p"]].to_string(index=False), flush=True)
    print(pd.DataFrame(failure_rows)[["variant", "site", "predicted_batch", "top_model_score", "n_observed_inputs"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
