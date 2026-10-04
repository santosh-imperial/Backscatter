"""E41S fixed comparison; save known stage before reading revealed drop features."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.classification_m1_m2.run import sha, write_json, prediction_rows
from analysis.forest_geometry import models as m, geometry as g
from analysis.quality_policy_audit import policy as qp

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CLASSES = m.CLASSES
SCORE_COLS = [f"score_{c}" for c in CLASSES]


def join_geometry(base, extra, check_batch=True):
    """One-to-one join; preserve original flags/anchors instead of overwriting."""
    if not base.site.is_unique or not extra.site.is_unique or set(base.site) != set(extra.site):
        raise ValueError("Geometry join requires exactly the same unique sites")
    extra = extra.set_index("site").reindex(base.site)
    if check_batch and not np.array_equal(base.batch.to_numpy(), extra.batch.to_numpy()):
        raise ValueError("Geometry label metadata disagrees with frozen known labels")
    out = base.copy()
    for column in extra:
        if column not in out:
            out[column] = extra[column].to_numpy()
    if set(g.FEATURES) - set(out):
        raise ValueError("Missing fixed geometry inputs")
    for column in ["bse_p1", "bright_low_contrast", "th_lo", "th_hi"]:
        if column in base and column in extra:
            a, b = base[column].to_numpy(), extra[column].to_numpy()
            equal = np.allclose(a, b, atol=1e-8, rtol=0, equal_nan=True) if column != "bright_low_contrast" else np.array_equal(a, b)
            if not equal:
                raise ValueError(f"Frozen/raw geometry acquisition field mismatch: {column}")
    return out


def current_protection():
    frozen = json.loads((ROOT / "analysis/feedback_drop_01/output/receipt.json").read_text())["protected_artifacts_sha256"]
    for dirname in ["analysis/classification_m1_m2", "analysis/source_crop_audit"]:
        for p in sorted((ROOT / dirname).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                frozen[str(p.relative_to(ROOT))] = sha(p)
    raw = {str(p.relative_to(ROOT)): sha(p) for dirname in ["Dataset", "Hackathon-Polaron-test"]
           for p in sorted((ROOT / dirname).rglob("*.tif"))}
    # Files use .tiff on some exports; both extensions are covered.
    raw.update({str(p.relative_to(ROOT)): sha(p) for dirname in ["Dataset", "Hackathon-Polaron-test"]
                for p in sorted((ROOT / dirname).rglob("*.tiff"))})
    check_hashes(frozen)
    return frozen, raw


def check_hashes(hashes):
    for rel, digest in hashes.items():
        if sha(ROOT / rel) != digest:
            raise AssertionError(f"Protected source/artifact changed: {rel}")


def source_files():
    paths = list(HERE.glob("*.py")) + [HERE / "protocol.md", HERE / "protocol_geometry.md",
        ROOT / "analysis/classification_m1_m2/models.py", ROOT / "analysis/classification_m1_m2/run.py",
        ROOT / "analysis/quality_policy_audit/policy.py", ROOT / "polaron_qc/features.py",
        ROOT / "analysis/morphology/benchmark_methods.py", ROOT / "analysis/morphology/k_pilot/descriptors.py",
        ROOT / "analysis/ml_options/e_graph/graph.py", ROOT / "tests/test_forest_models.py",
        ROOT / "tests/test_forest_geometry.py"]
    return sorted(set(paths))


def explanations(candidate, cohort, model, frame, X, names, scores):
    result = []
    kind = m.CANDIDATES[candidate][1]
    for i, row in frame.iterrows():
        chosen = int(scores[i].argmax())
        other = 2 if chosen != 2 else int(np.argsort(scores[i])[-2])
        terms, root, margin = m.explain(model, X[i:i+1], names, chosen, other, kind)
        for rank, term in enumerate(terms, 1):
            result.append(dict(candidate=candidate, cohort=cohort, site=row.site,
                predicted_batch=CLASSES[chosen], comparison_batch=CLASSES[other], rank=rank,
                root_or_intercept_margin=root, total_margin=margin, **term))
    return result


def save_models(out, name, full, folds):
    target = out / "fits" / name
    target.mkdir(parents=True)
    joblib.dump(full, target / "known31.joblib", compress=3)
    for i, model in enumerate(folds):
        joblib.dump(model, target / f"fold_{i:02d}.joblib", compress=3)


def known_stage(out, extraction):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh evaluation directory; never overwrite a trial")
    protected, raw = current_protection()
    known_path = ROOT / "analysis/submission_v2/freeze/known_sites.csv"
    geometry_path = extraction / "geometry_known.csv"
    verification = json.loads((extraction / "geometry_known_verification.json").read_text())
    if verification["fresh_nominal_matches"] != verification["fresh_nominal_example_cells"] or not verification["inputs_unchanged"]:
        raise ValueError("Geometry verification failed")
    known = join_geometry(pd.read_csv(known_path), pd.read_csv(geometry_path))
    if known.groupby("batch").size().to_dict() != {"Batch_1": 7, "Batch_2": 7, "Batch_3": 17}:
        raise ValueError("Known 31-label contract changed")
    known = known.iloc[qp.shared_order(known)].reset_index(drop=True)
    y = known.batch.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(int)
    out.mkdir(parents=True)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_files()}
    for rel in sources:
        target = out / "sources" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    known.to_csv(out / "known_measurements.csv", index=False)
    _, _, old_audit, q = qp.prepare(known, "dependency_gate")
    _, _, extra_audit, _ = g.prepare(known)
    old_audit["batch"] = old_audit.site.map(known.set_index("site").batch)
    cell_audit = pd.concat([old_audit, extra_audit], ignore_index=True)
    cell_audit.to_csv(out / "cell_availability.csv", index=False)
    cell_audit.groupby(["batch", "feature"]).agg(n_crops=("site", "size"),
        observed=("observed_model_input", "sum"), rejected=("policy_eligible", lambda x: (~x).sum())).reset_index().to_csv(out / "availability_by_feature.csv", index=False)
    summaries, predictions, attrs, subgroup = [], [], [], []
    outer = {}
    for name, (view, kind) in m.CANDIDATES.items():
        X, names = m.matrix(known, view)
        P, choices, folds = m.nested_loo(X, y, kind)
        full = m.fit(known, name)
        save_models(out, name, full, folds)
        outer[name] = (P, choices, folds)
        summary = m.metrics(y, P)
        confusion = summary.pop("confusion")
        summary.update(candidate=name, n_features=len(names), C_values="|".join(str(c) for c in sorted(set(choices), key=str)))
        summaries.append(summary)
        predictions.extend(prediction_rows(name, known, P, X, choices))
        pd.DataFrame(confusion, index=CLASSES, columns=CLASSES).to_csv(out / f"confusion_{name}.csv")
        for i, model in enumerate(folds):
            attrs.extend(explanations(name, "known_oof", model, known.iloc[i:i+1].reset_index(drop=True), X[i:i+1], names, P[i:i+1]))
        for label, selected in {"quality_unflagged": ~q.bright_bad & ~q.pore_bad,
                                "bright_bad_or_unknown": q.bright_bad, "pore_bad_or_unknown": q.pore_bad,
                                "any_quality_flag": q.bright_bad | q.pore_bad}.items():
            idx = selected.to_numpy(bool)
            subgroup.append(dict(candidate=name, subgroup=label, n_crops=int(idx.sum()),
                n_correct=int((P[idx].argmax(axis=1) == y[idx]).sum()),
                n_Batch_1=int((y[idx] == 0).sum()), n_Batch_2=int((y[idx] == 1).sum()), n_Batch_3=int((y[idx] == 2).sum())))
        print(f"{name}: BA={summary['balanced_accuracy']:.4f}, correct={summary['n_correct']}/31, recalls=" +
              "/".join(f"{summary['recall_'+c]:.3f}" for c in CLASSES), flush=True)
    predictions = pd.DataFrame(predictions)
    pd.DataFrame(summaries).to_csv(out / "known_summary.csv", index=False)
    predictions.to_csv(out / "known_oof_predictions.csv", index=False)
    pd.DataFrame(attrs).to_csv(out / "explanations_known.csv", index=False)
    pd.DataFrame(subgroup).to_csv(out / "quality_subgroups.csv", index=False)
    old = pd.read_csv(ROOT / "analysis/classification_m1_m2/output/evaluation/known_oof_predictions.csv")
    replay = {}
    for name, old_name in [("legacy_l1", "legacy_l1"), ("base_l1", "gated_l1")]:
        a = predictions[predictions.candidate == name].set_index("site")[SCORE_COLS].sort_index().to_numpy()
        b = old[old.candidate == old_name].set_index("site")[SCORE_COLS].sort_index().to_numpy()
        delta = float(np.abs(a-b).max())
        replay[name] = dict(max_score_difference=delta, matches=delta < 1e-12)
        if delta >= 1e-12:
            raise AssertionError("Matched L1 replay differs")
    comparisons = []
    for candidate, reference in [("base_rf", "base_l1"), ("geometry_l1", "base_l1"),
                                 ("geometry_rf", "base_rf"), ("geometry_rf", "base_l1"), ("geometry_rf", "legacy_l1")]:
        a, b = outer[candidate][0].argmax(axis=1), outer[reference][0].argmax(axis=1)
        comparisons.append(dict(candidate=candidate, reference=reference, changed=int((a != b).sum()),
            newly_correct=int(((a == y) & (b != y)).sum()), newly_wrong=int(((a != y) & (b == y)).sum()),
            balanced_accuracy_delta=m.metrics(y, outer[candidate][0])["balanced_accuracy"] - m.metrics(y, outer[reference][0])["balanced_accuracy"]))
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    check_hashes(protected); check_hashes(raw); check_hashes(sources)
    registry = (ROOT / "experiments/registry.csv").read_bytes()
    write_json(out / "known_stage_receipt.json", dict(experiment="E41S", stage="known_only_complete",
        version=m.VERSION, completed=datetime.now(timezone.utc).isoformat(), candidate_order=list(m.CANDIDATES),
        primary="geometry_rf", known_fit_n=31, source_parent_ids_available=False,
        no_human_annotations_used=True, first_drop_features_or_truth_used=False, sources_sha256=sources,
        known_inputs_sha256={str(known_path.relative_to(ROOT)): sha(known_path), str(geometry_path.relative_to(ROOT)): sha(geometry_path)},
        matched_L1_reproduction=replay, protected_artifacts_sha256=protected, raw_sha256=raw,
        registry_prefix_bytes=len(registry), registry_prefix_sha256=sha(ROOT / "experiments/registry.csv"),
        validation="crop-LOO development; unknown shared-parent leakage; no IID p-values/intervals",
        submission_promotion=False))
    print("Known-only complete; receipt saved. Revealed drop may now be extracted for failure diagnostics.", flush=True)


def load_fits(out, name):
    target = out / "fits" / name
    return joblib.load(target / "known31.joblib"), [joblib.load(target / f"fold_{i:02d}.joblib") for i in range(31)]


def diagnostic_stage(out, extraction):
    receipt_path = out / "known_stage_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["stage"] != "known_only_complete" or (out / "receipt.json").exists():
        raise ValueError("Requires untouched completed known stage, and no existing diagnostics")
    check_hashes(receipt["sources_sha256"])
    check_hashes(receipt["known_inputs_sha256"])
    check_hashes(receipt["protected_artifacts_sha256"]); check_hashes(receipt["raw_sha256"])
    known = pd.read_csv(out / "known_measurements.csv")
    original = pd.read_csv(out / "known_oof_predictions.csv")
    variants = pd.read_csv(extraction / "geometry_known_variants.csv")
    y = known.batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
    fits = {name: load_fits(out, name) for name in m.CANDIDATES}
    sensitivity = []
    for variant in g.VARIANTS:
        # Replace ALL geometry metadata so changed coverage is rechecked, leaving frozen29/flags fixed.
        extra = variants[variants.variant == variant].copy()
        base_columns = [c for c in known if c not in extra.columns or c in ["site", "batch", "bse_p1", "bright_low_contrast", "th_lo", "th_hi"] or c in qp.FEATURES]
        changed_frame = join_geometry(known[base_columns], extra)
        for name in ["geometry_l1", "geometry_rf"]:
            view, _ = m.CANDIDATES[name]
            nominal, names = m.matrix(known, view)
            X, variant_names = m.matrix(changed_frame, view)
            assert names == variant_names
            X = m.preserve_nominal_availability(nominal, X)
            old = original[original.candidate == name].set_index("site").reindex(known.site)[SCORE_COLS].to_numpy()
            for i, model in enumerate(fits[name][1]):
                P = m.probabilities(model, X[i:i+1])[0]
                sensitivity.append(dict(candidate=name, variant=variant, site=known.iloc[i].site,
                    true_batch=known.iloc[i].batch, original_bet=CLASSES[old[i].argmax()], perturbed_bet=CLASSES[P.argmax()],
                    bet_changed=bool(P.argmax() != old[i].argmax()), max_absolute_score_change=float(np.abs(P-old[i]).max()),
                    nominal_observed_inputs=int(np.isfinite(nominal[i]).sum()), perturbed_observed_inputs=int(np.isfinite(X[i]).sum()),
                    **{f"score_{c}":float(P[j]) for j,c in enumerate(CLASSES)}))
    pd.DataFrame(sensitivity).to_csv(out / "geometry_oof_sensitivity.csv", index=False)
    X, names = m.matrix(known, "geometry")
    primary = original[original.candidate == "geometry_rf"].set_index("site").reindex(known.site)[SCORE_COLS].to_numpy()
    seed_rows, seed_summary = [], []
    for seed in [1, 2]:
        P, _, _ = m.nested_loo(X, y, "rf", seed=seed)
        met = m.metrics(y, P); met.pop("confusion")
        seed_summary.append(dict(seed=seed, bet_changes_from_seed0=int((P.argmax(axis=1) != primary.argmax(axis=1)).sum()), **met))
        for i, row in known.iterrows():
            seed_rows.append(dict(seed=seed, site=row.site, true_batch=row.batch, original_bet=CLASSES[primary[i].argmax()],
                perturbed_bet=CLASSES[P[i].argmax()], bet_changed=bool(P[i].argmax() != primary[i].argmax()),
                max_absolute_score_change=float(np.abs(P[i]-primary[i]).max()), **{f"score_{c}":float(P[i,j]) for j,c in enumerate(CLASSES)}))
        print(f"Diagnostic seed{seed}: BA={met['balanced_accuracy']:.4f}; no seed selection", flush=True)
    pd.DataFrame(seed_rows).to_csv(out / "seed_sensitivity.csv", index=False)
    pd.DataFrame(seed_summary).to_csv(out / "seed_summary.csv", index=False)
    replacement = []
    for i, model in enumerate(fits["geometry_rf"][1]):
        chosen = int(primary[i].argmax()); other = 2 if chosen != 2 else int(np.argsort(primary[i])[-2])
        for group, features in m.GROUPS.items():
            changed, count = m.replace_observed_group(model, X[i:i+1], names, features)
            P = m.probabilities(model, changed)[0]
            replacement.append(dict(candidate="geometry_rf", site=known.iloc[i].site, feature_group=group,
                replacement_count=count, original_bet=CLASSES[chosen], perturbed_bet=CLASSES[P.argmax()],
                bet_changed=bool(chosen != P.argmax()), max_absolute_score_change=float(np.abs(P-primary[i]).max()),
                original_margin=float(primary[i, chosen]-primary[i, other]), perturbed_margin=float(P[chosen]-P[other]),
                empty_train_column_fallback="zero; unavailable query cells remain unavailable",
                **{f"score_{c}":float(P[j]) for j,c in enumerate(CLASSES)}))
    pd.DataFrame(replacement).to_csv(out / "group_replacement.csv", index=False)
    # Only now load previously revealed development drop; never train on these labels.
    drop = join_geometry(pd.read_csv(ROOT / "analysis/submission_v2/first_run/scored/features.csv"),
                         pd.read_csv(extraction / "geometry_first_drop.csv"), check_batch=False)
    truth = json.loads((ROOT / "analysis/feedback_drop_01/truth.json").read_text())["labels"]
    if set(drop.site) != set(truth):
        raise ValueError("Revealed drop IDs differ")
    drop["batch"] = drop.site.map(truth)
    drop = drop.sort_values("site").reset_index(drop=True)
    drop.to_csv(out / "feedback_measurements.csv", index=False)
    prediction, attrs = [], pd.read_csv(out / "explanations_known.csv").to_dict("records")
    for name, (view, _) in m.CANDIDATES.items():
        full = fits[name][0]
        X, names = m.matrix(drop, view)
        P = full.predict(drop)
        prediction.extend(prediction_rows(name, drop, P, X))
        attrs.extend(explanations(name, "feedback_development", full.pipeline, drop, X, names, P))
    pd.DataFrame(prediction).to_csv(out / "feedback_development_predictions.csv", index=False)
    pd.DataFrame(attrs).to_csv(out / "explanations.csv", index=False)
    check_hashes(receipt["protected_artifacts_sha256"]); check_hashes(receipt["raw_sha256"])
    check_hashes(receipt["sources_sha256"])
    write_json(out / "receipt.json", dict(**receipt, diagnostic_completed=datetime.now(timezone.utc).isoformat(),
        diagnostic_stage="complete", known_stage_receipt_sha256=sha(receipt_path), feedback_diagnostic_n=3,
        feedback_inputs_sha256={str((extraction / "geometry_first_drop.csv").relative_to(ROOT)):sha(extraction / "geometry_first_drop.csv"),
                               "analysis/feedback_drop_01/truth.json": sha(ROOT / "analysis/feedback_drop_01/truth.json")},
        score_interpretation="uncalibrated balanced-target model probabilities; not correctness confidence",
        family_replacement="fixed-model train-median input sensitivity; correlated hybrid may be off support; noncausal",
        geometry_variants="partial-input mask sensitivity, old29 fixed; nominal unavailable stays unavailable"))
    print("All fixed diagnostics saved; original submission untouched.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["known", "diagnostics"], required=True)
    parser.add_argument("--out", type=Path, default=HERE / "output/evaluation")
    parser.add_argument("--extraction", type=Path, default=HERE / "output")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (known_stage if args.stage == "known" else diagnostic_stage)(args.out, args.extraction)


if __name__ == "__main__":
    main()
