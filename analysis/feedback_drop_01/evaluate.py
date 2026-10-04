"""Evaluate saved first-drop bets against separately recorded organiser feedback.

Reads saved features/models to check class mapping and explanations; fits nothing.
Run from the repository root in a fresh Python process:
    /opt/anaconda3/bin/python3 -m analysis.feedback_drop_01.evaluate
Use --out with a new directory for a verification replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analysis.submission_v2.runtime import load_runtime

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CLASSES = ["Batch_1", "Batch_2", "Batch_3"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "output")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Preserve feedback outputs; choose a new directory")

    evidence = json.loads((HERE / "truth.json").read_text())
    truth = pd.read_csv(HERE / "truth.csv")
    assert truth.sample_id.is_unique
    assert truth.set_index("sample_id").true_batch.to_dict() == evidence["labels"]
    assert truth.sample_id.tolist() == evidence["submitted_order"]
    assert sorted(truth.true_batch.tolist()) == CLASSES

    protected = {}
    for directory in ["analysis/organiser_drop_01", "analysis/submission_v2"]:
        receipt = ROOT / directory / "verification.json"
        protected[str(receipt.relative_to(ROOT))] = sha(receipt)
        for rel, expected in json.loads(receipt.read_text())["artifact_sha256"].items():
            path = ROOT / directory / rel
            assert sha(path) == expected, path
            protected[str(path.relative_to(ROOT))] = expected
    for rel in ["analysis/submission_test/predictions.csv",
                "analysis/submission_test/submission.md",
                "analysis/submission_test/categoriser/site_table.csv"]:
        protected[rel] = sha(ROOT / rel)
    registry_before = (ROOT / "experiments/registry.csv").read_bytes()

    snapshot, cat = load_runtime(ROOT / "analysis/submission_v2/freeze")
    saved = joblib.load(ROOT / "analysis/submission_v2/freeze/models.joblib")
    features = pd.read_csv(ROOT / "analysis/submission_v2/first_run/scored/features.csv")
    table = pd.read_csv(ROOT / "analysis/submission_v2/first_run/scored/site_table.csv")
    bets = pd.read_csv(ROOT / "analysis/submission_v2/first_run/submission/predictions.csv")
    drivers = pd.read_csv(ROOT / "analysis/submission_v2/first_run/scored/driver_contrasts.csv")
    assert set(features.site) == set(truth.sample_id) == set(table.site) == set(bets.sample_id)
    assert all(frame.site.is_unique for frame in [features, table])
    mapping_checks = {}
    for family, model in saved["models"].items():
        assert model.classes == CLASSES
        assert model.pipeline["lr"].classes_.tolist() == [0, 1, 2]
        encoded_classes, y = cat._encode(saved["known"])
        assert encoded_classes == CLASSES
        assert np.array_equal(np.array(CLASSES)[y], saved["known"].batch.to_numpy())
        replay = cat._proba_full(model.pipeline, features[model.features].to_numpy(float), 3)
        recorded = table.set_index("site").loc[features.site,
                    [f"mp_{c}__{family}" for c in CLASSES]].to_numpy(float)
        np.testing.assert_allclose(replay, recorded, rtol=1e-12, atol=1e-12)
        assert np.array_equal(np.array(CLASSES)[replay.argmax(axis=1)],
                              table.set_index("site").loc[features.site, f"argmax__{family}"])
        mapping_checks[family] = dict(class_names=model.classes, estimator_indices=[0, 1, 2],
                                     saved_scores_reproduced=True, encoded_training_labels_match=True)

    rows = []

    def add_rows(name, family, frame, id_col, bet_col, score_cols, role):
        assert frame[id_col].is_unique and set(frame[id_col]) == set(truth.sample_id)
        for _, row in frame.iterrows():
            site = row[id_col]
            label = evidence["labels"][site]
            scores = np.array([float(row[score_cols[c]]) for c in CLASSES])
            assert np.isfinite(scores).all() and (scores >= 0).all()
            # The original E33 CSV stored probabilities rounded to 3 decimals.
            assert abs(scores.sum() - 1) <= .002
            assert row[bet_col] == CLASSES[int(scores.argmax())]
            rows.append(dict(model_record=name, family=family, role=role, sample_id=site,
                             true_batch=label, predicted_batch=row[bet_col],
                             correct=bool(row[bet_col] == label), top_model_score=float(scores.max()),
                             true_batch_model_score=float(scores[CLASSES.index(label)])))

    original = pd.read_csv(ROOT / "analysis/submission_test/predictions.csv")
    assert original["sample"].tolist() == evidence["submitted_order"]
    assert original.assigned_batch.tolist() == CLASSES
    add_rows("E33_original_v1", "combined", original, "sample", "assigned_batch",
             {c: f"p_{c}" for c in CLASSES}, "original submitted primary; scores rounded")
    first = pd.read_csv(ROOT / "analysis/organiser_drop_01/first_run/submission_sites.csv")
    add_rows("E34R_frozen_v1", "combined", first, "sample_id", "predicted_batch",
             {c: f"model_score_{c}" for c in CLASSES}, "independent v1 replay; same three sites")
    for family in saved["models"]:
        add_rows("D52_frozen_v2", family, table, "site", f"argmax__{family}",
                 {c: f"mp_{c}__{family}" for c in CLASSES},
                 "declared v2 primary" if family == "material" else "saved comparator; not selected using truth")
    evaluation = pd.DataFrame(rows)
    summary = evaluation.groupby(["model_record", "family", "role"], sort=False).agg(
        n_sites=("sample_id", "size"), n_correct=("correct", "sum"), accuracy=("correct", "mean")).reset_index()
    primary = evaluation.query("model_record == 'D52_frozen_v2' and family == 'material'")
    assert primary.correct.tolist() == [False, False, True]
    assert bets.predicted_batch.tolist() == primary.predicted_batch.tolist()
    # Aligned detector images and repeated models are not additional test samples.
    primary = primary.merge(bets[["sample_id", "deletion_same_assignment", "deletion_fit_count",
                                 "acquisition_flags", "baseline_morphology_percentile"]],
                            on="sample_id", validate="one_to_one")
    confusion = pd.crosstab(primary.true_batch, primary.predicted_batch).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0).fillna(0).astype(int)
    family_rows = []
    for _, bet in bets.iterrows():
        group = drivers[drivers.sample_id == bet.sample_id]
        assert len(group) == 29 and group.feature.is_unique
        assert group.assigned_batch.unique().tolist() == [bet.predicted_batch]
        assert group.runner_up.unique().tolist() == [bet.runner_up]
        model = saved["models"]["material"]
        x = features.set_index("site").loc[bet.sample_id, model.features].to_numpy(float)
        logits = model.pipeline.decision_function(x[None, :])[0]
        a, b = model.classes.index(bet.predicted_batch), model.classes.index(bet.runner_up)
        intercept = float(group.intercept_difference.iloc[0])
        assert group.intercept_difference.nunique() == 1
        np.testing.assert_allclose(intercept + group.contribution.sum(), logits[a] - logits[b], atol=1e-10)
        for family, values in group.groupby("family", sort=False):
            family_rows.append(dict(sample_id=bet.sample_id, assigned_batch=bet.predicted_batch,
                                    comparison_batch=bet.runner_up, true_batch=evidence["labels"][bet.sample_id],
                                    comparison_is_truth=bool(bet.runner_up == evidence["labels"][bet.sample_id]),
                                    family=family, contribution_sum=float(values.contribution.sum()),
                                    intercept_difference=intercept))

    args.out.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(args.out / "evaluation.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    primary.to_csv(args.out / "primary_outcomes.csv", index=False)
    confusion.to_csv(args.out / "confusion_v2.csv")
    pd.DataFrame(family_rows).to_csv(args.out / "driver_family_contrasts.csv", index=False)
    for rel, expected in protected.items():
        assert sha(ROOT / rel) == expected, rel
    assert (ROOT / "experiments/registry.csv").read_bytes() == registry_before
    inputs = [HERE / "truth.json", HERE / "truth.csv", Path(__file__),
              ROOT / "analysis/submission_v2/freeze/snapshot.json"]
    receipt = dict(experiment="E36S", decision="D55S", evaluated_utc=datetime.now(timezone.utc).isoformat(),
                   organiser_received_date="2026-10-04", truth_mapping=evidence["mapping_basis"],
                   inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                   protected_artifacts_sha256=protected, protected_artifact_count=len(protected),
                   artifact_sha256={p.name:sha(p) for p in sorted(args.out.glob("*.csv"))},
                   model_sha256=snapshot["models_sha256"], class_mapping_checks=mapping_checks,
                   n_labelled_sites=3, aligned_images_are_not_extra_samples=True,
                   summary=summary.to_dict("records"), no_fit_or_model_change=True,
                   exact_pairwise_logits_verified=True, scores_calibrated=False,
                   feedback_selected_comparator=False, registry_unchanged_during_evaluation=True,
                   future_feedback_informed_versions_use_drop_as_development=True)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(summary[["model_record", "family", "n_sites", "n_correct", "accuracy"]].to_string(index=False))
    print(f"Preserved {len(protected)} original artifacts; class mapping and exact logits verified; fitted nothing.")


if __name__ == "__main__":
    main()
