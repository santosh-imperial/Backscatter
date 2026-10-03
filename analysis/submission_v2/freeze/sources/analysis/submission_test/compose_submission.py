"""Compose a version-bound submission into a new directory.

Requires a frozen scorer receipt and explicit model version. Existing E33 files are
historical records; this command never overwrites a nonempty output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("Batch_1", "Batch_2", "Batch_3")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_run(run, model_version, primary_family):
    run = Path(run)
    receipt = json.loads((run / "prediction_receipt.json").read_text())
    if receipt.get("model_version") != model_version or receipt.get("primary_family") != primary_family:
        raise ValueError("Scored model version/family does not match the declared submission version")
    snapshot_path = Path(receipt["snapshot_path"])
    if not snapshot_path.is_absolute():
        snapshot_path = ROOT / snapshot_path
    if sha(snapshot_path) != receipt["source_snapshot_sha256"]:
        raise ValueError("Frozen snapshot hash mismatch")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("model_version") != model_version or snapshot.get("primary_family") != primary_family:
        raise ValueError("Frozen snapshot declares a different model version/family")
    if snapshot["models_sha256"] != receipt["model_sha256"] or sha(snapshot_path.parent / "models.joblib") != receipt["model_sha256"]:
        raise ValueError("Frozen model hash mismatch")
    for name, expected in receipt["output_sha256"].items():
        if Path(name).name != name or sha(run / name) != expected:
            raise ValueError(f"Scored artifact hash mismatch: {name}")
    for name, expected in snapshot["evaluation_sha256"].items():
        if sha(run / name) != expected:
            raise ValueError(f"Evaluation evidence does not belong to this frozen model: {name}")
    table = pd.read_csv(run / "site_table.csv")
    required = ["site", "role", f"argmax__{primary_family}", f"cat_top_features__{primary_family}"]
    required += [f"mp_{c}__{primary_family}" for c in CLASSES]
    if not set(required) <= set(table):
        raise ValueError(f"Missing declared primary-family columns: {sorted(set(required) - set(table))}")
    query = table.loc[table.role == "scored"].copy()
    if query.empty or query.site.isna().any() or query.site.duplicated().any():
        raise ValueError("Expected nonempty scored rows with unique site IDs")
    scores = query[[f"mp_{c}__{primary_family}" for c in CLASSES]].to_numpy(float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError("Primary scores must be finite, nonnegative and sum to one")
    assigned = np.asarray(CLASSES)[scores.argmax(1)]
    if not np.array_equal(assigned, query[f"argmax__{primary_family}"].to_numpy()):
        raise ValueError("Stored primary argmax disagrees with its model scores")
    submitted = pd.read_csv(run / "submission_sites.csv").set_index("sample_id")
    if set(submitted.index) != set(query.site) or submitted.index.duplicated().any():
        raise ValueError("Submission rows do not match scored sites")
    for i, site in enumerate(query.site):
        row = submitted.loc[site]
        if row.predicted_batch != assigned[i] or not np.allclose([row[f"model_score_{c}"] for c in CLASSES], scores[i], rtol=0, atol=1e-12):
            raise ValueError("Scored submission disagrees with declared primary family")
    summary = pd.read_csv(run / "categoriser_summary.csv")
    primary = summary.loc[summary.family == primary_family]
    if len(primary) != 1 or not bool(primary.iloc[0].primary):
        raise ValueError("Known-site evaluation does not declare this primary family")
    reliability = pd.read_csv(run / f"reliability_{primary_family}.csv")
    if int(reliability.n.sum()) != int(primary.iloc[0].n_sites):
        raise ValueError("Reliability evidence has inconsistent site coverage")
    return receipt, snapshot, query, submitted, primary.iloc[0], reliability


def compose(run, out, model_version, primary_family="material"):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Preserve existing outputs; choose a new empty directory: {out}")
    receipt, snapshot, query, submitted, summary, reliability = read_run(run, model_version, primary_family)
    predictions = submitted.reset_index().sort_values("sample_id").copy()
    predictions.insert(1, "model_version", model_version)
    predictions.insert(2, "primary_family", primary_family)
    predictions["confidence_qualification"] = "uncalibrated model score; not probability of correctness"
    images = pd.read_csv(run / "submission_images.csv")
    if set(images.sample_id) != set(predictions.sample_id) or images.filename.duplicated().any() or not (images.groupby('sample_id').size() == 3).all():
        raise ValueError("Image submission must map exactly three distinct detector files per site")
    lookup = predictions.set_index("sample_id")
    if not (images.predicted_batch == images.sample_id.map(lookup.predicted_batch)).all():
        raise ValueError("Image assignments do not match sample assignments")
    for c in CLASSES:
        if not np.allclose(images[f"model_score_{c}"], images.sample_id.map(lookup[f"model_score_{c}"]), rtol=0, atol=1e-12):
            raise ValueError("Image scores do not match sample scores")
    images.insert(2, "model_version", model_version)
    images.insert(3, "primary_family", primary_family)
    lines = [f"# Batch assignments — {model_version}", "",
             f"Declared primary: **{primary_family}**. Source model SHA256: `{receipt['model_sha256']}`.",
             "The primary uses morphology and acquisition-sensitive ETD/Inlens appearance. Material origin of texture remains unresolved. "
             "Every sample receives a bet. Model scores are uncalibrated; deletion fits measure training-set sensitivity, not correctness probability.", "",
             "| Sample | Bet | Scores B1 / B2 / B3 | Runner-up | Margin | Same bet in deletion fits | Quality flags |",
             "|---|---|---|---|---|---|---|"]
    for row in predictions.itertuples():
        scores = " / ".join(f"{getattr(row, 'model_score_'+c):.3f}" for c in CLASSES)
        lines.append(f"| {row.sample_id} | **{row.predicted_batch}** | {scores} | {row.runner_up} | {row.margin:.3f} | {row.deletion_same_assignment}/{row.deletion_fit_count} | {row.acquisition_flags} |")
    lines += ["", "## Matching known-site evaluation", "",
              f"Leave-one-site-out on {int(summary.n_sites)} known sites; balanced accuracy {summary.balanced_accuracy:.3f}; "
              f"recall B1/B2/B3 {summary.recall_Batch_1:.3f}/{summary.recall_Batch_2:.3f}/{summary.recall_Batch_3:.3f}; "
              f"site-label permutation p {summary.perm_p:.4g}. These are development results, not new-session accuracy.", "",
              "Reliability bins below belong to this model family. Small bins and overlapping LOO fits do not establish calibration.", "",
              "| Score bin | Known sites | Mean top score | Observed correct fraction |", "|---|---|---|---|"]
    for row in reliability.itertuples():
        lines.append(f"| {row.bin} | {int(row.n)} | {row.mean_top_prob:.3f} | {row.observed_top_accuracy:.3f} |")
    lines += ["", "## Per-sample explanation", "",
              "The linked driver table exactly decomposes assigned-class minus runner-up linear logits. Baseline/class medians are "
              "descriptive and include known quality limitations. Signs explain this model, not material causes.", ""]
    for row in predictions.itertuples():
        lines += [f"### {row.sample_id}", "",
                  f"Primary score drivers: `{row.explanation}`. Morphology-only bet: {row.morphology_only_bet}; "
                  f"acquisition-only comparator: {row.acquisition_only_bet}.",
                  f"Baseline morphology percentile: {row.baseline_morphology_percentile:.1f}. It is a descriptive distance rank, "
                  "not calibrated membership or a production release verdict.", ""]
    lines += ["[All image assignments](predictions_images.csv) · [Exact pairwise driver table](driver_contrasts.csv) · "
              "[Illustrated report](report.html) · [Prediction provenance](submission_receipt.json)", "",
              "The input folder may mix batches and is not pooled into a manufacturing-batch QC verdict. "
              "Truth is pending; future label feedback must be saved separately without editing these bets.", "",
              "Version 2 was revised after inspecting the first drop, before its truth, and declared before final evaluation. "
              "E33 v1 predictions remain a separate historical record."]
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    images.to_csv(out / "predictions_images.csv", index=False)
    (out / "driver_contrasts.csv").write_bytes((run / "driver_contrasts.csv").read_bytes())
    (out / "submission.md").write_text("\n".join(lines) + "\n")
    for name in ["categoriser_summary.csv", f"reliability_{primary_family}.csv"]:
        (out / name).write_bytes((run / name).read_bytes())
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), model_version=model_version, primary_family=primary_family,
                  model_sha256=receipt['model_sha256'], scored_receipt_sha256=sha(run/'prediction_receipt.json'),
                  scored_directory=str(run), snapshot_path=receipt['snapshot_path'], source_snapshot_sha256=receipt['source_snapshot_sha256'],
                  n_sites=len(predictions), n_images=len(images), confidence='uncalibrated model scores',
                  output_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out / "submission_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directory produced by the frozen scorer")
    parser.add_argument("--out", type=Path, required=True, help="New empty submission directory")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--primary-family", default="material")
    args = parser.parse_args(argv)
    result = compose(args.run, args.out, args.model_version, args.primary_family)
    print(result[["sample_id", "predicted_batch", "top_model_score", "margin"]].to_string(index=False))


if __name__ == "__main__":
    main()
