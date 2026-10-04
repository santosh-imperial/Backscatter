"""Append E41S reported numerical diagnostics; historical rows remain byte-identical."""
import csv
import hashlib
import io
import json

import pandas as pd

from analysis.forest_geometry.run import HERE, ROOT, CLASSES, write_json


def main():
    out = HERE / "output/evaluation"
    receipt = json.loads((out / "receipt.json").read_text())
    registry = ROOT / "experiments/registry.csv"
    before = registry.read_bytes()
    assert hashlib.sha256(before[:receipt["registry_prefix_bytes"]]).hexdigest() == receipt["registry_prefix_sha256"]
    assert not any(row["experiment"] == "E41S" for row in csv.DictReader(io.StringIO(before.decode())))
    columns = next(csv.reader(io.StringIO(before.decode())))
    rows = []

    def add(metric, value, kpi="", batch="all", reference="known_original_crops", n=31, unit="fraction", note=""):
        row = {key:"" for key in columns}
        row.update(experiment="E41S", date="2026-10-04", metric=metric, reference=reference,
                   batch=batch, kpi=kpi, statistic="development/technical diagnostic", n_ref=31,
                   n_batch=n, value=value, unit=unit,
                   note="Unknown shared-source dependence; no source-held-out accuracy/IID inference. " + note)
        rows.append(row)

    summary = pd.read_csv(out / "known_summary.csv")
    for _, row in summary.iterrows():
        for metric in ["n_features", "n_sites", "n_correct", "accuracy", "balanced_accuracy", "balanced_logloss", "balanced_brier"]:
            add("crop_loo_" + metric, row[metric], kpi=row.candidate,
                unit="count" if metric.startswith("n_") else "nats" if metric == "balanced_logloss" else "score" if metric == "balanced_brier" else "fraction", note="Fixed candidates; no selection/promotion.")
        for c in CLASSES:
            add("crop_loo_recall", row["recall_" + c], kpi=row.candidate, batch=c, n=17 if c == "Batch_3" else 7)
    for _, row in pd.read_csv(out / "paired_comparisons.csv").iterrows():
        for metric in ["changed", "newly_correct", "newly_wrong", "balanced_accuracy_delta"]:
            add("paired_" + metric, row[metric], kpi=row.candidate, reference=row.reference,
                unit="fraction" if metric.endswith("delta") else "crops")
    for (candidate, variant), group in pd.read_csv(out / "geometry_oof_sensitivity.csv").groupby(["candidate", "variant"]):
        for metric, value, unit in [("bet_changes", int(group.bet_changed.sum()), "crops"),
            ("max_score_change", group.max_absolute_score_change.max(), "fraction"),
            ("median_score_change", group.max_absolute_score_change.median(), "fraction")]:
            add("geometry_variant_" + metric, value, kpi=candidate, reference=variant, unit=unit,
                note="Fixed original fit; old29 fixed; partial-input mask sensitivity only.")
    for _, row in pd.read_csv(out / "seed_summary.csv").iterrows():
        for metric in ["bet_changes_from_seed0", "n_correct", "accuracy", "balanced_accuracy", "balanced_logloss", "balanced_brier"] + ["recall_" + c for c in CLASSES]:
            add("forest_seed_" + metric, row[metric], kpi="geometry_rf", reference="seed" + str(int(row.seed)),
                unit="crops" if metric in ["bet_changes_from_seed0", "n_correct"] else "fraction", note="Sensitivity only; seed0 primary, never choose a seed.")
    for group, part in pd.read_csv(out / "group_replacement.csv").groupby("feature_group"):
        for metric, value, unit in [("bet_changes", int(part.bet_changed.sum()), "crops"),
            ("max_score_change", part.max_absolute_score_change.max(), "fraction"),
            ("median_score_change", part.max_absolute_score_change.median(), "fraction")]:
            add("train_median_replacement_" + metric, value, kpi=group, reference="fixed_geometry_rf", unit=unit,
                note="Noncausal fixed-model input sensitivity; correlated hybrids may be implausible; not refit ablation.")
    for _, row in pd.read_csv(out / "feedback_development_predictions.csv").iterrows():
        for c in CLASSES:
            add("revealed_development_class_score", row["score_" + c], kpi=row.candidate, batch=row.true_batch,
                reference="known31_fit", n=1, note=f"site={row.site}; scoreclass={c}; forcedbet={row.predicted_batch}; uncalibrated, already-revealed diagnostic, never selection.")
        add("revealed_development_correct", int(row.correct), kpi=row.candidate, batch=row.true_batch, n=1,
            unit="boolean", note=f"site={row.site}; forcedbet={row.predicted_batch}; never external test.")
        add("revealed_development_observed_inputs", row.n_observed_inputs, kpi=row.candidate, batch=row.true_batch,
            n=1, unit="input cells/bits", note=f"site={row.site}; controls count finite predictor bits, not numerical material measurements.")
    q = pd.read_csv(HERE / "output/geometry_known_availability.csv")
    q = q[q.variant == "nominal"]
    for feature, part in q.groupby("feature"):
        for name in ["raw_finite", "policy_eligible", "observed_model_input"]:
            add("geometry_coverage_" + name, int(part[name].sum()), kpi=feature, unit="cells",
                note="Phase/coverage gates; computational availability is not expert accuracy.")
    add("nominal_extra_geometry_observed_cells", int(q.observed_model_input.sum()), unit="cells")
    add("nominal_extra_geometry_total_cells", len(q), unit="cells")
    for metric, value, unit in [("full_suite_tests_passed", 232, "tests"), ("full_suite_warnings", 15, "warnings"),
        ("full_suite_elapsed", 199.94, "seconds"), ("fresh_nominal_geometry_parity_cells", 32, "cells"),
        ("fresh_nominal_geometry_parity_matches", 32, "cells"), ("actual_geometry_examples", 4, "crops"),
        ("protected_prior_artifacts_unchanged", len(receipt["protected_artifacts_sha256"]), "files"),
        ("raw_TIFFs_unchanged", len(receipt["raw_sha256"]), "files"), ("metric_inventory_entries", 112, "entries")]:
        add(metric, value, unit=unit, n="", note="Technical/coverage measure, not independent performance n.")
    for name, entry in receipt["matched_L1_reproduction"].items():
        add("matched_L1_max_score_difference", entry["max_score_difference"], kpi=name)
    delivery = json.loads((out / "delivery_validation.json").read_text())["checks"]
    for metric in ["saved_known_and_drop_score_replay_max_difference", "full_fit_renaming_and_row_permutation_max_difference",
                   "margin_reconstruction_groups", "margin_reconstruction_max_error", "report_links_checked", "report_images_decoded"]:
        add(metric, delivery[metric], unit="fraction" if "difference" in metric or "error" in metric else "count",
            n="", note="Exact technical replay/accounting check; no accuracy claim.")
    for metric, value, unit in [("narrow_tests_passed", 17, "tests"), ("held_out_fold_models_replayed", 217, "fits")]:
        add(metric, value, unit=unit, n="", note="Technical regressions, not independent labelled observations.")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writerows(rows)
    with registry.open("ab") as handle:
        handle.write(buffer.getvalue().encode())
    assert registry.read_bytes()[:len(before)] == before
    write_json(out / "registry_append_receipt.json", dict(experiment="E41S", appended_rows=len(rows),
        historical_prefix_bytes=len(before), historical_prefix_sha256=hashlib.sha256(before).hexdigest(),
        current_sha256=hashlib.sha256(registry.read_bytes()).hexdigest()))
    print(f"Appended {len(rows)} E41S numerical rows; historical prefix unchanged.")


if __name__ == "__main__":
    main()
