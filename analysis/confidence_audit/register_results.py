"""Append E42S descriptive score/coverage results without changing prior ledger rows."""
import csv
import hashlib
import io
import json

import numpy as np
import pandas as pd

from analysis.confidence_audit.probe import OUT, ROOT
from analysis.classification_m1_m2.run import write_json, sha


def main():
    registry = ROOT / "experiments/registry.csv"
    original = registry.read_bytes()
    assert not any(row["experiment"] == "E42S" for row in csv.DictReader(io.StringIO(original.decode())))
    fields = next(csv.reader(io.StringIO(original.decode())))
    rows = []

    def add(metric, value, candidate="", cohort="known_oof", reference="known31", n=31, unit="score", extra=""):
        if not np.isfinite(value):
            return
        row = {key:"" for key in fields}
        row.update(experiment="E42S",date="2026-10-04",metric=metric,reference=reference,batch=cohort,kpi=candidate,
            statistic="development/technical diagnostic",n_ref=31,n_batch=n,value=value,unit=unit,
            note=("Unknown shared-source dependence; uncalibrated; exact organiser utility unavailable; no selection/promotion. " + extra).strip())
        rows.append(row)

    for filename, label in [("confidence_summary.csv","candidate"),("controls_summary.csv","control")]:
        frame = pd.read_csv(OUT / filename)
        for _, row in frame.iterrows():
            for metric in frame.select_dtypes(include="number"):
                unit = "nats" if "logloss" in metric or "nll" in metric else "count" if metric.startswith("n_") else "score"
                add(label + "_" + metric,row[metric],row.candidate,row.cohort,
                    reference=row.get("control","common_true_label_and_own_bet_separate"),n=row.n_sites,unit=unit,
                    extra="Own-bet target changes when bet changes; balanced/ordinary targets separate. Controls retain declared semantics.")
    for filename, prefix in [("reliability_bins.csv","reliability"),("threshold_counts.csv","threshold"),("paired_changes.csv","paired")]:
        frame = pd.read_csv(OUT / filename)
        for _, row in frame.iterrows():
            ref = (f"bin[{row.bin_low},{row.bin_high}]" if prefix == "reliability" else
                   f"fixed_threshold={row.threshold}" if prefix == "threshold" else row.reference)
            for metric in frame.select_dtypes(include="number"):
                if metric in ["bin_low","bin_high","threshold"]:
                    continue
                add(prefix + "_" + metric,row[metric],row.candidate,row.cohort,reference=ref,
                    n=31 if row.cohort == "known_oof" else 3,
                    unit="nats" if "logloss" in metric else "crops" if metric.startswith(("n_","high_","newly_","same_")) and not "mean" in metric else "score",
                    extra="Descriptive fixed bins/thresholds and paired same-bet changes; not calibration or a judging utility.")
    probe = pd.read_csv(OUT / "probe_known_summary.csv").iloc[0]
    for metric in ["balanced_accuracy","accuracy","balanced_logloss","balanced_brier","n_correct","full_fit_C","n_features"] + ["recall_Batch_"+str(i) for i in [1,2,3]]:
        add("geometry_L2_" + metric,probe[metric],"geometry_l2",unit="nats" if metric.endswith("logloss") else "score")
    predictions = pd.read_csv(OUT / "predictions.csv")
    for _, row in predictions[predictions.cohort == "feedback_development"].iterrows():
        for metric in ["score_Batch_1","score_Batch_2","score_Batch_3","bet_score","true_class_score","n_observed_inputs"]:
            add("revealed_"+metric,row[metric],row.candidate,row.cohort,n=1,
                unit="cells" if metric == "n_observed_inputs" else "score",extra=f"site={row.site}; bet={row.predicted_batch}; truth={row.true_batch}; already revealed development.")
    validation = json.loads((OUT / "delivery_validation.json").read_text())
    for metric,value in validation.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            add(metric,value,cohort="technical",n="",unit="count" if not "error" in metric else "score",extra="Technical replay/coverage, not independent n.")
    for metric,value,unit in [("full_suite_tests_passed",232,"tests"),("full_suite_warnings",15,"warnings"),
                             ("full_suite_elapsed",189.09,"seconds"),("metric_inventory_entries",112,"entries")]:
        add(metric,value,cohort="technical",n="",unit=unit)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer,fieldnames=fields,lineterminator="\n");writer.writerows(rows)
    with registry.open("ab") as handle:
        handle.write(buffer.getvalue().encode())
    assert registry.read_bytes()[:len(original)] == original
    write_json(OUT / "registry_append_receipt.json",dict(experiment="E42S",appended_rows=len(rows),
        historical_prefix_bytes=len(original),historical_prefix_sha256=hashlib.sha256(original).hexdigest(),
        current_sha256=sha(registry)))
    print(f"Appended{len(rows)}E42S numerical rows; historical prefix unchanged.")


if __name__ == "__main__":
    main()
