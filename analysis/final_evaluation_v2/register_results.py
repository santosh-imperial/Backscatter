"""Append evaluation measurements. Preserve earlier registry bytes."""
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.submission_v2.runtime import sha


def main():
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    registry = root / "experiments/registry.csv"
    before = registry.read_bytes()
    preflight = json.loads((here / "preflight.json").read_text())
    assert hashlib.sha256(before[:preflight["registry_before_bytes"]]).hexdigest() == preflight["registry_before_sha256"]
    assert b"\nE44S," not in before, "E44S already has registry rows."
    fields = next(csv.reader(StringIO(before.decode().splitlines()[0])))
    rows = []

    def add(metric, batch, kpi, value, unit="", n_ref=31, n_batch=1, note=""):
        rows.append(dict(experiment="E44S", date="2026-10-04", metric=metric,
                         reference="Batch_3", batch=batch, kpi=kpi, statistic="frozen_v2_inference",
                         n_ref=n_ref, n_batch=n_batch, value=value, unit=unit,
                         note="Uncalibrated scores; labels pending; no evaluation fitting. " + note))

    scores = pd.read_csv(here / "submission/predictions.csv")
    metrics = ["model_score_Batch_1", "model_score_Batch_2", "model_score_Batch_3", "top_model_score",
               "margin", "deletion_same_assignment", "deletion_fit_count", "assigned_score_deletion_min",
               "assigned_score_deletion_max", "baseline_morphology_percentile", "exceeds_observed_baseline_max"]
    for _, r in scores.iterrows():
        for m in metrics:
            unit = "percentile" if m == "baseline_morphology_percentile" else "count" if m.startswith("deletion_") else ""
            value = int(r[m]) if isinstance(r[m], (bool, np.bool_)) else r[m]
            add(m, r.sample_id, "", value, unit, n_ref=17 if m.startswith(("baseline_", "exceeds_")) else 31)
        add("assigned_class_index", r.sample_id, "", int(r.predicted_batch[-1]), note="1=Batch_1, 2=Batch_2, 3=Batch_3; highest model score.")
        add("listed_quality_flags", r.sample_id, "", 0, "count", note="No listed flags; phase truth remains unreviewed.")
    drivers = pd.read_csv(here / "scored/driver_contrasts.csv")
    for _, r in drivers.iterrows():
        for metric, key in (("driver_measurement", "value"), ("pairwise_contribution", "contribution")):
            add(metric, r.sample_id, r.feature, r[key],
                note=f"Assigned {r.assigned_batch} versus {r.runner_up}; noncausal explanation.")
    add("n_samples", "Hackathon-Polaron-eval", "", 6, "count", n_batch=6)
    add("n_images", "Hackathon-Polaron-eval", "", 18, "count", n_batch=6)
    add("primary_features", "Hackathon-Polaron-eval", "", 29, "count", n_batch=6)
    add("colour_border_files", "Hackathon-Polaron-eval", "", 6, "count", n_batch=6)
    add("missing_primary_measurements", "Hackathon-Polaron-eval", "", 0, "count", n_batch=6)
    verification = json.loads((here / "verification.json").read_text())
    add("max_score_replay_error", "Hackathon-Polaron-eval", "", verification["max_score_replay_error"], n_batch=6)
    buf = StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writerows(rows)
    appended = buf.getvalue().encode()
    assert before.endswith(b"\n")
    with registry.open("ab") as f:
        f.write(appended)
    after = registry.read_bytes()
    assert after == before + appended
    receipt = dict(completed_utc=datetime.now(timezone.utc).isoformat(), rows_appended=len(rows),
                   earlier_bytes=len(before), earlier_sha256=hashlib.sha256(before).hexdigest(),
                   earlier_bytes_preserved=True, append_sha256=hashlib.sha256(appended).hexdigest(),
                   registry_sha256=sha(registry))
    (here / "registry_append_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
