"""E42S one fixed L2 geometry candidate; known first, no calibration or selection."""
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.classification_m1_m2 import models as old
from analysis.classification_m1_m2.run import prediction_rows, write_json, sha
from analysis.forest_geometry import models as geometry_models
from analysis.forest_geometry.run import check_hashes, explanations, CLASSES, ROOT
from analysis.quality_policy_audit import policy as qp

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
NAME = "geometry_l2"


@dataclass
class Probe:
    pipeline: object
    feature_names: list
    C: float
    version: str = "E42S-fixed-geometry-L2-v1"

    def predict(self, frame):
        X, names = geometry_models.matrix(frame, "geometry")
        if names != self.feature_names:
            raise ValueError("Geometry positions changed")
        return old.probabilities(self.pipeline, X)


def explain(model, frame, X, names, P, cohort):
    rows = []
    for i, row in frame.iterrows():
        chosen = int(P[i].argmax())
        other = 2 if chosen != 2 else int(np.argsort(P[i])[-2])
        terms, root, margin = old.margin_contributions(model, X[i:i+1], names, chosen, other)
        for rank, term in enumerate(terms, 1):
            term.update(feature_group=geometry_models.group(term["feature"]))
            rows.append(dict(candidate=NAME, cohort=cohort, site=row.site,
                predicted_batch=CLASSES[chosen], comparison_batch=CLASSES[other], rank=rank,
                root_or_intercept_margin=root, total_margin=margin,
                explanation_type="linear_decision_margin", **term))
    return rows


def execute():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Use a fresh study; never overwrite E42S")
    previous_receipt = json.loads((ROOT / "analysis/forest_geometry/output/evaluation/receipt.json").read_text())
    protected = previous_receipt["protected_artifacts_sha256"].copy()
    for p in sorted((ROOT / "analysis/forest_geometry").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            protected[str(p.relative_to(ROOT))] = sha(p)
    check_hashes(protected); check_hashes(previous_receipt["raw_sha256"])
    known_path = ROOT / "analysis/forest_geometry/output/evaluation/known_measurements.csv"
    known = pd.read_csv(known_path)
    known = known.iloc[qp.shared_order(known)].reset_index(drop=True)
    assert known.groupby("batch").size().to_dict() == {"Batch_1":7, "Batch_2":7, "Batch_3":17}
    X, names = geometry_models.matrix(known, "geometry")
    y = known.batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
    assert X.shape == (31, 37)
    OUT.mkdir(parents=True)
    sources = [HERE / "probe.py", HERE / "protocol.md", ROOT / "analysis/classification_m1_m2/models.py",
        ROOT / "analysis/forest_geometry/models.py", ROOT / "analysis/forest_geometry/geometry.py",
        ROOT / "analysis/quality_policy_audit/policy.py"]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    for p in sources:
        target = OUT / "sources" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    P, choices, folds = old.nested_loo(X, y, "l2")
    C = old.select_C(X, y, "l2")
    full = Probe(old.pipeline("l2", C).fit(X, y), names, C)
    # Persist the sklearn pipeline, not this script's __main__ dataclass.
    joblib.dump(full.pipeline, OUT / "known31.joblib", compress=3)
    attrs = []
    for i, model in enumerate(folds):
        joblib.dump(model, OUT / f"fold_{i:02d}.joblib", compress=3)
        attrs.extend(explain(model, known.iloc[i:i+1].reset_index(drop=True), X[i:i+1], names, P[i:i+1], "known_oof"))
    pd.DataFrame(prediction_rows(NAME, known, P, X, choices)).to_csv(OUT / "probe_known_predictions.csv", index=False)
    known.to_csv(OUT / "known_measurements.csv", index=False)
    met = old.metrics(y, P)
    pd.DataFrame(met.pop("confusion"), index=CLASSES, columns=CLASSES).to_csv(OUT / "probe_confusion.csv")
    met.update(candidate=NAME, C_values="|".join(map(str, sorted(set(choices)))), full_fit_C=C, n_features=37)
    pd.DataFrame([met]).to_csv(OUT / "probe_known_summary.csv", index=False)
    pd.DataFrame(attrs).to_csv(OUT / "probe_known_explanations.csv", index=False)
    check_hashes(hashes); check_hashes(protected)
    outputs = {str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    write_json(OUT / "known_stage_receipt.json", dict(experiment="E42S", stage="known_only_complete",
        known_fit_n=31, first_drop_used_for_fitting=False, full_C=C, feature_names=names, sources_sha256=hashes,
        known_input_sha256={str(known_path.relative_to(ROOT)):sha(known_path)}, known_outputs_sha256=outputs,
        protected_artifacts_sha256=protected, raw_sha256=previous_receipt["raw_sha256"],
        source_group_ids_available=False, no_calibration_fit=True, submission_promotion=False))
    print(f"Known-only saved: geometry_l2 BA={met['balanced_accuracy']:.6f}, logloss={met['balanced_logloss']:.6f}, Brier={met['balanced_brier']:.6f}", flush=True)
    # Read the saved, already-revealed drop only after known receipt.
    drop_path = ROOT / "analysis/forest_geometry/output/evaluation/feedback_measurements.csv"
    drop = pd.read_csv(drop_path)
    X, names = geometry_models.matrix(drop, "geometry")
    P = full.predict(drop)
    pd.DataFrame(prediction_rows(NAME, drop, P, X)).to_csv(OUT / "probe_feedback_predictions.csv", index=False)
    attrs.extend(explain(full.pipeline, drop, X, names, P, "feedback_development"))
    pd.DataFrame(attrs).to_csv(OUT / "probe_explanations.csv", index=False)
    known_receipt = json.loads((OUT / "known_stage_receipt.json").read_text())
    check_hashes(known_receipt["known_outputs_sha256"]); check_hashes(hashes)
    check_hashes(protected); check_hashes(previous_receipt["raw_sha256"])
    write_json(OUT / "probe_receipt.json", dict(**known_receipt, stage_after_known="development_diagnostics_complete",
        first_drop_diagnostic_n=3, drop_inputs_sha256={str(drop_path.relative_to(ROOT)):sha(drop_path)},
        scores="uncalibrated balanced-target softmax, not probability of correctness"))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        execute()
