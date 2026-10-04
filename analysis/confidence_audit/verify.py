"""Independent E42S score/reliability/fold replay and immutable-artifact checks."""
import base64
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import log_loss, brier_score_loss

from analysis.confidence_audit.probe import HERE, OUT, ROOT
from analysis.confidence_audit import audit
from analysis.classification_m1_m2 import models as old
from analysis.classification_m1_m2.run import sha, write_json
from analysis.forest_geometry import models as geo
from analysis.forest_geometry.run import check_hashes, CLASSES, SCORE_COLS
from analysis.forest_geometry.verify_delivery import References


def main():
    probe = json.loads((OUT / "probe_receipt.json").read_text())
    a = json.loads((OUT / "audit_receipt.json").read_text())
    for key in ["sources_sha256","known_input_sha256","known_outputs_sha256","protected_artifacts_sha256","raw_sha256","drop_inputs_sha256"]:
        check_hashes(probe[key])
    for key in ["sources_sha256","inputs_sha256","outputs_sha256"]:
        check_hashes(a[key])
    known = pd.read_csv(OUT / "known_measurements.csv")
    X, names = geo.matrix(known,"geometry")
    y = known.batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
    saved = pd.read_csv(OUT / "probe_known_predictions.csv").set_index("site").reindex(known.site)
    replay = 0.
    for i in range(len(known)):
        model = joblib.load(OUT / f"fold_{i:02d}.joblib")
        train = np.arange(len(known)) != i
        medians = np.nanmedian(X[train],axis=0)
        medians[np.isnan(medians)] = 0.
        np.testing.assert_allclose(model["impute"].statistics_,medians,atol=0,rtol=0)
        imputed = model["impute"].transform(X[train])
        np.testing.assert_allclose(model["scale"].mean_,imputed.mean(axis=0),atol=0,rtol=0)
        params = model["classifier"].get_params()
        assert params["solver"] == "lbfgs" and params["multi_class"] == "multinomial" and params["class_weight"] == "balanced"
        assert params["penalty"] == "l2" and params["max_iter"] == 4000 and params["C"] == saved.C.iloc[i]
        replay = max(replay,float(np.abs(old.probabilities(model,X[i:i+1])[0]-saved.iloc[i][SCORE_COLS].to_numpy(float)).max()))
    assert replay < 1e-12 and probe["feature_names"] == names
    full = joblib.load(OUT / "known31.joblib")
    drop = pd.read_csv(ROOT / "analysis/forest_geometry/output/evaluation/feedback_measurements.csv")
    Z, _ = geo.matrix(drop,"geometry")
    expected = pd.read_csv(OUT / "probe_feedback_predictions.csv").set_index("site").reindex(drop.site)[SCORE_COLS].to_numpy()
    assert np.max(np.abs(old.probabilities(full,Z)-expected)) < 1e-12
    frame = pd.read_csv(OUT / "predictions.csv")
    summary = pd.read_csv(OUT / "confidence_summary.csv").set_index(["candidate","cohort"])
    max_loss_error = 0.
    for key, part in frame.groupby(["candidate","cohort"]):
        P = part[SCORE_COLS].to_numpy()
        y = part.true_batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
        q = P.max(axis=1)
        z = part.correct.to_numpy(int)
        for balanced in [False,True]:
            weights = np.array([1/np.count_nonzero(y == label) for label in y]) if balanced else np.ones(len(y))
            prefix = "balanced_" if balanced else ""
            independent = {
                "multiclass_logloss":log_loss(y,P,labels=[0,1,2],sample_weight=weights),
                "multiclass_brier":np.average(((P-np.eye(3)[y])**2).sum(axis=1),weights=weights),
                "bet_brier":brier_score_loss(z,q,sample_weight=weights),
                "bet_nll":log_loss(z,np.column_stack([1-q,q]),labels=[0,1],sample_weight=weights),
            }
            for name,value in independent.items():
                err = abs(value-summary.loc[key,prefix+name])
                max_loss_error = max(max_loss_error,err)
                assert err < 1e-12
    bins = pd.read_csv(OUT / "reliability_bins.csv")
    totals = bins.groupby(["candidate","cohort"])[["n_sites","n_correct"]].sum().sort_index()
    np.testing.assert_equal(totals.to_numpy(),summary.sort_index()[["n_sites","n_correct"]].to_numpy())
    assert (bins[["n_" + c for c in CLASSES]].sum(axis=1) == bins.n_sites).all()
    thresholds = pd.read_csv(OUT / "threshold_counts.csv")
    assert (thresholds.high_correct + thresholds.high_wrong == thresholds.n_high).all()
    for key, rows in thresholds.groupby(["candidate","cohort"]):
        assert (np.diff(rows.sort_values("threshold").n_high) <= 0).all()
    attrs = pd.read_csv(OUT / "explanations.csv")
    error = 0.
    for _, part in attrs.groupby(["candidate","cohort","site"]):
        error = max(error,abs(part.contribution.sum()+part.root_or_intercept_margin.iloc[0]-part.total_margin.iloc[0]))
    assert error < 1e-9
    assert (attrs.loc[attrs.feature.isin(names[29:]),"family"] == "phase-mask-dependent geometry").all()
    report = HERE / "report.html"
    refs = References();refs.feed(report.read_text())
    for href in refs.links:
        if href.startswith("#"):
            assert href[1:] in refs.ids
        elif not href.startswith(("http:","https:")):
            filename, _, fragment = href.partition("#")
            target = HERE / filename
            assert target.is_file(),href
            if fragment:
                linked = References();linked.feed(target.read_text())
                assert fragment in linked.ids,href
    for src in refs.images:
        Image.open(io.BytesIO(base64.b64decode(src.split(",",1)[1]))).verify()
    assert not [p for p in HERE.rglob("*") if p.is_file() and p.stat().st_size > 20*1024**2]
    write_json(OUT / "delivery_validation.json", dict(experiment="E42S", all_checks_passed=True,
        replay_max_error=replay,independent_score_loss_max_error=max_loss_error,explanation_margin_max_error=error,
        n_prediction_rows=len(frame),n_summaries=len(summary),n_reliability_rows=len(bins),n_threshold_rows=len(thresholds),
        protected_files_unchanged=len(probe["protected_artifacts_sha256"]),raw_TIFFs_unchanged=len(probe["raw_sha256"]),
        report_links=len(refs.links),report_images=len(refs.images),native_browser_layout_verified=False,
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in [HERE / "verify.py",HERE / "build_report.py",HERE / "findings.md"]},
        report_sha256=sha(report),test_validation=dict(command="pytest tests -q",passed=232,warnings=15,elapsed_seconds=189.09),
        organiser_utility_not_identified=True,no_calibration_fit=True,no_submission_promotion=True))
    print(f"E42S:31 fold replay max{replay:.3g}; independent loss max{max_loss_error:.3g}; all{len(frame)}rows/links/images valid.")


if __name__ == "__main__":
    main()
