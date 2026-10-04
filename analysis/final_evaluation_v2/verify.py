"""Verify the saved evaluation outputs. Do not fit a model."""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analysis.submission_v2.runtime import load_runtime, sha


def main():
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    scored, submitted = here / "scored", here / "submission"
    receipt = json.loads((scored / "prediction_receipt.json").read_text())
    freeze = Path(receipt["snapshot_path"]).parent
    snapshot, _ = load_runtime(freeze)
    checks = {}
    for folder, name in ((scored, "prediction_receipt.json"),
                         (submitted, "submission_receipt.json"),
                         (submitted, "report_receipt.json")):
        r = json.loads((folder / name).read_text())
        for filename, expected in r["output_sha256"].items():
            assert sha(folder / filename) == expected, filename
        checks[name] = True
    preflight = json.loads((here / "preflight.json").read_text())
    assert sha(here / "protocol.md") == preflight["protocol_sha256"]
    assert sha(freeze / "models.joblib") == preflight["model_sha256"]
    assert json.loads((scored / "input_manifest.json").read_text()) == preflight["input_manifest"]
    for image in preflight["input_manifest"]:
        assert sha(root / "Hackathon-Polaron-eval" / image["file"]) == image["sha256"]
    saved = joblib.load(freeze / "models.joblib")
    model = saved["models"]["material"]
    assert model.features == snapshot["primary_features"] and len(model.features) == 29
    assert "H" not in model.features
    features = pd.read_csv(scored / "features.csv").set_index("site")
    predictions = pd.read_csv(submitted / "predictions.csv").set_index("sample_id")
    assert len(predictions) == 6 and predictions.index.is_unique
    features = features.loc[predictions.index]
    x = features[model.features].to_numpy(float)
    actual = model.pipeline.predict_proba(x)
    assert model.pipeline["lr"].classes_.tolist() == [0, 1, 2]
    scores = predictions[["model_score_" + c for c in model.classes]].to_numpy(float)
    np.testing.assert_allclose(actual, scores, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scores.sum(axis=1), 1, rtol=0, atol=1e-12)
    assert predictions.predicted_batch.tolist() == [model.classes[j] for j in actual.argmax(axis=1)]
    images = pd.read_csv(submitted / "predictions_images.csv")
    assert len(images) == 18 and images.filename.is_unique
    assert (images.groupby("sample_id").size() == 3).all()
    assert images.predicted_batch.equals(images.sample_id.map(predictions.predicted_batch))
    drivers = pd.read_csv(scored / "driver_contrasts.csv")
    z = model.pipeline["scale"].transform(model.pipeline["impute"].transform(x))
    lr = model.pipeline["lr"]
    for i, (sample, row) in enumerate(predictions.iterrows()):
        a, b = model.classes.index(row.predicted_batch), model.classes.index(row.runner_up)
        expected = pd.Series((lr.coef_[a] - lr.coef_[b]) * z[i], index=model.features)
        got = drivers.loc[drivers.sample_id == sample].set_index("feature")
        np.testing.assert_allclose(got.loc[model.features].contribution, expected, rtol=0, atol=1e-12)
    old = json.loads((root / "analysis/demo_freeze_20261004/manifest.json").read_text())
    for filename, expected in old["artifacts_sha256"].items():
        assert sha(root / filename) == expected, filename
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
                  n_sites=6, n_images=18, primary_features=29,
                  max_score_replay_error=float(np.abs(actual - scores).max()),
                  all_assignments_are_argmax=True, pairwise_contributions_replayed=True,
                  old_frozen_files_unchanged=len(old["artifacts_sha256"]),
                  input_hashes_unchanged=True, model_fitted=False, labels_available=False)
    target = here / "verification.json"
    assert not target.exists(), "Preserve the existing verification file."
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
