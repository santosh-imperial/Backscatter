"""Post-fit delivery checks; never searches models or changes experiment outputs."""
from __future__ import annotations

import base64
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analysis.forest_geometry import models as m
from analysis.forest_geometry.run import HERE, ROOT, SCORE_COLS, check_hashes, load_fits, sha, write_json
from analysis.quality_policy_audit import policy as qp


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.images, self.ids = [], [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img":
            self.images.append(attrs["src"])


def main():
    out = HERE / "output/evaluation"
    receipt = json.loads((out / "receipt.json").read_text())
    checks = {}
    for key in ["sources_sha256", "known_inputs_sha256", "known_outputs_sha256", "protected_artifacts_sha256", "raw_sha256", "feedback_inputs_sha256"]:
        check_hashes(receipt[key]); checks[key] = len(receipt[key])
    known = pd.read_csv(out / "known_measurements.csv")
    drop = pd.read_csv(out / "feedback_measurements.csv")
    predictions = pd.read_csv(out / "known_oof_predictions.csv")
    feedback = pd.read_csv(out / "feedback_development_predictions.csv")
    score_replay = 0.
    full_order_replay = 0.
    for name, (view, _) in m.CANDIDATES.items():
        full, folds = load_fits(out, name)
        X, names = m.matrix(known, view)
        assert names == full.feature_names
        P = np.vstack([m.probabilities(model, X[i:i+1])[0] for i, model in enumerate(folds)])
        saved = predictions[predictions.candidate == name].set_index("site").reindex(known.site)[SCORE_COLS].to_numpy()
        score_replay = max(score_replay, float(np.abs(P-saved).max()))
        saved_drop = feedback[feedback.candidate == name].set_index("site").reindex(drop.site)[SCORE_COLS].to_numpy()
        score_replay = max(score_replay, float(np.abs(full.predict(drop)-saved_drop).max()))
        renamed = known.sample(frac=1, random_state=17).reset_index(drop=True)
        renamed["site"] = ["neutral" + str(i) for i in range(len(renamed))]
        # Refit EXACT same candidate/order; this is an invariance regression, not selection.
        refitted = m.fit(renamed, name)
        full_order_replay = max(full_order_replay, float(np.abs(full.predict(known)-refitted.predict(known)).max()))
    assert score_replay < 1e-12 and full_order_replay < 1e-12
    checks["saved_known_and_drop_score_replay_max_difference"] = score_replay
    checks["full_fit_renaming_and_row_permutation_max_difference"] = full_order_replay
    attrs = pd.read_csv(out / "explanations.csv")
    errors = []
    for key, part in attrs.groupby(["candidate", "cohort", "site"]):
        assert part.feature.is_unique
        assert part.root_or_intercept_margin.nunique() == 1 and part.total_margin.nunique() == 1
        errors.append(abs(part.contribution.sum() + part.root_or_intercept_margin.iloc[0] - part.total_margin.iloc[0]))
    assert max(errors) < 1e-9
    checks["margin_reconstruction_groups"] = len(errors)
    checks["margin_reconstruction_max_error"] = max(errors)
    invalid = pd.read_csv(out / "cell_availability.csv")
    assert not invalid.loc[~invalid.policy_eligible, "observed_model_input"].any()
    X, names = m.matrix(known, "geometry")
    bad = known.copy()
    for _, cell in invalid[~invalid.observed_model_input].iterrows():
        bad.loc[bad.site == cell.site, cell.feature] = 9e12
    # Coverage-invalid fields deliberately remain unavailable despite numerical mutations.
    mutated, _ = m.matrix(bad, "geometry")
    np.testing.assert_equal(X, mutated)
    checks["rejected_cell_numeric_mutation_invariance"] = True
    assert len(receipt["raw_sha256"]) == 102
    prefix = (ROOT / "experiments/registry.csv").read_bytes()[:receipt["registry_prefix_bytes"]]
    assert hashlib.sha256(prefix).hexdigest() == receipt["registry_prefix_sha256"]
    checks["registry_historical_prefix_unchanged"] = True
    report = HERE / "report.html"
    parser = References(); parser.feed(report.read_text())
    for href in parser.links:
        if href.startswith("#"):
            assert href[1:] in parser.ids
        elif not href.startswith(("https:", "http:", "mailto:")):
            assert (report.parent / href).is_file(), href
    for src in parser.images:
        if src.startswith("data:"):
            Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1]))).verify()
        else:
            Image.open(report.parent / src).verify()
    checks["report_links_checked"] = len(parser.links)
    checks["report_images_decoded"] = len(parser.images)
    oversized = [str(p.relative_to(ROOT)) for p in HERE.rglob("*") if p.is_file() and p.stat().st_size > 20 * 1024**2]
    assert not oversized, oversized
    checks["new_file_size_under_20MiB"] = True
    summaries = pd.read_csv(out / "known_summary.csv")
    for name, part in predictions.groupby("candidate"):
        assert (part[SCORE_COLS].to_numpy().argmax(axis=1) == part.predicted_batch.map({c:i for i,c in enumerate(m.CLASSES)}).to_numpy()).all()
        assert part.correct.sum() == summaries.set_index("candidate").loc[name, "n_correct"]
    checks["class_mapping_counts_and_argmax"] = True
    register = json.loads((ROOT / "analysis/morphology/metric_register.json").read_text())
    selected = [item for item in register["metrics"] if item["id"] in names[29:]]
    assert len(selected) == 8 and all(item["review_status"] == "unreviewed" for item in selected)
    assert all("E41S" in item["experiments"] for item in selected)
    checks["existing_eight_inventory_entries_linked_without_expert_promotion"] = True
    checks["inventory_entries"] = len(register["metrics"])
    delivery_sources = [HERE / "verify_delivery.py", HERE / "delivery_report.py", HERE / "findings.md"]
    write_json(out / "delivery_validation.json", dict(experiment="E41S", checks=checks, all_checks_passed=True,
        report_sha256=sha(report), delivery_sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in delivery_sources},
        native_browser_layout_verified=False, no_commit_requested=True,
        source_group_generalisation_established=False, submission_promotion=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
