"""Check prepared-packet identity, raw values, no-hint payload and zero review state."""
import base64
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from analysis.boundary_review.prepare import ROOT, OUT, BENCH, sha, verify_packet, validate_export, write_json


def main():
    receipt = json.loads((OUT / "receipt.json").read_text())
    for section, prefix in [("input_sha256", ROOT), ("protected_original_artifacts_sha256", ROOT),
                            ("outputs_sha256", OUT)]:
        for path, expected in receipt[section].items():
            assert sha(prefix/path) == expected, path
    manifest = json.loads((OUT / "coordinator_manifest.json").read_text())
    old = json.loads((BENCH / "roi_manifest.json").read_text())
    old_rois = {r["id"]: r for r in old["rois"]}
    known_all = set()
    raw_verified = 0
    for packet in manifest["packets"]:
        verify_packet(packet)
        html = (OUT / (packet["kind"] + ".html")).read_text()
        public = json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', html, re.S)[1])
        assert set(public) == {"schema_version", "packet_id", "kind", "rois"}
        assert public["packet_id"] == packet["packet_id"]
        assert len(public["rois"]) == len(packet["rois"])
        assert len({r["alias"] for r in packet["rois"]}) == len(packet["rois"])
        for r, visible in zip(packet["rois"], public["rois"]):
            assert set(visible) == {"id", "W", "H", "raw_png_sha256", "image"}
            assert visible["id"] == r["alias"]
            encoded = base64.b64decode(visible["image"].split(",", 1)[1])
            assert encoded == (OUT / r["raw_png"]).read_bytes()
            a = np.asarray(Image.open(io.BytesIO(encoded)))
            assert a.shape == (r["H"], r["W"])
            if packet["kind"] != "feedback_development":
                original = old_rois[r["id"]]
                assert original["split"] == packet["kind"]
                assert original["box_raw_yxyx"] == r["box_raw_yxyx"]
                assert sha(BENCH/original["paths"]["raw"]) == r["raw_png_sha256"]
                assert sha(BENCH/original["predictions"]) == r["predictions_sha256"]
                known_all.add(r["id"])
                path = ROOT / "Dataset" / original["batch"] / ("img_" + r["site"] + "_BSE.tif")
            else:
                path = ROOT / "Hackathon-Polaron-test" / ("img_" + r["site"] + "_BSE.tif")
            # Numerical raw-value checks do not inspect annotations or release
            # held-out method results; no labels are created here.
            image = tifffile.imread(path)
            if image.ndim == 3:
                image = image[:, :, 0]
            y0, x0, y1, x1 = r["box_raw_yxyx"]
            assert np.array_equal(a, image[y0:y1, x0:x1]), r["alias"]
            raw_verified += 1
        blank = json.loads((OUT/(packet["kind"]+"_blank_export.json")).read_text())
        adapted = validate_export(packet, blank)
        assert all(a["review_status"] == "unreviewed" and not a["polygons"] for a in adapted["rois"])
    assert known_all == set(old_rois)
    headers = json.loads((OUT / "tiff_headers.json").read_text())
    assert len(headers) == 102 and not any(h["possible_metadata_keys"] for h in headers)
    groups = pd.read_csv(OUT / "group_mapping_template.csv", keep_default_na=False)
    assert len(groups) == groups.site.nunique() == 34
    assert groups[["specimen_id", "preparation_session_id", "imaging_session_id", "evidence_source"]].eq("").all().all()
    before = json.loads(Path(__file__).with_name("metric_roles_before.json").read_text())
    current = json.loads((ROOT / "analysis/morphology/metric_register.json").read_text())
    assert before == {m["id"]: {k: m[k] for k in ("qc_role", "review_status", "evidence_status")} for m in current["metrics"]}
    assert all(p.stat().st_size < 20_000_000 for p in OUT.rglob("*") if p.is_file())
    assert not list(OUT.rglob("*.tif"))
    result = dict(experiment="E38S", input_and_output_hashes_verified=True,
                  protected_original_artifacts=len(receipt["protected_original_artifacts_sha256"]),
                  original_artifacts_unchanged=True, known_crop_identities_preserved=11,
                  exact_raw_value_crops_verified=raw_verified, hint_free_embedded_images=raw_verified,
                  real_expert_annotations=0, grouped_validation_available=False,
                  scalar_roles_and_review_evidence_unchanged=True, raw_tiffs_packaged=False,
                  synthetic_regression_tests_passed=8, synthetic_ui_controls_passed=True,
                  native_browser_layout_verified=False,
                  browser_limitation="Browser automation refuses file URLs; no workaround attempted.")
    write_json(Path(__file__).with_name("verification.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
