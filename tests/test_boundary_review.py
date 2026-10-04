"""Safety checks use synthetic annotations only; no expert truth is fabricated."""
import copy
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import tifffile

from analysis.boundary_review import prepare as review


def fixture(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    raw = out / "raw.png"
    Image.fromarray(np.zeros((20, 30), np.uint8)).save(raw)
    pred = out / "p.npz"
    a = np.ones((20, 30), np.uint8)
    np.savez(pred, baseline=a, hysteresis=a)
    roi = dict(alias="D01", id="private_original", site="private_site", split="development",
               W=30, H=20, raw_png="raw.png", predictions="p.npz",
               raw_png_sha256=review.sha(raw), predictions_sha256=review.sha(pred))
    packet = dict(kind="development", packet_id="synthetic-packet", evaluation_manifest_id="original-manifest",
                  rois=[roi], input_sha256={})
    export = dict(schema_version=1, packet_id=packet["packet_id"], rois=[dict(
        id="D01", review_status="unreviewed", measurable=None, reviewer="", notes="",
        background="unlabelled", polygons=[], independent_drawing=False, prior_exposure="not_answered")])
    return out, packet, export


def reviewed(export):
    value = copy.deepcopy(export)
    value["rois"][0].update(review_status="reviewed", measurable=True, reviewer="Synthetic test fixture",
                           independent_drawing=True, prior_exposure="none",
                           polygons=[dict(class_id=0, points=[[5, 5], [10, 5], [10, 10], [5, 10]])])
    return value


def test_public_packet_is_strictly_raw_only(tmp_path, monkeypatch):
    out, packet, _ = fixture(tmp_path)
    packet["rois"][0].update(batch="SECRET_BATCH", bright_low_contrast=True, th_hi=99, prediction_label="SECRET_LABEL")
    monkeypatch.setattr(review, "OUT", out)
    public = review.public_packet(packet)
    assert set(public) == {"schema_version", "packet_id", "kind", "rois"}
    assert set(public["rois"][0]) == {"id", "W", "H", "image", "raw_png_sha256"}
    payload = json.dumps(public)
    assert all(k not in payload for k in ("private_original", "private_site", "SECRET_BATCH", "SECRET_LABEL", "th_hi"))


def test_identity_missing_duplicate_aliases_and_declarations_fail(tmp_path):
    _, packet, export = fixture(tmp_path)
    adapted = review.validate_export(packet, export)
    assert adapted["manifest_id"] == "original-manifest" and adapted["rois"][0]["id"] == "private_original"
    bad = copy.deepcopy(export)
    bad["packet_id"] = "another"
    with pytest.raises(ValueError, match="identity"):
        review.validate_export(packet, bad)
    bad = copy.deepcopy(export)
    bad["rois"] = []
    with pytest.raises(ValueError, match="Every"):
        review.validate_export(packet, bad)
    bad = reviewed(export)
    bad["rois"][0]["independent_drawing"] = False
    with pytest.raises(ValueError, match="declaration"):
        review.validate_export(packet, bad)
    for field, value in [("measurable", 1), ("prior_exposure", "not_answered"), ("reviewer", "")]:
        bad = reviewed(export)
        bad["rois"][0][field] = value
        with pytest.raises(ValueError):
            review.validate_export(packet, bad)


def test_bounds_uncertain_only_and_numeric_types_fail(tmp_path):
    _, packet, export = fixture(tmp_path)
    for point in ([-1, 3], [30, 3], [float("nan"), 3], ["5", 3], [True, 3]):
        bad = reviewed(export)
        bad["rois"][0]["polygons"][0]["points"][0] = point
        with pytest.raises(ValueError):
            review.validate_export(packet, bad)
    bad = reviewed(export)
    bad["rois"][0]["polygons"][0]["class_id"] = 255
    with pytest.raises(ValueError, match="labelled"):
        review.validate_export(packet, bad)


def test_existing_evaluator_partial_and_unreviewed_contract(tmp_path, monkeypatch):
    out, packet, export = fixture(tmp_path)
    monkeypatch.setattr(review, "OUT", out)
    review.write_json(out / "coordinator_manifest.json", {"packets": [packet]})
    path = tmp_path / "synthetic.json"
    review.write_json(path, export)
    dest = tmp_path / "unreviewed"
    summary = review.import_review(path, dest)
    assert summary["reviewed_measurable_rois"] == 0
    assert pd.read_csv(dest / "expert_evaluation.csv").empty
    with pytest.raises(FileExistsError):
        review.import_review(path, dest)
    review.write_json(path, reviewed(export))
    summary = review.import_review(path, tmp_path / "partial")
    rows = pd.read_csv(tmp_path / "partial/expert_evaluation.csv")
    assert len(rows) == 2 and rows.void_iou.eq(0).all()
    assert "bright_d50_error_px" not in rows
    assert summary["development_sites_scored"] == ["private_site"]
    assert not summary["held_out_sites_scored"]


def test_unmeasurable_is_an_abstention_and_exposure_is_retained(tmp_path, monkeypatch):
    out, packet, export = fixture(tmp_path)
    monkeypatch.setattr(review, "OUT", out)
    review.write_json(out / "coordinator_manifest.json", {"packets": [packet]})
    a = reviewed(export)
    a["rois"][0].update(measurable=False, polygons=[], prior_exposure="prior_exposure")
    path = tmp_path / "synthetic.json"
    review.write_json(path, a)
    dest = tmp_path / "abstention"
    review.import_review(path, dest)
    rows = pd.read_csv(dest / "expert_evaluation.csv")
    assert rows.status.tolist() == ["expert_unmeasurable"]
    receipt = json.loads((dest / "review_receipt.json").read_text())
    assert receipt["unmeasurable_rois"] == receipt["prior_exposure_rois"] == 1
    assert receipt["unseen_batch_validation"] is False


def test_held_out_requires_release_and_changed_sources_fail(tmp_path, monkeypatch):
    out, packet, export = fixture(tmp_path)
    packet["kind"] = "held_out_site"
    packet["rois"][0]["split"] = "held_out_site"
    monkeypatch.setattr(review, "OUT", out)
    review.write_json(out / "coordinator_manifest.json", {"packets": [packet]})
    path = tmp_path / "synthetic.json"
    review.write_json(path, reviewed(export))
    with pytest.raises(ValueError, match="release"):
        review.import_review(path, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    summary = review.import_review(path, tmp_path / "released", release_held_out=True)
    assert summary["held_out_sites_scored"] == ["private_site"]
    (out / "raw.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        review.import_review(path, tmp_path / "changed", release_held_out=True)
    assert not (tmp_path / "changed").exists()


def test_whole_crop_bright_d50_excludes_clipped_objects_on_both_masks(tmp_path, monkeypatch):
    out, packet, export = fixture(tmp_path)
    monkeypatch.setattr(review, "OUT", out)
    a = reviewed(export)
    a["rois"][0].update(background="solid", polygons=[
        dict(class_id=2, points=[[12, 5], [19, 5], [19, 12], [12, 12]])])
    pred = np.ones((20, 30), np.uint8)
    pred[5:13, 12:20] = 2
    pred[1:19, 0:8] = 2  # Large clipped object must not change the object D50.
    np.savez(out / "p.npz", baseline=pred, hysteresis=pred)
    packet["rois"][0]["predictions_sha256"] = review.sha(out / "p.npz")
    review.write_json(out / "coordinator_manifest.json", {"packets": [packet]})
    path = tmp_path / "synthetic.json"
    review.write_json(path, a)
    dest = tmp_path / "whole_crop"
    review.import_review(path, dest)
    rows = pd.read_csv(dest / "expert_evaluation.csv")
    assert rows.bright_d50_error_px.eq(0).all()
    assert rows.manual_bright_objects_retained.eq(1).all()
    assert rows.predicted_bright_objects_retained.eq(1).all()
    assert rows.predicted_bright_objects_edge_excluded.eq(1).all()
    d50, n, excluded = review.interior_bright_summary(pred == 2)
    assert np.isclose(d50, np.sqrt(4 * 64 / np.pi)) and n == excluded == 1
    only_edge = np.zeros((20, 30), bool)
    only_edge[:, :5] = True
    d50, n, excluded = review.interior_bright_summary(only_edge)
    assert np.isnan(d50) and n == 0 and excluded == 1


def test_headers_preserve_explicit_metadata_leads_without_inferred_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, np.zeros((15, 20), np.uint8), metadata={"specimen_id": "fixture-specimen"})
    header = review.inspect_header(path)
    assert "specimen_id" in header["possible_metadata_keys"]
    assert "ImageLength" in header["tag_names"]
    assert "specimen_id" not in header  # Leads require interpretation, not automatic grouping.
    assert header["file_sha256"] == review.sha(path)
