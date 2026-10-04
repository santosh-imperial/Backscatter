"""Inspection server (D57U): request validation, upload guards and the input check, without running the pipeline."""
import importlib
import json

import numpy as np
import pytest
import tifffile

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("POLARON_INSPECTIONS", str(tmp_path / "runs"))
    monkeypatch.setenv("POLARON_INBOX", str(tmp_path / "inbox"))
    import app.server as server
    server = importlib.reload(server)
    (tmp_path / "runs").mkdir(); (tmp_path / "inbox").mkdir()
    return server, TestClient(server.app), tmp_path


def _tif(path, seed=0):
    tifffile.imwrite(path, np.random.default_rng(seed).integers(0, 255, (40, 60), dtype=np.uint8))


def test_create_rejects_bad_and_training_names(srv):
    _, c, _ = srv
    assert c.post("/api/runs", json={"name": "../etc", "mode": "lot", "source": "upload"}).status_code == 400
    assert c.post("/api/runs", json={"name": "Batch_1", "mode": "lot", "source": "upload"}).status_code == 400
    assert c.post("/api/runs", json={"name": "Lot-A", "mode": "pooled", "source": "upload"}).status_code == 400
    assert c.post("/api/runs", json={"name": "Lot-A", "mode": "lot", "source": "inbox", "folder": "../x"}).status_code == 400


def test_upload_accepts_only_detector_file_names_and_never_overwrites(srv):
    _, c, tmp = srv
    rec = c.post("/api/runs", json={"name": "Lot-A", "mode": "lot", "source": "upload"}).json()
    assert rec["status"] == "uploading" and [s["key"] for s in rec["steps"]][0] == "check"
    url = f"/api/runs/{rec['id']}/files/"
    assert c.put(url + "notes.txt", content=b"x").status_code == 400
    assert c.put(url + "img_new1_BSE.tif", content=b"abc").status_code == 200
    assert c.put(url + "img_new1_BSE.tif", content=b"abc").status_code == 409
    assert c.get("/api/runs/../../etc").status_code == 404
    listed = c.get("/api/runs").json()
    assert listed[0]["name"] == "Lot-A"


def test_input_check_needs_three_detectors_and_rejects_training_site_ids(srv, tmp_path):
    server, _, _ = srv
    d = tmp_path / "lot"; d.mkdir()
    _tif(d / "img_new1_BSE.tif"); _tif(d / "img_new1_ETD.tif", 1)
    with pytest.raises(RuntimeError, match="expected exactly one BSE"):
        server.check_input(d, "lot")
    _tif(d / "img_new1_Inlens.tif", 2)
    info = server.check_input(d, "lot")
    assert info["sites"] == ["new1"] and "quality abstention" in info["note"]
    k = tmp_path / "known"; k.mkdir()
    for det, seed in (("BSE", 3), ("ETD", 4), ("Inlens", 5)):
        _tif(k / f"img_4ih2ggld_{det}.tif", seed)
    with pytest.raises(RuntimeError, match="training set"):
        server.check_input(k, "samples")


def test_tabs_are_injected_after_body(srv):
    server, _, _ = srv
    html = server.with_tabs("<html><body><main></main></body></html>", "built")
    assert html.index("apptabs") > html.index("<body>") and 'href="/built" aria-current=page' in html


def test_known_sites_rehearsal_flag_and_plain_log_route(srv):
    server, c, _ = srv
    known = c.get("/api/known-sites").json()
    assert "4ih2ggld" in known and len(known) == 31
    rec = c.post("/api/runs", json={"name": "Rehearsal-x", "mode": "samples", "source": "upload"}).json()
    assert rec["rehearsal"] is True
    assert c.get("/api/runs").json()[0]["rehearsal"] is True
    r = c.get(f"/runs/{rec['id']}/log/score")
    assert r.status_code == 200 and r.text == "No log for this step."
    assert c.get(f"/runs/{rec['id']}/log/../../etc").status_code == 404


def test_failed_subprocess_step_shows_plain_text_not_a_traceback(srv, tmp_path, monkeypatch):
    server, c, _ = srv
    d = tmp_path / "lot"; d.mkdir()
    for det, seed in (("BSE", 1), ("ETD", 2), ("Inlens", 3)):
        _tif(d / f"img_zz9_{det}.tif", seed)
    rec = c.post("/api/runs", json={"name": "Lot-B", "mode": "samples", "source": "upload"}).json()
    run = server.RUNS / rec["id"]
    r = server.load(run); r["input_dir"] = str(d); server.save(run, r)
    def boom(dd, key, cmd):
        raise RuntimeError("Traceback (most recent call last):\n  File x\nValueError: secret internals")
    monkeypatch.setattr(server, "_sub", boom)
    server.pipeline(run)
    out = server.load(run)
    failed = [s for s in out["steps"] if s["status"] == "failed"][0]
    assert out["status"] == "failed" and failed["key"] == "score" and failed["log"] is True
    assert "Traceback" not in failed["detail"] and "log" in failed["detail"]
