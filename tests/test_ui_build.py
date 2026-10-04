"""Customer QC interface (D56U): wording guards, STE text rules, fail-closed inputs and an end-to-end build when the
locally exported bundles exist (they need the raw Dataset/, so the integration test is skipped without them)."""
import json
import os
import re

import pytest

from app import build_ui, export_lot
from polaron_qc import PRIMARY_KPIS, decision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLES = os.path.join(ROOT, "ui", "bundles")
EVAL = os.path.join(ROOT, "analysis", "submission_v2", "freeze", "evaluation")

# Technical names that end in -ing and are permitted in STE as names (docs/ui_plan.md §4.4)
STE_ING_NAMES = {"calendering", "coating"}
NOT_ING_FORMS = {"during"}  # prepositions that only look like -ing forms
# Words STE does not approve in these meanings; the drafts used them before the STE rewrite
STE_AVOID = {"suggest", "suggests", "may", "might", "matter", "matters", "hypothesis", "relates", "relative", "weaker", "origin"}


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _all_ste_texts():
    s = build_ui.STE
    return [s["intro"], s["no_driver"], s["tier2_sub"], s["tier3_sub"], *s["kpi"].values()]


def test_every_decision_verdict_has_a_label_and_unknown_fails():
    for v in (decision.CONSISTENT, decision.INVESTIGATE_DRIFT, decision.INVESTIGATE_LOCAL, decision.REJECT):
        label, state = build_ui.verdict_label(v)
        assert state in ("ok", "warn", "bad") and "accept" not in label.lower()
    with pytest.raises(SystemExit):
        build_ui.verdict_label("accept")


def test_wording_guard_catches_forbidden_words_and_ignores_image_data():
    assert build_ui.check_wording("Batch accepted.") == ["accepted"]
    assert build_ui.check_wording("High Confidence") == ["Confidence"]
    assert build_ui.check_wording('<img src="data:image/jpeg;base64,acceptconfidence">') == []
    assert build_ui.check_wording("Consistent within detectable limits") == []


def test_ste_texts_cover_primary_kpis_and_follow_sentence_rules():
    assert set(build_ui.STE["kpi"]) == set(PRIMARY_KPIS)
    for text in _all_ste_texts():
        assert len(_sentences(text)) <= 6, text
        for sent in _sentences(text):
            words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", sent)
            assert len(words) <= 25, sent
            ing = {w.lower() for w in words if w.lower().endswith("ing") and len(w) > 4} - STE_ING_NAMES - NOT_ING_FORMS
            assert not ing, (sent, ing)
            assert not ({w.lower() for w in words} & STE_AVOID), sent
            assert "'" not in sent, sent  # no contractions


def test_builder_fails_closed_on_missing_inputs(tmp_path):
    with pytest.raises(SystemExit):
        build_ui.main(["--lot", str(tmp_path / "nope"), "--track-record", EVAL, "--out", str(tmp_path / "x.html")])
    with pytest.raises(SystemExit):
        build_ui.main(["--track-record", EVAL, "--out", str(tmp_path / "x.html")])


def test_export_refuses_non_empty_output(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(SystemExit):
        export_lot._new_out(str(tmp_path))


def test_json_cleaner_handles_numpy_and_nan():
    import numpy as np
    out = export_lot._j({"a": np.float64("nan"), "b": np.int64(3), "c": (np.bool_(True), 1.5)})
    assert out == {"a": None, "b": 3, "c": [True, 1.5]}
    json.dumps(out)


@pytest.mark.skipif(not os.path.exists(os.path.join(BUNDLES, "lot_Batch_1", "baseline.json")), reason="UI bundles not exported locally")
def test_end_to_end_build(tmp_path):
    out = tmp_path / "index.html"
    lots = [os.path.join(BUNDLES, d) for d in sorted(os.listdir(BUNDLES)) if d.startswith("lot_")]
    args = ["--track-record", EVAL, "--out", str(out)]
    for d in lots:
        args += ["--lot", d]
    build_ui.main(args)
    html = out.read_text()
    assert build_ui.check_wording(html) == []
    payload = re.search(r"const DATA = (\{.*?\});\nconst IMG", html, re.S).group(1)
    data = json.loads(payload.replace("<\\/", "</"))
    for lot in data["lots"]:
        assert lot["verdict_label"] and lot["distance"]["sha256"]
        # C07: every "not detected" lane carries the MDC of each primary KPI
        assert set(lot["mdc"]) == set(PRIMARY_KPIS)
        assert all(v["feasible"] is not None for v in lot["mdc"].values())
    receipt = json.load(open(tmp_path / "index_receipt.json"))
    assert receipt["output_sha256"] and all(len(h) == 64 for h in receipt["inputs_sha256"].values())
    with pytest.raises(SystemExit):
        build_ui.main(args)  # an existing page is never overwritten without --replace


@pytest.mark.skipif(not os.path.exists(os.path.join(BUNDLES, "lot_Batch_1", "lot.json")), reason="UI bundles not exported locally")
def test_layout_fixtures_never_enter_a_demo_build(tmp_path):
    from app import make_layout_fixtures
    fx = tmp_path / "fx"
    make_layout_fixtures.main(["--from", os.path.join(BUNDLES, "lot_Batch_1"), "--out", str(fx)])
    names = sorted(os.listdir(fx))
    assert names == ["fixture_abstention", "fixture_drift", "fixture_localized", "fixture_reject"]
    for n in names:
        doc = json.load(open(fx / n / "lot.json"))
        assert doc["fixture"] is True and doc["lot"].startswith("Fixture-") and "Not a result" in doc["verdict_reason"]
        build_ui.verdict_label(doc["summary"]["verdict"])  # every fixture verdict is a real decision-module state
    with pytest.raises(SystemExit):  # a fixture passed as a real lot is refused
        build_ui.main(["--lot", str(fx / "fixture_drift"), "--track-record", EVAL, "--out", str(tmp_path / "x.html")])
    with pytest.raises(SystemExit):  # a real lot passed as a fixture is refused
        build_ui.main(["--lot", os.path.join(BUNDLES, "lot_Batch_1"), "--fixture-lot", os.path.join(BUNDLES, "lot_Batch_1"),
                       "--track-record", EVAL, "--out", str(tmp_path / "y.html")])


def test_view_ids_never_contain_the_deep_link_separator():
    import re as _re
    assert _re.sub(r"[^A-Za-z0-9_.-]", "_", "Batch~1 x") == "Batch_1_x"
    src = open(os.path.join(ROOT, "app", "build_ui.py")).read()
    assert "~" not in "".join(_re.findall(r"re\.sub\(r\"(\[[^\]]+\])\"", src))
