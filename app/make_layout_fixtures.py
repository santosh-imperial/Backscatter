"""app.make_layout_fixtures — layout-only copies of a real lot bundle that show the verdict states the known lots never
reach (lot-wide drift, localized anomaly pending review, provisional reject, quality abstention).

    python -m app.make_layout_fixtures --from ui/bundles/lot_Batch_1 --out ui/fixtures

Every fixture carries ``"fixture": true`` and a lot name starting with "Fixture", and the page shows a banner on it.
``app.build_ui`` accepts fixtures only through ``--fixture-lot``, never ``--lot``, so a demo build cannot include one
by accident. The altered fields are only those the page reads for the verdict; the measurements and images are the
source lot's, unchanged. These files are not results and never enter the experiment log or registry.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil

from polaron_qc import decision

FIXTURE_NOTE = "Layout fixture: an altered copy of {src} made to show this verdict state. Not a result."


def _variants(src: dict) -> dict[str, dict]:
    lot = src["lot"]
    out = {}

    def base(name, verdict, reason, action):
        d = copy.deepcopy(src)
        d["fixture"] = True
        d["lot"] = f"Fixture-{name}"
        d["summary"]["verdict"] = verdict
        d["verdict_reason"] = FIXTURE_NOTE.format(src=lot) + " " + reason
        d["next_action"] = action
        return d

    d = base("drift", decision.INVESTIGATE_DRIFT, "bright_d50 is set as a driver.",
             "examine the bright-phase images of the furthest sites and ask the supplier about particle size")
    d["summary"]["outcome_columns"]["drift_alert"] = True
    d["summary"]["drivers"] = ["bright_d50"]
    for r in d["compare"]:
        if r["kpi"] == "bright_d50":
            r["p_holm"] = 0.012
    out["drift"] = d

    d = base("localized", decision.INVESTIGATE_LOCAL, "One site is set 3.4 MAD above the ordinary-baseline maximum.",
             "review the marked crop and confirm or refute it")
    d["summary"]["outcome_columns"]["localized"] = "pending_review"
    for f in d["summary"]["local_anomaly"]["flags"]:
        f["margin_in_mad"], f["severity_ok"], f["review_status"] = 3.4, True, "unreviewed"
    out["localized"] = d

    d = base("reject", decision.REJECT, "Drift and a confirmed localized anomaly are both set.",
             "hold the lot and send the evidence to the supplier")
    d["summary"]["outcome_columns"].update(drift_alert=True, localized="credible")
    d["summary"]["drivers"] = ["bright_d50"]
    for f in d["summary"]["local_anomaly"]["flags"]:
        f["margin_in_mad"], f["severity_ok"], f["review_status"] = 3.4, True, "confirmed"
    out["reject"] = d

    d = base("abstention", decision.INVESTIGATE_DRIFT, "Too few usable sites are set for every primary KPI.",
             "image more sites before any decision")
    d["summary"]["outcome_columns"]["quality_abstention"] = True
    d["summary"]["abstention"] = {"abstain": True, "reasons": ["fixture: usable sites below the minimum"]}
    out["abstention"] = d
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", required=True, help="a real lot bundle directory")
    ap.add_argument("--out", required=True, help="directory for the fixture bundles (must not exist)")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; fixtures are written to a new directory")
    src = json.load(open(os.path.join(a.src, "lot.json")))
    if src.get("fixture"):
        raise SystemExit("source is already a fixture")
    for name, doc in _variants(src).items():
        d = os.path.join(a.out, f"fixture_{name}")
        os.makedirs(os.path.join(d, "img"))
        for fn in os.listdir(os.path.join(a.src, "img")):
            s, t = os.path.join(a.src, "img", fn), os.path.join(d, "img", fn)
            try:
                os.link(s, t)
            except OSError:
                shutil.copy2(s, t)
        with open(os.path.join(d, "lot.json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=1)
        print("wrote", d, doc["summary"]["verdict"])


if __name__ == "__main__":
    main()
