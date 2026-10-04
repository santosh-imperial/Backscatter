"""app.build_ui — build the customer QC interface (docs/ui_plan.md, D56U) as one self-contained HTML file.

    python -m app.build_ui --lot ui/bundles/lot_Batch_1 --lot ui/bundles/lot_Batch_2 \
        --samples ui/bundles/samples_drop01 --track-record analysis/submission_v2/freeze/evaluation \
        --feedback /path/to/analysis/feedback_drop_01 --out ui/index.html

The builder reads saved bundles only (``app.export_lot``): it never runs a model, refits or re-scores. Missing inputs
fail; nothing falls back to another model or a default verdict (C34S). It trims every bundle to the fields the page
shows, embeds the images as data URIs, checks the page wording and writes a receipt with the SHA256 of every input.

The display text of the "What this could mean for the cell" panel is ASD-STE100 Simplified Technical English,
approved by Santosh (docs/ui_plan.md §4). Edit it here and in the plan together.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys

import pandas as pd

from polaron_qc import BATCH_COLORS, PRIMARY_KPIS, UNSEEN_COLOR
from polaron_qc import decision

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "ui_template.html")
BUILDER_VERSION = "ui-1.0.0"

# Verdict strings come from the frozen decision module; an unknown string is an error, never a default label.
VERDICT_LABELS = {
    decision.CONSISTENT: ("Consistent within detectable limits", "ok"),
    decision.INVESTIGATE_DRIFT: ("Investigate: lot-wide drift", "warn"),
    decision.INVESTIGATE_LOCAL: ("Investigate: localized anomaly", "warn"),
    decision.REJECT: ("Reject (provisional)", "bad"),
}

# ---- ASD-STE100 display text (docs/ui_plan.md §4.1–4.3, approved by Santosh 2026-10-04) -----------------------
STE = {
    "intro": ("This panel shows possible effects on the cell. It shows text only for the KPIs that cause the verdict. "
              "The text gives only the direction of the change. It does not give a performance value."),
    "no_driver": ("No primary KPI has a change that is larger than the minimum detectable change. "
                  "The system does not show possible effects for smaller changes."),
    "kpi": {
        "crack_frac": ("This value is the area of long voids in the section. These voids are similar to delamination. "
                       "An increase can show a decrease in coating cohesion or adhesion. This condition can be important "
                       "during calendering and formation. Sample preparation can also cause these voids. "
                       "Examine the images to find the cause."),
        "pore_max_d": ("This value is the diameter of the largest void in the section. One large void can be important. "
                       "This is also true when the lot average does not change. The local defect rule examines this value."),
        "pore_frac": ("This value is the area fraction of the voids that the image shows. This value can change with the "
                      "calendering density. It can also have an effect on electrolyte access. The image does not show fine "
                      "pores or binder. Use this value only to compare lots. It is not the total porosity."),
        "bright_frac": ("This value is the area of the bright additive phase in the section. Possibly, this additive is Si "
                        "or SiOx. This system does not identify the chemistry. A change can show a change in formulation or "
                        "dispersion. Ask the supplier about this change. This value does not measure capacity."),
        "bright_d50": ("This value is the median diameter of the additive particles in the section. An increase can show a "
                       "change in the particle size from the supplier. It can also show agglomeration. A change can have an "
                       "effect on mechanical properties during cell operation. This system does not measure this effect."),
    },
    "tier2_title": "Context values", "tier2_sub": "The verdict does not use these values.",
    "tier3_title": "Experimental battery geometry",
    "tier3_sub": "An expert must examine these values before use. The verdict does not use these values.",
}
assert set(STE["kpi"]) == set(PRIMARY_KPIS)

# Words the page must never show (CLAUDE.md Style; checklist C17). Checked on the final HTML minus image data.
FORBIDDEN = [r"\baccept(?:ed|s|ance)?\b", r"\bconfidence\b", r"\bconfident\b"]
PROVENANCE = "Developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived."


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _need(path: str) -> str:
    if not os.path.exists(path):
        raise SystemExit(f"missing input {path}")
    return path


class Images:
    """Collects bundle JPEGs as data URIs under short keys, so the page data refers to images by key."""

    def __init__(self):
        self.uris: dict[str, str] = {}

    def add(self, bundle: str, rel: str) -> str:
        path = _need(os.path.join(bundle, "img", rel))
        key = f"i{len(self.uris)}"
        with open(path, "rb") as fh:
            self.uris[key] = "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()
        return key


def _views(img: Images, bundle: str, images: dict) -> dict:
    return {name: {"key": img.add(bundle, v["file"]), "w": v["w"], "h": v["h"]} for name, v in images["views"].items()}


def verdict_label(v: str) -> tuple[str, str]:
    if v not in VERDICT_LABELS:
        raise SystemExit(f"unknown verdict string {v!r}; add it to VERDICT_LABELS from polaron_qc.decision")
    return VERDICT_LABELS[v]


# ---------------------------------------------------------------------------------------------------------------
# loaders: each returns only the fields the page shows
# ---------------------------------------------------------------------------------------------------------------
def load_lot(bundle: str, img: Images, inputs: dict, fixture: bool = False) -> dict:
    p = _need(os.path.join(bundle, "lot.json"))
    inputs[p] = sha256(p)
    d = json.load(open(p))
    if d.get("kind") != "lot":
        raise SystemExit(f"{p} is not a lot bundle")
    if bool(d.get("fixture")) != fixture:
        raise SystemExit(f"{p}: layout fixtures go through --fixture-lot only, real lots through --lot only")
    s = d["summary"]
    label, state = verdict_label(s["verdict"])
    if s["outcome_columns"].get("quality_abstention"):
        # The frozen rules return an investigate string for an abstention; the page names the real reason and keeps
        # the frozen string in the footer and the details.
        label = "Investigate: too few usable sites (quality abstention)"
    if s["meta"]["thresholds_hash"] != decision.Thresholds().hash():
        raise SystemExit(f"{p}: thresholds hash {s['meta']['thresholds_hash']} differs from the frozen rules ({decision.Thresholds().hash()}); re-export the lot")
    if d.get("distance") and os.path.isabs(str(d["distance"].get("source", ""))):
        src = d["distance"]["source"]  # show repository-relative paths, never a home directory
        root = os.path.dirname(HERE)
        d["distance"]["source"] = os.path.relpath(src, root) if src.startswith(root) else os.path.basename(src)
    if d.get("distance") is None:
        raise SystemExit(f"{p} has no lane-3 distance table; export it with --distance-table")
    sites = [{k: v for k, v in site.items() if k != "images"} | {"views": _views(img, bundle, site["images"]),
             "images": {"trim_rows": site["images"]["trim_rows"], "shape": site["images"]["shape"]}} for site in d["sites"]]
    crops = [{"key": img.add(bundle, c["file"]), **{k: c[k] for k in ("site", "kpi", "what", "area_px", "major_axis_px", "rows_trimmed_frame", "cols", "band_top")},
              "margin_in_mad": c["flag"]["margin_in_mad"], "review_status": c["flag"]["review_status"],
              "severity_ok": c["flag"]["severity_ok"], "measurement_note": c["flag"]["measurement_note"]} for c in d["evidence_crops"]]
    return {
        "id": "lot-" + re.sub(r"[^A-Za-z0-9_.-]", "_", d["lot"]), "lot": d["lot"], "reference": d["reference"], "fixture": fixture,
        "color": BATCH_COLORS.get(d["lot"], UNSEEN_COLOR), "exported_utc": d["exported_utc"],
        "verdict": s["verdict"], "verdict_label": label, "verdict_state": state, "reason": d["verdict_reason"],
        "next_action": d["next_action"], "what_would_move_it": s.get("what_would_move_it"),
        "outcome": s["outcome_columns"], "drivers": s.get("drivers", []), "abstention": s.get("abstention"),
        "stability": {k: s["stability"][k] for k in ("share", "n_runs")}, "energy": d["energy"],
        "mdc": s["mdc"], "usable_n": s.get("usable_n"), "compare": d["compare"], "excluded": d["excluded_by_flag"],
        "ref_values": d["ref_values"], "sites": sites, "crops": crops, "local": d["local"],
        "local_flags": s["local_anomaly"]["flags"], "descriptive_flags": d.get("descriptive_flags", []),
        "acquisition": d["acquisition"], "physics": d["physics"], "battery": d["battery_secondary"],
        "distance": d["distance"], "meta": s["meta"], "limits": d.get("limits"), "reviews_applied": d.get("reviews_applied", []),
        "frozen_text": {"reason": d["verdict_reason"], "what_would_move_it": s.get("what_would_move_it")},
    }


def load_baseline(bundle: str, img: Images, inputs: dict) -> dict:
    p = _need(os.path.join(bundle, "baseline.json"))
    inputs[p] = sha256(p)
    d = json.load(open(p))
    d["examples"] = [{k: v for k, v in e.items() if k != "images"} | {"views": _views(img, bundle, e["images"])} for e in d["examples"]]
    d["color"] = BATCH_COLORS.get(d["reference"], UNSEEN_COLOR)
    return d


def load_samples(bundle: str, img: Images, inputs: dict, truth: dict) -> dict:
    p = _need(os.path.join(bundle, "samples.json"))
    inputs[p] = sha256(p)
    d = json.load(open(p))
    out = []
    for s in d["samples"]:
        pr = s["prediction"]
        scores = {b: pr[f"model_score_{b}"] for b in ("Batch_1", "Batch_2", "Batch_3")}
        drv = s["drivers"]
        out.append({
            "sample_id": s["sample_id"], "bet": pr["predicted_batch"], "runner_up": pr["runner_up"], "scores": scores,
            "margin": pr["margin"], "deletion_same": pr["deletion_same_assignment"], "deletion_n": pr["deletion_fit_count"],
            "morph_bet": pr["morphology_only_bet"], "acq_bet": pr["acquisition_only_bet"],
            "baseline_pct": pr["baseline_morphology_percentile"], "exceeds_baseline": pr["exceeds_observed_baseline_max"],
            "flags": s["flags"], "acq_pct": s["distance"].get("acq_pct"),
            "drivers": [{k: r.get(k) for k in ("feature", "family", "value", "quality", "contribution", "median_Batch_1", "median_Batch_2", "median_Batch_3",
                                               "measurement_missing_imputed")} for r in drv],
            "intercept": drv[0]["intercept_difference"] if drv else None,
            "truth": truth.get(s["sample_id"]),
            "views": _views(img, bundle, s["images"]),
        })
    return {"id": "samples-" + re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(os.path.normpath(bundle))), "name": d["name"],
            "model_version": d["model_version"], "primary_family": d["primary_family"], "model_sha256": d["model_sha256"],
            "source_snapshot_sha256": d["source_snapshot_sha256"], "scored_utc": d["scored_utc"], "samples": out,
            "truth_available": bool(truth)}


def load_track(eval_dir: str, inputs: dict) -> dict:
    files = {k: _need(os.path.join(eval_dir, f)) for k, f in (("summary", "categoriser_summary.csv"), ("confusion", "confusion_material.csv"),
                                                             ("reliability", "reliability_material.csv"))}
    for f in files.values():
        inputs[f] = sha256(f)
    summ = pd.read_csv(files["summary"])
    cols = ["family", "primary", "n_features", "n_sites", "balanced_accuracy", "recall_Batch_1", "recall_Batch_2", "recall_Batch_3",
            "perm_p", "n_perm", "accuracy", "accuracy_ci95", "label"]
    conf = pd.read_csv(files["confusion"], index_col=0)
    return {"summary": summ[cols].to_dict("records"),
            "confusion": {"rows": [r.replace("true ", "") for r in conf.index], "cols": [c.replace("pred ", "") for c in conf.columns],
                          "values": conf.values.tolist()},
            "reliability": pd.read_csv(files["reliability"]).to_dict("records")}


def load_feedback(fb_dir: str, inputs: dict) -> tuple[dict, dict]:
    files = {k: _need(os.path.join(fb_dir, f)) for k, f in (("truth", "truth.csv"), ("summary", "output/summary.csv"))}
    for f in files.values():
        inputs[f] = sha256(f)
    t = pd.read_csv(files["truth"])
    truth = {str(r.sample_id): {"true_batch": r.true_batch, "source": r.source} for r in t.itertuples()}
    s = pd.read_csv(files["summary"])
    return truth, {"source": t.source.iloc[0], "rows": s.to_dict("records")}


def load_registry(path: str, experiment: str, inputs: dict) -> list[dict]:
    inputs[_need(path)] = sha256(path)
    r = pd.read_csv(path, dtype=str)
    return r[r.experiment == experiment].fillna("").to_dict("records")


# ---------------------------------------------------------------------------------------------------------------
def _clean(o):
    """NaN → None so the embedded JSON is valid."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


def check_wording(html: str) -> list[str]:
    """Forbidden words in the page, ignoring embedded image data. Returns the offending matches."""
    text = re.sub(r"data:image/[^\"']+", "", html)
    hits = []
    for pat in FORBIDDEN:
        hits += [m.group(0) for m in re.finditer(pat, text, flags=re.I)]
    return hits


def build(args) -> str:
    inputs: dict[str, str] = {}
    img = Images()
    truth, feedback = load_feedback(args.feedback, inputs) if args.feedback else ({}, None)
    lots = [load_lot(b, img, inputs) for b in args.lot] + [load_lot(b, img, inputs, fixture=True) for b in args.fixture_lot]
    if not lots and not args.samples:
        raise SystemExit("nothing to show: pass at least one --lot or --samples bundle")
    base_dirs = ([args.baseline] if args.baseline else []) + [b for b in args.lot if os.path.exists(os.path.join(b, "baseline.json"))]
    if not base_dirs:
        raise SystemExit("no lot bundle carries baseline.json; export one lot with --with-baseline")
    baseline = load_baseline(base_dirs[0], img, inputs)
    samplesets = [load_samples(b, img, inputs, truth) for b in args.samples]
    track = load_track(args.track_record, inputs)
    loco = load_registry(args.registry, "E35", inputs)
    try:
        git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        git = None
    data = _clean({
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "builder": BUILDER_VERSION, "git": git,
        "provenance": PROVENANCE, "inspection": bool(getattr(args, "inspection", False)), "primary_kpis": PRIMARY_KPIS, "batch_colors": BATCH_COLORS, "unseen_color": UNSEEN_COLOR,
        "ste": STE, "thresholds": {k: v for k, v in dataclasses.asdict(decision.Thresholds()).items() if k != "provenance"} | {"hash": decision.Thresholds().hash()},
        "baseline": baseline, "lots": lots, "samplesets": samplesets, "track": track, "feedback": feedback, "loco": loco,
    })
    tpl = open(TEMPLATE, encoding="utf-8").read()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    images = json.dumps(img.uris)
    html = tpl.replace("/*__DATA__*/null", payload).replace("/*__IMAGES__*/null", images)
    hits = check_wording(html)
    if hits:
        raise SystemExit(f"forbidden wording in the page: {sorted(set(hits))}")
    if os.path.exists(args.out) and not args.replace:
        raise SystemExit(f"{args.out} exists; pass --replace to rebuild it")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    receipt = {"built_utc": data["built_utc"], "builder": BUILDER_VERSION, "git": git, "template_sha256": sha256(TEMPLATE),
               "builder_sha256": sha256(os.path.abspath(__file__)), "inputs_sha256": inputs, "output": args.out,
               "output_sha256": sha256(args.out), "size_mb": round(os.path.getsize(args.out) / 1e6, 2)}
    with open(os.path.splitext(args.out)[0] + "_receipt.json", "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=1)
    print(f"wrote {args.out} ({receipt['size_mb']} MB): {len(lots)} lots, {len(samplesets)} sample sets, {len(img.uris)} images")
    return args.out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lot", action="append", default=[], help="lot bundle directory (repeatable)")
    ap.add_argument("--samples", action="append", default=[], help="sample-set bundle directory (repeatable)")
    ap.add_argument("--fixture-lot", action="append", default=[], help="layout-only fixture bundle (app.make_layout_fixtures); never for a demo build")
    ap.add_argument("--inspection", action="store_true", help="page for one inspection run (app.server): no known-lot verdict table")
    ap.add_argument("--baseline", help="bundle directory holding baseline.json (default: the first --lot that has one)")
    ap.add_argument("--track-record", required=True, help="frozen categoriser evaluation directory")
    ap.add_argument("--feedback", help="organiser feedback directory with truth.csv and output/summary.csv")
    ap.add_argument("--registry", default="experiments/registry.csv")
    ap.add_argument("--out", default="ui/index.html")
    ap.add_argument("--replace", action="store_true", help="overwrite an existing page")
    build(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
