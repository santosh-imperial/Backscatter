"""E38S: raw-only review packets and an explicit grouping-metadata evidence sheet.

Build once: python -m analysis.boundary_review.prepare
Import: python -m analysis.boundary_review.prepare --annotations FILE --out FRESH_DIR
Held-out import additionally requires --release-held-out (after method choices are fixed).
No expert labels are created by preparation. No classifier is fit or changed.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from skimage.measure import label

from analysis.morphology import build_benchmark as benchmark
from analysis.morphology.benchmark_methods import bright_hysteresis, semantic_mask, component_summary
from polaron_qc import features

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
BENCH = ROOT / "analysis/morphology/benchmark"
METHOD_SOURCES = [ROOT / p for p in (
    "analysis/morphology/build_benchmark.py",
    "analysis/morphology/benchmark_methods.py", "polaron_qc/features.py")]
PACKETS = ("development", "held_out_site", "feedback_development")
SIGNAL = re.compile(r"specimen|sample.?id|session|preparat|acquisit|datetime|voltage|magnification|working.?distance|detector.?gain", re.I)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_keys(value):
    if isinstance(value, dict):
        return [str(k) for k in value] + [k for v in value.values() for k in json_keys(v)]
    if isinstance(value, list):
        return [k for v in value for k in json_keys(v)]
    return []


def inspect_header(path):
    """Export metadata only; explicit signal keys are leads, never inferred group IDs."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = {t.name: t.value for t in page.tags.values()}
        desc = str(tags.get("ImageDescription", ""))
        try:
            desc_keys = json_keys(json.loads(desc))
        except (ValueError, TypeError):
            desc_keys = []
        keys = sorted(set(tags) | set(desc_keys))
        leads = sorted(k for k in keys if SIGNAL.search(k))
        if SIGNAL.search(desc) and not desc_keys:
            leads.append("ImageDescription:text")
        # Structural strip offsets/counts have no grouping meaning and are omitted
        # from values, while their tag names remain in the full inventory.
        useful = {k: str(v) for k, v in tags.items()
                  if k not in ("StripOffsets", "StripByteCounts", "TileOffsets", "TileByteCounts")}
        return dict(file=str(path.relative_to(ROOT)), pages=len(tif.pages),
                    shape=list(page.shape), dtype=str(page.dtype), tag_names=sorted(tags),
                    description_keys=desc_keys, possible_metadata_keys=leads,
                    tag_values=useful, file_sha256=sha(path))


def public_packet(packet):
    """Strict allowlist: never put coordinator IDs, methods or flags in reviewer HTML."""
    rois = []
    for r in packet["rois"]:
        raw = OUT / r["raw_png"]
        rois.append(dict(id=r["alias"], W=r["W"], H=r["H"],
                         raw_png_sha256=sha(raw),
                         image="data:image/png;base64," + base64.b64encode(raw.read_bytes()).decode()))
    return dict(schema_version=1, packet_id=packet["packet_id"], kind=packet["kind"], rois=rois)


def validate_export(packet, exported):
    """Validate before writing/scoring; the human declaration is recorded, not certified."""
    if exported.get("schema_version") != 1 or exported.get("packet_id") != packet["packet_id"]:
        raise ValueError("Wrong packet identity/version")
    rois = exported.get("rois")
    expected = {r["alias"]: r for r in packet["rois"]}
    if not isinstance(rois, list) or len(rois) != len(expected):
        raise ValueError("Every packet ROI must be present, including unreviewed ones")
    seen = set()
    adapted = []
    for a in rois:
        rid = a.get("id")
        if rid not in expected or rid in seen:
            raise ValueError("Unknown or duplicate alias")
        seen.add(rid)
        r = expected[rid]
        if a.get("review_status") not in ("unreviewed", "reviewed"):
            raise ValueError("Unknown review status")
        measurable = a.get("measurable")
        if measurable is not None and type(measurable) is not bool:
            raise ValueError("Measurability must be an explicit boolean or null")
        if type(a.get("independent_drawing")) is not bool:
            raise ValueError("Independent-drawing declaration must be boolean")
        if a.get("prior_exposure") not in ("not_answered", "none", "prior_exposure"):
            raise ValueError("Record prior exposure explicitly")
        if not isinstance(a.get("reviewer"), str) or not isinstance(a.get("notes"), str):
            raise ValueError("Reviewer and notes must be text")
        if not isinstance(a.get("polygons"), list):
            raise ValueError("Polygons must be a list")
        for polygon in a["polygons"]:
            if type(polygon.get("class_id")) is not int:
                raise ValueError("Class IDs must be integers")
            points = polygon.get("points")
            if (not isinstance(points, list) or any(not isinstance(q, list) or len(q) != 2
                    or any(type(v) not in (int, float) for v in q) for q in points)):
                raise ValueError("Polygon coordinates must be numeric pairs")
        mask = benchmark.manual_mask(r, a)
        reviewed = a["review_status"] == "reviewed"
        if reviewed:
            if (not a["reviewer"].strip() or measurable is None
                    or not a["independent_drawing"] or a["prior_exposure"] == "not_answered"):
                raise ValueError("Reviewed ROI needs identity, measurability, independent drawing and exposure declaration")
            if measurable and not (mask != 255).any():
                raise ValueError("Measurable ROI has no independent labelled pixels")
        adapted.append({**a, "id": r["id"]})
    return dict(schema_version=1, manifest_id=packet["evaluation_manifest_id"], rois=adapted)


def verify_packet(packet):
    for path, expected in packet["input_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Review input changed: " + path)
    for r in packet["rois"]:
        for key in ("raw_png", "predictions"):
            if sha(OUT / r[key]) != r[key + "_sha256"]:
                raise ValueError("Packet image/prediction changed: " + r[key])


def interior_bright_summary(mask):
    """E38S object comparison: >=50 px², eight-connected, no ROI-clipped objects."""
    labels = label(np.asarray(mask, bool), connectivity=2)
    areas = np.bincount(labels.ravel())
    keep = areas >= 50
    keep[0] = False
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    excluded = int(keep[edge].sum())
    keep[edge] = False
    return component_summary(keep[labels])["d50"], int(keep.sum()), excluded


def correct_roi_object_scope(dest, packet, annotations):
    """Keep E24 pixel/width results; repair its bright-object clipping contract locally."""
    table = pd.read_csv(dest / "expert_evaluation.csv")
    by_id = {a["id"]: a for a in annotations["rois"]}
    for r in packet["rois"]:
        a = by_id[r["id"]]
        if a["review_status"] != "reviewed" or not a["measurable"]:
            continue
        manual = benchmark.manual_mask(r, a)
        if (manual == 255).any():
            continue  # Partial polygons cannot define object extent.
        truth_d50, truth_n, truth_excluded = interior_bright_summary(manual == 2)
        with np.load(OUT / r["predictions"]) as predictions:
            for method in ("baseline", "hysteresis"):
                pred_d50, pred_n, pred_excluded = interior_bright_summary(predictions[method] == 2)
                use = (table.roi == r["id"]) & (table.method == method)
                table.loc[use, "bright_d50_error_px"] = pred_d50 - truth_d50
                table.loc[use, "manual_bright_objects_retained"] = truth_n
                table.loc[use, "manual_bright_objects_edge_excluded"] = truth_excluded
                table.loc[use, "predicted_bright_objects_retained"] = pred_n
                table.loc[use, "predicted_bright_objects_edge_excluded"] = pred_excluded
    table.to_csv(dest / "expert_evaluation.csv", index=False)


def import_review(path, dest, release_held_out=False):
    exported = json.loads(Path(path).read_text())
    packets = json.loads((OUT / "coordinator_manifest.json").read_text())["packets"]
    matches = [p for p in packets if p["packet_id"] == exported.get("packet_id")]
    if len(matches) != 1:
        raise ValueError("Export belongs to another prepared packet")
    packet = matches[0]
    if packet["kind"] == "held_out_site" and not release_held_out:
        raise ValueError("Held-out results remain separate: fix development choices, then explicitly release")
    verify_packet(packet)
    adapted = validate_export(packet, exported)
    dest = Path(dest).resolve()
    if dest.exists():
        raise FileExistsError("Use a fresh review output directory; do not overwrite evidence")
    dest.mkdir(parents=True)
    annotated = dest / "annotations.json"
    write_json(annotated, adapted)
    # Reuse the evaluator with private, saved prediction arrays. It writes ONLY
    # into this fresh output folder; E24's original evaluations remain intact.
    eval_rois = [{**r, "predictions": str((OUT / r["predictions"]).resolve())}
                 for r in packet["rois"]]
    write_json(dest / "roi_manifest.json", dict(manifest_id=packet["evaluation_manifest_id"], rois=eval_rois))
    with patch.object(benchmark, "OUT", dest):
        benchmark.evaluate_annotations(annotated)
    correct_roi_object_scope(dest, packet, adapted)
    summary = json.loads((dest / "expert_evaluation_summary.json").read_text())
    reviewed = [a for a in adapted["rois"] if a["review_status"] == "reviewed"]
    write_json(dest / "review_receipt.json", dict(
        experiment="E38S", packet_id=packet["packet_id"], kind=packet["kind"],
        export_sha256=sha(path), created_utc=datetime.now(timezone.utc).isoformat(),
        reviewed_rois=len(reviewed), unmeasurable_rois=sum(not a["measurable"] for a in reviewed),
        prior_exposure_rois=sum(a["prior_exposure"] == "prior_exposure" for a in reviewed),
        human_declarations_not_independently_certified=True,
        bright_object_scope="E38S: area-weighted D50 of >=50 px2 eight-connected, non-ROI-clipped objects on whole-crop labels only",
        historical_E24_results_unchanged=True,
        unseen_batch_validation=False, summary=summary,
        outputs_sha256={p.name: sha(p) for p in dest.iterdir() if p.is_file()}))
    return summary


def build():
    if OUT.exists():
        raise FileExistsError("Prepared packet already exists; preserve it, do not overwrite reviews")
    OUT.mkdir()
    (OUT / "raw").mkdir()
    (OUT / "predictions").mkdir()
    protected = json.loads((ROOT / "analysis/quality_policy_audit/output/receipt.json").read_text())["protected_artifacts_sha256"]
    protected.update({str(p.relative_to(ROOT)): sha(p) for p in BENCH.rglob("*") if p.is_file()})
    old_manifest = json.loads((BENCH / "roi_manifest.json").read_text())
    source_paths = [Path(__file__), HERE / "review.html.in", HERE / "protocol.md", *METHOD_SOURCES,
                    BENCH / "roi_manifest.json", ROOT / "analysis/quality_policy_audit/image_evidence.json",
                    ROOT / "analysis/submission_v2/first_run/scored/features.csv"]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    headers = [inspect_header(p) for p in sorted((ROOT / "Dataset").rglob("*.tif"))]
    headers += [inspect_header(p) for p in sorted((ROOT / "Hackathon-Polaron-test").glob("*.tif"))]
    write_json(OUT / "tiff_headers.json", headers)
    pd.DataFrame([dict(file=h["file"], pages=h["pages"],
                       description_keys=";".join(h["description_keys"]),
                       possible_metadata_keys=";".join(h["possible_metadata_keys"]),
                       software=h["tag_values"].get("Software", ""), sha256=h["file_sha256"])
                  for h in headers]).to_csv(OUT / "metadata_inventory.csv", index=False)
    inputs.update({h["file"]: h["file_sha256"] for h in headers})
    mapping = []
    for parent, scope in [(ROOT / "Dataset", "known"), (ROOT / "Hackathon-Polaron-test", "feedback_development")]:
        for p in sorted(parent.rglob("*_BSE.tif")):
            site = p.name.split("_")[1]
            mapping.append(dict(scope=scope, folder=p.parent.name, site=site,
                                specimen_id="", preparation_session_id="", imaging_session_id="",
                                evidence_source="", status="unknown"))
    pd.DataFrame(mapping).to_csv(OUT / "group_mapping_template.csv", index=False)
    packets = []
    for kind, prefix in [("development", "D"), ("held_out_site", "H")]:
        rois = []
        # Content order hides the hand-picked original list order, with no batch balancing.
        selected = [r for r in old_manifest["rois"] if r["split"] == kind]
        selected.sort(key=lambda r: sha(BENCH / r["paths"]["raw"]))
        for i, source in enumerate(selected, 1):
            alias = f"{prefix}{i:02d}"
            raw, pred = f"raw/{alias}.png", f"predictions/{alias}.npz"
            (OUT / raw).write_bytes((BENCH / source["paths"]["raw"]).read_bytes())
            (OUT / pred).write_bytes((BENCH / source["predictions"]).read_bytes())
            rois.append(dict(id=source["id"], alias=alias, site=source["site"], split=kind,
                             W=source["W"], H=source["H"], box_raw_yxyx=source["box_raw_yxyx"],
                             raw_png=raw, predictions=pred,
                             raw_png_sha256=sha(OUT / raw), predictions_sha256=sha(OUT / pred)))
        packets.append(dict(kind=kind, evaluation_manifest_id=old_manifest["manifest_id"], rois=rois))
    evidence = {r["site"]: r for r in json.loads((ROOT / "analysis/quality_policy_audit/image_evidence.json").read_text())}
    table = pd.read_csv(ROOT / "analysis/submission_v2/first_run/scored/features.csv")
    raw_hashes = {h["file"].split("/")[-1].split("_")[1]: h["file_sha256"]
                  for h in headers if h["file"].startswith("Hackathon-Polaron-test/") and h["file"].endswith("_BSE.tif")}
    table = table.assign(_raw_order=table.site.map(raw_hashes)).sort_values("_raw_order")
    rois = []
    for i, row in enumerate(table.itertuples(), 1):
        e, alias = evidence[row.site], f"F{i:02d}"
        raw = features.load_image(str(ROOT / "Hackathon-Polaron-test"), row.site)
        top, bottom = features.bright_bands(raw)
        bse = features.trimmed(raw)
        y, x, side = e["raw_crop_y"], e["raw_crop_x"], e["side"]
        sl = np.s_[y-top:y-top+side, x:x+side]
        if y < top or y + side > raw.shape[0] - bottom:
            raise ValueError("Saved crop leaves the trimmed measurement frame")
        pore, bright, sm = features.segment(bse, row.th_lo, row.th_hi)
        candidate = bright_hysteresis(sm, row.th_hi)
        raw_name, pred = f"raw/{alias}.png", f"predictions/{alias}.npz"
        Image.fromarray(raw[y:y+side, x:x+side]).save(OUT / raw_name)
        np.savez_compressed(OUT / pred, baseline=semantic_mask(pore[sl], bright[sl]),
                            hysteresis=semantic_mask(pore[sl], candidate[sl]))
        rois.append(dict(id="feedback_"+row.site, alias=alias, site=row.site,
                         split="development", W=side, H=side, box_raw_yxyx=[y, x, y+side, x+side],
                         raw_png=raw_name, predictions=pred,
                         raw_png_sha256=sha(OUT/raw_name), predictions_sha256=sha(OUT/pred)))
        print("Prepared first-drop development crop", alias, flush=True)
    packets.append(dict(kind="feedback_development", rois=rois))
    for p in packets:
        identity = dict(kind=p["kind"], rois=p["rois"], input_sha256=inputs)
        p["packet_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        p.setdefault("evaluation_manifest_id", p["packet_id"])
        p["input_sha256"] = inputs
        public = public_packet(p)
        page = (HERE / "review.html.in").read_text().replace("__DATA__", json.dumps(public).replace("</", "<\\/"))
        (OUT / (p["kind"] + ".html")).write_text(page)
        write_json(OUT / (p["kind"] + "_blank_export.json"), dict(schema_version=1, packet_id=p["packet_id"], rois=[
            dict(id=r["alias"], review_status="unreviewed", measurable=None, reviewer="", notes="",
                 independent_drawing=False, prior_exposure="not_answered", background="unlabelled", polygons=[])
            for r in p["rois"]]))
    write_json(OUT / "coordinator_manifest.json", dict(experiment="E38S", decision="D57S", packets=packets))
    summary = dict(tiff_files=len(headers), sites=len(mapping), known_sites=sum(r["scope"] == "known" for r in mapping),
                   first_drop_development_sites=sum(r["scope"] != "known" for r in mapping),
                   files_with_possible_grouping_or_settings_keys=sum(bool(h["possible_metadata_keys"]) for h in headers),
                   explicit_specimen_ids_available=0, explicit_preparation_ids_available=0, explicit_imaging_ids_available=0,
                   mapping_available=False, source="Santosh: no mapping available yet; full TIFF header audit",
                   grouped_validation_available=False, independent_annotation_exports_received=0,
                   reviewed_phase_masks=0, packet_sizes={p["kind"]: len(p["rois"]) for p in packets})
    # A non-empty lead requires manual inspection before reporting all IDs missing.
    if summary["files_with_possible_grouping_or_settings_keys"]:
        raise ValueError("Metadata leads found: inspect before declaring grouping unavailable")
    write_json(OUT / "status.json", summary)
    for path, expected in protected.items():
        if sha(ROOT/path) != expected:
            raise ValueError("Protected original artifact changed: " + path)
    for path, expected in inputs.items():
        if sha(ROOT/path) != expected:
            raise ValueError("Input changed during preparation: " + path)
    write_json(OUT / "receipt.json", dict(experiment="E38S", decision="D57S",
        created_utc=datetime.now(timezone.utc).isoformat(), input_sha256=inputs,
        protected_original_artifacts_sha256=protected, originals_unchanged=True,
        annotations_created=False, classifier_fit=False, live_pipeline_changed=False,
        outputs_sha256={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file()}))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--release-held-out", action="store_true")
    args = parser.parse_args()
    if args.annotations:
        if not args.out:
            parser.error("Import needs --out FRESH_DIR")
        print(json.dumps(import_review(args.annotations, args.out, args.release_held_out), indent=2))
    elif args.out or args.release_held_out:
        parser.error("--out / --release-held-out require --annotations")
    else:
        build()


if __name__ == "__main__":
    main()
