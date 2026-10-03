"""Score a complete three-channel organiser folder with the saved E31 models.

This wrapper does not fit or select anything on incoming data. Existing model APIs
perform all measurement and scoring. Existing output folders are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FREEZE = HERE / "freeze"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_frozen_sources():
    snapshot = json.loads((FREEZE / "snapshot.json").read_text())
    for path, expected in snapshot["source_sha256"].items():
        if sha(FREEZE / "sources" / path) != expected:
            raise ValueError(f"Frozen source snapshot changed: {path}")
    if sha(HERE / "protocol.md") != snapshot["source_sha256"]["analysis/organiser_drop_01/protocol.md"]:
        raise ValueError("Frozen protocol changed")
    return snapshot


# Keep this model version independent of concurrent edits to the live package.
# Do not replace already imported live modules in a long-lived Python session.
validate_frozen_sources()
frozen_package = (FREEZE / "sources" / "polaron_qc").resolve()
for name, module in tuple(sys.modules.items()):
    if name == "polaron_qc" or name.startswith("polaron_qc."):
        if not Path(module.__file__).resolve().is_relative_to(frozen_package):
            raise RuntimeError("Run the frozen scorer in a fresh Python process; live polaron_qc is already imported")
sys.path.insert(0, str(FREEZE / "sources"))
from polaron_qc import categorise as CAT


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def inspect_input(folder):
    files = sorted(folder.glob("*.tif"))
    if not files:
        raise ValueError(f"No TIFFs found in {folder}")
    rows = []
    for path in files:
        match = re.fullmatch(r"img_(\w+)_(BSE|ETD|SE|Inlens)\.tif", path.name)
        if not match:
            raise ValueError(f"Unsupported filename: {path.name}")
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            row = dict(file=path.name, site=match[1], detector="ETD" if match[2] == "SE" else match[2],
                       shape=list(series.shape), dtype=str(series.dtype), sha256=sha(path))
        rows.append(row)
    for site in sorted({r["site"] for r in rows}):
        group = [r for r in rows if r["site"] == site]
        if len(group) != 3 or {r["detector"] for r in group} != {"BSE", "ETD", "Inlens"}:
            raise ValueError(f"{site}: expected exactly one BSE, ETD/SE and Inlens image")
        if len({tuple(r["shape"]) for r in group}) != 1 or any(r["dtype"] != "uint8" for r in group):
            raise ValueError(f"{site}: expected matching uint8 detector geometry")
        shape = group[0]["shape"]
        if not (len(shape) == 2 or (len(shape) == 3 and shape[-1] == 3)):
            raise ValueError(f"{site}: expected a grayscale or RGB image, got shape {shape}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Hackathon-Polaron-test")
    parser.add_argument("--out", type=Path, default=HERE / "first_run")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "_scratch/organiser_drop_01_features")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Preserve prior outputs; choose a new empty --out directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    snapshot = validate_frozen_sources()
    for dependency, expected in snapshot["dependencies"].items():
        actual = importlib.metadata.version(dependency)
        if actual != expected:
            raise ValueError(f"Frozen dependency changed: {dependency} {actual}, expected {expected}")
    if sha(FREEZE / "models.joblib") != snapshot["models_sha256"]:
        raise ValueError("Frozen model hash mismatch")
    files = inspect_input(args.input)
    original = {Path(r["path"]).name: r["sha256"] for r in snapshot["raw_manifest"] if r["role"] == "test"}
    if args.input.resolve() == (ROOT / "Hackathon-Polaron-test").resolve():
        if {r["file"]: r["sha256"] for r in files} != original:
            raise ValueError("First-drop input differs from its pre-prediction snapshot")
    saved = joblib.load(FREEZE / "models.joblib")  # trusted local artifact created from known sites
    known, models = saved["known"], saved["models"]
    if {r["site"] for r in files} & set(known.site):
        raise ValueError("Incoming site IDs overlap training IDs; do not score as a holdout")
    print(f"Scoring {len(files)//3} sites with frozen known-only models; extracting incoming features...", flush=True)
    sites = CAT.load_known([str(args.input.resolve())], cache_dir=str(args.cache_dir.resolve()), verbose=False)
    table = CAT.categorise_sites(sites, models, saved["ood_models"], role="scored", top=10)
    sites.to_csv(args.out / "features.csv", index=False)
    table.to_csv(args.out / "site_table.csv", index=False)
    primary = models["combined"]
    X = sites[primary.features].to_numpy(float)
    deletion_scores = np.stack([CAT._proba_full(pipe, X, len(primary.classes))
                               for pipe in saved["primary_deletion_models"]])
    records, details = [], {}
    for i, row in table.iterrows():
        probabilities = np.array([row[f"mp_{c}__combined"] for c in primary.classes])
        winner = int(np.argmax(probabilities)); order = np.argsort(-probabilities, kind="stable")
        assignment = primary.classes[winner]
        counts = {c: int((deletion_scores[:, i].argmax(axis=1) == j).sum()) for j, c in enumerate(primary.classes)}
        flags = [f for f in ("bright_low_contrast", "grey_pore", "raised_black_level", "contrast_stretched_bse")
                 if pd.notna(row.get(f)) and bool(row.get(f))]
        contributions = CAT.categoriser_contributions(primary.pipeline, X[i], primary.features, winner,
                                                     primary.ref_median, primary.ref_mad, top=len(primary.features))
        for item in contributions:
            item["family"] = ("morphology" if item["feature"] in CAT.MORPH_FEATURES
                              else "acquisition-sensitive texture" if item["feature"] in CAT.TEXTURE_FEATURES
                              else "acquisition")
            item["measurement_missing_imputed"] = bool(np.isnan(X[i, primary.features.index(item["feature"])]))
        note = ("Uncalibrated model scores; deletion counts measure training-set sensitivity, not independent votes "
                "or probabilities of correctness. True batch label pending.")
        record = dict(sample_id=row.site, predicted_batch=assignment,
                      **{f"model_score_{c}": float(probabilities[j]) for j, c in enumerate(primary.classes)},
                      top_model_score=float(probabilities[winner]),
                      runner_up=primary.classes[int(order[1])], margin=float(probabilities[order[0]] - probabilities[order[1]]),
                      deletion_same_assignment=counts[assignment], deletion_fit_count=len(deletion_scores),
                      **{f"deletion_votes_{c}": n for c, n in counts.items()},
                      assigned_score_deletion_min=float(deletion_scores[:, i, winner].min()),
                      assigned_score_deletion_max=float(deletion_scores[:, i, winner].max()),
                      morphology_only_bet=row.argmax__morph, acquisition_only_bet=row.argmax__acq,
                      baseline_morphology_percentile=float(row.ood_pct_knn__morph),
                      exceeds_observed_baseline_max=bool(row.ood_exceeds_knn__morph),
                      acquisition_flags="|".join(flags) or "none of listed flags",
                      explanation=row.cat_top_features__combined, uncertainty_note=note)
        records.append(record)
        details[str(row.site)] = dict(contributions=contributions, deletion_counts=counts,
                                    missing_primary_features=[f for f, v in zip(primary.features, X[i]) if np.isnan(v)],
                                    baseline_nearest_sites=row.ood_nearest_ref__morph,
                                    baseline_drivers=row.ood_top_features__morph)
    submission = pd.DataFrame(records)
    submission.to_csv(args.out / "submission_sites.csv", index=False)
    image_rows = pd.DataFrame(files).rename(columns={"site": "sample_id", "file": "filename"})
    image_rows.merge(submission, on="sample_id", validate="many_to_one").to_csv(args.out / "submission_images.csv", index=False)
    write_json(args.out / "explanations.json", details)
    write_json(args.out / "input_manifest.json", files)
    for r in files:
        if sha(args.input / r["file"]) != r["sha256"]:
            raise ValueError("Input mutated during scoring; first outputs are not valid")
    receipt = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_s=time.monotonic()-t0, input_directory=str(args.input.resolve()),
                   source_snapshot_sha256=sha(FREEZE / "snapshot.json"), model_sha256=sha(FREEZE / "models.joblib"),
                   scorer_sha256=sha(Path(__file__)), n_sites=len(sites), n_images=len(files),
                   training_only=True, first_drop_labels_pending=True, no_test_fitting_or_model_selection=True,
                   source_execution="preserved source snapshot under freeze/sources; live package edits ignored",
                   bet_rule="existing E31 combined argmax; no post-drop overrides",
                   output_sha256={p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()})
    write_json(args.out / "prediction_receipt.json", receipt)
    print(submission[["sample_id", "predicted_batch", "top_model_score", "margin", "deletion_same_assignment",
                      "deletion_fit_count", "baseline_morphology_percentile", "acquisition_flags"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
