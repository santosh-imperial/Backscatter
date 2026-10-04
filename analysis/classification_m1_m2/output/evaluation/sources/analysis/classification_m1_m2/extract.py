"""Extract the fixed E39S appearance panel and acquisition challenges.

Run from the repository root:
  /opt/anaconda3/bin/python3 analysis/classification_m1_m2/extract.py

Original per-site features are saved before sensitivity processing so the
matched classification audit can start independently. No labels drive image
sampling, filters, normalization or aggregation, and no classifier is fit here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.classification_m1_m2.appearance import (
    CHANNELS, FEATURE_NAMES, definition, extract_site,
)
from polaron_qc import features

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
ASSETS = HERE / "assets"
KNOWN = ROOT / "analysis/submission_v2/freeze/known_sites.csv"
DROP = ROOT / "analysis/submission_v2/first_run/scored/features.csv"
VARIANTS = ("original", "gain_offset", "gamma_0p7", "gamma_1p4", "quantise_step8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_rows() -> pd.DataFrame:
    known = pd.read_csv(KNOWN)[["batch", "site"]].assign(cohort="known")
    drop = pd.read_csv(DROP)[["batch", "site"]].assign(cohort="first_drop_development")
    frame = pd.concat([known, drop], ignore_index=True)
    if frame.duplicated(["cohort", "batch", "site"]).any():
        raise ValueError("source site identifiers must be unique")
    return frame.sort_values(["cohort", "batch", "site"]).reset_index(drop=True)


def load_row(row: dict):
    folder = ROOT / "Dataset" / row["batch"] if row["cohort"] == "known" else ROOT / "Hackathon-Polaron-test"
    paths = {channel: Path(features._image_path(folder, row["site"], channel))
             for channel in CHANNELS}
    images = {channel: features.load_image(folder, row["site"], channel)
              for channel in CHANNELS}
    return images, paths


def transform(images: dict[str, np.ndarray], variant: str) -> dict[str, np.ndarray]:
    """Fixed all-channel nuisance challenges; pixels/coordinates remain paired."""
    if variant == "original":
        return images
    result = {}
    for channel, image in images.items():
        a = np.asarray(image, dtype=float)
        if variant == "gain_offset":
            value = 1.35 * a + 17.0  # deliberately float, without clipping
        elif variant == "gamma_0p7":
            value = 255.0 * np.power(a / 255.0, 0.7)
        elif variant == "gamma_1p4":
            value = 255.0 * np.power(a / 255.0, 1.4)
        elif variant == "quantise_step8":
            value = 8.0 * np.floor(a / 8.0)  # 32 bins, no rescaling or clipping
        else:
            raise ValueError(f"unknown challenge {variant}")
        result[channel] = value
    return result


def write_response_maps(row: dict, maps: dict, diagnostics: dict) -> str:
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for index, channel in enumerate(CHANNELS):
        raw = maps[channel]["raw"]
        lo, hi = np.percentile(raw, [1, 99])
        axes[index, 0].imshow(raw, cmap="gray", vmin=lo, vmax=hi)
        fine = maps[channel]["fine_band"]
        bound = max(float(np.percentile(np.abs(fine), 99)), 1e-9)
        axes[index, 1].imshow(fine, cmap="RdBu_r", vmin=-bound, vmax=bound)
        magnitude = maps[channel]["gradient_magnitude"]
        axes[index, 2].imshow(magnitude, cmap="magma", vmin=0,
                             vmax=max(float(np.percentile(magnitude, 99)), 1e-9))
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[index, 0].set_ylabel(channel)
    for axis, title in zip(axes[0], ("Raw (1–99% display only)", "G2−G8 response", "sigma8 gradient magnitude")):
        axis.set_title(title, fontsize=10)
    raw_origin = diagnostics["tile_origins_raw_yx"][4]
    fig.suptitle(f"{row['batch']} / {row['site']} — central 512px tile\n"
                 f"raw y,x={tuple(raw_origin)}; colour scales are display-only; gradients are image appearance")
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"appearance_responses_{row['site']}.jpg"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def extract_original(row: dict, examples: set[str]):
    images, paths = load_row(row)
    measured, diag, maps = extract_site(images, return_maps=row["site"] in examples)
    raw_receipt = {channel: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                   for channel, path in paths.items()}
    return row | measured, row | diag, raw_receipt, maps


def extract_challenges(row: dict):
    images, _ = load_row(row)
    trim = features.bright_bands(images["BSE"])
    records = []
    for variant in VARIANTS[1:]:
        values, diagnostics, _ = extract_site(transform(images, variant), trim=trim)
        records.append(row | {"variant": variant,
                              "fixed_trim_top": int(trim[0]), "fixed_trim_bottom": int(trim[1]),
                              "available_features": diagnostics["available_features"]} | values)
    return records


def summarize_sensitivity(frame: pd.DataFrame) -> dict:
    original = frame[frame.variant == "original"].set_index("site")
    summary = {}
    for variant in VARIANTS[1:]:
        changed = frame[frame.variant == variant].set_index("site").loc[original.index]
        delta = changed[list(FEATURE_NAMES)].to_numpy() - original[list(FEATURE_NAMES)].to_numpy()
        scale = original[list(FEATURE_NAMES)].std(axis=0, ddof=1).to_numpy()
        standardized = np.divide(delta, scale, out=np.full_like(delta, np.nan), where=scale > 0)
        summary[variant] = {
            "n_supplied_crops": len(changed),
            "maximum_absolute_feature_change": float(np.nanmax(np.abs(delta))),
            "median_absolute_standardized_feature_change": float(np.nanmedian(np.abs(standardized))),
            "p90_absolute_standardized_feature_change": float(np.nanpercentile(np.abs(standardized), 90)),
            "note": "scale uses descriptive known-crop feature standard deviation; not independent-site uncertainty",
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--originals-only", action="store_true")
    parser.add_argument("--sensitivity-only", action="store_true")
    args = parser.parse_args()
    if args.originals_only and args.sensitivity_only:
        parser.error("choose at most one stage-only option")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = source_rows()
    records = frame.to_dict("records")
    known = frame[frame.cohort == "known"]
    examples = set(known.groupby("batch", sort=True).first().site) | set(frame[frame.cohort != "known"].site)
    started = time.monotonic()
    receipt_path = OUTPUT / "appearance_extraction_receipt.json"
    if not args.sensitivity_only:
        receipt = {
            "definition": definition(),
            "source_files": {str(p.relative_to(ROOT)): sha256(p) for p in (
                Path(__file__), HERE / "appearance.py", ROOT / "polaron_qc/features.py", KNOWN, DROP)},
            "n_supplied_crops": len(records), "n_known_crops": int(len(known)),
            "first_drop_role": "development; no truth read by feature extraction",
            "source_images": {}, "response_maps": [],
        }
        atomic_json(OUTPUT / "appearance_definition.json", receipt["definition"])
        feature_rows, diagnostic_rows = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, (values, diag, sources, maps) in enumerate(
                    pool.map(lambda row: extract_original(row, examples), records), start=1):
                feature_rows.append(values)
                diagnostic_rows.append(diag)
                receipt["source_images"][values["site"]] = sources
                if maps:
                    path = write_response_maps(values, maps, diag)
                    receipt["response_maps"].append({"path": path, "sha256": sha256(ROOT / path)})
                print(f"original {index}/{len(records)}: {values['site']} ({diag['available_features']}/24 available)", flush=True)
        features_frame = pd.DataFrame(feature_rows)
        features_frame.to_csv(OUTPUT / "appearance_features.csv", index=False)
        # Coordinate arrays remain JSON strings inside CSV diagnostics only.
        diagnostics_frame = pd.DataFrame(diagnostic_rows)
        for name in ("tile_origins_trimmed_yx", "tile_origins_raw_yx"):
            diagnostics_frame[name] = diagnostics_frame[name].map(json.dumps)
        diagnostics_frame.to_csv(OUTPUT / "appearance_diagnostics.csv", index=False)
        receipt["originals_elapsed_seconds"] = time.monotonic() - started
        receipt["measurement_coverage_range"] = [float(diagnostics_frame.measurement_coverage.min()),
                                                  float(diagnostics_frame.measurement_coverage.max())]
        receipt["outputs"] = {name: sha256(OUTPUT / name) for name in (
            "appearance_features.csv", "appearance_diagnostics.csv", "appearance_definition.json")}
        atomic_json(receipt_path, receipt)
        print("ORIGINAL FEATURE PANEL READY", flush=True)
    else:
        receipt = json.loads(receipt_path.read_text())
        for path, digest in receipt["source_files"].items():
            if sha256(ROOT / path) != digest:
                raise RuntimeError(f"source changed since original extraction: {path}")
        features_frame = pd.read_csv(OUTPUT / "appearance_features.csv")
    if args.originals_only:
        return
    originals = features_frame[features_frame.cohort == "known"].assign(variant="original")
    challenge_rows = originals.to_dict("records")
    known_records = known.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, values in enumerate(pool.map(extract_challenges, known_records), start=1):
            challenge_rows.extend(values)
            print(f"challenge {index}/{len(known_records)}: {values[0]['site']} (all four fixed variants)", flush=True)
    challenges = pd.DataFrame(challenge_rows)
    challenges.to_csv(OUTPUT / "appearance_sensitivity.csv", index=False)
    receipt["sensitivity_summary"] = summarize_sensitivity(challenges)
    receipt["sensitivity_rows"] = len(challenges)
    receipt["challenge_policy"] = {
        "applied_to": "all31 known crops, all3 channels, same fixed windows/shared original trim",
        "gain_offset": "float1.35*x+17; no clipping/rounding",
        "gamma_0p7": "255*(x/255)^0.7; float",
        "gamma_1p4": "255*(x/255)^1.4; float",
        "quantise_step8": "8*floor(x/8); 32 output bins, no clipping/rescaling",
        "role": "paired appearance-input nuisance checks; no geometry changes, synthetic crops or independent observations",
    }
    receipt["outputs"]["appearance_sensitivity.csv"] = sha256(OUTPUT / "appearance_sensitivity.csv")
    receipt["total_elapsed_seconds"] = time.monotonic() - started
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt["sensitivity_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
