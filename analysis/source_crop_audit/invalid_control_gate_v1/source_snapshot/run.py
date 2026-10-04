"""Run the fixed overlap protocol into a fresh directory; no models/labels read."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
import tifffile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.source_crop_audit.match import (  # noqa: E402
    Translation, best_translation, compare_patch, components, confirmed,
    refine_translation, small_image, verification_patches,
)

STRIDE = 8
BORDER = 4
SIGMA = 2.0
MIN_HEIGHT = 512
MIN_WIDTH = 1024
CANDIDATE_NCC = 0.85
REFINE_RADIUS = 16
DETECTORS = ("BSE", "ETD", "Inlens")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def discover():
    files = sorted((ROOT/"Dataset").glob("*/*.tif")) + sorted((ROOT/"Hackathon-Polaron-test").glob("*.tif"))
    sites, rows = {}, []
    for path in files:
        match = re.fullmatch(r"img_(\w+)_(BSE|ETD|SE|Inlens)\.tif", path.name)
        if not match:
            raise ValueError(f"unrecognised TIFF name: {path}")
        site, detector = match.groups()
        detector = "ETD" if detector == "SE" else detector
        if detector in sites.setdefault(site, {}):
            raise ValueError(f"duplicate detector for site {site}")
        sites[site][detector] = path
        with tifffile.TiffFile(path) as tif:
            shape = list(tif.pages[0].shape)
        rows.append(dict(site=site, detector=detector, path=str(path.relative_to(ROOT)),
                         sha256=sha(path), bytes=path.stat().st_size, height=shape[0], width=shape[1]))
    if any(set(paths) != set(DETECTORS) for paths in sites.values()):
        raise ValueError("all sites must have exactly the three detector channels")
    return sites, rows


def load(path):
    a = tifffile.imread(path)
    return a[..., 0] if a.ndim == 3 else a


def control_receipt():
    rng = np.random.default_rng(418)
    source = ndimage.gaussian_filter(rng.normal(size=(1000, 2800)), 3)
    source = 110 + 32*source/source.std()
    a = source[:800, :2200]
    b = source[93:893, 233:2433] * 0.65 + 31
    coarse = best_translation(small_image(a), small_image(b),
                              min_height=MIN_HEIGHT//STRIDE, min_width=MIN_WIDTH//STRIDE)
    t = Translation(coarse.dy*STRIDE, coarse.dx*STRIDE, coarse.ncc, coarse.height*STRIDE, coarse.width*STRIDE)
    fine = refine_translation(a, b, t, radius=REFINE_RADIUS)
    patches = verification_patches(a.shape, b.shape, fine)
    checks = [dict(detector=d, **compare_patch(a, b, p)) for d in DETECTORS for p in patches]
    unrelated = ndimage.gaussian_filter(rng.normal(size=a.shape), 3)
    unrelated = 110 + 32*unrelated/unrelated.std()
    null = best_translation(small_image(a), small_image(unrelated),
                            min_height=MIN_HEIGHT//STRIDE, min_width=MIN_WIDTH//STRIDE)
    c = dict(a_y=100, a_x=100, b_y=100, b_x=100, height=256, width=256)
    null_check = compare_patch(a, unrelated, c)
    passed = ((fine.dy, fine.dx) == (93, 233) and confirmed(checks)
              and null.ncc < CANDIDATE_NCC and not null_check["passed"]
              and best_translation(np.ones((64, 128)), np.ones((64, 128)), min_height=64, min_width=128) is None)
    return dict(passed=passed, seed=418, expected_translation=[93, 233],
                recovered_translation=[fine.dy, fine.dx], synthetic_coarse_ncc=coarse.ncc,
                synthetic_fine_ncc=fine.ncc, synthetic_multichannel_verification=confirmed(checks),
                unrelated_max_coarse_ncc=null.ncc, unrelated_patch=null_check,
                limitation="Synthetic success is not a recall estimate for unknown organiser parent crops.")


def matched_panel(path, image_a, image_b, c, title):
    h, w = c["height"], c["width"]
    a = np.asarray(image_a)[c["a_y"]:c["a_y"]+h, c["a_x"]:c["a_x"]+w]
    b = np.asarray(image_b)[c["b_y"]:c["b_y"]+h, c["b_x"]:c["b_x"]+w]
    # Preserve raw grey levels. Display difference is a labelled diagnostic only.
    canvas = Image.new("RGB", (3*w+24, h+65), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black")
    for i, (arr, label) in enumerate(((a, "A: raw"), (b, "B: raw"), (np.abs(a.astype(float)-b.astype(float)), "abs raw difference"))):
        x = 8+i*(w+4)
        canvas.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), (x, 50))
        draw.text((x, 30), label, fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"output")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("write to a fresh output directory to preserve previous receipts")
    out.mkdir(parents=True)
    started = time.time()
    sources = [HERE/"protocol.md", HERE/"run.py", HERE/"match.py", ROOT/"tests/test_source_crop_audit.py"]
    snapshot = out/"source_snapshot"
    snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot/source.name)
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    protected = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/"analysis/submission_v2").rglob("*")) if p.is_file()}
    sites, inventory = discover()
    pd.DataFrame(inventory).to_csv(out/"file_manifest.csv", index=False)
    write_json(out/"pre_score_receipt.json", dict(
        purpose="Pixel-overlap diagnostic; not parent reconstruction or source-held-out validation",
        site_count=len(sites), tiff_count=len(inventory), pairs=len(sites)*(len(sites)-1)//2,
        source_hashes=source_hashes, protocol_sha256=sha(HERE/"protocol.md"),
        input_hashes={r["path"]: r["sha256"] for r in inventory},
        protected_submission_hashes=protected,
        classifier_labels_predictions_read=False, dimensions_used_as_grouping_features=False,
        settings=dict(stride=STRIDE, border=BORDER, blur_sigma_px=SIGMA,
                      min_overlap_height_px=MIN_HEIGHT, min_overlap_width_px=MIN_WIDTH,
                      candidate_ncc=CANDIDATE_NCC, refinement_radius_px=REFINE_RADIUS,
                      verification_patch_px=256, patch_std_min_gray=1.0,
                      verification_pearson_min=0.995, verification_spearman_min=0.995,
                      verification_affine_residual_target_sd_max=0.10)))
    controls = control_receipt()
    write_json(out/"controls.json", controls)
    if not controls["passed"]:
        raise RuntimeError("synthetic control failed; do not interpret data audit")
    smalls = {}
    for site, paths in sorted(sites.items()):
        smalls[site] = small_image(load(paths["BSE"]), stride=STRIDE, sigma=SIGMA, border=BORDER)
        print(f"Loaded BSE {site}", flush=True)
    pairs, candidates = [], []
    for i, (a, b) in enumerate(itertools.combinations(sorted(sites), 2)):
        best = best_translation(smalls[a], smalls[b], min_height=MIN_HEIGHT//STRIDE, min_width=MIN_WIDTH//STRIDE)
        row = dict(site_a=a, site_b=b, coarse_ncc=best.ncc if best else None,
                   a_minus_b_dy_coarse_px=best.dy*STRIDE if best else None,
                   a_minus_b_dx_coarse_px=best.dx*STRIDE if best else None,
                   coarse_overlap_height_px=best.height*STRIDE if best else None,
                   coarse_overlap_width_px=best.width*STRIDE if best else None,
                   candidate=bool(best and best.ncc >= CANDIDATE_NCC))
        pairs.append(row)
        if row["candidate"]:
            candidates.append((a, b, best))
        if (i+1) % 25 == 0:
            print(f"Scored {i+1} pairs; {len(candidates)} candidates", flush=True)
    pd.DataFrame(pairs).to_csv(out/"all_pair_scores.csv", index=False)
    checks, verified, edges = [], [], []
    evidence = out/"verified_panels"
    evidence.mkdir()
    for a, b, coarse in candidates:
        aa, bb = load(sites[a]["BSE"]), load(sites[b]["BSE"])
        coarse_raw = Translation(coarse.dy*STRIDE, coarse.dx*STRIDE, coarse.ncc, coarse.height*STRIDE, coarse.width*STRIDE)
        fine = refine_translation(aa, bb, coarse_raw, radius=REFINE_RADIUS)
        local, panel_paths = [], []
        coordinates = verification_patches(aa.shape, bb.shape, fine) if fine else []
        if fine and fine.height >= MIN_HEIGHT and fine.width >= MIN_WIDTH and len(coordinates) == 3:
            for detector in DETECTORS:
                if detector != "BSE":
                    aa, bb = load(sites[a][detector]), load(sites[b][detector])
                for c in coordinates:
                    result = dict(site_a=a, site_b=b, detector=detector, **c, **compare_patch(aa, bb, c))
                    local.append(result)
            passed = confirmed(local)
            if passed:
                for detector in DETECTORS:
                    aa, bb = load(sites[a][detector]), load(sites[b][detector])
                    c = coordinates[1]
                    p = evidence/f"{a}__{b}__{detector}.png"
                    matched_panel(p, aa, bb, c, f"{a} / {b} | {detector} | dy={fine.dy}, dx={fine.dx}")
                    panel_paths.append(str(p.relative_to(out)))
                edges.append((a, b))
        else:
            passed = False
        checks.extend(local)
        verified.append(dict(site_a=a, site_b=b, coarse_ncc=coarse.ncc,
                             fine_dy_px=fine.dy if fine else None, fine_dx_px=fine.dx if fine else None,
                             central_patch_ncc=fine.ncc if fine else None, passed=passed,
                             verification_patch_count=len(local), coordinates=coordinates, panel_paths=panel_paths))
        print(f"Verified {a}/{b}: {passed}", flush=True)
    pd.DataFrame(checks, columns=["site_a", "site_b", "detector", "patch", "a_y", "a_x", "b_y", "b_x", "height", "width",
                                "valid", "passed", "pearson", "spearman", "exact_pixel_fraction", "affine_gain_b_from_a",
                                "affine_offset_b_from_a", "affine_residual_target_sd", "std_a", "std_b", "reason"]).to_csv(out/"patch_verification.csv", index=False)
    write_json(out/"candidate_verification.json", verified)
    component_list = components(sites, edges)
    write_json(out/"confirmed_overlap_components.json", dict(
        components=component_list, confirmed_edges=[list(e) for e in edges],
        grouping_status="diagnostic_only_not_parent_or_specimen_IDs",
        isolates_are_unknown_parents=True, classifier_grouping_changed=False,
        same_parent_nonoverlapping_crops_not_identifiable=True))
    unchanged_inputs = all(sha(ROOT/r["path"]) == r["sha256"] for r in inventory)
    unchanged_submission = all(sha(ROOT/path) == old for path, old in protected.items())
    unchanged_source = all(sha(ROOT/path) == old for path, old in source_hashes.items())
    verification = dict(raw_tiffs_unchanged=unchanged_inputs,
                        frozen_submission_unchanged=unchanged_submission,
                        protected_submission_file_count=len(protected),
                        source_unchanged_after_protocol=unchanged_source,
                        synthetic_controls_passed=controls["passed"],
                        site_count=len(sites), tiff_count=len(inventory), pair_count=len(pairs),
                        candidate_pair_count=len(candidates), confirmed_edge_count=len(edges),
                        nontrivial_overlap_component_count=len(component_list),
                        maximum_coarse_ncc=max(r["coarse_ncc"] for r in pairs if r["coarse_ncc"] is not None),
                        wall_seconds=time.time()-started,
                        true_parent_heldout_validation_available=False)
    write_json(out/"verification.json", verification)
    if not all((unchanged_inputs, unchanged_submission, unchanged_source)):
        raise RuntimeError("source or protected input changed during run")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
