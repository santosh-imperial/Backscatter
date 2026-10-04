"""Supplementary real-image control under its fixed pre-specified protocol."""
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.source_crop_audit.run import (  # noqa: E402
    BORDER, CANDIDATE_NCC, DETECTORS, MIN_HEIGHT, MIN_WIDTH, REFINE_RADIUS,
    SIGMA, STRIDE, load, matched_panel, sha, write_json,
)
from analysis.source_crop_audit.match import (  # noqa: E402
    Translation, best_translation, compare_patch, confirmed,
    refine_translation, small_image, verification_patches,
)


def main():
    out = HERE/"output/real_image_controls"
    if out.exists():
        raise FileExistsError("preserve old controls; choose a fresh output before rerun")
    out.mkdir()
    source_files = [HERE/"control_real.py", HERE/"control_real_protocol.md", HERE/"run.py", HERE/"match.py"]
    snapshot = out/"source_snapshot"
    snapshot.mkdir()
    for path in source_files:
        shutil.copy2(path, snapshot/path.name)
    frame = pd.read_csv(HERE/"output/file_manifest.csv")
    selected = frame[frame.detector == "BSE"].sort_values("sha256").iloc[0]
    source = frame[frame.site == selected.site].set_index("detector")
    hashes = {d: source.loc[d, "sha256"] for d in DETECTORS}
    if not all(sha(ROOT/source.loc[d, "path"]) == hashes[d] for d in DETECTORS):
        raise RuntimeError("raw image hash differs from valid audit receipt")
    images = {}
    for detector in DETECTORS:
        raw = load(ROOT/source.loc[detector, "path"])
        a = raw[160:960, 500:2700]
        b = np.rint(raw[253:1053, 733:2933].astype(float)*0.65+31).astype(np.uint8)
        if a.shape != (800, 2200) or b.shape != a.shape:
            raise RuntimeError("fixed control geometry unavailable; do not choose another image")
        images[detector] = a, b
    a, b = images["BSE"]
    coarse = best_translation(small_image(a, stride=STRIDE, sigma=SIGMA, border=BORDER),
                              small_image(b, stride=STRIDE, sigma=SIGMA, border=BORDER),
                              min_height=MIN_HEIGHT//STRIDE, min_width=MIN_WIDTH//STRIDE)
    if coarse is None or coarse.ncc < CANDIDATE_NCC:
        raise RuntimeError("real-image positive control failed the actual candidate gate")
    coarse_raw = Translation(coarse.dy*STRIDE, coarse.dx*STRIDE, coarse.ncc, coarse.height*STRIDE, coarse.width*STRIDE)
    fine = refine_translation(a, b, coarse_raw, radius=REFINE_RADIUS)
    patches = verification_patches(a.shape, b.shape, fine)
    checks = [dict(detector=d, **c, **compare_patch(*images[d], c)) for d in DETECTORS for c in patches]
    rng = np.random.default_rng(419)
    shuffled = rng.permutation(a.ravel()).reshape(a.shape)
    negative = best_translation(small_image(a, stride=STRIDE, sigma=SIGMA, border=BORDER),
                                small_image(shuffled, stride=STRIDE, sigma=SIGMA, border=BORDER),
                                min_height=MIN_HEIGHT//STRIDE, min_width=MIN_WIDTH//STRIDE)
    passed = (confirmed(checks) and (fine.dy, fine.dx) == (93, 233)
              and negative is not None and negative.ncc < CANDIDATE_NCC)
    matched_panel(out/"synthetic_affine_quantisation_positive_BSE.png", a, b, patches[1],
                  "Positive control only: two constructed crops of one supplied BSE image")
    pd.DataFrame(checks).to_csv(out/"multichannel_patch_checks.csv", index=False)
    write_json(out/"receipt.json", dict(
        passed=passed, selected_site=selected.site, selection="minimum_BSE_file_SHA256_no_label_or_visual_screening",
        source_paths={d: str(source.loc[d, "path"]) for d in DETECTORS}, source_sha256=hashes,
        source_code_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_files},
        expected_translation=[93, 233], recovered_translation=[fine.dy, fine.dx],
        source_crop_a=dict(y=160, x=500, height=800, width=2200),
        source_crop_b=dict(y=253, x=733, height=800, width=2200),
        synthetic_transform_b=dict(gain=0.65, offset=31, quantisation="nearest_integer_uint8_no_clipping"),
        coarse_ncc=coarse.ncc, candidate_gate_passed=coarse.ncc >= CANDIDATE_NCC,
        fine_central_ncc=fine.ncc, all_nine_checks_passed=confirmed(checks),
        shuffled_negative_max_coarse_ncc=negative.ncc, negative_seed=419,
        raw_sources_unchanged=all(sha(ROOT/source.loc[d, "path"]) == hashes[d] for d in DETECTORS),
        organiser_source_mapping_validated=False,
        limitation="Constructed control confirms one relationship; no recall estimate on unknown source crops."))
    if not passed:
        raise RuntimeError("real-image control failed; audit needs further diagnosis")
    print(f"Real-image gate/full verification passed: coarse={coarse.ncc:.6f}, translation={(fine.dy, fine.dx)}")


if __name__ == "__main__":
    main()
