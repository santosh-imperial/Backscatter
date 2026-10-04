"""app.export_lot — save one incoming lot, or one categoriser sample set, as a UI bundle (docs/ui_plan.md, D56U).

Two subcommands::

    python -m app.export_lot lot --reference Dataset/Batch_3 --batch Dataset/Batch_1 --out ui_bundles/lot_Batch_1 \
        --distance-table analysis/submission_v2/freeze/evaluation/site_table.csv --distance-label "..." [--with-baseline]
    python -m app.export_lot samples --run analysis/submission_v2/first_run --input /path/to/test_folder --out ui_bundles/samples_drop01

``lot`` runs ``polaron_qc.report.build_result`` (the frozen verdict rules; thresholds hash checked into the bundle) and
writes ``lot.json`` plus display JPEGs. ``samples`` reads a saved categoriser run (it never scores or refits) and adds
display JPEGs from the input TIFFs. Lane 3 (morphology distance from baseline) is read from a saved categoriser site
table; the bundle records which table and its SHA256. Nothing here changes a verdict input.

Output directories must be new or empty, so a bundle is never overwritten in place. Images: trimmed frame (bright
edge bands removed, ``features.bright_bands``), at most ``--width`` px wide. Crack-like voids are the pixels counted in
``crack_frac`` (red); outlines are the bright-phase particles counted in ``bright_frac`` / ``bright_d50``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

from polaron_qc import KPI_TRUST, PRIMARY_KPIS, physics
from polaron_qc.features import bright_bands, load_image, segment
from polaron_qc.report import crack_void_mask, outline_bright, paint_crack_voids, summary

SCHEMA = 1
DEFAULT_WIDTH = 1000
OUTLINE_PX = 7          # outline width at full resolution, so it survives the ~7x display downsample
CROP_H, CROP_W = 600, 1200
DISTANCE_COLS = {"pct": "ood_pct_knn__morph", "exceeds": "ood_exceeds_knn__morph", "top": "ood_top_features__morph",
                 "nearest": "ood_nearest_ref__morph", "acq_pct": "ood_pct_knn__acq"}


# ---------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------
def _j(o):
    """Plain-JSON copy: numpy scalars → Python, NaN/inf → None, DataFrames → records, tuples → lists."""
    if isinstance(o, dict):
        return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _j(o.to_dict("records"))
    if isinstance(o, pd.Series):
        return _j(o.to_dict())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, np.ndarray):
        return _j(o.tolist())
    return str(o)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _new_out(path: str) -> None:
    if os.path.exists(path) and os.listdir(path):
        raise SystemExit(f"refusing to write into non-empty {path}; bundles are never overwritten (choose a new --out)")
    os.makedirs(os.path.join(path, "img"), exist_ok=True)


def save_jpg(rgb: np.ndarray, path: str, width: int = DEFAULT_WIDTH, quality: int = 80) -> dict:
    im = Image.fromarray(np.asarray(rgb).astype(np.uint8))
    if im.mode != "RGB":
        im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, max(1, int(im.height * width / im.width))), Image.LANCZOS)
    im.save(path, "JPEG", quality=quality, optimize=True)
    return {"file": os.path.basename(path), "w": im.width, "h": im.height}


def _grey_rgb(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(a, np.uint8)[..., None], 3, axis=2)


def _trim_rows(bse: np.ndarray) -> tuple[int, int]:
    t, b = bright_bands(bse)
    return int(t), int(bse.shape[0] - b if b else bse.shape[0])


def site_views(folder: str, site: str, th_lo: float, th_hi: float, out_dir: str, prefix: str, width: int) -> dict:
    """Four display views of one aligned site: BSE with crack-like voids, BSE with bright outlines, ETD, Inlens."""
    bse = load_image(folder, site, "BSE")
    r0, r1 = _trim_rows(bse)
    trim = bse[r0:r1]
    views = {
        "voids": paint_crack_voids(trim, th_lo, th_hi),
        "bright": outline_bright(trim, th_lo, th_hi, width=OUTLINE_PX),
        "etd": _grey_rgb(load_image(folder, site, "ETD")[r0:r1]),
        "inlens": _grey_rgb(load_image(folder, site, "Inlens")[r0:r1]),
    }
    meta = {name: save_jpg(img, os.path.join(out_dir, "img", f"{prefix}{site}_{name}.jpg"), width) for name, img in views.items()}
    return {"views": meta, "trim_rows": [r0, r1], "shape": list(bse.shape)}


def evidence_crop(folder: str, site: str, kpi: str, th_lo: float, th_hi: float, out_path: str) -> dict | None:
    """Native-resolution 1200 x 600 crop on the object behind a localized flag (largest crack-like void, largest
    pore component, or largest bright-phase particle)."""
    from scipy import ndimage as ndi
    from skimage.measure import label, regionprops_table
    bse = load_image(folder, site, "BSE")
    r0, r1 = _trim_rows(bse)
    trim = bse[r0:r1]
    pore, bright, _ = segment(trim, th_lo, th_hi)
    if kpi in ("crack_frac", "crack_count_per_Mpx", "pore_max_d"):
        mask = pore if kpi == "pore_max_d" else crack_void_mask(pore)
        what = "largest pore component" if kpi == "pore_max_d" else "longest crack-like void"
    else:
        mask, what = bright, "largest bright-phase particle"
    lab = label(mask)
    if lab.max() == 0:
        return None
    rp = regionprops_table(lab, properties=("label", "area", "centroid", "major_axis_length"))
    i = int(np.argmax(rp["major_axis_length"] if kpi == "crack_frac" else rp["area"]))
    H, W = trim.shape
    h, w = min(CROP_H, H), min(CROP_W, W)
    y0 = int(np.clip(int(rp["centroid-0"][i]) - h // 2, 0, H - h))
    x0 = int(np.clip(int(rp["centroid-1"][i]) - w // 2, 0, W - w))
    crop = trim[y0:y0 + h, x0:x0 + w]
    if kpi in ("crack_frac", "crack_count_per_Mpx"):
        rgb = paint_crack_voids(crop, th_lo, th_hi)
    elif kpi == "pore_max_d":
        rgb = _grey_rgb(crop)
        sub = lab[y0:y0 + h, x0:x0 + w] == rp["label"][i]
        rgb[sub & ~ndi.binary_erosion(sub, iterations=3)] = (227, 73, 72)
    else:
        rgb = outline_bright(crop, th_lo, th_hi, width=2)
    info = save_jpg(rgb, out_path, width=CROP_W)
    info.update(site=site, kpi=kpi, what=what, area_px=float(rp["area"][i]), major_axis_px=float(rp["major_axis_length"][i]),
                rows_trimmed_frame=[y0, y0 + h], cols=[x0, x0 + w], band_top=r0)
    return info


def read_distance(table: str, sites: list[str], label: str) -> dict:
    """Lane 3/4 per-site distance columns from a saved categoriser site table (no scoring here)."""
    if not os.path.exists(table):
        raise SystemExit(f"distance table {table} not found")
    d = pd.read_csv(table)
    missing = [c for c in DISTANCE_COLS.values() if c not in d.columns]
    if missing:
        raise SystemExit(f"distance table {table} lacks {missing}")
    d = d[d.site.astype(str).isin(sites)]
    if set(d.site.astype(str)) != set(sites):
        raise SystemExit(f"distance table {table} lacks sites {sorted(set(sites) - set(d.site.astype(str)))}")
    rows = [{"site": str(r["site"]), **{k: r[c] for k, c in DISTANCE_COLS.items()}} for _, r in d.iterrows()]
    return {"label": label, "source": table, "sha256": sha256(table), "variant": "morphology kNN (primary OOD variant)",
            "sites": _j(rows)}


# ---------------------------------------------------------------------------------------------------------------
# lot
# ---------------------------------------------------------------------------------------------------------------
def _site_values(sites: pd.DataFrame, flags: pd.DataFrame, kpis) -> list[dict]:
    f = flags.set_index("site")
    out = []
    for _, r in sites.iterrows():
        s = str(r.site)
        fr = f.loc[s] if s in f.index else {}
        out.append({"site": s, "group": str(fr.get("acquisition_group", "ordinary")),
                    "bright_low_contrast": bool(fr.get("bright_low_contrast", False)), "grey_pore": bool(fr.get("grey_pore", False)),
                    "contrast_stretched_bse": bool(fr.get("contrast_stretched_bse", False)),
                    "raised_black_level": bool(fr.get("raised_black_level", False)), "cracked": bool(fr.get("cracked", False)),
                    **{k: r.get(k) for k in kpis}})
    return out


def _excluded(res: dict) -> dict:
    """Sites whose value of a primary KPI is excluded by a quality flag (from stats.usable_n, same rule as the verdict)."""
    from polaron_qc import stats
    return {k: list(stats.usable_n(res["sites_batch"], k)["excluded_sites"]) for k in PRIMARY_KPIS}


def lot_bundle(res: dict, out: str, width: int, distance: dict | None) -> dict:
    m, v = res["meta"], res["verdict"]
    batch_dir = m["batch_dir"]
    sb = res["sites_batch"]
    cmp_ = res["compare"]
    compare = [{"kpi": r.kpi, "label": physics.KPI_LABELS.get(r.kpi, r.kpi), "trust": KPI_TRUST.get(r.kpi),
                "weight": physics.CONSEQUENCE_WEIGHTS.get(r.kpi), "is_primary": bool(r.is_primary),
                **{c: getattr(r, c) for c in ("ref_median", "batch_median", "ref_mad", "shift_mad", "ci_low", "ci_high", "cliffs_delta",
                                              "p_reported", "p_holm", "n_ref_usable", "n_batch_usable", "n_ref_fallback", "n_batch_excluded")}}
               for r in cmp_.itertuples()]
    drift = res["drift"]
    dsite = drift[drift.group == "batch"].set_index("site")
    sites = []
    for s in _site_values(sb, res["flags_batch"], PRIMARY_KPIS):
        th = sb[sb.site == s["site"]].iloc[0]
        s["drift_distance"] = dsite.distance.get(s["site"]) if s["site"] in dsite.index else None
        s["drift_top_driver"] = dsite.top_driver.get(s["site"]) if s["site"] in dsite.index else None
        s["images"] = site_views(batch_dir, s["site"], float(th.th_lo), float(th.th_hi), out, "lot_", width)
        sites.append(s)
    crops = []
    for n, f in enumerate(res["check_b"].get("flags", [])):
        th = sb[sb.site == f["site"]].iloc[0]
        c = evidence_crop(batch_dir, f["site"], f["kpi"], float(th.th_lo), float(th.th_hi),
                          os.path.join(out, "img", f"crop_{n}_{f['site']}_{f['kpi']}.jpg"))
        if c:
            crops.append({**c, "flag": f})
    local = {k: {kk: d.get(kk) for kk in ("ref_max", "ref_mad", "n_ref", "n_batch", "n_exceed", "n_credible", "iid_flag_probability", "margin_mad")}
             for k, d in res["local"].items() if k in PRIMARY_KPIS}
    acq = res["acquisition"]
    acquisition = {"note": acq.get("note"), "attenuation": acq.get("attenuation"),
                   "unadjusted_p": (acq.get("unadjusted") or {}).get("energy", {}).get("p"),
                   "stratified": {k: (acq.get("stratified") or {}).get(k) for k in ("n_ref_kept", "n_batch_kept", "reason")},
                   "stratified_p": ((acq.get("stratified") or {}).get("energy") or {}).get("p"),
                   "adjusted_p": ((acq.get("adjusted") or {}).get("energy_adjusted") or {}).get("p"),
                   "covariates": (acq.get("adjusted") or {}).get("covariates")} if "error" not in acq else {"error": acq["error"]}
    ph = res["physics"]
    pst = ph["sites"]
    void_geom = pst.groupby("batch").agg(sites=("site", "size"), crack_like=("n_crack_like", "median"),
                                         delamination_index=("delamination_index", "median"),
                                         longest_void_over_thickness=("longest_void_over_thickness", "median")).reset_index()
    stereo = (pst.groupby("batch").agg(vol_frac_bright=("vol_frac_bright", "median"), vol_frac_pore=("vol_frac_pore", "median"),
                                       mass_frac_additive_if_si=("nominal_mass_frac_additive_if_Si", "median")).reset_index()
              if "vol_frac_bright" in pst else pd.DataFrame())
    san = ph["sanity"]
    physics_out = {
        "statements": {k: ph["statements"][k] for k in PRIMARY_KPIS if k in ph["statements"]},
        "sanity": {"n_sites": int(len(san)), "n_sum_to_one": int(san.fractions_sum_to_one.sum()),
                   "max_abs_residual": float(san.fraction_sum_residual.abs().max()),
                   "n_porosity_below_typical": int((san.porosity_vs_typical == "below_typical").sum())},
        "void_geometry": void_geom, "stereology": stereo,
        "fraction_connected_note": ph.get("fraction_connected_note"), "caveats": ph.get("caveats"),
    }
    bs = res.get("battery_secondary")
    from polaron_qc import secondary
    battery = [] if bs is None else [{**r, "label": secondary.KPI_DEFINITIONS.get(r["kpi"], {}).get("label", r["kpi"])}
                                    for r in bs.to_dict("records")]
    doc = {
        "schema": SCHEMA, "kind": "lot", "exported_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "lot": m["batch"], "reference": m["reference"],
        "summary": summary(res), "next_action": v.get("next_action"), "verdict_reason": v.get("reason"),
        "energy": {k: res["energy"].get(k) for k in ("statistic", "p", "n_perm", "method", "n_ref", "n_batch", "n_dropped_batch")},
        "compare": compare, "excluded_by_flag": _excluded(res),
        "ref_values": _site_values(res["sites_ref"], res["flags_ref"], PRIMARY_KPIS),
        "sites": sites, "evidence_crops": crops, "local": local,
        "descriptive_flags": res["check_b"].get("descriptive_flags", []),
        "acquisition": acquisition, "physics": physics_out, "battery_secondary": battery,
        "distance": distance, "limits": res.get("limits"),
    }
    return _j(doc)


def baseline_bundle(res: dict, out: str, width: int) -> dict:
    """Approved-baseline view: reference KPI values with sub-populations and three example sites (most typical
    ordinary, one grey-pore, one cracked), drawn from the same result."""
    sr, fr = res["sites_ref"], res["flags_ref"]
    d = res["drift"]; d = d[d.group == "reference"].set_index("site")
    groups = fr.set_index("site").acquisition_group
    pick = []
    ordn = [s for s in d.sort_values("distance").index if groups.get(s) == "ordinary"]
    if ordn:
        pick.append((ordn[0], "most typical ordinary site (leave-one-site-out distance)"))
    for g, why in (("grey_pore", "grey-pore group: pore KPIs on a fallback threshold"), ("cracked", "cracked site: large delamination-like voids")):
        cand = [s for s in d.sort_values("distance").index if groups.get(s) == g]
        if cand:
            pick.append((cand[-1] if g == "cracked" else cand[0], why))
    ref_dir = res["meta"]["reference_dir"]
    examples = []
    for s, why in pick:
        th = sr[sr.site == s].iloc[0]
        examples.append({"site": s, "group": groups.get(s), "why": why,
                         "images": site_views(ref_dir, s, float(th.th_lo), float(th.th_hi), out, "ref_", width)})
    return _j({"schema": SCHEMA, "kind": "baseline", "reference": res["meta"]["reference"], "n_sites": int(len(sr)),
               "groups": groups.value_counts().to_dict(), "values": _site_values(sr, fr, PRIMARY_KPIS),
               "trust": {k: KPI_TRUST.get(k) for k in PRIMARY_KPIS},
               "labels": {k: physics.KPI_LABELS.get(k, k) for k in PRIMARY_KPIS}, "examples": examples})


def cmd_lot(a) -> None:
    from polaron_qc.report import build_result
    _new_out(a.out)
    from polaron_qc.report import _parse_reviews
    reviews = _parse_reviews(a.review) if a.review else None
    res = build_result(a.reference, a.batch, config=dict(image_reviewed=reviews) if reviews else None, cache_dir=a.cache_dir)
    distance = None
    if a.distance_table:
        distance = read_distance(a.distance_table, res["sites_batch"].site.astype(str).tolist(), a.distance_label)
    doc = lot_bundle(res, a.out, a.width, distance)
    doc["reviews_applied"] = [{"site": s, "kpi": k, "confirmed": v} for (s, k), v in (reviews or {}).items()]
    with open(os.path.join(a.out, "lot.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    if a.with_baseline:
        with open(os.path.join(a.out, "baseline.json"), "w", encoding="utf-8") as fh:
            json.dump(baseline_bundle(res, a.out, a.width), fh, ensure_ascii=False, indent=1)
    print("wrote", a.out, doc["summary"]["verdict"])


# ---------------------------------------------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------------------------------------------
SAMPLE_FILES = {"predictions": "submission/predictions.csv", "drivers": "submission/driver_contrasts.csv",
                "receipt": "submission/submission_receipt.json", "prediction_receipt": "scored/prediction_receipt.json",
                "features": "scored/features.csv", "site_table": "scored/site_table.csv"}


def cmd_samples(a) -> None:
    paths = {k: os.path.join(a.run, p) for k, p in SAMPLE_FILES.items()}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"categoriser run {a.run} is incomplete; missing {missing}")
    _new_out(a.out)
    pred = pd.read_csv(paths["predictions"])
    drv = pd.read_csv(paths["drivers"])
    feat = pd.read_csv(paths["features"]).set_index("site")
    st = pd.read_csv(paths["site_table"]).set_index("site")
    receipt = json.load(open(paths["prediction_receipt"]))
    versions = set(pred.model_version)
    if versions != {receipt["model_version"]}:
        raise SystemExit(f"model version mismatch: predictions {versions} vs receipt {receipt['model_version']}")
    samples = []
    for _, p in pred.iterrows():
        s = str(p.sample_id)
        f = feat.loc[s]
        flags = {k: bool(st.loc[s].get(k, False)) for k in ("bright_low_contrast", "grey_pore", "contrast_stretched_bse", "raised_black_level")}
        samples.append({
            "sample_id": s, "prediction": p.to_dict(),
            "drivers": drv[drv.sample_id == s].to_dict("records"),
            "flags": flags,
            "distance": {k: st.loc[s].get(c) for k, c in DISTANCE_COLS.items()},
            "images": site_views(a.input, s, float(f.th_lo), float(f.th_hi), a.out, "smp_", a.width),
        })
    doc = {"schema": SCHEMA, "kind": "samples", "exported_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "name": a.name, "run": a.run, "model_version": receipt["model_version"], "primary_family": receipt.get("primary_family"),
           "model_sha256": receipt.get("model_sha256"), "source_snapshot_sha256": receipt.get("source_snapshot_sha256"),
           "scored_utc": receipt.get("completed_utc"), "input_sha256": {k: sha256(p) for k, p in paths.items()},
           "samples": samples}
    with open(os.path.join(a.out, "samples.json"), "w", encoding="utf-8") as fh:
        json.dump(_j(doc), fh, ensure_ascii=False, indent=1)
    print("wrote", a.out, len(samples), "samples")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("lot", help="one incoming lot vs the approved baseline (runs the frozen verdict pipeline)")
    lp.add_argument("--reference", required=True); lp.add_argument("--batch", required=True); lp.add_argument("--out", required=True)
    lp.add_argument("--cache-dir", default="analysis_cache/features")
    lp.add_argument("--distance-table", help="saved categoriser site_table.csv with ood_* columns for this lot's sites")
    lp.add_argument("--distance-label", default="", help="how the distance table was produced (shown in the UI)")
    lp.add_argument("--with-baseline", action="store_true", help="also write baseline.json and reference example images")
    lp.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    lp.add_argument("--review", action="append", default=[], metavar="SITE:KPI=yes|no",
                    help="a human image review of a routed or pending crop (repeatable); yes = confirmed, no = refuted")
    sp = sub.add_parser("samples", help="a saved categoriser run (sample set; may mix lots, no lot verdict)")
    sp.add_argument("--run", required=True, help="run root containing scored/ and submission/")
    sp.add_argument("--input", required=True, help="the folder of TIFFs that was scored (for display images only)")
    sp.add_argument("--out", required=True); sp.add_argument("--name", default="sample set")
    sp.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    a = ap.parse_args(argv)
    if a.cmd == "lot" and a.distance_table and not a.distance_label:
        ap.error("--distance-label is required with --distance-table")
    {"lot": cmd_lot, "samples": cmd_samples}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
