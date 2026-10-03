#!/usr/bin/env python3
"""Run the documented one-command drop procedure on each rehearsal fixture and tabulate the result (task H, E25).

For every fixture folder this runs exactly
    python3 -m polaron_qc.report <reference> <fixture> <out>/qc_<name>.html --cache-dir <cache> --summary <out>/qc_<name>.json [--review ...]
as a subprocess, measures wall time, keeps the stdout log (<out>/qc_<name>.log) and reads the JSON summary back. The
cache directory should be EMPTY for the incoming fixture (cold run); whether the reference is already cached there is
reported separately (pre-warm it first to measure the real drop condition: reference warm, incoming cold).

Usage
  python3 analysis/rehearsal_h/run_rehearsal.py --fixtures _fixtures/Batch_R _fixtures/Batch_Q --cache _scratch/cache --out _scratch/out
  python3 analysis/rehearsal_h/run_rehearsal.py --fixtures _fixtures/Batch_C --review c20a68de:crack_frac=yes --tag yes
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_one(reference: str, fixture: str, out_dir: str, cache: str, reviews: list[str], tag: str, python: str) -> dict:
    name = os.path.basename(os.path.normpath(fixture)) + (f"_{tag}" if tag else "")
    html, js, log = (os.path.join(out_dir, f"qc_{name}.{ext}") for ext in ("html", "json", "log"))
    cmd = [python, "-m", "polaron_qc.report", reference, fixture, html, "--cache-dir", cache, "--summary", js]
    for r in reviews:
        cmd += ["--review", r]
    ref_cached = any(f.startswith(os.path.basename(os.path.normpath(reference)) + "_") for f in os.listdir(cache)) if os.path.isdir(cache) else False
    fx_cached = any(f.startswith(os.path.basename(os.path.normpath(fixture)) + "_") for f in os.listdir(cache)) if os.path.isdir(cache) else False
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + f"\n# wall {wall:.1f} s, exit {proc.returncode}, reference cached before run: {ref_cached}, fixture cached before run: {fx_cached}\n")
        fh.write(proc.stdout); fh.write(proc.stderr)
    row = dict(fixture=name, command=" ".join(cmd), wall_s=round(wall, 1), exit=proc.returncode, reference_cached_before=ref_cached,
               fixture_cached_before=fx_cached, html=html, summary=js, log=log, html_mb=round(os.path.getsize(html) / 1e6, 2) if os.path.exists(html) else None)
    if proc.returncode == 0 and os.path.exists(js):
        s = json.load(open(js))
        fb = s["flags_batch"]
        row.update(verdict=s["verdict"], outcome_columns=s["outcome_columns"], abstention_reasons=(s.get("abstention") or {}).get("reasons"),
                   stability=f"{s['stability']['share']:.2f} ({round(s['stability']['share'] * s['stability']['n_runs'])}/{s['stability']['n_runs']})",
                   attenuation=s.get("attenuation"), usable_n=s.get("usable_n"),
                   derived_flags=dict(grey_pore=[f["site"] for f in fb if f.get("grey_pore")], low_contrast=[f["site"] for f in fb if f.get("bright_low_contrast")],
                                      raised_black_level=[f["site"] for f in fb if f.get("raised_black_level")],
                                      groups={f["site"]: f.get("acquisition_group") for f in fb}),
                   mdc={k: (round(v["mdc_mad"], 2) if v.get("feasible") and v.get("mdc_mad") is not None else f"n/a: {v.get('reason')}") for k, v in s["mdc"].items()},
                   local_flags=[dict(site=f["site"], kpi=f["kpi"], margin_in_mad=f.get("margin_in_mad"), status=f.get("review_status"),
                                     reliable=f.get("measurement_reliable"), severity_ok=f.get("severity_ok")) for f in s["local_anomaly"]["flags"]],
                   n_pending=s["local_anomaly"]["n_sites_pending"], n_credible=s["local_anomaly"]["n_sites_credible"], n_refuted=s["local_anomaly"]["n_sites_refuted"],
                   c2st=s.get("c2st_material"), notes=s["notes"], config_hash=s["meta"]["config_hash"], thresholds_hash=s["meta"]["thresholds_hash"],
                   git_describe=s["meta"]["git_describe"], runtime_s_reported=s["meta"]["runtime_s"], drivers=s.get("drivers"), what_would_move_it=s.get("what_would_move_it"))
    else:
        row.update(error=(proc.stderr or proc.stdout)[-2000:])
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="Dataset/Batch_3")
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("--cache", default="_scratch/cache")
    ap.add_argument("--out", default="_scratch/out")
    ap.add_argument("--review", action="append", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
    rows = []
    for fx in a.fixtures:
        r = run_one(a.reference, fx, a.out, a.cache, a.review, a.tag, a.python)
        rows.append(r)
        print(json.dumps(r, indent=1, ensure_ascii=False), flush=True)
    table = os.path.join(a.out, f"rehearsal_rows{('_' + a.tag) if a.tag else ''}.json")
    prev = json.load(open(table)) if os.path.exists(table) else []
    json.dump(prev + rows, open(table, "w"), indent=1, ensure_ascii=False)
    print("appended to", table)
