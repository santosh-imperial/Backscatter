#!/usr/bin/env python3
"""Rehearsal fixtures for next_steps task H (experiment log E25): HARD LINKS to the raw TIFFs under new 8-char site ids.

Nothing inside Dataset/ is created, renamed or modified; the fixtures live outside Dataset/ (default ./_fixtures,
git-ignored). A hard link shares the file's bytes with the original (same inode, no copy); removing the link leaves the
original untouched. Re-running is idempotent; ``--clean`` removes the links and the fixture folders.

Fixtures (site membership comes from the registry lists in polaron_qc, never typed here)
  Batch_R  all seven Batch_1 sites                                       expect: Batch_1 verdict; low-contrast flags derived
                                                                         from the data on the two renamed low-contrast sites
  Batch_Q  the four grey-pore Batch_3 sites + the two low-contrast        expect: quality abstention (more than half the sites
           Batch_1 sites                                                 flagged) from data-derived flags alone
  Batch_S  the first three ordinary Batch_1 sites (alphabetical)          expect: abstention for n < 5
  Batch_C  the three cracked Batch_3 sites                                expect: localized flags plus the n < 5 abstention

New ids are deterministic: one prefix letter (r / q / s / c) + the first 7 hex characters of sha1("<fixture>:<old id>"),
so a map can always be re-derived from this script; it is also written to <out>/<fixture>_map.csv and printed.

Usage
  python3 analysis/rehearsal_h/make_fixtures.py                  # build all four under ./_fixtures
  python3 analysis/rehearsal_h/make_fixtures.py --only Batch_R   # one fixture
  python3 analysis/rehearsal_h/make_fixtures.py --clean          # remove the links (originals keep their data)
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from polaron_qc import CRACKED_SITES, GREY_PORE_SITES, LOW_CONTRAST_SITES  # noqa: E402

_FILE_RE = re.compile(r"img_(\w+)_(\w+)\.tif$")   # same pattern as polaron_qc.features


def sites_in(batch_dir: str) -> list[str]:
    return sorted({_FILE_RE.match(os.path.basename(f)).group(1) for f in glob.glob(os.path.join(batch_dir, "img_*_*.tif"))})


def fixture_plan(dataset: str) -> dict[str, list[tuple[str, str]]]:
    """fixture name -> [(source batch, site id), ...]"""
    b1 = sites_in(os.path.join(dataset, "Batch_1"))
    ordinary_b1 = [s for s in b1 if s not in LOW_CONTRAST_SITES]
    return {
        "Batch_R": [("Batch_1", s) for s in b1],
        "Batch_Q": [("Batch_3", s) for s in GREY_PORE_SITES] + [("Batch_1", s) for s in LOW_CONTRAST_SITES],
        "Batch_S": [("Batch_1", s) for s in ordinary_b1[:3]],
        "Batch_C": [("Batch_3", s) for s in CRACKED_SITES],
    }


def new_id(fixture: str, old: str) -> str:
    return fixture[-1].lower() + hashlib.sha1(f"{fixture}:{old}".encode()).hexdigest()[:7]


def build(dataset: str, out: str, only: list[str] | None = None) -> None:
    plan = fixture_plan(dataset)
    for fx, members in plan.items():
        if only and fx not in only:
            continue
        dst_dir = os.path.join(out, fx)
        os.makedirs(dst_dir, exist_ok=True)
        rows, n_files, n_bytes = [], 0, 0
        ids = [new_id(fx, s) for _, s in members]
        assert len(set(ids)) == len(ids), f"{fx}: new-id collision"
        for (src_batch, site), nid in zip(members, ids):
            files = sorted(glob.glob(os.path.join(dataset, src_batch, f"img_{site}_*.tif")))
            assert files, f"{src_batch}/{site}: no TIFFs found"
            for src in files:
                det = _FILE_RE.match(os.path.basename(src)).group(2)       # detector label kept as is (SE stays SE)
                dst = os.path.join(dst_dir, f"img_{nid}_{det}.tif")
                if os.path.exists(dst) and os.path.samefile(src, dst):
                    pass                                                   # idempotent
                else:
                    if os.path.lexists(dst):
                        os.unlink(dst)
                    os.link(src, dst)
                st_src, st_dst = os.stat(src), os.stat(dst)
                assert st_src.st_ino == st_dst.st_ino and st_dst.st_nlink >= 2, f"{dst}: not a hard link of {src}"
                n_files += 1; n_bytes += st_src.st_size
            rows.append(dict(fixture=fx, new_id=nid, source_batch=src_batch, source_site=site, n_files=len(files)))
        with open(os.path.join(out, f"{fx}_map.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"{fx}: {len(rows)} sites, {n_files} hard links, {n_bytes / 1e6:.0f} MB shared with the originals (0 bytes copied)")
        for r in rows:
            print(f"  {r['new_id']}  <-  {r['source_batch']}/{r['source_site']}  ({r['n_files']} files)")


def clean(out: str) -> None:
    plan_names = ("Batch_R", "Batch_Q", "Batch_S", "Batch_C")
    for fx in plan_names:
        d = os.path.join(out, fx)
        if os.path.isdir(d):
            n = 0
            for f in glob.glob(os.path.join(d, "img_*_*.tif")):
                os.unlink(f); n += 1
            shutil.rmtree(d)
            print(f"{fx}: removed {n} links")
        m = os.path.join(out, f"{fx}_map.csv")
        if os.path.exists(m):
            os.unlink(m)
    if os.path.isdir(out) and not os.listdir(out):
        os.rmdir(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(ROOT, "Dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "_fixtures"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    if a.clean:
        clean(a.out)
    else:
        build(a.dataset, a.out, a.only)
