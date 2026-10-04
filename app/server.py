"""app.server — local inspection server for the customer QC interface (docs/ui_plan.md §10, D57U).

    /opt/anaconda3/bin/python3 -m app.server            # http://127.0.0.1:8770

Two tabs. "Inspect a lot" takes a new lot (browser upload, or a folder in ``inbox/``), runs the frozen pipeline and
opens the lot review. "How it was built" serves the committed analysis page ``ui/index.html``.

Each inspection runs the same frozen commands an operator would run by hand, as subprocesses, each into a new
directory under ``inspections/<run id>/``:

1. input check: file names, one BSE/ETD/Inlens set per site, uint8 geometry (``score_folder.inspect_input``), and no
   image or site ID from the training set (the frozen snapshot's manifest);
2. ``analysis.submission_v2.score_folder``: frozen v2 sample scoring (also the distance-from-baseline lane);
3. ``analysis.submission_test.compose_submission``;
4. ``app.export_lot lot`` against the approved baseline with the frozen thresholds (one-lot mode only);
5. ``app.export_lot samples``;
6. ``app.build_ui`` into ``inspections/<run id>/index.html``.

Nothing is refitted on the incoming lot. Runs are never overwritten. The server binds to 127.0.0.1 only and accepts
only file names of the form ``img_<site>_<BSE|ETD|SE|Inlens>.tif``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("POLARON_INSPECTIONS", ROOT / "inspections"))
INBOX = Path(os.environ.get("POLARON_INBOX", ROOT / "inbox"))
FREEZE = ROOT / "analysis" / "submission_v2" / "freeze"
EVAL = FREEZE / "evaluation"
BASELINE_BUNDLE = ROOT / "ui" / "bundles" / "lot_Batch_1"
MODEL_VERSION = "categoriser-v2-D52"
FNAME = re.compile(r"img_(\w+)_(BSE|ETD|SE|Inlens)\.tif")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}")
MAX_FILE_BYTES = 400 * 1024 * 1024
DISTANCE_LABEL = ("Frozen v2 morphology distance model fitted on Batch 3 only (analysis/submission_v2/freeze). "
                  "This lot was not in the fit.")


def dataset_dir() -> Path:
    """The raw dataset (approved baseline images). Worktrees keep it in the main checkout."""
    env = os.environ.get("POLARON_DATASET")
    for p in ([Path(env)] if env else []) + [ROOT / "Dataset", ROOT.parents[2] / "Dataset"]:
        if (p / "Batch_3").is_dir():
            return p
    raise SystemExit("approved baseline images not found; set POLARON_DATASET to the folder holding Batch_3")


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------------------
# run records
# ---------------------------------------------------------------------------------------------------------------
STEPS = [("check", "Check the input"), ("score", "Measure sites and score sample resemblance (model v2)"),
         ("compose", "Assemble per-site resemblance"), ("lot", "Compare the lot with the approved baseline (frozen rules)"),
         ("samples", "Prepare the sample images"), ("build", "Build the report")]
LOCK = threading.Lock()          # one pipeline at a time; the steps are CPU-heavy
STATE_LOCK = threading.Lock()


def run_dir(rid: str) -> Path:
    if not re.fullmatch(r"\d{8}T\d{6}Z_[A-Za-z0-9_-]+", rid):
        raise HTTPException(404, "unknown inspection")
    d = RUNS / rid
    if not (d / "run.json").exists():
        raise HTTPException(404, "unknown inspection")
    return d


def load(d: Path) -> dict:
    return json.loads((d / "run.json").read_text())


def save(d: Path, rec: dict) -> None:
    with STATE_LOCK:
        tmp = d / "run.json.tmp"
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(d / "run.json")


def set_step(d: Path, key: str, **kw) -> dict:
    rec = load(d)
    for s in rec["steps"]:
        if s["key"] == key:
            s.update(kw)
    save(d, rec)
    return rec


# ---------------------------------------------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------------------------------------------
def check_input(folder: Path, mode: str) -> dict:
    """Same guards as the frozen scorer, run first so a bad lot fails in seconds with a plain message."""
    sys.path.insert(0, str(ROOT))
    from analysis.submission_v2.score_folder import inspect_input
    try:
        rows = inspect_input(folder)
    except ValueError as e:
        raise RuntimeError(str(e))
    snap = json.loads((FREEZE / "snapshot.json").read_text())
    train = {r["sha256"] for r in snap["raw_training_manifest"]}
    dup = sorted({r["file"] for r in rows if r["sha256"] in train})
    if dup:
        raise RuntimeError(f"{len(dup)} image(s) are copies of training images (Batches 1–3), for example {dup[0]}. "
                           "The tool inspects new lots only.")
    known = set(pd.read_csv(FREEZE / "known_sites.csv").site.astype(str))
    overlap = sorted({r["site"] for r in rows} & known)
    if overlap:
        raise RuntimeError(f"Site IDs already used in the training set: {', '.join(overlap[:5])}. Rename only if these are new images.")
    sites = sorted({r["site"] for r in rows})
    note = ""
    from polaron_qc.decision import Thresholds
    if mode == "lot" and len(sites) < Thresholds().min_usable_sites:
        note = (f"{len(sites)} sites is fewer than the {Thresholds().min_usable_sites} sites the frozen rules need in a lot, "
                "so expect a quality abstention.")
    return {"sites": sites, "images": len(rows), "note": note}


def _sub(d: Path, key: str, cmd: list[str]) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLCONFIGDIR=str(RUNS / ".mpl"), PYTHONUNBUFFERED="1")
    log = d / f"{key}.log"
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        tail = log.read_text().strip().splitlines()[-6:]
        raise RuntimeError("\n".join(tail))


def step_commands(d: Path, rec: dict) -> dict:
    folder, mode, name = Path(rec["input_dir"]), rec["mode"], rec["name"]
    py = sys.executable
    v2 = d / "v2"
    return {
        "score": [py, "-m", "analysis.submission_v2.score_folder", "--model-version", MODEL_VERSION, "--input", str(folder),
                  "--out", str(v2 / "scored"), "--cache-dir", str(ROOT / "_scratch" / "inspect_cache" / "v2")],
        "compose": [py, "-m", "analysis.submission_test.compose_submission", "--model-version", MODEL_VERSION,
                    "--primary-family", "material", "--run", str(v2 / "scored"), "--out", str(v2 / "submission")],
        "lot": [py, "-m", "app.export_lot", "lot", "--reference", str(dataset_dir() / "Batch_3"), "--batch", str(folder),
                "--out", str(d / "bundle_lot"), "--distance-table", str(v2 / "scored" / "site_table.csv"),
                "--distance-label", DISTANCE_LABEL, "--with-baseline"],
        "samples": [py, "-m", "app.export_lot", "samples", "--run", str(v2), "--input", str(folder), "--out", str(d / "bundle_samples"),
                    "--name", f"{name}: sample resemblance"],
        "build": [py, "-m", "app.build_ui", "--samples", str(d / "bundle_samples"), "--track-record", str(EVAL), "--out", str(d / "index.html"), "--inspection"]
                 + (["--lot", str(d / "bundle_lot")] if mode == "lot" else ["--baseline", str(BASELINE_BUNDLE)]),
    }


FAIL_TEXT = {"score": "Measuring the sites or scoring them with model v2 stopped.",
             "compose": "Assembling the per-sample bets stopped.",
             "lot": "Comparing the lot with the approved baseline stopped.",
             "samples": "Preparing the sample images stopped.",
             "build": "Building the report stopped."}


def pipeline(d: Path) -> None:
    rec = load(d)
    folder, mode = Path(rec["input_dir"]), rec["mode"]
    cmds = step_commands(d, rec)
    rec.update(status="running", started_utc=now())
    save(d, rec)
    with LOCK:
        for key, _ in STEPS:
            if key == "lot" and mode != "lot":
                set_step(d, key, status="skipped", detail="sample set: no lot verdict")
                continue
            t0 = time.monotonic()
            set_step(d, key, status="running", started_utc=now())
            try:
                detail = ""
                if key == "check":
                    info = check_input(folder, mode)
                    detail = f"{len(info['sites'])} sites, {info['images']} images. {info['note']}".strip()
                    r = load(d); r["sites"] = info["sites"]; save(d, r)
                else:
                    _sub(d, key, cmds[key])
                set_step(d, key, status="done", seconds=round(time.monotonic() - t0, 1), detail=detail)
            except Exception as e:  # report the failing step and stop; earlier outputs stay on disk
                plain = str(e)[-600:] if key == "check" else f"{FAIL_TEXT[key]} The log has the technical details."
                set_step(d, key, status="failed", seconds=round(time.monotonic() - t0, 1), detail=plain, log=key != "check")
                r = load(d); r.update(status="failed", finished_utc=now()); save(d, r)
                return
    r = load(d)
    r.update(status="done", finished_utc=now(), summary=summarise(d, mode))
    save(d, r)


def summarise(d: Path, mode: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from app.build_ui import ABSTENTION_LABEL, VERDICT_LABELS
    out = {}
    if mode == "lot":
        lot = json.loads((d / "bundle_lot" / "lot.json").read_text())
        label, state = VERDICT_LABELS[lot["summary"]["verdict"]]
        if lot["summary"]["outcome_columns"].get("quality_abstention"):
            label, state = ABSTENTION_LABEL, "abst"
            out["frozen_verdict"] = lot["summary"]["verdict"]
            out["abstention"] = True
        out.update(verdict_label=label, verdict_state=state, lot_id="lot-" + re.sub(r"[^A-Za-z0-9_.-]", "_", lot["lot"]))
    pred = pd.read_csv(d / "v2" / "submission" / "predictions.csv")
    out["bets"] = pred.predicted_batch.value_counts().to_dict()
    # samples whose bet rests on the first-drop miss pattern (low contrast, unreliable bright masks)
    miss = pred[pred.acquisition_flags.astype(str).str.contains("bright_low_contrast")].sample_id.astype(str).tolist()
    out["miss_pattern"] = miss
    return out


# ---------------------------------------------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------------------------------------------
app = FastAPI(title="Backscatter", docs_url=None, redoc_url=None, openapi_url=None)
TABS_CSS = """<style>.apptabs{display:flex;gap:6px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line,#dce0e5);background:var(--bg,#f4f5f6);font:14px/1.4 "IBM Plex Sans",system-ui,sans-serif}
.apptabs a{padding:6px 12px;border-radius:8px;text-decoration:none;color:var(--ink,#15191e)}.apptabs a[aria-current=page]{background:var(--accent,#2f3e52);color:var(--accent-ink,#fff)}
.apptabs a:hover:not([aria-current]){background:var(--sunk,#eceef1)}.apptabs.sticky{position:sticky;top:0;z-index:4}.apptabs .who a{padding:0;color:inherit}.apptabs .who{margin-left:auto;color:var(--muted,#535b67);font-size:12.5px}
.apptabs .mark{font-weight:600;letter-spacing:.01em;margin-right:14px}
@media (max-width:600px){.apptabs .who{display:none}}</style>"""


def tabs(active: str, extra: str = "", sticky: bool = False) -> str:
    a = lambda href, label, key: f'<a href="{href}"{" aria-current=page" if key == active else ""}>{label}</a>'
    return (TABS_CSS + f'<div class="apptabs{" sticky" if sticky else ""}" role="navigation" aria-label="Sections"><span class="mark">Backscatter</span>' + a("/inspect", "Inspect a lot", "inspect")
            + a("/approach", "Our approach", "approach") + a("/built", "How it was built", "built") + f'<span class="who">{extra}</span></div>')


def with_tabs(html: str, active: str, extra: str = "", sticky: bool = False) -> str:
    i = html.find("<body>")
    return html if i < 0 else html[: i + 6] + tabs(active, extra, sticky) + html[i + 6:]


@app.get("/")
def root():
    return RedirectResponse("/inspect")


@app.get("/inspect", response_class=HTMLResponse)
def inspect_page():
    return with_tabs((HERE / "inspect.html").read_text(encoding="utf-8"), "inspect", "Local server · frozen rules b4f4da2e357c", sticky=True)


@app.get("/approach", response_class=HTMLResponse)
def approach_page():
    p = ROOT / "ui" / "approach.html"
    if not p.exists():
        raise HTTPException(404, "ui/approach.html has not been built (python -m app.build_approach)")
    return with_tabs(p.read_text(encoding="utf-8"), "approach", "How the model and workflow were built", sticky=True)


@app.get("/built", response_class=HTMLResponse)
def built_page():
    p = ROOT / "ui" / "index.html"
    if not p.exists():
        raise HTTPException(404, "ui/index.html has not been built")
    return with_tabs(p.read_text(encoding="utf-8"), "built", "Development record: Batches 1–3 and the first drop")


@app.get("/runs/{rid}/", response_class=HTMLResponse)
def run_page(rid: str):
    d = run_dir(rid)
    rec = load(d)
    if rec["status"] != "done":
        return RedirectResponse(f"/inspect#{rid}")
    html = (d / "index.html").read_text(encoding="utf-8")
    if rec["mode"] == "lot" and rec.get("summary", {}).get("lot_id"):
        html = html.replace("</body>", f'<script>if(!location.hash)location.replace("#{rec["summary"]["lot_id"]}")</script></body>', 1)
    else:
        html = html.replace("</body>", f'<script>if(!location.hash)location.replace("#samples-bundle_samples")</script></body>', 1)
    crumb = f'<a href="/inspect">Inspect a lot</a> › <a href="/inspect#{rec["id"]}">{rec["name"]}</a> · {rec["started_utc"][:16].replace("T", " ")} UTC'
    return with_tabs(html, "", crumb)


@app.get("/api/known-sites")
def api_known_sites():
    """Training site IDs, so the browser can refuse them before a long upload (the server checks again)."""
    return sorted(pd.read_csv(FREEZE / "known_sites.csv").site.astype(str))


@app.get("/runs/{rid}/log/{key}", response_class=PlainTextResponse)
def run_log(rid: str, key: str):
    d = run_dir(rid)
    if key not in dict(STEPS):
        raise HTTPException(404, "unknown step")
    p = d / f"{key}.log"
    return p.read_text() if p.exists() else "No log for this step."


@app.get("/api/inbox")
def api_inbox():
    INBOX.mkdir(exist_ok=True)
    out = []
    for p in sorted(INBOX.iterdir()):
        if p.is_dir() and NAME.fullmatch(p.name):
            tifs = [f for f in p.iterdir() if FNAME.fullmatch(f.name)]
            out.append({"folder": p.name, "images": len(tifs), "sites": len({FNAME.fullmatch(f.name)[1] for f in tifs})})
    return out


@app.get("/api/runs")
def api_runs():
    RUNS.mkdir(exist_ok=True)
    recs = []
    for p in sorted(RUNS.iterdir(), reverse=True):
        if (p / "run.json").exists():
            r = load(p)
            item = {k: r.get(k) for k in ("id", "name", "mode", "source", "status", "created_utc", "started_utc", "finished_utc", "summary", "sites", "steps")}
            item["rehearsal"] = bool(r.get("rehearsal", str(r.get("name", "")).lower().startswith("rehearsal")))
            recs.append(item)
    return recs


@app.post("/api/runs")
async def api_create(request: Request):
    body = await request.json()
    name, mode, source = str(body.get("name", "")).strip(), body.get("mode"), body.get("source")
    if not NAME.fullmatch(name):
        raise HTTPException(400, "Use letters, digits, - or _ for the lot name (up to 48 characters).")
    if name in {"Batch_1", "Batch_2", "Batch_3"}:
        raise HTTPException(400, "That name belongs to a training lot. Choose another name.")
    if mode not in ("lot", "samples") or source not in ("upload", "inbox"):
        raise HTTPException(400, "mode must be lot or samples; source must be upload or inbox")
    rid = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + name
    d = RUNS / rid
    if d.exists():
        raise HTTPException(409, "An inspection with this name started this second; try again.")
    if source == "inbox":
        folder = INBOX / str(body.get("folder", ""))
        if not (NAME.fullmatch(folder.name) and folder.parent == INBOX and folder.is_dir()):
            raise HTTPException(400, "Choose a folder listed in the inbox.")
        input_dir = folder
    else:
        input_dir = d / "lot" / name
        input_dir.mkdir(parents=True)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"id": rid, "name": name, "mode": mode, "source": source, "input_dir": str(input_dir), "created_utc": now(),
           "rehearsal": name.lower().startswith("rehearsal"),
           "status": "uploading" if source == "upload" else "ready",
           "steps": [{"key": k, "label": l, "status": "queued"} for k, l in STEPS]}
    save(d, rec)
    return rec


@app.put("/api/runs/{rid}/files/{fname}")
async def api_upload(rid: str, fname: str, request: Request):
    d = run_dir(rid)
    rec = load(d)
    if rec["status"] != "uploading":
        raise HTTPException(409, "This inspection is no longer accepting files.")
    if not FNAME.fullmatch(fname):
        raise HTTPException(400, f"{fname}: expected img_<site>_<BSE|ETD|SE|Inlens>.tif")
    dest = Path(rec["input_dir"]) / fname
    if dest.exists():
        raise HTTPException(409, f"{fname} was already uploaded")
    n = 0
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as fh:
        async for chunk in request.stream():
            n += len(chunk)
            if n > MAX_FILE_BYTES:
                fh.close(); tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"{fname} is larger than {MAX_FILE_BYTES // 2**20} MB")
            fh.write(chunk)
    tmp.replace(dest)
    return {"file": fname, "bytes": n}


@app.post("/api/runs/{rid}/start")
def api_start(rid: str):
    d = run_dir(rid)
    rec = load(d)
    if rec["status"] not in ("uploading", "ready"):
        raise HTTPException(409, "This inspection has already started.")
    rec["status"] = "queued"
    save(d, rec)
    threading.Thread(target=pipeline, args=(d,), daemon=True).start()
    return rec


@app.get("/api/runs/{rid}")
def api_run(rid: str):
    return load(run_dir(rid))


@app.get("/api/runs/{rid}/log/{key}")
def api_log(rid: str, key: str):
    d = run_dir(rid)
    if key not in dict(STEPS):
        raise HTTPException(404, "unknown step")
    p = d / f"{key}.log"
    return JSONResponse({"log": p.read_text()[-6000:] if p.exists() else ""})


@app.get("/runs/{rid}/report.html")
def run_report_file(rid: str):
    d = run_dir(rid)
    p = d / "index.html"
    if not p.exists():
        raise HTTPException(404, "report not built")
    return FileResponse(p, filename=f"{load(d)['name']}_report.html", media_type="text/html")


def rebuild_pages() -> list[str]:
    """Rebuild every finished inspection page from its saved bundles with the current template (no step re-runs;
    bundles, bets and verdicts are untouched)."""
    done = []
    for d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        if not (d / "run.json").exists() or load(d).get("status") != "done":
            continue
        cmd = step_commands(d, load(d))["build"] + ["--replace"]
        _sub(d, "build", cmd)
        r = load(d)
        labels = dict(STEPS)
        for st in r["steps"]:
            st["label"] = labels.get(st["key"], st["label"])
        r["summary"] = summarise(d, r["mode"])
        save(d, r)
        done.append(d.name)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--rebuild-pages", action="store_true", help="rebuild stored inspection pages with the current template, then exit")
    a = ap.parse_args(argv)
    if a.rebuild_pages:
        for name in rebuild_pages():
            print("rebuilt", name)
        return
    dataset_dir()
    RUNS.mkdir(exist_ok=True); INBOX.mkdir(exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
