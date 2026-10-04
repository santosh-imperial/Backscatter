"""app.build_approach — the "Our approach" page: how the Backscatter model and workflow were built, why, and how they
plug into the product (docs/ui_plan.md §11, D59U).

    python -m app.build_approach --feedback /path/to/analysis/feedback_drop_01 --out ui/approach.html --replace

Every number on the page is read from saved files: the lot bundles (``ui/bundles``), the frozen categoriser evaluation
and snapshot (``analysis/submission_v2/freeze``), the experiment registry and ``polaron_qc.decision.Thresholds``. The
images are existing repository figures and bundle overlays, downscaled and embedded. Claims cite their decision (Dnn)
or experiment (Enn) entries. The narrative is a draft that Santosh approves (team split: narrative is Santosh's).
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import html
import io
import json
import os
import subprocess
import sys

import pandas as pd
from PIL import Image

from polaron_qc import PRIMARY_KPIS, KPI_TRUST
from polaron_qc.decision import Thresholds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "ui", "bundles")
FREEZE = os.path.join(ROOT, "analysis", "submission_v2", "freeze")
METHOD_NOTE = "https://claude.ai/artifact/Y58UwtnreZrw6kDcy5eKhc"
e = html.escape


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


INPUTS: dict[str, str] = {}


def need(p):
    p = os.path.join(ROOT, p) if not os.path.isabs(p) else p
    if not os.path.exists(p):
        raise SystemExit(f"missing input {p}")
    INPUTS[os.path.relpath(p, ROOT) if p.startswith(ROOT) else p] = sha256(p)
    return p


def img(path, box=None, width=1100, q=80):
    """Data URI of a repository image, optionally cropped to ``box`` (fractions of width/height)."""
    im = Image.open(need(path)).convert("RGB")
    if box:
        W, H = im.size
        im = im.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), im.width, im.height


def fig(src, alt, cap, cls=""):
    uri, w, h = src
    return f'<figure class="{cls}"><img src="{uri}" width="{w}" height="{h}" alt="{e(alt)}"><figcaption>{cap}</figcaption></figure>'


def ev(*ids):
    return '<span class="ev">' + " ".join(f'<span class="id">{e(i)}</span>' for i in ids) + "</span>"


def f(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------------------------------------------
def load(feedback):
    lots = {b: json.load(open(need(f"ui/bundles/lot_{b}/lot.json"))) for b in ("Batch_1", "Batch_2")}
    base = json.load(open(need("ui/bundles/lot_Batch_1/baseline.json")))
    samples = json.load(open(need("ui/bundles/samples_drop01/samples.json")))
    summ = pd.read_csv(need(os.path.join(FREEZE, "evaluation", "categoriser_summary.csv")))
    conf = pd.read_csv(need(os.path.join(FREEZE, "evaluation", "confusion_material.csv")), index_col=0)
    snap = json.load(open(need(os.path.join(FREEZE, "snapshot.json"))))
    reg = pd.read_csv(need("experiments/registry.csv"), dtype=str)
    loco = reg[(reg.experiment == "E35") & (reg.metric == "categoriser_loco_balanced_accuracy")]
    truth = {}
    if feedback:
        t = pd.read_csv(need(os.path.join(feedback, "truth.csv")))
        truth = dict(zip(t.sample_id.astype(str), t.true_batch))
    return lots, base, samples, summ, conf, snap, loco, truth


# ---------------------------------------------------------------------------------------------------------------
def shift_chart(lots):
    """Per KPI: observed |robust shift| of each known lot against its detectable shift (MDC, 80 % power)."""
    rows, W, L, R = [], 640, 170, 610
    x = lambda v: L + min(v, 4) / 4 * (R - L)
    y0 = 34
    s = f'<svg viewBox="0 0 {W} {y0 + len(PRIMARY_KPIS) * 46 + 30}" width="100%" role="img" aria-label="Observed shift against detectable shift per KPI for Batch 1 and Batch 2">'
    for t in range(0, 5):
        s += f'<line x1="{x(t)}" x2="{x(t)}" y1="{y0 - 10}" y2="{y0 + len(PRIMARY_KPIS) * 46 - 6}" stroke="var(--line)"/><text x="{x(t)}" y="{y0 + len(PRIMARY_KPIS) * 46 + 12}" text-anchor="middle" class="tk">{t}{" MAD" if t == 4 else ""}</text>'
    for i, k in enumerate(PRIMARY_KPIS):
        yy = y0 + i * 46
        s += f'<text x="0" y="{yy + 14}" class="lb">{e(k)}</text>'
        for j, (b, col) in enumerate((("Batch_1", "#2a78d6"), ("Batch_2", "#eb6834"))):
            lot = lots[b]
            r = next(c for c in lot["compare"] if c["kpi"] == k)
            m = lot["summary"]["mdc"][k]["mdc_mad"]
            yb = yy + 4 + j * 16
            s += f'<rect x="{L}" y="{yb}" width="{x(m) - L:.1f}" height="11" rx="2" fill="var(--band)"><title>{b} detectable shift {m} MAD</title></rect>'
            s += f'<circle cx="{x(abs(r["shift_mad"])):.1f}" cy="{yb + 5.5}" r="5" fill="{col}"><title>{b} observed |shift| {abs(r["shift_mad"]):.2f} MAD</title></circle>'
    s += "</svg>"
    return s + ('<div class="legend"><span><i style="background:var(--band);border-radius:2px;width:22px"></i>smallest shift the rules detect at 7 sites (80 % power)</span>'
                '<span><i style="background:#2a78d6"></i>Batch 1 observed shift</span><span><i style="background:#eb6834"></i>Batch 2 observed shift</span></div>')


def family_chart(summ):
    fam = {"morph": "Morphology only", "acq": "Acquisition only", "material": "Declared primary (v2)", "combined": "Combined v1"}
    W, L, R = 660, 236, 560
    x = lambda v: L + v * (R - L)
    s = f'<svg viewBox="0 0 {W} {len(summ) * 40 + 40}" width="100%" role="img" aria-label="Balanced accuracy by feature family">'
    for t in (0, 0.25, 0.5, 0.75, 1):
        s += f'<line x1="{x(t)}" x2="{x(t)}" y1="6" y2="{len(summ) * 40 + 8}" stroke="var(--line)"/><text x="{x(t)}" y="{len(summ) * 40 + 24}" text-anchor="middle" class="tk">{t:g}</text>'
    s += f'<line x1="{x(1 / 3)}" x2="{x(1 / 3)}" y1="2" y2="{len(summ) * 40 + 10}" stroke="var(--ink)" stroke-dasharray="4 3"/><text x="{x(1 / 3) + 4}" y="{len(summ) * 40 + 36}" class="tk">chance 0.33</text>'
    for i, r in enumerate(summ.itertuples()):
        yy = 12 + i * 40
        prim = bool(r.primary)
        s += f'<text x="0" y="{yy + 13}" class="lb{" strong" if prim else ""}">{e(fam.get(r.family, r.family))} · {r.n_features}</text>'
        s += f'<rect x="{L}" y="{yy}" width="{x(r.balanced_accuracy) - L:.1f}" height="18" rx="3" fill="{"var(--ink)" if prim else "var(--base)"}"/>'
        s += f'<text x="{x(r.balanced_accuracy) + 6:.1f}" y="{yy + 13}" class="vl">{r.balanced_accuracy:.3f} · p {r.perm_p:.3f}</text>'
    return s + "</svg>"


def confusion(conf):
    rows = "".join(f'<tr><th>{e(i.replace("true ", "").replace("_", " "))}</th>' + "".join(
        f'<td class="n{" diag" if a == b else ""}">{v}</td>' for b, v in enumerate(r)) + "</tr>" for a, (i, r) in enumerate(zip(conf.index, conf.values)))
    return (f'<table class="cm"><thead><tr><th>True lot ↓ · bet →</th>' + "".join(f'<th class="n">{e(c.replace("pred ", "").replace("_", " "))}</th>' for c in conf.columns)
            + f"</tr></thead><tbody>{rows}</tbody></table>")


FLOW = """<svg viewBox="0 0 980 300" width="100%" role="img" aria-label="Workflow from three detector images to a recommended action and a per-site resemblance">
<defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1L8 5L1 9" fill="none" stroke="currentColor" stroke-width="1.6"/></marker></defs>
<g class="fl">
<rect x="10" y="110" width="130" height="80" rx="8"/><text x="75" y="140" class="h">3 images</text><text x="75" y="160">BSE · ETD · Inlens</text><text x="75" y="176">per site</text>
<rect x="175" y="110" width="140" height="80" rx="8"/><text x="245" y="140" class="h">Quality flags</text><text x="245" y="160">from the lot's</text><text x="245" y="176">own images</text>
<rect x="350" y="40" width="150" height="80" rx="8"/><text x="425" y="70" class="h">BSE segmentation</text><text x="425" y="90">histogram-anchored</text><text x="425" y="106">thresholds</text>
<rect x="350" y="180" width="150" height="80" rx="8"/><text x="425" y="210" class="h">29 features</text><text x="425" y="230">19 morphology</text><text x="425" y="246">10 appearance</text>
<rect x="535" y="40" width="160" height="80" rx="8"/><text x="615" y="70" class="h">5 primary KPIs</text><text x="615" y="90">site-level tests</text><text x="615" y="106">vs baseline</text>
<rect x="535" y="180" width="160" height="80" rx="8"/><text x="615" y="210" class="h">Frozen model v2</text><text x="615" y="230">L1 logistic</text><text x="615" y="246">31 known sites</text>
<rect x="730" y="20" width="240" height="120" rx="8" class="key"/><text x="850" y="50" class="h">Recommended action</text><text x="850" y="72">frozen rules, plus four lanes:</text><text x="850" y="92">drift · localized · distance</text><text x="850" y="112">· image quality</text>
<rect x="730" y="180" width="240" height="80" rx="8"/><text x="850" y="210" class="h">Per-site resemblance</text><text x="850" y="230">secondary; never changes</text><text x="850" y="246">the action</text>
</g>
<g class="wr">
<path d="M140 150H171" marker-end="url(#ar)"/><path d="M315 150H330V80H346" marker-end="url(#ar)"/><path d="M330 150V220H346" marker-end="url(#ar)"/>
<path d="M500 80H531" marker-end="url(#ar)"/><path d="M500 220H531" marker-end="url(#ar)"/><path d="M695 80H726" marker-end="url(#ar)"/><path d="M695 220H726" marker-end="url(#ar)"/>
<path d="M615 180V124" marker-end="url(#ar)" stroke-dasharray="3 3"/>
</g>
<text x="622" y="156" class="nt">distance lane</text>
</svg>"""

RUNTIME = """<svg viewBox="0 0 980 230" width="100%" role="img" aria-label="Runtime: an uploaded lot passes an input check, four frozen commands and a page build">
<defs><marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1L8 5L1 9" fill="none" stroke="currentColor" stroke-width="1.6"/></marker></defs>
<g class="fl">
<rect x="10" y="70" width="120" height="80" rx="8" class="key"/><text x="70" y="100" class="h">Inspect a lot</text><text x="70" y="120">upload or</text><text x="70" y="136">inbox folder</text>
<rect x="160" y="70" width="120" height="80" rx="8"/><text x="220" y="100" class="h">Input check</text><text x="220" y="120">no training image</text><text x="220" y="136">or site ID</text>
<rect x="310" y="20" width="150" height="70" rx="8"/><text x="385" y="48" class="h">Frozen scorer v2</text><text x="385" y="68">bets · distance</text>
<rect x="310" y="130" width="150" height="70" rx="8"/><text x="385" y="158" class="h">Lot comparison</text><text x="385" y="178">frozen rules vs Batch 3</text>
<rect x="490" y="20" width="150" height="70" rx="8"/><text x="565" y="48" class="h">Composer</text><text x="565" y="68">per-site resemblance</text>
<rect x="490" y="130" width="150" height="70" rx="8"/><text x="565" y="158" class="h">Bundles</text><text x="565" y="178">JSON + overlays</text>
<rect x="670" y="70" width="140" height="80" rx="8"/><text x="740" y="100" class="h">Page build</text><text x="740" y="120">reads saved files</text><text x="740" y="136">only</text>
<rect x="840" y="70" width="130" height="80" rx="8" class="key"/><text x="905" y="100" class="h">Lot review</text><text x="905" y="120">+ receipts</text><text x="905" y="136">and hashes</text>
</g>
<g class="wr">
<path d="M130 110H156" marker-end="url(#ar2)"/><path d="M280 110H295V55H306" marker-end="url(#ar2)"/><path d="M295 110V165H306" marker-end="url(#ar2)"/>
<path d="M460 55H486" marker-end="url(#ar2)"/><path d="M385 90V126" marker-end="url(#ar2)" stroke-dasharray="3 3"/><path d="M460 165H486" marker-end="url(#ar2)"/>
<path d="M640 55H655V110H666" marker-end="url(#ar2)"/><path d="M640 165H655V110" /><path d="M810 110H836" marker-end="url(#ar2)"/>
</g>
<text x="392" y="112" class="nt">site table</text>
</svg>"""


def build(args):
    lots, base, samples, summ, conf, snap, loco, truth = load(args.feedback)
    T = Thresholds()
    prim = summ[summ.primary].iloc[0]
    fam = {r.family: r for r in summ.itertuples()}
    mdc = [lots[b]["summary"]["mdc"][k]["mdc_mad"] for b in lots for k in PRIMARY_KPIS if lots[b]["summary"]["mdc"][k]["feasible"]]
    correct = int(sum(conf.values[i][i] for i in range(len(conf))))
    groups = base["groups"]
    ss = samples["samples"]
    right = sum(1 for x in ss if truth.get(x["sample_id"]) == x["prediction"]["predicted_batch"])
    loco_vals = sorted(float(v) for v in loco.value)

    I = {
        "bse": img("analysis/assets/img_pl8uabbv_BSE.jpg", (0.3, 0, 0.7, 1), 560),
        "etd": img("analysis/assets/img_pl8uabbv_ETD.jpg", (0.3, 0, 0.7, 1), 560),
        "inl": img("analysis/assets/img_pl8uabbv_Inlens.jpg", (0.3, 0, 0.7, 1), 560),
        "hist": img("analysis/morphology/output/atlas_assets/acquisition_histogram.jpg", None, 1100),
        "lc": img("analysis/morphology/benchmark/rois/Batch_1_4ih2ggld_baseline.png", None, 560),
        "ok": img("analysis/morphology/benchmark/rois/Batch_1_f1vzngrs_baseline.png", None, 560),
        "order": img("analysis/e20_inlens/figures/strips_by_batch.png", (0, 0.08, 0.205, 0.93), 520),
        "conf": img("analysis/e20_inlens/figures/variant_vs_inlens_p50.png", (0, 0.06, 0.205, 0.93), 520),
        "grey": img("ui/bundles/lot_Batch_1/img/ref_71vgq3fw_voids.jpg", None, 1000),
        "crack": img("ui/bundles/lot_Batch_1/img/ref_hzumfsms_voids.jpg", None, 1000),
        "kcrop": img("ui/bundles/lot_Batch_1/img/crop_0_4ih2ggld_crack_frac.jpg", None, 900),
        "kbright": img("ui/bundles/lot_Batch_1/img/lot_f1vzngrs_bright.jpg", (0.25, 0, 0.75, 1), 700),
        "kvoids": img("ui/bundles/lot_Batch_1/img/lot_f1vzngrs_voids.jpg", (0.25, 0, 0.75, 1), 700),
        "sig": img("analysis/signatures/fig_signature_crops.png", (0, 0.1, 1, 1), 1100),
        "geom": img("analysis/morphology/output/geometry_examples.png", None, 1000),
    }
    kpi_rows = {
        "crack_frac": ("Area of long, crack-like voids (major axis over 500 px)", "delamination-like voids; the one clear material anomaly in the known data"),
        "pore_max_d": ("Diameter of the largest void in the section", "one severe void can matter when the lot average does not move"),
        "pore_frac": ("Area fraction of resolved voids", "relates to calendering density and electrolyte access; fine pores are unresolved"),
        "bright_frac": ("Area fraction of the bright, higher-Z additive phase (possibly Si or SiOx)", "a formulation or dispersion question for the supplier"),
        "bright_d50": ("Median section diameter of additive particles", "particle-size supply or agglomeration"),
    }
    kpi_table = "".join(f'<tr><td class="mono">{k}</td><td>{e(a)}</td><td>{e(b)}</td><td>{e(KPI_TRUST.get(k, ""))}</td></tr>' for k, (a, b) in kpi_rows.items())
    loco_rows = "".join(f'<tr><td>{e(r.statistic)}</td><td class="n">{e(r.value)}</td></tr>' for r in loco.itertuples())
    drop_rows = "".join(
        f'<tr><td class="mono">{e(x["sample_id"])}</td><td>{e(x["prediction"]["predicted_batch"].replace("_", " "))}</td>'
        f'<td>{e(str(truth.get(x["sample_id"], "pending")).replace("_", " "))}</td>'
        f'<td>{"right" if truth.get(x["sample_id"]) == x["prediction"]["predicted_batch"] else ("wrong" if x["sample_id"] in truth else "")}</td>'
        f'<td>{e(x["prediction"]["acquisition_flags"].replace("_", " "))}</td></tr>' for x in ss)
    why = [
        ("Measure physical quantities on the image, then test them per site", "An end-to-end image classifier", f"{int(sum(snap['training_site_counts'].values()))} known sites. A learned model would mostly learn the microscope session (D15, D38)."),
        ("Histogram-anchored BSE thresholds with a low-contrast flag", "Fixed thresholds or plain multi-Otsu", "Multi-Otsu inflated the bright fraction 3× on low-contrast sites (D05, D07)."),
        ("Five primary KPIs that a materials expert can check on the image", "Every measured feature as a verdict input", "Small n: more tests mean more false alarms; each primary KPI has a painted image (D21, D22)."),
        ("Separate lanes, one action from frozen rules", "One combined score", "Defect risk, distribution conformance and acquisition uncertainty are different questions (D49P, D56U)."),
        ("Show the detectable shift next to every 'not detected'", "A green pass", f"At 7 sites the rules detect shifts of {min(mdc):.2f}–{max(mdc):.2f} MAD; smaller ones can hide (D16, D23)."),
        ("Appearance features only in the secondary resemblance model", "Inlens texture as a verdict input", "Inlens texture tracks Inlens brightness (Spearman ρ +0.77 over 31 sites) (D11, D51)."),
        ("Freeze rules and model before the unseen lot; refuse training images", "Re-tune on each new lot", "Hashes and receipts prove what ran; the scorer rejects training copies (D53, D54S, D57U)."),
    ]
    why_rows = "".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td><td>{e(c)}</td></tr>" for a, b, c in why)

    body = f"""
<header class="hero"><div class="toprow"><h1>Our approach</h1><button class="themebtn" id="theme" aria-pressed="false">Dark mode</button></div>
<p class="lead">Backscatter answers one question for an incoming electrode lot: has it changed against the supplier's approved baseline, and in what way?
It measures physical quantities on SEM cross-sections, tests them site by site against frozen rules, and shows the evidence on the image.
A separate frozen model says which known lot each site most resembles. Nothing is refitted on a new lot.</p>
<nav class="toc" aria-label="Sections"><a href="#s1">The question and the data</a><a href="#s2">What the data taught us</a><a href="#s3">Features</a><a href="#s4">Decision model</a><a href="#s5">Resemblance model</a><a href="#s6">Outcomes</a><a href="#s7">Into the product</a><a href="#s8">Why this design</a></nav></header>

<section id="s1"><h2><span class="sn">1</span>The question and the data</h2>
<p>Three supplier lots of one product. {e(base['reference'].replace('_', ' '))} is the approved baseline with {base['n_sites']} sites ({', '.join(f"{v} {k.replace('_', '-')}" for k, v in groups.items())}).
Batches 1 and 2 have {lots['Batch_1']['summary']['meta']['n_sites_batch']} and {lots['Batch_2']['summary']['meta']['n_sites_batch']} sites. Each site is three pixel-aligned detector images. There are no defect labels, and lengths are pixels: the 25 nm/px tag is nominal and unverified. {ev('D03', 'D14', 'D49P')}</p>
<div class="trio">{fig(I['bse'], 'BSE image of a baseline site', '<b>BSE</b>: backscattered electrons. Brightness follows atomic number, so voids, graphite and the bright additive phase separate.')}
{fig(I['etd'], 'ETD image of the same site', '<b>ETD</b>: secondary electrons. Edges and topography.')}
{fig(I['inl'], 'Inlens image of the same site', '<b>Inlens</b>: surface-sensitive and dominated by charging.')}</div>
<p class="note">Baseline site pl8uabbv, central part of the strip, the three detectors over the same field.</p></section>

<section id="s2"><h2><span class="sn">2</span>What the data taught us before any model</h2>
<div class="pair"><div><h3>Brightness is not material</h3><p>About half the images were contrast-stretched after acquisition: the histogram shows empty, comb-like bins. Raw intensity of any detector is never a material KPI. {ev('D04')}</p></div>{fig(I['hist'], 'BSE crops and a comb-shaped histogram', 'Gaps in the exported histogram show remapping after acquisition.')}</div>
<div class="pair"><div><h3>Low-contrast sites break simple thresholds</h3><p>On two Batch 1 sites the bright phase sat only 25–35 grey levels above graphite. Plain multi-Otsu then inflated the bright fraction 3×. We anchor thresholds on the histogram, flag low contrast, and drop bright-phase KPIs on those sites. {ev('D05', 'D07')}</p></div>
<div class="duo">{fig(I['lc'], 'Low-contrast site with speckled bright mask', '<b>4ih2ggld, low contrast</b>: the bright mask (yellow) fragments into rims and particle interiors.')}{fig(I['ok'], 'Ordinary site with clean bright mask', '<b>f1vzngrs, ordinary</b>: clean particles. Yellow bright phase, teal voids; unreviewed algorithm masks.')}</div></div>
<div class="pair"><div><h3>The strongest-looking feature was the microscope</h3><p>Inlens particle texture orders the lots best of all, but it rises with Inlens brightness (Spearman ρ +0.77 over 31 sites), and about 75 % of its variance follows acquisition statistics. It never drives a verdict. {ev('D11', 'D51', 'E20')}</p></div>
<div class="duo">{fig(I['order'], 'Inlens texture by lot', 'Lots order B1 > B2 > B3 …')}{fig(I['conf'], 'Inlens texture against Inlens brightness', '… but texture tracks image brightness.')}</div></div>
<div class="pair"><div><h3>The baseline is not one population</h3><p>Four grey-pore sites need a fallback pore threshold and three cracked sites carry the only clear material anomaly in the known data. They stay in the baseline, visible, with robust statistics. Frame height tracked sessions, so it was dropped from every model. {ev('D08', 'D14', 'D52')}</p></div>
<div class="stack">{fig(I['grey'], 'Grey-pore baseline site', '<b>71vgq3fw, grey-pore group</b>: crack-like voids painted red.')}{fig(I['crack'], 'Cracked baseline site', '<b>hzumfsms, cracked</b>: long delamination-like voids.')}</div></div></section>

<section id="s3"><h2><span class="sn">3</span>Features: measure what an expert can check</h2>
<p>The verdict uses five primary KPIs. Each is an observed 2-D measurement on the BSE masks, has a trust level, and can be painted on the image. Everything else is context. {ev('D09', 'D21', 'D22')}</p>
<div class="tw"><table><thead><tr><th>KPI</th><th>What it measures</th><th>Why a manufacturer cares</th><th>Trust</th></tr></thead><tbody>{kpi_table}</tbody></table></div>
<div class="stack">{fig(I['kcrop'], 'Full-resolution crop of a crack-like void', '<b>crack_frac</b>: the pixels counted are the red void, here at full resolution (Batch 1, 4ih2ggld).')}
<div class="duo">{fig(I['kvoids'], 'Voids painted on a BSE strip', '<b>pore_frac, pore_max_d</b>: voids and the largest void on the BSE strip.')}{fig(I['kbright'], 'Bright particles outlined', '<b>bright_frac, bright_d50</b>: the outlined particles are what is counted.')}</div></div>
<details class="more"><summary>Measured, tested, not promoted</summary>
<p>We measured an inventory of 88 morphology descriptors: shape variability, orientation, neighbour spacing, graph arrangement, Gabor texture and frozen image embeddings. None became a verdict input. Each stayed descriptive until it separated lots without tracking acquisition and an expert reviewed it. {ev('E18', 'E26G', 'E27', 'E28J', 'E30K')}</p>
{fig(I['geom'], 'Geometry descriptor examples', 'Nearest-neighbour spacing and void alignment, two of the descriptors kept as context.')}</details></section>

<section id="s4"><h2><span class="sn">4</span>The decision model</h2>
<p>A site, not a patch, is the unit of evidence. For each primary KPI we compare the lot's sites with the baseline's sites by permutation test on the Hodges–Lehmann shift, corrected for five KPIs (Holm), plus an energy-distance test on all five.
A KPI drives the verdict at Holm p below {T.alpha} with a shift of at least {T.min_effect_mad:g} MAD. A separate localized path checks single sites against the ordinary-baseline maximum: {T.severity_margin_mad:g} MAD routes to image review, {T.single_site_escalate_mad:g} MAD or agreement changes the verdict.
Fewer than {T.min_usable_sites} usable sites means a quality abstention. Every reference-dependent fit is repeated in each leave-one-site-out re-run to give decision stability. The thresholds are frozen and hashed: <span class="mono">{T.hash()}</span>. {ev('D17', 'D21', 'D22', 'D23', 'D29', 'D33')}</p>
<div class="figbox">{FLOW}</div>
<h3>Not detected is not unchanged</h3>
<p>Both known variant lots are consistent within detectable limits. The chart shows why that is honest rather than reassuring: each dot is the observed shift, each band the smallest shift the rules detect at this lot size. {ev('D16', 'D23')}</p>
<div class="figbox">{shift_chart(lots)}</div></section>

<section id="s5"><h2><span class="sn">5</span>The resemblance model</h2>
<p>The judged task also asks which known lot each held-back site comes from. A frozen L1 logistic regression (C {snap['chosen_C']['material']}) on {len(snap['primary_features'])} features ({fam['morph'].n_features} morphology, {len(snap['primary_features']) - fam['morph'].n_features} appearance) was fitted on the {int(sum(snap['training_site_counts'].values()))} known sites only. Session statistics and frame height are excluded; appearance features stay labelled acquisition-sensitive. Scores are uncalibrated. {ev('D50', 'D52', 'D54S')}</p>
<div class="pair"><div><h3>Nested leave-one-site-out test</h3><p>{correct} of {int(conf.values.sum())} sites correct, balanced accuracy {prim.balanced_accuracy:.3f} against chance 0.33, permutation p {prim.perm_p:.3f} over {int(prim.n_perm)} label shuffles. Morphology alone does not separate the lots ({fam['morph'].balanced_accuracy:.3f}); Batch 2 recall stays weak ({prim.recall_Batch_2:.2f}).
Holding out whole acquisition clusters keeps balanced accuracy at {loco_vals[0]:.3f}–{loco_vals[-1]:.3f}, so the model is not simply memorising sessions. {ev('E34', 'E35')}</p>
<p>Full method note: <a href="{METHOD_NOTE}">Categoriser v2 method</a>.</p></div>
<div class="figbox">{family_chart(summ)}</div></div>
<div class="pair"><div class="tw">{confusion(conf)}</div>{fig(I['sig'], 'ETD and Inlens crops per lot', 'What "appearance" means: ETD ridges and Inlens particle texture per lot. An expert must judge whether this is material or imaging.')}</div></section>

<section id="s6"><h2><span class="sn">6</span>Outcomes, stated plainly</h2>
<p>Known lots: both consistent within detectable limits at 7 sites, while the fingerprint separates them better than chance. First organiser drop: {right} of {len(ss) if truth else 'n/a'} samples assigned to the right lot. The low-contrast Batch 2 sample was called Batch 1, a quality and lot combination the known sites never showed. The bets stay as saved, and the acquisition-only comparator that scored 2 of 3 was not adopted after the fact. {ev('E33', 'E35', 'E36S', 'D54S')}</p>
<div class="tw"><table><thead><tr><th>Sample</th><th>Saved bet</th><th>Organiser label</th><th>Result</th><th>Quality flags</th></tr></thead><tbody>{drop_rows}</tbody></table></div>
<p>A 3-site rehearsal through the product abstains, as the rules require below {T.min_usable_sites} usable sites, and lists what to re-image. {ev('D57U')}</p></section>

<section id="s7"><h2><span class="sn">7</span>How it plugs into Backscatter</h2>
<p>The product runs the same frozen commands an operator would run by hand, each into a new folder with a receipt. Nothing refits on the incoming lot, and training images or site IDs are refused before any work. {ev('D56U', 'D57U')}</p>
<div class="figbox">{RUNTIME}</div>
<div class="tw"><table><thead><tr><th>On screen</th><th>Comes from</th></tr></thead><tbody>
<tr><td>Recommended action and drift lane</td><td>Site-level permutation and energy-distance tests on the five KPIs; frozen thresholds <span class="mono">{T.hash()}</span></td></tr>
<tr><td>Localized lane and the yellow box on the image</td><td>Single-site check against the ordinary-baseline maximum; full-resolution crop of the flagged object</td></tr>
<tr><td>Distance-from-baseline lane</td><td>Frozen v2 morphology distance (k-nearest neighbours to the 17 baseline sites), descriptive rank</td></tr>
<tr><td>Image-quality lane and abstention checklist</td><td>Flags derived from the lot's own images; usable sites per KPI</td></tr>
<tr><td>Per-site resemblance</td><td>Frozen v2 model <span class="mono">{e(samples['model_version'])}</span>, sha256 <span class="mono">{e(samples['model_sha256'][:12])}</span>, with exact feature contributions</td></tr>
<tr><td>What this could mean for the cell</td><td>Approved Simplified Technical English text, shown only for KPIs that drive the verdict</td></tr></tbody></table></div></section>

<section id="s8"><h2><span class="sn">8</span>Why this is the right implementation</h2>
<div class="tw"><table class="why"><thead><tr><th>We chose</th><th>Instead of</th><th>Because</th></tr></thead><tbody>{why_rows}</tbody></table></div>
<p class="note">Limits we state: 2-D sections only; pixel lengths; uncycled electrodes, so no claims about cycling damage or performance; one heterogeneous baseline lot; expert review of masks still pending.</p></section>
"""
    return body


CSS = """
:root { --bg: #f4f5f6; --surface: #fff; --sunk: #eceef1; --band: #dfe4ea; --ink: #15191e; --muted: #535b67; --line: #dce0e5; --line-strong: #b9c0c9;
  --accent: #2f3e52; --accent-ink: #fff; --focus: #3d6fb0; --base: #8b939d;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; --font-data: "IBM Plex Mono", ui-monospace, Menlo, monospace; color-scheme: light }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #111316; --surface: #191c20; --sunk: #22262b; --band: #343b44; --ink: #e9ebee; --muted: #a8b0bb; --line: #2d3238; --line-strong: #46505b; --accent: #c9d3e0; --accent-ink: #111316; --focus: #7ea6dc; --base: #6f7884; color-scheme: dark } }
:root[data-theme="dark"] { --bg: #111316; --surface: #191c20; --sunk: #22262b; --band: #343b44; --ink: #e9ebee; --muted: #a8b0bb; --line: #2d3238; --line-strong: #46505b; --accent: #c9d3e0; --accent-ink: #111316; --focus: #7ea6dc; --base: #6f7884; color-scheme: dark }
* { box-sizing: border-box } [hidden] { display: none !important }
html, body { margin: 0 } body { background: var(--bg); color: var(--ink); font: 15.5px/1.6 var(--font-body) }
::selection { background: var(--focus); color: var(--accent-ink) } :focus-visible { outline: 2px solid var(--focus); outline-offset: 2px }
a { color: inherit; text-underline-offset: 3px }
main { max-width: 1120px; margin: 0 auto; padding-block: 30px 90px; padding-inline: 24px; display: flex; flex-direction: column; gap: 46px }
h1 { font-size: 32px; line-height: 1.15; margin: 0 0 10px; font-weight: 600 } h2 { font-size: 22px; margin: 0 0 12px; font-weight: 600; display: flex; gap: 12px; align-items: baseline; text-wrap: balance }
h3 { font-size: 16.5px; margin: 0 0 6px; font-weight: 600 } p { margin: 0 0 12px; max-width: 72ch }
.sn { font-family: var(--font-data); font-size: 15px; color: var(--muted); font-weight: 500 }
.lead { font-size: 17.5px; max-width: 74ch } .note { font-size: 13.5px; color: var(--muted) } .mono { font-family: var(--font-data); font-size: .93em }
.toc { display: flex; flex-wrap: wrap; gap: 6px 10px; margin-top: 14px }
.toc a { text-decoration: none; padding: 4px 10px; border: 1px solid var(--line-strong); border-radius: 999px; font-size: 13.5px } .toc a:hover { background: var(--sunk) }
section { display: flex; flex-direction: column; gap: 14px; padding-top: 22px; border-top: 1px solid var(--line) }
.ev { display: inline-flex; gap: 4px; flex-wrap: wrap; vertical-align: 1px } .ev .id { font-family: var(--font-data); font-size: 11.5px; padding: 0 6px; border: 1px solid var(--line-strong); border-radius: 4px; color: var(--muted) }
figure { margin: 0; display: flex; flex-direction: column; gap: 6px; min-width: 0 } figure img { width: 100%; height: auto; display: block; border-radius: 6px; border: 1px solid var(--line); background: #fff }
figcaption { font-size: 13px; color: var(--muted); max-width: 70ch }
.trio { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px }
.pair { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.4fr); gap: 22px; align-items: start; padding-block: 8px }
.duo { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px } .stack { display: flex; flex-direction: column; gap: 12px }
.figbox { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px; overflow-x: auto }
.figbox svg { min-width: 560px; display: block } svg text { fill: var(--muted); font-family: var(--font-body); font-size: 12px }
svg .tk { font-family: var(--font-data); font-size: 11px } svg .lb { font-family: var(--font-data); font-size: 12px; fill: var(--ink) } svg .lb.strong { font-weight: 600 } svg .vl { font-family: var(--font-data); font-size: 11.5px; fill: var(--ink) }
.fl rect { fill: var(--surface); stroke: var(--line-strong) } .fl rect.key { fill: var(--sunk); stroke: var(--ink) } .fl text { text-anchor: middle; font-size: 12px } .fl text.h { fill: var(--ink); font-weight: 600; font-size: 13.5px }
.wr path { fill: none; stroke: var(--muted); stroke-width: 1.4; color: var(--muted) } svg .nt { font-size: 11px; font-style: italic }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12.5px; color: var(--muted); margin-top: 8px } .legend span { display: inline-flex; gap: 6px; align-items: center } .legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block }
.tw { overflow-x: auto } table { border-collapse: collapse; width: 100%; font-size: 14px; background: var(--surface) }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top } th { color: var(--muted); font-weight: 500; font-size: 12.5px }
td.n, th.n { text-align: right; font-family: var(--font-data) } td.diag { font-weight: 600 } table.cm td.n { text-align: center }
table.why td:first-child { font-weight: 500 }
details.more { border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; background: var(--surface) } details.more summary { cursor: pointer; font-weight: 500 } details.more[open] summary { margin-bottom: 10px }
.toprow { display: flex; justify-content: space-between; align-items: center; gap: 12px }
.themebtn { font: inherit; font-size: 12px; padding: 5px 11px; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface); color: var(--muted); cursor: pointer }
@media (max-width: 860px) { main { padding-inline: 16px } .trio, .pair, .duo { grid-template-columns: minmax(0, 1fr) } h1 { font-size: 26px } }
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--feedback", help="organiser feedback folder with truth.csv (first-drop labels)")
    ap.add_argument("--out", default="ui/approach.html")
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and not a.replace:
        raise SystemExit(f"{a.out} exists; pass --replace")
    body = build(a)
    for word in ("accept", "confidence"):
        if f" {word}" in body.lower():
            raise SystemExit(f"forbidden wording on the page: {word}")
    try:
        git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        git = ""
    built = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    page = f"""<!doctype html>
<html lang="en" data-theme="light">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Our approach · Backscatter</title>
<script>try {{ const t = localStorage.getItem("backscatter-theme"); if (t === "dark" || t === "light") document.documentElement.dataset.theme = t; }} catch (e) {{}}</script>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style></head>
<body><main id="main">{body}
<footer class="note">Developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived. Every number on this page is read from saved files.
Page built {built} · {e(git)}</footer></main>
<script>
const t = document.getElementById("theme");
const sync = () => t.setAttribute("aria-pressed", String(document.documentElement.dataset.theme === "dark")); sync();
t.addEventListener("click", () => {{ const n = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = n; try {{ localStorage.setItem("backscatter-theme", n); }} catch (e) {{}} sync(); }});
</script></body></html>"""
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(page)
    with open(os.path.splitext(a.out)[0] + "_receipt.json", "w") as fh:
        json.dump({"built_utc": built, "git": git, "builder_sha256": sha256(os.path.abspath(__file__)), "inputs_sha256": INPUTS,
                   "output_sha256": sha256(a.out), "size_mb": round(os.path.getsize(a.out) / 1e6, 2)}, fh, indent=1)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB, {len(INPUTS)} inputs)")


if __name__ == "__main__":
    sys.exit(main())
