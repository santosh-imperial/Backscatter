"""app.build_approach — the "Our approach" page: how the Backscatter model and workflow were built, why, and how they
plug into the product (docs/ui_plan.md §11, D59U).

    python -m app.build_approach --feedback /path/to/analysis/feedback_drop_01 --out ui/approach.html --replace

Numbers on the page are read from saved files, or typed with the decision or experiment entry they come from: the lot bundles (``ui/bundles``), the frozen categoriser evaluation
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
from polaron_qc.decision import Thresholds, LOCAL_KPIS_PROMOTE

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
    ns = [lots[b]["summary"]["mdc"][k]["n_incoming"] for b in lots for k in PRIMARY_KPIS if lots[b]["summary"]["mdc"][k]["feasible"]]
    nrange = f"{min(ns)}–{max(ns)}" if min(ns) != max(ns) else f"{min(ns)}"
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
            nk = lot["summary"]["mdc"][k]["n_incoming"]
            yb = yy + 4 + j * 16
            s += f'<rect x="{L}" y="{yb}" width="{x(m) - L:.1f}" height="11" rx="2" fill="var(--band)"><title>{b} detectable shift {m} MAD at {nk} usable sites</title></rect>'
            s += f'<circle cx="{x(abs(r["shift_mad"])):.1f}" cy="{yb + 5.5}" r="5" fill="{col}"><title>{b} observed shift {r["shift_mad"]:+.2f} MAD (size plotted)</title></circle>'
    s += "</svg>"
    return s + (f'<div class="legend"><span><i style="background:var(--band);border-radius:2px;width:22px"></i>smallest shift the rules detect at the usable sites ({nrange} per KPI; 80 % power)</span>'
                '<span><i style="background:#2a78d6"></i>Batch 1 observed shift size</span><span><i style="background:#eb6834"></i>Batch 2 observed shift size</span></div>')


def family_chart(summ):
    fam = {"morph": "Morphology only", "acq": "Acquisition only", "material": "Declared primary (v2)", "combined": "Combined, no frame height"}
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


FLOW = """<svg viewBox="0 0 980 300" width="100%" role="img" aria-label="Workflow from three detector images to a recommended action set by the frozen rules only, with the distance from baseline and the per-site resemblance shown beside it">
<defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1L8 5L1 9" fill="none" stroke="currentColor" stroke-width="1.6"/></marker></defs>
<g class="fl">
<rect x="10" y="110" width="130" height="80" rx="8"/><text x="75" y="140" class="h">3 images</text><text x="75" y="160">BSE · ETD · Inlens</text><text x="75" y="176">per site</text>
<rect x="175" y="110" width="140" height="80" rx="8"/><text x="245" y="140" class="h">Quality flags</text><text x="245" y="160">from the lot's</text><text x="245" y="176">own images</text>
<rect x="350" y="40" width="150" height="80" rx="8"/><text x="425" y="70" class="h">BSE segmentation</text><text x="425" y="90">histogram-anchored</text><text x="425" y="106">thresholds</text>
<rect x="350" y="180" width="150" height="80" rx="8"/><text x="425" y="210" class="h">29 features</text><text x="425" y="230">19 morphology</text><text x="425" y="246">10 appearance</text>
<rect x="535" y="40" width="160" height="80" rx="8"/><text x="615" y="70" class="h">5 primary KPIs</text><text x="615" y="90">site-level tests</text><text x="615" y="106">vs baseline</text>
<rect x="535" y="180" width="160" height="80" rx="8"/><text x="615" y="210" class="h">Frozen v2 snapshot</text><text x="615" y="230">L1 logistic model</text><text x="615" y="246">Batch 3 distance model</text>
<rect x="730" y="20" width="240" height="90" rx="8" class="key"/><text x="850" y="48" class="h">Recommended action</text><text x="850" y="70">frozen rules only:</text><text x="850" y="88">drift · localized · image quality</text>
<rect x="730" y="126" width="240" height="62" rx="8" class="side"/><text x="850" y="152" class="h">Distance from baseline</text><text x="850" y="172">descriptive; never sets the action</text>
<rect x="730" y="204" width="240" height="72" rx="8" class="side"/><text x="850" y="234" class="h">Per-site resemblance</text><text x="850" y="254">secondary; never sets the action</text>
</g>
<g class="wr">
<path d="M140 150H171" marker-end="url(#ar)"/><path d="M315 150H330V80H346" marker-end="url(#ar)"/><path d="M330 150V220H346" marker-end="url(#ar)"/>
<path d="M500 80H531" marker-end="url(#ar)"/><path d="M500 220H531" marker-end="url(#ar)"/><path d="M695 80H726" marker-end="url(#ar)"/>
<path d="M695 205H712V157H726" marker-end="url(#ar)" stroke-dasharray="3 3"/><path d="M695 240H726" marker-end="url(#ar)" stroke-dasharray="3 3"/>
</g>
<text x="535" y="294" class="nt">Dashed boxes are shown beside the action and never set it.</text>
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
<path d="M460 55H486" marker-end="url(#ar2)"/><path d="M385 90V110H475V150H486" marker-end="url(#ar2)" stroke-dasharray="3 3"/><path d="M460 165H486" marker-end="url(#ar2)"/>
<path d="M640 55H655V110H666" marker-end="url(#ar2)"/><path d="M640 165H655V110" /><path d="M810 110H836" marker-end="url(#ar2)"/>
</g>
<text x="430" y="104" class="nt" text-anchor="middle">distance</text>
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
    mdc_n = [lots[b]["summary"]["mdc"][k]["n_incoming"] for b in lots for k in PRIMARY_KPIS if lots[b]["summary"]["mdc"][k]["feasible"]]
    nrange = f"{min(mdc_n)}–{max(mdc_n)}" if min(mdc_n) != max(mdc_n) else f"{min(mdc_n)}"
    krus = pd.read_csv(need(os.path.join(FREEZE, "evaluation", "kruskal_context.csv"))).set_index("feature")
    reg_items = json.load(open(need("analysis/morphology/metric_register.json")))["metrics"]
    reg_items = reg_items if isinstance(reg_items, list) else list(reg_items.values())
    n_reg, n_reg_primary = len(reg_items), sum(1 for m in reg_items if m.get("qc_role") == "primary")

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
        "sigstrip": img("analysis/signatures/fig_signature_strips.png", None, 1100),
    }
    kpi_rows = {
        "crack_frac": ("Area of long, crack-like voids (major axis over 500 px)", "delamination-like voids; the one clear material anomaly in the known data"),
        "pore_max_d": ("Equivalent diameter of the largest void in the section (px)", "one severe void can matter when the lot average does not move"),
        "pore_frac": ("Area fraction of resolved voids", "relates to calendering density and electrolyte access; fine pores are unresolved"),
        "bright_frac": ("Area fraction of the bright, higher-Z additive phase (possibly Si or SiOx)", "a formulation or dispersion question for the supplier"),
        "bright_d50": ("Area-weighted median section diameter of bright-phase particles (px)", "particle-size supply or agglomeration"),
    }
    kpi_table = "".join(f'<tr><td class="mono">{k}</td><td>{e(a)}</td><td>{e(b)}</td><td>{e(KPI_TRUST.get(k, ""))}{" when the site is not low-contrast" if k.startswith("bright_") else ""}</td></tr>' for k, (a, b) in kpi_rows.items())
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
        ("Show the detectable shift next to every 'not detected'", "A green pass", f"At {nrange} usable sites per KPI the rules detect shifts of {min(mdc):.2f}–{max(mdc):.2f} MAD; smaller ones can hide (D16, D23)."),
        ("Appearance features only in the secondary resemblance model", "Inlens texture as a verdict input", "Inlens texture tracks Inlens brightness (Spearman ρ +0.77 over 31 sites) (D11, D51)."),
        ("Freeze the rules before any test data and model v2 before the final evaluation; refuse training images", "Re-tune on each new lot", "Hashes and receipts prove what ran; the scorer rejects training copies. Model v2 was revised after the first-drop images, before their labels (D53, D54S, D57U)."),
    ]
    why_rows = "".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td><td>{e(c)}</td></tr>" for a, b, c in why)
    att_rows = "".join(
        f'<tr><td>{b.replace("_", " ")}</td><td class="n">{lots[b]["acquisition"]["attenuation"]["stratified"]:+.2f}</td>'
        f'<td class="n">{lots[b]["acquisition"]["attenuation"]["adjusted"]:+.2f}</td><td>{"yes" if lots[b]["acquisition"]["attenuation"]["available"] else "no"}</td></tr>' for b in lots)
    ph1 = lots["Batch_1"]["physics"]; vg = {r["batch"]: r for b in lots for r in lots[b]["physics"]["void_geometry"]}
    san = {b: lots[b]["physics"]["sanity"] for b in lots}
    phys_rows = "".join([
        f'<tr><td>Phase fractions sum to 1 on every site</td><td>{", ".join(f"{b.replace(chr(95), chr(32))}: {san[b]["n_sum_to_one"]} of {san[b]["n_sites"]} sites" for b in san)}</td><td>sanity check, passed</td></tr>',
        f'<tr><td>Connected void path from the top to the bottom of the section</td><td>none on any known site</td><td>no tortuosity value is shown</td></tr>',
        f'<tr><td>Delamination index (crack-like void area per section)</td><td>{", ".join(f"{b.replace(chr(95), chr(32))}: {vg[b]["delamination_index"]:.2f}" for b in ("Batch_1", "Batch_2", "Batch_3") if b in vg)}</td><td>context value; the verdict does not use it</td></tr>',
        f'<tr><td>Longest void as a fraction of the section height</td><td>{", ".join(f"{b.replace(chr(95), chr(32))}: {vg[b]["longest_void_over_thickness"]:.2f}" for b in ("Batch_1", "Batch_2", "Batch_3") if b in vg)}</td><td>context value; the row count is not a confirmed thickness</td></tr>',
        f'<tr><td>Volume fraction from the area fraction (Delesse method)</td><td>shown in the drawer</td><td>context value; it assumes a random section</td></tr>',
    ])

    body = f"""
<header class="hero"><div class="toprow"><h1>Our approach</h1><button class="themebtn" id="theme" aria-pressed="false">Dark mode</button></div>
<p class="lead">Backscatter examines an incoming electrode lot against the supplier's approved baseline. A lot is a batch; the organisers use the word batch.
It gives three answers, and it keeps them apart: the QC action from frozen rules, the distance from the baseline, and the known batch that each sample resembles.
It measures physical quantities on SEM cross-sections, tests them site by site, and shows the evidence on the image. Nothing is refitted on a new lot.</p>
<nav class="toc" aria-label="Sections"><a href="#s1">The question and the data</a><a href="#s2">What the data taught us</a><a href="#s3">Features</a><a href="#s4">Decision model</a><a href="#s5">Acquisition and review</a><a href="#s6">Physics and the cell</a><a href="#s7">Resemblance model</a><a href="#s8">What is different</a><a href="#s9">Outcomes</a><a href="#s10">Into the product</a><a href="#s11">Why this design</a></nav></header>

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
<div class="pair"><div><h3>The strongest-looking feature was the microscope</h3><p>Inlens particle texture orders the lots B1 > B2 > B3 and separates them strongly (Kruskal p {krus.loc['inlens_particle_texture', 'p']:.3f}). But it rises with Inlens brightness (Spearman ρ +0.77 over 31 sites). About 75 % of its variance follows acquisition statistics. It never drives a verdict.
A later test (E32) showed that the contrast is not only a gain or offset effect. It stays linked to brightness, and a gamma change moves it by a third of the batch gap. The Batch 1 over Batch 2 part of the order comes from the amplitude. {ev('D11', 'D51', 'E20', 'E32')}</p></div>
<div class="duo">{fig(I['order'], 'Inlens texture by lot', 'Lots order B1 > B2 > B3 …')}{fig(I['conf'], 'Inlens texture against Inlens brightness', '… but texture tracks image brightness.')}</div></div>
<div class="pair"><div><h3>The baseline is not one population</h3><p>Four grey-pore sites need a fallback pore threshold and three cracked sites carry the only clear material anomaly in the known data. They stay in the baseline, visible, with robust statistics. Frame height tracked sessions, so it was dropped from every model. {ev('D08', 'D14', 'D52')}</p></div>
<div class="stack">{fig(I['grey'], 'Grey-pore baseline site', '<b>71vgq3fw, grey-pore group</b>: the pores are grey, so the pore threshold falls back. Voids are painted for reference; this site is not a cracked site.')}{fig(I['crack'], 'Cracked baseline site', '<b>hzumfsms, cracked</b>: long delamination-like voids.')}</div></div></section>

<section id="s3"><h2><span class="sn">3</span>Features: measure what an expert can check</h2>
<p>The verdict uses five primary KPIs. Each is an observed 2-D measurement on the BSE masks, has a trust level, and can be painted on the image. Everything else is context. {ev('D09', 'D21', 'D22')}</p>
<div class="tw"><table><thead><tr><th>KPI</th><th>What it measures</th><th>Why a manufacturer cares</th><th>Trust</th></tr></thead><tbody>{kpi_table}</tbody></table></div>
<div class="stack">{fig(I['kcrop'], 'Full-resolution crop of a crack-like void', '<b>crack_frac</b>: the pixels counted are the red void, here at full resolution (Batch 1, 4ih2ggld).')}
<div class="duo">{fig(I['kvoids'], 'Voids painted on a BSE strip', '<b>pore_frac, pore_max_d</b>: voids and the largest void on the BSE strip.')}{fig(I['kbright'], 'Bright particles outlined', '<b>bright_frac, bright_d50</b>: the outlined particles are what is counted.')}</div></div>
<details class="more"><summary>Measured, tested, not promoted</summary>
<p>Our metric register lists {n_reg} measured descriptors. They include the {n_reg_primary} primary KPIs above, shape variability, orientation, neighbour spacing, graph arrangement, Gabor texture and frozen image embeddings. None of the other {n_reg - n_reg_primary} became a verdict input. Each stays descriptive until it separates lots without tracking acquisition and an expert reviews it. {ev('E18', 'E26G', 'E27', 'E28J', 'E30K')}</p>
{fig(I['geom'], 'Geometry descriptor examples', 'Nearest-neighbour spacing and void alignment, two of the descriptors kept as context.')}</details></section>

<section id="s4"><h2><span class="sn">4</span>The decision model</h2>
<p>A site, not a patch, is the unit of evidence. For each primary KPI we compare the lot's sites with the baseline's sites. The test is a permutation test on the Hodges–Lehmann shift, corrected for five KPIs (Holm). An energy-distance test covers all five KPIs together.
A KPI drives the verdict at Holm p below {T.alpha} with a shift of at least {T.min_effect_mad:g} MAD. A separate localized path checks single sites against the ordinary-baseline maximum. It promotes only the two void KPIs ({', '.join(LOCAL_KPIS_PROMOTE)}). A site at {T.severity_margin_mad:g} MAD goes to image review. A site at {T.single_site_escalate_mad:g} MAD, or agreement of two sites or two KPIs, changes the verdict. The other extreme-value KPIs are flagged and never promote.
Fewer than {T.min_usable_sites} usable sites means a quality abstention. Every reference-dependent fit is repeated in each leave-one-site-out re-run to give decision stability. The thresholds are frozen and hashed: <span class="mono">{T.hash()}</span>. {ev('D17', 'D21', 'D22', 'D23', 'D29', 'D33')}</p>
<div class="figbox">{FLOW}</div>
<h3>Not detected is not unchanged</h3>
<p>Both known variant lots are consistent within detectable limits. The chart shows why that is honest and not reassuring. Each dot is the observed shift. Each band is the smallest shift the rules detect with the lot's usable sites ({nrange} per KPI, after quality flags). {ev('D16', 'D23')}</p>
<div class="figbox">{shift_chart(lots)}</div></section>

<section id="s5"><h2><span class="sn">5</span>Acquisition sensitivity and the review loop</h2>
<p>The microscope can change between sessions. The system does not let an imaging change become a reject. It derives quality flags from the lot's own images before any statistic runs. The flags are low bright-phase contrast, a raised black level with grey pores, and contrast stretching. {ev('D31')}</p>
<p>It then repeats the multivariate test in three views: unadjusted, stratified to ordinary acquisition groups, and adjusted for three acquisition covariates with the regression refitted inside every permutation. The attenuation is the share by which the statistic drops. A drop larger than {T.strong_attenuation:g} holds the verdict at investigate. If no view is available, the system withholds a reject. A negative attenuation means that the adjustment did not explain the shift away. {ev('D26', 'D30')}</p>
<div class="tw"><table><thead><tr><th>Lot</th><th>Stratified attenuation</th><th>Adjusted attenuation</th><th>Views available</th></tr></thead><tbody>{att_rows}</tbody></table></div>
<p>A person closes the loop. The localized path routes a flagged site to image review with a painted crop. The review has three states: not reviewed, confirmed, and refuted. A refuted review closes the flag. Each verdict ends with one next QC action in plain words. The action is never a release decision, because no tolerances exist. {ev('D32', 'D33', 'D45')}</p></section>

<section id="s6"><h2><span class="sn">6</span>Physics and the cell</h2>
<p>The physics layer gives the direction of a change, not a performance value. For each KPI that drives the verdict, it writes one sentence. The sentence says higher or lower than the baseline, and what that direction can mean in the cell. It writes nothing for a KPI that does not drive the verdict. The sentence is withheld when the shift is smaller than the ±5 grey-level threshold band of that KPI. {ev('D19', 'D22', 'D46')}</p>
<p>The specimens are new electrodes. Santosh confirmed a fresh graphite anode with a Si or SiOx additive. The text therefore speaks about structure as manufactured and about possible susceptibility, never about cycle damage, SEI or lithium plating. It writes "possibly Si or SiOx", because the system does not identify the chemistry. Contact between an additive particle and a void is not electrical contact. All display text for the cell is in Simplified Technical English and was approved by Santosh. {ev('D51', 'D56U')}</p>
<div class="tw"><table><thead><tr><th>Check or context value</th><th>Result on the known lots</th><th>Status</th></tr></thead><tbody>{phys_rows}</tbody></table></div>
<p>Eight experimental battery geometry KPIs exist: void distance from the additive, local bright dispersion, additive and pore association, and long-void burden. They sit in a drawer with site-bootstrap intervals and threshold sensitivity. The verdict does not use them. An expert must examine them before use. {ev('E25', 'D47K')}</p></section>

<section id="s7"><h2><span class="sn">7</span>The resemblance model</h2>
<p>The judged task also asks which known lot each held-back site comes from. A frozen L1 logistic regression uses {len(snap['primary_features'])} features ({fam['morph'].n_features} morphology, {len(snap['primary_features']) - fam['morph'].n_features} appearance). Its C of {snap['chosen_C']['material']} was chosen by inner cross-validation on the known sites. It was fitted on the {int(sum(snap['training_site_counts'].values()))} known sites only. Session statistics and frame height are excluded; appearance features stay labelled acquisition-sensitive. Scores are uncalibrated. {ev('D50', 'D52', 'D54S')}</p>
<div class="pair"><div><h3>Nested leave-one-site-out test</h3><p>{correct} of {int(conf.values.sum())} sites correct, balanced accuracy {prim.balanced_accuracy:.3f} against chance 0.33, permutation p {prim.perm_p:.3f} over {int(prim.n_perm)} label shuffles. Morphology alone does not separate the lots ({fam['morph'].balanced_accuracy:.3f}); Batch 2 recall stays weak ({prim.recall_Batch_2:.2f}).
The nested test chooses C inside each fold. A second test holds out whole acquisition clusters, with C fixed at 0.5. It keeps the balanced accuracy at {loco_vals[0]:.3f}–{loco_vals[-1]:.3f}. Within the known data the model does not simply memorise sessions. With 31 sites this is a weak test. {ev('E34', 'E35')}</p>
<p>Full method note: <a href="{METHOD_NOTE}">Categoriser v2 method</a>.</p></div>
<div class="figbox">{family_chart(summ)}</div></div>
<div class="pair"><div class="tw">{confusion(conf)}</div>{fig(I['sig'], 'ETD and Inlens crops per lot', 'What "appearance" means: ETD ridges and Inlens particle texture per lot. An expert must judge whether this is material or imaging.')}</div></section>

<section id="s8"><h2><span class="sn">8</span>What is different, and in what way</h2>
<p>The organisers ask what is different about each later batch. We tested 119 descriptors against the baseline, site by site. Twenty-four separate Batch 1 or Batch 2 from the baseline at p below 0.05; about six would do so by chance. Thirteen of the 24 are ETD or Inlens texture descriptors, seven are BSE morphology. The five primary KPIs move by at most 1.4 MAD. {ev('E32')}</p>
<p>Batch 1 shows more texture inside the additive particles, smoother graphite faces in ETD, and more additive in the lower part of the strip. Batch 2 shows fewer resolved pores per area and additive particles that lie farther from voids. Texture descriptors are acquisition-sensitive. The first organiser drop showed that these cues are not exclusive to one batch. The full table is in <span class="mono">docs/batch_signatures.md</span>.</p>
{fig(I['sigstrip'], 'Strip plots of eight descriptors per batch', 'Site values per batch for eight descriptors; the bar is the median. Shifts are in baseline MADs with a site-level permutation p.')}</section>

<section id="s9"><h2><span class="sn">9</span>Outcomes, stated plainly</h2>
<p>Known lots: both consistent within detectable limits with {nrange} usable sites per KPI, while the fingerprint separates them better than chance. First organiser drop: {right} of {len(ss) if truth else 'n/a'} samples assigned to the right lot. The low-contrast Batch 2 sample was called Batch 1, a quality and lot combination the known sites never showed. Its strongest drivers were measured inside an unreliable bright mask. The decision model drops such measurements on flagged sites. The resemblance model only marks them. The bets stay as saved, and the acquisition-only comparator that scored 2 of 3 was not adopted after the fact. The next audit defines a quality policy for every mask-dependent input on the known data, inside the folds. {ev('E33', 'E35', 'E36S', 'D54S')}</p>
<p>Model v2 was revised after the first-drop images were examined and before their labels arrived, then frozen before the final evaluation. The first drop is development evidence for v2, not an untouched test. {ev('D52', 'D54S')}</p>
<div class="tw"><table><thead><tr><th>Sample</th><th>Saved bet</th><th>Organiser label</th><th>Result</th><th>Quality flags</th></tr></thead><tbody>{drop_rows}</tbody></table></div>
<p>A 3-site rehearsal through the product abstains, as the rules require below {T.min_usable_sites} usable sites, and lists what to re-image. {ev('D57U')}</p></section>

<section id="s10"><h2><span class="sn">10</span>How it plugs into Backscatter</h2>
<p>The product runs the same frozen commands an operator would run by hand, each into a new folder with a receipt. Nothing refits on the incoming lot, and training images or site IDs are refused before any work. {ev('D56U', 'D57U')}</p>
<div class="figbox">{RUNTIME}</div>
<div class="tw"><table><thead><tr><th>On screen</th><th>Comes from</th></tr></thead><tbody>
<tr><td>Recommended action and drift lane</td><td>Site-level permutation and energy-distance tests on the five KPIs; frozen thresholds <span class="mono">{T.hash()}</span></td></tr>
<tr><td>Localized lane and the yellow box on the image</td><td>Single-site check against the ordinary-baseline maximum; full-resolution crop of the flagged object</td></tr>
<tr><td>Distance-from-baseline lane</td><td>Frozen v2 morphology distance (k-nearest neighbours to the 17 baseline sites), descriptive rank; never sets the action</td></tr>
<tr><td>Image-quality lane and abstention checklist</td><td>Flags derived from the lot's own images; usable sites per KPI</td></tr>
<tr><td>Per-site resemblance</td><td>Frozen v2 model <span class="mono">{e(samples['model_version'])}</span>, sha256 <span class="mono">{e(samples['model_sha256'][:12])}</span>, with exact feature contributions</td></tr>
<tr><td>What this could mean for the cell</td><td>Approved Simplified Technical English text, shown only for KPIs that drive the verdict</td></tr></tbody></table></div></section>

<section id="s11"><h2><span class="sn">11</span>Why this is the right implementation</h2>
<div class="tw"><table class="why"><thead><tr><th>We chose</th><th>Instead of</th><th>Because</th></tr></thead><tbody>{why_rows}</tbody></table></div>
<h3>How we keep ourselves honest</h3>
<p>Every consequential decision has a numbered entry with its rationale and the alternatives: D01 to D59U. Every number comes from a logged experiment and a registry row: E01 to E36S. A review checklist, C01 to C35, grew from each mistake a reviewer caught. Eighteen interpretive assumptions, A1 to A18, sit in an assumption register with an annotated image and a review state each. Two cold rehearsals of the drop procedure found three failures before the organisers' drop. Each exploratory pilot was pre-registered before its first run. {ev('D21', 'E23', 'E28', 'D34')}</p>
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
.fl rect { fill: var(--surface); stroke: var(--line-strong) } .fl rect.key { fill: var(--sunk); stroke: var(--ink) } .fl rect.side { stroke-dasharray: 5 4 } .fl text { text-anchor: middle; font-size: 12px } .fl text.h { fill: var(--ink); font-weight: 600; font-size: 13.5px }
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
<footer class="note">Decision rules: developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived. Resemblance model v2: revised after the first-drop images were examined and before their labels; frozen before the final evaluation. Every number on this page is read from saved files or cites the logged decision or experiment it comes from.
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
