"""Generator for notebooks/02_batch_qc.ipynb (plan: docs/qc_plan.md). Edit this file, then:

    python3 notebooks/_build_02_batch_qc.py
    cd notebooks && PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 02_batch_qc.ipynb

Sections 0–3 (configuration, features, reference characterisation) are implemented. Sections 4–10 are scaffolded and
filled in as the acquisition and report modules land (see docs/workflow.md).
"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 02 · Batch QC — does an incoming batch differ from the working reference, and does it matter?

**What this notebook does.** Compares electrode-coating batches against a working reference using interpretable microstructure KPIs, states what it could and could not have detected, separates "the material changed" from "the microscope or preparation changed", and returns one of four outcomes with its evidence:

- **consistent with the working reference, within detectable limits** — never "accept" (acceptance needs tolerances we do not have);
- **investigate — batch-wide drift**;
- **investigate — localized anomaly**;
- **reject (provisional)**.

**Read first.** `docs/qc_plan.md` (method), `docs/problem_and_findings.md` (what we know about the data), `docs/assumption_register.html` (every interpretive assumption with an image), `docs/decision_log.md` (why each choice was made and the review checklist), `docs/experiment_log.md` (numbers and progress).

**Provenance.** All thresholds, KPI lists and weights below were *developed using exploratory analysis of Batches 1–3 and frozen before the unseen batch arrived*. Their hash is asserted at run time; the hash proves stability after the freeze, not independence before it.""")

md("""## 0 · Configuration (frozen)""")
code(r'''import os, sys, json, hashlib, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, ROOT)
from polaron_qc import (NM_PER_PX, BATCH_COLORS, LOW_CONTRAST_SITES, GREY_PORE_SITES, CRACKED_SITES, PRIMARY_KPIS, MATERIAL_KPIS, KPI_TRUST)
from polaron_qc import features as FE, stats as ST, decision as DE, physics as PH
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "legend.frameon": False})

CONFIG = dict(
    reference = "Batch_3",
    compare   = ["Batch_1", "Batch_2"],          # add the unseen batch folder name here when it lands (§10)
    alpha = 0.05, power = 0.80, seed = 0,
    statistic = "hl_shift",                       # verdict test statistic (D23); median_diff kept as the reported effect size
    primary_kpis = list(PRIMARY_KPIS),            # five, fixed (plan §2.0 rule 1)
    secondary_kpis = [k for k in MATERIAL_KPIS if k not in PRIMARY_KPIS],
    known_anomalous = dict(grey_pore=list(GREY_PORE_SITES), cracked=list(CRACKED_SITES), low_contrast=list(LOW_CONTRAST_SITES)),
    anomalous_handling = "flag-and-show",         # or "exclude"
    thresholds = DE.Thresholds(),                 # decision thresholds (D22)
    provenance = "developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived",
)
FROZEN_HASH = None   # set to the printed value once frozen; the assert below then guards the unseen-batch run
cfg_hash = hashlib.sha1(json.dumps({k: (v if not hasattr(v, "hash") else v.hash()) for k, v in CONFIG.items()}, sort_keys=True, default=str).encode()).hexdigest()[:12]
print("configuration hash:", cfg_hash, "| thresholds hash:", CONFIG["thresholds"].hash())
if FROZEN_HASH is not None:
    assert cfg_hash == FROZEN_HASH, f"configuration changed after freeze ({cfg_hash} != {FROZEN_HASH})"
DATA = os.path.join(ROOT, "Dataset"); FCACHE = os.path.join(ROOT, "analysis_cache", "features"); REPORTS = os.path.join(ROOT, "reports"); os.makedirs(REPORTS, exist_ok=True)''')

md("""## 1 · Features

One call per batch folder (`polaron_qc.features.extract_batch`) returns four tables — sites, images, particles, 512-px patches — and caches them by content hash. Lengths are in pixels; µm are nominal at 25 nm/px. Definitions and trust levels are in the KPI catalogue (`docs/problem_and_findings.md` §4) and in `polaron_qc.KPI_TRUST`.""")
code(r'''t0 = time.time()
BATCHES = {b: FE.extract_batch(os.path.join(DATA, b), cache_dir=FCACHE, verbose=False) for b in [CONFIG["reference"]] + CONFIG["compare"]}
print({b: {k: v.shape for k, v in t.items()} for b, t in BATCHES.items()}, f"\n{time.time()-t0:.0f} s")

def sites_of(b):
    s = BATCHES[b]["sites"].copy()
    s["grey_pore"] = s.site.isin(CONFIG["known_anomalous"]["grey_pore"]) | s.get("grey_pore", False)
    s["cracked_known"] = s.site.isin(CONFIG["known_anomalous"]["cracked"])
    s["low_contrast"] = s.bright_low_contrast.astype(bool)
    return s
REF = sites_of(CONFIG["reference"]); ORD_MASK = ~(REF.grey_pore | REF.cracked_known)
print(f"reference {CONFIG['reference']}: {len(REF)} sites, {int(ORD_MASK.sum())} ordinary, {int(REF.grey_pore.sum())} grey-pore, {int(REF.cracked_known.sum())} cracked, {int(REF.low_contrast.sum())} low-contrast")
display(REF[["site"] + CONFIG["primary_kpis"] + ["low_contrast", "grey_pore", "cracked_known"]].round(4))''')

md("""## 2 · Reference characterisation — how much does the reference constrain?

Batch 3 is the closest thing to a baseline we have, not a clean one: it is a single production batch with 17 sites, of which four were imaged or prepared differently (grey-pore group) and three contain large delamination-like voids. Everything below is shown **with and without** those known-anomalous sites.

- **Spread**: median and MAD per primary KPI.
- **Minimum detectable change (MDC)**: the smallest shift, in reference MADs and in KPI units, that a batch of *n* sites would be detected with 80 % power at α = 0.05 by the test used in §3. Obtained by simulation: n-site pseudo-batches drawn from the reference without replacement (D23), shifted, compared against the remaining sites.
- A null result in §3 must always be read next to the MDC: it means *not detectable at this sensitivity*, never *no change*.""")
code(r'''prim = CONFIG["primary_kpis"]; ref_all = REF; ref_ord = REF[ORD_MASK]
fig, axes = plt.subplots(1, len(prim), figsize=(4 * len(prim), 3.8))
rng = np.random.default_rng(1)
for ax, k in zip(axes, prim):
    for i, (lab, sub, alpha) in enumerate([(f"all {len(ref_all)}", ref_all, .5), (f"ordinary {len(ref_ord)}", ref_ord, .9)]):
        y = sub[k].values; x = np.full(len(y), i) + rng.uniform(-.12, .12, len(y))
        ax.scatter(x, y, s=24, color=BATCH_COLORS["Batch_3"], alpha=alpha); m, d = np.median(y), ST.mad(y)
        ax.hlines(m, i - .3, i + .3, color="k", lw=1.5); ax.vlines(i + .32, m - d, m + d, color="k", lw=3, alpha=.35)
    g = ref_all[ref_all.grey_pore]; c = ref_all[ref_all.cracked_known]
    ax.scatter(np.full(len(g), .2), g[k], s=60, facecolors="none", edgecolors=BATCH_COLORS["Batch_3 (grey-pore group)"], lw=1.5, label="grey-pore")
    ax.scatter(np.full(len(c), -.2), c[k], s=60, facecolors="none", edgecolors="#e34948", lw=1.5, label="cracked")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"all {len(ref_all)}", f"ordinary {len(ref_ord)}"]); ax.set_title(k)
axes[0].legend(fontsize=8); plt.suptitle("Reference spread per primary KPI (bar = median, grey bar = ± MAD)", y=1.02); plt.tight_layout(); plt.show()''')

code(r'''# MDC per primary KPI, for the usable n of each compared batch (computed on the full usable reference; D23)
def usable_batch_n(batch_sites, k):
    return int(ST.usable_n(batch_sites, k)["n_usable"]) if "n_usable" in ST.usable_n(batch_sites, k) else int(batch_sites[k].notna().sum())
rows = []
for k in prim:
    x = REF[k].dropna().values
    for b in CONFIG["compare"]:
        n_in = usable_batch_n(sites_of(b), k)
        m = ST.mdc(x, n_in, alpha=CONFIG["alpha"], power=CONFIG["power"], test=dict(statistic=CONFIG["statistic"], n_mc=999), seed=CONFIG["seed"])
        rows.append(dict(kpi=k, batch=b, n_incoming=n_in, ref_median_all=np.median(x), ref_mad_all=ST.mad(x),
                         ref_median_ordinary=np.median(ref_ord[k]), ref_mad_ordinary=ST.mad(ref_ord[k]),
                         mdc_mad=m["mdc_mad"], mdc_abs=m.get("mdc_abs", m["mdc_mad"] * ST.mad(x)), design=m.get("design", "")))
MDC = pd.DataFrame(rows)
display(MDC.round(4))
print("Reading: an absolute MDC larger than the reference median (e.g. crack_frac) means only a batch that more than doubles that KPI could register as batch-wide drift; the localized path (§3) exists for exactly this case.")''')

md("""### 2b · Reference-split diagnostics (internal)

Repeated random splits of the reference into *n* pseudo-incoming sites versus the rest, run through the same comparison and decision as §3–§4. Outcomes are tallied separately — statistical drift alerts, local flags, quality abstentions — and the split (7 vs 10) is smaller than the real comparison (7 vs 17), so this is a diagnostic with its own uncertainty, not a verified false-alarm rate. *(Filled in after §4 is wired; placeholder.)*""")
code(r'''# placeholder — wired after the decision pipeline function exists (see §4)
SPLIT_DIAG = None''')

md("""## 3 · Pairwise comparison — *(wired next: compare_kpis, energy distance, per-site drift, local-anomaly flags)*""")
code(r'''COMPARE = {}
for b in CONFIG["compare"]:
    bs = sites_of(b)
    tab = ST.compare_kpis(REF, bs, CONFIG["primary_kpis"] + CONFIG["secondary_kpis"], primary=CONFIG["primary_kpis"], seed=CONFIG["seed"], statistic=CONFIG["statistic"], alpha=CONFIG["alpha"])
    COMPARE[b] = tab
    print(f"\n{b} vs {CONFIG['reference']} — primary KPIs ({CONFIG['statistic']}, site-level permutation, Holm on 5)")
    display(tab[tab.is_primary][["kpi", "n_ref_usable", "n_batch_usable", "ref_median", "batch_median", "shift_mad", "ci_low", "ci_high", "p_perm", "p_method", "n_perm", "p_holm", "direction"]].round(4))''')

md("""## 4 · ML corroboration — *(to wire: material-only c2st for Check A(iv); novelty as evidence flag)*""")
code(r'''pass''')
md("""## 5 · Sensitivity to acquisition adjustment — *(to wire from polaron_qc.acquisition: unadjusted / stratified / adjusted)*""")
code(r'''pass''')
md("""## 6 · Physics reading — *(to wire: consequence weights, qualitative statements gated on the threshold band, sanity checks)*""")
code(r'''pass''')
md("""## 7 · Decision — *(to wire: Check A, Check B, abstention, verdict, jackknife stability)*""")
code(r'''pass''')
md("""## 8 · Explanation and per-batch reports — *(to wire from polaron_qc.report)*""")
code(r'''pass''')
md("""## 9 · Self-test: false-alarm diagnostics, power, synthetic shifts, heterogeneity check — *(to wire)*""")
code(r'''pass''')
md("""## 10 · Unseen batch

Add the folder name to `CONFIG["compare"]`, set `FROZEN_HASH`, re-run all. The batch gets its own report in `reports/`. Nothing else changes.""")
code(r'''pass''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = __file__.replace("_build_02_batch_qc.py", "02_batch_qc.ipynb")
nbf.write(nb, out); print("wrote", out, len(cells), "cells")
