"""Generator for notebooks/02_batch_qc.ipynb (plan: docs/qc_plan.md). Edit this file, then:

    python3 notebooks/_build_02_batch_qc.py
    cd notebooks && PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 02_batch_qc.ipynb

All ten sections are implemented against polaron_qc (see docs/workflow.md for the data flow).
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

md("""## 3 · Pairwise comparison

Each compared batch goes through one call, `polaron_qc.report.build_result`, which runs the comparison engine, the decision layer and the leave-one-site-out stability check and collects the ML, physics and acquisition evidence. Per KPI: robust shift in reference MADs, an approximate bootstrap interval, a site-level permutation p-value on the Hodges–Lehmann shift (exact enumeration at these n), Holm-adjusted across the five primary KPIs only. Multivariate: energy distance on the primary KPIs with site-label permutation and the scaling re-fitted inside every permutation. Usable n per KPI (after quality flags) is what enters every test.""")
code(r'''from polaron_qc import report as RP, acquisition as AC
%matplotlib inline
RP_CONFIG = dict(alpha=CONFIG["alpha"], power=CONFIG["power"], seed=CONFIG["seed"], statistic=CONFIG["statistic"], primary=CONFIG["primary_kpis"])
RESULTS = {}
for b in CONFIG["compare"]:
    t0 = time.time()
    RESULTS[b] = RP.build_result(os.path.join(DATA, CONFIG["reference"]), os.path.join(DATA, b), config=RP_CONFIG, cache_dir=FCACHE, verbose=False)
    print(f"{b}: built in {time.time()-t0:.0f} s → {RESULTS[b]['verdict']['verdict']}")

show_cols = ["kpi", "n_ref_usable", "n_batch_usable", "ref_median", "batch_median", "shift_mad", "ci_low", "ci_high", "p_perm", "p_method", "n_perm", "p_holm", "direction"]
for b, res in RESULTS.items():
    cmp = res["compare"]
    print(f"\n### {b} vs {CONFIG['reference']} — primary KPIs")
    display(cmp[cmp.is_primary][show_cols].round(4).reset_index(drop=True))
    e = res["energy"]; print(f"energy distance on primary KPIs: statistic {e['statistic']:.3f}, p = {e['p']:.3f} ({e.get('n_perm')} permutations)")''')

md("""### 3b · Secondary KPIs (descriptive, uncorrected)""")
code(r'''for b, res in RESULTS.items():
    cmp = res["compare"]; sec = cmp[~cmp.is_primary].copy(); sec["trust"] = sec.kpi.map(KPI_TRUST)
    print(f"\n### {b} — secondary KPIs (no multiplicity correction; never drive a verdict)")
    display(sec[["kpi", "trust", "n_ref_usable", "n_batch_usable", "shift_mad", "ci_low", "ci_high", "p_perm", "direction"]].round(4).reset_index(drop=True))''')

md("""### 3c · Per-site drift — batch-wide or a few sites?

Robust diagonal-MAD distance of every site to the reference centre on the primary KPIs; reference sites are scored leave-one-out so they are out of sample like the batch sites. Percentile = rank within the reference distribution.""")
code(r'''fig, axes = plt.subplots(1, len(RESULTS), figsize=(7 * len(RESULTS), 3.8), sharey=True)
for ax, (b, res) in zip(np.atleast_1d(axes), RESULTS.items()):
    d = res["drift"]
    ref_d = d[d.group == "reference"]; bat_d = d[d.group == "batch"]
    is_grey = ref_d.site.isin(GREY_PORE_SITES); is_cr = ref_d.site.isin(CRACKED_SITES)
    ax.scatter(np.zeros(len(ref_d)) + np.random.default_rng(0).uniform(-.1, .1, len(ref_d)), ref_d.distance, s=26, color=BATCH_COLORS[CONFIG["reference"]], alpha=.7, label="reference (LOO)")
    ax.scatter(np.zeros(is_grey.sum()) + .18, ref_d[is_grey].distance, s=70, facecolors="none", edgecolors=BATCH_COLORS["Batch_3 (grey-pore group)"], label="grey-pore")
    ax.scatter(np.zeros(is_cr.sum()) - .18, ref_d[is_cr].distance, s=70, facecolors="none", edgecolors="#e34948", label="cracked")
    ax.scatter(np.ones(len(bat_d)) + np.random.default_rng(1).uniform(-.1, .1, len(bat_d)), bat_d.distance, s=30, color=BATCH_COLORS[b], label=b)
    for _, r in bat_d.iterrows(): ax.annotate(r.site, (1, r.distance), fontsize=7, xytext=(6, 0), textcoords="offset points")
    ax.axhline(np.percentile(ref_d.distance, 95), color="#888", ls="--", lw=.8); ax.set_xticks([0, 1]); ax.set_xticklabels([CONFIG["reference"], b]); ax.set_title(f"{b}: per-site drift (dashed = reference 95th pct)")
np.atleast_1d(axes)[0].set_ylabel("robust distance (primary KPIs)"); np.atleast_1d(axes)[0].legend(fontsize=8); plt.tight_layout(); plt.show()
for b, res in RESULTS.items():
    d = res["drift"]; bd = d[d.group == "batch"]
    print(f"{b}: {int((bd.percentile_in_ref > 95).sum())} of {len(bd)} sites beyond the reference 95th percentile; sites with no drift score (unusable bright KPIs): {sorted(set(sites_of(b).site) - set(bd.site))}")''')

md("""### 3d · Local-anomaly evidence flags

Per-site exceedance of the **ordinary-reference maximum** on the extreme-semantics KPIs (`decision.LOCAL_KPIS`). Under an i.i.d. null, at least one of 7 sites exceeds the max of 10 with probability 7/17 ≈ 41 % on a single KPI, so a flag is evidence, not a defect. A flag is promoted only with a severity margin ≥ 1 MAD of the reference per-site values, a reliable measurement on that site, and a reviewed image. Exceedances on batch-mean KPIs are listed separately as outlying sites.""")
code(r'''for b, res in RESULTS.items():
    cb = res["check_b"]
    print(f"\n### {b}: {cb['n_sites_flagged']} site(s) flagged · {cb['n_sites_pending']} pending image review · {cb['n_sites_credible']} credible · i.i.d. flag probability {ST.iid_flag_probability(len(res['ordinary_ref_sites']), res['meta']['n_sites_batch']):.2f}")
    if cb["flags"]: display(pd.DataFrame(cb["flags"]).round(4))
    if cb["descriptive_flags"]: print("outlying sites on batch-mean KPIs (not localized defects):"); display(pd.DataFrame(cb["descriptive_flags"]).round(4))''')

md("""### 3e · Batch 1 vs Batch 2 (7 vs 7)

Reported because the task is differentiation, with the caveat that at 7 vs 7 the smallest attainable two-sided p is 2/3432 ≈ 0.0006 and power is low; most KPIs will return "cannot distinguish".""")
code(r'''b1, b2 = sites_of("Batch_1"), sites_of("Batch_2")
c12 = ST.compare_kpis(b1, b2, CONFIG["primary_kpis"] + CONFIG["secondary_kpis"], primary=CONFIG["primary_kpis"], seed=CONFIG["seed"], statistic=CONFIG["statistic"], alpha=CONFIG["alpha"])
display(c12[c12.is_primary][show_cols].round(4).reset_index(drop=True))
X1 = RP._primary_matrix(RP._mask_unusable(b1, CONFIG["primary_kpis"]), CONFIG["primary_kpis"]); X2 = RP._primary_matrix(RP._mask_unusable(b2, CONFIG["primary_kpis"]), CONFIG["primary_kpis"])
e12 = ST.energy_distance_test(X1.dropna().values, X2.dropna().values, weights=PH.weights_vector(CONFIG["primary_kpis"]), n_mc=5000, seed=CONFIG["seed"])
print(f"energy distance Batch_1 (n={len(X1.dropna())}) vs Batch_2 (n={len(X2.dropna())}): {e12['statistic']:.3f}, p = {e12['p']:.3f}")''')

md("""## 4 · ML corroboration (never the sole driver)

Two members. A grouped-cross-validated L1 logistic regression on **material KPIs only** (sites are the groups and carry equal weight; the null is a site-level label permutation) drives Check A(iv); the same classifier with the acquisition flag `etd_boundary_sharpness` included is shown as acquisition evidence because that one flag carries the whole Batch 1 vs Batch 3 separation (E06). DINOv2 patch-embedding novelty is exploratory: on this data it tracks acquisition variables (E07) and can only raise an evidence flag.""")
code(r'''rows = []
for b, res in RESULTS.items():
    for lab, d in [("material only (drives A iv)", res["c2st_material"]), ("flag-inclusive (acquisition evidence)", res["c2st_flag_inclusive"])]:
        if d: rows.append(dict(batch=b, run=lab, auc=d["auc"], null_lo=(d.get("null_band") or [np.nan, np.nan])[0], null_hi=(d.get("null_band") or [np.nan, np.nan])[1], p=d["p"], n_perm=d.get("n_perm"), selected=", ".join(d["coef"][d["coef"].selected].feature.tolist()) if "selected" in d["coef"] else ""))
display(pd.DataFrame(rows).round(3))
for b, res in RESULTS.items():
    nv = res.get("novelty")
    if nv and isinstance(nv.get("per_site"), pd.DataFrame):
        ps = nv["per_site"]; print(f"\n{b} novelty per site (percentile within reference LOO; exceedance of the LOO max has ≈ {res['meta']['n_sites_batch']/(res['meta']['n_sites_ref']+res['meta']['n_sites_batch']):.0%} i.i.d. probability):")
        display(ps[[c for c in ["site", "novelty_median", "pct_median", "exceeds_ref_loo_max"] if c in ps]].round(2))
    if nv and isinstance(nv.get("correlates"), pd.DataFrame):
        print("top novelty correlates (what the embedding is tracking):"); display(nv["correlates"].head(6).round(3))''')

md("""## 5 · Sensitivity to acquisition adjustment

Three views of each comparison: unadjusted; stratified to ordinary acquisition groups on both sides; adjusted by residualising every KPI on three a-priori acquisition covariates (bright-phase separation, BSE black level, ETD boundary sharpness) with the regression re-fitted inside every permutation. This is a sensitivity analysis, not attribution. On this reference the adjustment **amplifies** the shift because the covariates encode Batch 3's own sub-populations (D26); negative attenuation is read as "not explained away", nothing more. Alongside: the three-channel agreement pattern per site.""")
code(r'''ACQ = {}
flags_all = AC.derive_flags(pd.concat([REF] + [sites_of(b) for b in CONFIG["compare"]]), pd.concat([BATCHES[x]["images"] for x in [CONFIG["reference"]] + CONFIG["compare"]]))
for b in CONFIG["compare"]:
    t0 = time.time()
    tv = AC.three_views(REF, sites_of(b), CONFIG["primary_kpis"], flags_all, statistic=CONFIG["statistic"], seed=CONFIG["seed"], flags={"grey_pore": flags_all.site[flags_all.grey_pore].tolist()})
    ACQ[b] = tv
    rows = []
    for view in ["unadjusted", "stratified", "adjusted"]:
        e = (tv[view].get("energy") if view != "adjusted" else tv[view].get("energy_adjusted")) if tv.get(view) else None
        rows.append(dict(view=view, energy=None if not e else round(e["statistic"], 3), p=None if not e else round(e["p"], 3), n_ref=tv[view].get("n_ref_kept", tv[view].get("n_ref")), n_batch=tv[view].get("n_batch_kept", tv[view].get("n_batch")), note=tv[view].get("reason", "")))
    print(f"\n### {b}: three views ({time.time()-t0:.0f} s) — attenuation {tv['attenuation']}")
    display(pd.DataFrame(rows))
agree = AC.three_channel_agreement(pd.concat([REF] + [sites_of(b) for b in CONFIG["compare"]]), pd.concat([BATCHES[x]["images"] for x in [CONFIG["reference"]] + CONFIG["compare"]]))
print("three-channel agreement — sites with a pattern other than 'none':")
display(agree[agree.pattern != "none"][[c for c in ["batch", "site", "pattern", "intensity_disagreement", "texture_disagreement"] if c in agree]].round(2))
RESULTS = {b: {**RESULTS[b], "acquisition": ACQ[b]} for b in RESULTS}''')

md("""## 6 · Physics reading (qualitative, relative) and measurement sanity

Consequence weights (ordinal) enter the multivariate test and the decision. Qualitative statements are emitted only for driver KPIs and only when the shift exceeds the ±5-gray-level threshold band for that KPI. Sanity checks: phase fractions sum to one; the ±5-level band per site; the dataset-level porosity note (open macro-pore fraction 6–14 % is consistent with unresolved porosity in the binder network being counted as solid — segmentation and preparation remain alternatives); no site has a 2-D top-to-bottom macro-pore path, so no tortuosity index is reported.""")
code(r'''W = pd.DataFrame({"kpi": CONFIG["primary_kpis"] + CONFIG["secondary_kpis"]}); W["consequence_weight"] = W.kpi.map(PH.CONSEQUENCE_WEIGHTS); W["trust"] = W.kpi.map(KPI_TRUST); W["rationale"] = W.kpi.map(PH.CONSEQUENCE_RATIONALE)
display(W)
bands = pd.read_csv(os.path.join(ROOT, "analysis_cache", "acquisition_sites.csv"))
print("±5-level threshold band (relative) by batch — median pore_frac / bright_frac:")
display(bands.groupby("batch")[["pore_frac_band_rel", "bright_frac_band_rel"]].median().round(3))
for b, res in RESULTS.items():
    drivers = res["check_a"]["drivers"]
    print(f"{b}: drivers = {drivers if drivers else 'none → no physics statement is emitted (nothing exceeds the reference null)'}")
    for k in drivers:
        pk = res["check_a"]["per_kpi"][k]; print("  ", PH.qualitative_statement(k, pk["shift_mad"], pk["direction"], None, None))''')

md("""## 7 · Decision

Two checks, one verdict, three outcome columns. **Check A** (batch-wide drift on primary KPIs): beyond the reference null · carried by a primary KPI (Holm p < α and |shift| ≥ 1 MAD) · consistent across usable sites · corroborated by the material-only classifier. **Check B** (localized): credible exceedance on an extreme-semantics KPI. **Quality abstention** is its own column, never counted as an alarm. **Decision stability** is the share of leave-one-site-out re-runs returning the same verdict — not a probability of being right. Every verdict states what would move it.""")
code(r'''rows = []
for b, res in RESULTS.items():
    v = res["verdict"]; a = res["check_a"]; st_ = res["stability"]
    rows.append(dict(batch=b, verdict=v["verdict"], drift_alert=v["outcome_columns"]["drift_alert"], localized=v["outcome_columns"]["localized"], quality_abstention=v["outcome_columns"]["quality_abstention"],
                     A_i_beyond_null=a["i_beyond_null"], A_ii_carried=a["ii_carried_by_primary"], A_iii_consistent=a["iii_consistent"], A_iv_classifier=a["iv_classifier_corroborates"],
                     stability=f"{st_['share']:.2f} ({int(round(st_['share']*st_['n_runs']))}/{st_['n_runs']})", thresholds_hash=v["thresholds_hash"]))
display(pd.DataFrame(rows).T)
for b, res in RESULTS.items():
    print(f"\n{b}: {res['verdict']['verdict']}\n  reason: {res['verdict']['reason']}\n  what would move it: {res['verdict']['what_would_move_it']}")
    if res["abstention"]["reasons"]: print("  abstention reasons:", res["abstention"]["reasons"])''')

md("""## 8 · Per-batch reports

One self-contained HTML per compared batch, with the MDC on the first screen, evidence images, the local-anomaly table, ML and acquisition evidence, physics sanity and the limits paragraph.""")
code(r'''for b, res in RESULTS.items():
    out = os.path.join(REPORTS, f"qc_{b}.html"); RP.render_report(res, out); print(f"wrote {os.path.relpath(out, ROOT)} ({os.path.getsize(out)/1e6:.2f} MB)")''')

md("""## 9 · Self-test: reference-split diagnostics, synthetic shifts, heterogeneity check

- **Reference-split diagnostics**: random 7-vs-10 splits of the reference through compare → energy → decision; drift alerts, localized flags and quality abstentions tallied separately. Internal diagnostic with uncertainty — a 7-vs-10 split does not validate the 7-vs-17 comparison.
- **Synthetic shifts**: a copy of Batch 2 with one primary KPI scaled by a known factor, to show at what effect size the drift path fires (classifier corroboration is unavailable for synthetic data, so the ceiling is *investigate — drift*).
- **Heterogeneity check**: ordinary reference sites as "incoming" against the full reference; the known cracked sites must surface on the localized path when the roles are reversed.""")
code(r'''th = DE.Thresholds(); ord_sites = REF[ORD_MASK].site.tolist()
def pipeline_fn(ref_part, pseudo):
    r = RP._run_pipeline(ref_part, pseudo, [s for s in ord_sites if s in set(ref_part.site)], {**RP.DEFAULT_CONFIG, **RP_CONFIG}, th, CONFIG["primary_kpis"], c2st=None, n_boot=300, energy_n_mc=1000, local_extra=False)
    oc = r["verdict"]["outcome_columns"]; return dict(verdict=r["verdict"]["verdict"], drift_alert=oc["drift_alert"], localized=oc["localized"], abstention=oc["quality_abstention"], energy_p=r["energy"]["p"])
t0 = time.time(); SPLIT_DIAG = ST.reference_split_diagnostics(REF, CONFIG["primary_kpis"], n_incoming=7, pipeline_fn=pipeline_fn, n_splits=60, seed=CONFIG["seed"]); print(f"{len(SPLIT_DIAG)} splits in {time.time()-t0:.0f} s")
summary = dict(drift_alert_rate=SPLIT_DIAG.drift_alert.mean(), localized_flag_rate=(SPLIT_DIAG.localized != "none").mean(), localized_pending_or_credible_rate=SPLIT_DIAG.localized.isin(["pending_review", "credible"]).mean(), abstention_rate=SPLIT_DIAG.abstention.mean())
display(pd.Series(summary).round(3).to_frame("rate"))
n = len(SPLIT_DIAG); p_hat = summary["drift_alert_rate"]; print(f"drift-alert rate {p_hat:.3f} ± {1.96*np.sqrt(p_hat*(1-p_hat)/n):.3f} (95 % binomial) vs α = {CONFIG['alpha']} — diagnostic only")
from math import comb
print(f"localized flags on {summary['localized_pending_or_credible_rate']:.0%} of splits: a 7-site draw from the 17 contains at least one of the 3 known cracked sites with probability {1 - comb(14, 7)/comb(17, 7):.0%}, so this rate measures the reference's own heterogeneity, not a false-alarm rate.")
# the same diagnostic on the ORDINARY reference only (10 sites → 5 vs 5): the localized false-alarm rate on a homogeneous reference
REF_ORD = REF[ORD_MASK].copy()
def pipeline_fn_ord(ref_part, pseudo):
    r = RP._run_pipeline(ref_part, pseudo, ref_part.site.tolist(), {**RP.DEFAULT_CONFIG, **RP_CONFIG}, th, CONFIG["primary_kpis"], c2st=None, n_boot=300, energy_n_mc=1000, local_extra=False)
    oc = r["verdict"]["outcome_columns"]; return dict(verdict=r["verdict"]["verdict"], drift_alert=oc["drift_alert"], localized=oc["localized"], abstention=oc["quality_abstention"])
SPLIT_ORD = ST.reference_split_diagnostics(REF_ORD, CONFIG["primary_kpis"], n_incoming=5, pipeline_fn=pipeline_fn_ord, n_splits=60, seed=CONFIG["seed"])
s_ord = dict(drift_alert_rate=SPLIT_ORD.drift_alert.mean(), localized_flag_rate=(SPLIT_ORD.localized != "none").mean(), localized_pending_or_credible_rate=SPLIT_ORD.localized.isin(["pending_review", "credible"]).mean(), abstention_rate=SPLIT_ORD.abstention.mean())
print("\nordinary-only reference splits (5 vs 5 of the 10 ordinary sites; i.i.d. single-KPI flag probability 5/10 = 50 %):")
display(pd.Series(s_ord).round(3).to_frame("rate"))''')
code(r'''# synthetic shifts on a copy of Batch_2: scale one primary KPI, re-run compare → decision (no classifier → ceiling is 'investigate — drift')
rows = []
base = sites_of("Batch_2")
for k in ["crack_frac", "pore_frac", "bright_frac"]:
    for f in [1.0, 1.5, 2.0, 3.0]:
        syn = base.copy(); syn[k] = syn[k] * f
        r = RP._run_pipeline(REF, syn, ord_sites, {**RP.DEFAULT_CONFIG, **RP_CONFIG}, th, CONFIG["primary_kpis"], c2st=None, n_boot=300, energy_n_mc=1000, local_extra=False)
        row = r["compare"].set_index("kpi").loc[k]
        rows.append(dict(kpi=k, factor=f, shift_mad=row.shift_mad, p_holm=row.p_holm, energy_p=r["energy"]["p"], verdict=r["verdict"]["verdict"], localized=r["verdict"]["outcome_columns"]["localized"]))
SYN = pd.DataFrame(rows); display(SYN.round(3))''')
code(r'''# heterogeneity check: ordinary reference sites as the 'incoming' batch vs the full reference
r = RP._run_pipeline(REF, REF[ORD_MASK].copy(), ord_sites, {**RP.DEFAULT_CONFIG, **RP_CONFIG}, th, CONFIG["primary_kpis"], c2st=None, n_boot=300, energy_n_mc=1000, local_extra=False)
print("ordinary-10 vs full-17:", r["verdict"]["verdict"], "| energy p", round(r["energy"]["p"], 3))
# and the reverse: known cracked sites as a 3-site 'batch' vs the ordinary reference (expect localized flags; too few sites for Check A)
r2 = RP._run_pipeline(REF[ORD_MASK].copy(), REF[REF.cracked_known].copy(), ord_sites, {**RP.DEFAULT_CONFIG, **RP_CONFIG}, th, CONFIG["primary_kpis"], c2st=None, n_boot=300, energy_n_mc=1000, local_extra=False)
print("cracked-3 vs ordinary-10:", r2["verdict"]["verdict"], "| localized:", r2["verdict"]["outcome_columns"], "| flags:", [(f["site"], f["kpi"], round(f["margin_in_mad"], 1)) for f in r2["check_b"]["flags"]])''')

md("""## 10 · Unseen batch

Add the folder name to `CONFIG["compare"]`, set `FROZEN_HASH` to the configuration hash printed in §0, re-run all. The batch gets its own report in `reports/`. The ML caches are pre-computed for Batches 1–3 only; for a new batch run `python3 -m polaron_qc.ml` (embeddings + classifier) before this notebook, or the ML section reports "not available" and Check A(iv) cannot corroborate (which caps the verdict at *investigate*).""")
code(r'''print("configuration hash:", cfg_hash, "| frozen:", FROZEN_HASH)''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = __file__.replace("_build_02_batch_qc.py", "02_batch_qc.ipynb")
nbf.write(nb, out); print("wrote", out, len(cells), "cells")
