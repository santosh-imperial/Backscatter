"""E19 — review load of the localized path under candidate rules (run from the repo root).

Rules compared (all on the two promotable KPIs crack_frac and pore_max_d, margin m in MADs of the ordinary
per-site reference values beyond the ordinary maximum):
  single@m      : >= 1 batch site beyond max + m MAD on either KPI            (current rule at m = 2.0)
  two_sites@2   : >= 2 batch sites beyond max + 2 MAD on the same KPI
  two_kpis@2    : both KPIs have >= 1 site beyond max + 2 MAD
  tiered(e)     : single@e  OR  two_sites@2  OR  two_kpis@2   (verdict flips);  single@2 but not tiered = routed to review only
Designs:
  A  parametric, fixed real 10-site ordinary reference, 7 clean sites drawn from a lognormal fitted to the ordinary values
     (Gaussian copula between the two KPIs at the observed Spearman rho)
  B  parametric, reference re-drawn (10 sites) and batch (7 sites) both from the fit — reference max/MAD vary
  C  real ordinary reference, 5-vs-5 site splits through the actual pipeline (stats + decision), 60 splits
Sensitivity: the three known cracked reference sites vs the ordinary 10, and synthetic x1.5 / x2 / x3 crack_frac on Batch 2.
Writes experiments/e19_review_load.csv (one row per rule x design) and prints the tables.
"""
import os, sys, time, itertools
import numpy as np, pandas as pd
from scipy import stats as sps
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from polaron_qc import features, acquisition, stats as ST, decision as DE, report as RP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KPIS = list(DE.LOCAL_KPIS_PROMOTE)          # crack_frac, pore_max_d
MARGINS = [2.0, 2.5, 3.0]
N_BATCH, N_REF, N_MC, SEED = 7, 10, 200_000, 0
rng = np.random.default_rng(SEED)

def load(b):
    t = features.extract_batch(os.path.join(ROOT, "Dataset", b), cache_dir=os.path.join(ROOT, "analysis_cache", "features"), verbose=False)
    s = t["sites"].reset_index(drop=True); s["site"] = s.site.astype(str)
    fl = RP.derive_flags(s, t["images"])
    return RP.apply_derived_flags(s, fl), fl

ref, flags_ref = load("Batch_3")
b2, _ = load("Batch_2")
ordinary = flags_ref.site[flags_ref.acquisition_group == "ordinary"].tolist()
cracked = flags_ref.site[flags_ref.cracked_known].tolist()
ref_ord = ref[ref.site.isin(ordinary)].reset_index(drop=True)
assert len(ref_ord) == N_REF, len(ref_ord)

# ---------------------------------------------------------------- rules on a (n_sites x 2) margin matrix
def rules(M):
    """M: margins in MADs, shape (n_sim, n_sites, 2). Returns dict rule -> bool array (n_sim,)."""
    out = {}
    for m in MARGINS:
        out[f"single@{m}"] = (M >= m).any(axis=(1, 2))
    out["two_sites@2"] = ((M >= 2.0).sum(axis=1) >= 2).any(axis=1)
    out["two_kpis@2"] = (M >= 2.0).any(axis=1).all(axis=1)
    for e in (2.5, 3.0):
        out[f"tiered({e})"] = out[f"single@{e}"] | out["two_sites@2"] | out["two_kpis@2"]
        out[f"review_only({e})"] = out["single@2.0"] & ~out[f"tiered({e})"]
    return out

# ---------------------------------------------------------------- parametric designs A and B
fits, rho = {}, None
vals = ref_ord[KPIS].to_numpy(float)
for j, k in enumerate(KPIS):
    shape, loc, scale = sps.lognorm.fit(vals[:, j], floc=0)
    fits[k] = (shape, scale)
rho = sps.spearmanr(vals[:, 0], vals[:, 1]).statistic
r_pearson = 2 * np.sin(np.pi * rho / 6)      # Gaussian-copula parameter matching the Spearman rho
L = np.linalg.cholesky(np.array([[1, r_pearson], [r_pearson, 1]]))

def draw(n_sim, n_sites):
    z = rng.standard_normal((n_sim, n_sites, 2)) @ L.T
    u = sps.norm.cdf(z)
    x = np.empty_like(u)
    for j, k in enumerate(KPIS):
        shape, scale = fits[k]; x[..., j] = sps.lognorm.ppf(u[..., j], shape, scale=scale)
    return x

real_max = vals.max(axis=0); real_mad = np.array([ST.mad(vals[:, j]) for j in range(2)])
XA = draw(N_MC, N_BATCH)
MA = (XA - real_max) / real_mad
RA = rules(MA)
XB_ref, XB_bat = draw(N_MC, N_REF), draw(N_MC, N_BATCH)
mx = XB_ref.max(axis=1, keepdims=True)
md = np.stack([ST.mad(XB_ref[..., j], axis=1) for j in range(2)], axis=-1)[:, None, :]
MB = (XB_bat - mx) / md
RB = rules(MB)
pct_max = {k: float(sps.lognorm.cdf(real_max[j], *fits[k][:1], scale=fits[k][1])) for j, k in enumerate(KPIS)}

# ---------------------------------------------------------------- design C: real 5-v-5 ordinary splits through the pipeline
th = DE.Thresholds(); cfg = dict(RP.DEFAULT_CONFIG)
def pipeline_fn(ref_part, pseudo):
    r = RP._run_pipeline(ref_part, pseudo, ref_part.site.tolist(), cfg, th, list(DE.PRIMARY_KPIS), c2st=None, n_boot=300, energy_n_mc=1000, local_extra=False)
    M = np.full((len(pseudo), 2), -np.inf)
    for j, k in enumerate(KPIS):
        t = r["local"][k]["table"].set_index("site")
        M[:, j] = t.reindex(pseudo.site.astype(str)).margin_in_mad.fillna(-np.inf).to_numpy()
    rl = rules(M[None])
    return dict(drift_alert=r["verdict"]["outcome_columns"]["drift_alert"], localized=r["verdict"]["outcome_columns"]["localized"],
                max_margin=float(np.nanmax(M)), **{k: bool(v[0]) for k, v in rl.items()})
t0 = time.time()
C = ST.reference_split_diagnostics(ref_ord, list(DE.PRIMARY_KPIS), n_incoming=5, pipeline_fn=pipeline_fn, n_splits=60, seed=SEED)
print(f"design C: 60 splits in {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- table
rule_names = list(RA.keys())
tab = pd.DataFrame({"A_fixed_ref_7v10": {k: RA[k].mean() for k in rule_names},
                    "B_redrawn_ref_7v10": {k: RB[k].mean() for k in rule_names},
                    "C_real_splits_5v5": {k: C[k].mean() for k in rule_names}})
tab.index.name = "rule"
print("\nP(rule fires) on a CLEAN batch (both KPIs crack_frac + pore_max_d):"); print(tab.round(3).to_string())
print(f"\nlognormal fits (ordinary 10): " + ", ".join(f"{k}: sigma {fits[k][0]:.2f}, median {fits[k][1]:.4g}, real max at {100*pct_max[k]:.0f}th pct" for k in KPIS)
      + f"; Spearman rho between KPIs {rho:+.2f}; drift-alert rate on C {C.drift_alert.mean():.3f}")

# ---------------------------------------------------------------- sensitivity
def margins_for(batch_df):
    out = {}
    for k in KPIS:
        le = ST.local_exceedance(ST.usable_values(ref_ord, k), ST.usable_values(batch_df, k), margin_mad=th.severity_margin_mad)
        out[k] = le["table"].set_index("site").margin_in_mad
    return pd.DataFrame(out)
sens = []
Mc = margins_for(ref[ref.site.isin(cracked)])
for s, row in Mc.iterrows():
    sens.append(dict(case=f"cracked ref site {s}", crack_frac=row.crack_frac, pore_max_d=row.pore_max_d))
for f in (1.5, 2.0, 3.0):
    syn = b2.copy(); syn["crack_frac"] *= f
    Ms = margins_for(syn)
    sens.append(dict(case=f"Batch_2 crack_frac x{f} (max site)", crack_frac=Ms.crack_frac.max(), pore_max_d=Ms.pore_max_d.max()))
S = pd.DataFrame(sens).set_index("case")
for e in (2.5, 3.0):
    S[f"flips tiered({e}) single-site"] = S.crack_frac.ge(e) | S.pore_max_d.ge(e)
S["single@2 (current)"] = S.crack_frac.ge(2) | S.pore_max_d.ge(2)
print("\nSensitivity (margins in MAD beyond the ordinary max; one site at a time):"); print(S.round(2).to_string())

rows = [dict(rule=k, design=d, rate=float(tab.loc[k, d])) for k in rule_names for d in tab.columns]
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "experiments", "e19_review_load.csv"), index=False)
S.to_csv(os.path.join(ROOT, "experiments", "e19_sensitivity.csv"))
print("\nwrote experiments/e19_review_load.csv and e19_sensitivity.csv")
