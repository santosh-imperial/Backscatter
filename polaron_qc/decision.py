"""Decision layer — plan §2.7, built on the small-sample discipline of §2.0.

Pure functions over the tables produced by polaron_qc.stats (compare_kpis tidy table, energy-distance dict,
per-site drift table, local-exceedance tables) and the acquisition views (§2.6). Nothing here touches images.

Outcomes are always reported in three separate columns — drift alert, localized anomaly, quality abstention —
so that an abstention caused by image quality is never counted as an alarm.

Verdicts (plan §2.7):
  CONSISTENT  "consistent with the working reference, within detectable limits"   (never "accept")
  INVESTIGATE_DRIFT, INVESTIGATE_LOCAL, REJECT_PROVISIONAL
Provenance of every threshold: developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib, json
import numpy as np, pandas as pd
from . import PRIMARY_KPIS

CONSISTENT = "consistent with working reference (within detectable limits)"
INVESTIGATE_DRIFT = "investigate — batch-wide drift"
INVESTIGATE_LOCAL = "investigate — localized anomaly"
REJECT = "reject (provisional)"

# KPIs whose per-site reliability depends on a flag (plan §2.0 rules 1 and 5)
KPI_RELIABILITY_FLAG = {"bright_frac": "bright_low_contrast", "bright_d50": "bright_low_contrast", "bright_d90": "bright_low_contrast",
                        "bright_count_per_Mpx": "bright_low_contrast", "bright_circ": "bright_low_contrast", "bright_max_d": "bright_low_contrast",
                        "patch_crack_area_frac_max": "grey_pore", "patch_pore_max_d_max": "grey_pore",
                        "pore_frac": "grey_pore", "pore_d50": "grey_pore", "pore_max_d": "grey_pore", "crack_frac": "grey_pore",
                        "crack_count_per_Mpx": "grey_pore", "pore_elong": "grey_pore"}
# for grey_pore the measurement is "fallback threshold", not unusable: it reduces reliability for Check B only
SOFT_FLAGS = {"grey_pore"}
# KPIs with per-site EXTREME semantics — the only ones that can raise a localized-anomaly flag (plan §2.4 local check; D28).
# Batch-mean KPIs (pore_frac, bright_frac, bright_d50) describe a site, not a local defect: an outlying site on those
# shows up in the per-site drift score and in Check A, never in Check B.
LOCAL_KPIS = ["crack_frac", "crack_count_per_Mpx", "pore_max_d", "bright_max_d",
              "patch_crack_area_frac_max", "patch_pore_max_d_max", "etd_crack_density_particles"]
# Only these can PROMOTE a flag to pending/credible (D29). Calibration against the i.i.d. null (7 batch vs 10 ordinary
# reference, Gaussian): P(at least one pending flag) at a 2.0-MAD margin ≈ 4 % per KPI, ≈ 8 % for two KPIs, ≈ 15 % for four;
# on real 5-v-5 splits of the ordinary reference the rate stays high (heavy tails), so promotion is restricted to the two
# physically consequential void KPIs and the rest are reported as secondary local flags that never drive a verdict.
LOCAL_KPIS_PROMOTE = ["crack_frac", "pore_max_d"]


@dataclass
class Thresholds:
    """All decision thresholds in one place. Hashed; the notebook asserts the hash before running on the unseen batch."""
    alpha: float = 0.05                 # for permutation p-values (Holm-adjusted on primary KPIs)
    min_effect_mad: float = 1.0         # a primary KPI "carries" drift only if |robust shift| >= this (in reference MADs)
    consistency_share: float = 0.5      # share of usable batch sites beyond the ordinary-reference range, in the shift direction
    min_usable_sites: int = 5           # below this, Check A abstains for that KPI; below for all primary KPIs → quality abstention
    severity_margin_mad: float = 2.0    # Check B: margin beyond ordinary-reference max, in MADs of ordinary per-site values (D29; calibrated on the i.i.d. null)
    strong_attenuation: float = 0.5     # if the multivariate shift drops by more than this share under stratified/adjusted views → investigate, not reject
    reject_min_credible_sites: int = 2  # Check B alone can reject only with ≥ this many credible sites …
    reject_min_severity_mad: float = 2.0  # … each beyond the reference max by ≥ this many MADs
    # Tiered localized rule (D33, calibrated in E19): a single pending site between severity_margin_mad and
    # single_site_escalate_mad is ROUTED to image review without changing the batch verdict; the verdict flips to
    # 'investigate — localized' when one site reaches single_site_escalate_mad, or agreement_min_sites sites / agreement_min_kpis
    # promotable KPIs are pending at severity_margin_mad, or a reviewer confirms any pending flag.
    single_site_escalate_mad: float = 3.0
    agreement_min_sites: int = 2
    agreement_min_kpis: int = 2
    provenance: str = "developed using exploratory analysis of Batches 1–3; frozen before the unseen batch arrived"

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- helpers
def _ordinary_range(ref_values: pd.Series) -> tuple[float, float]:
    v = pd.Series(ref_values).dropna().astype(float)
    return (float(v.min()), float(v.max())) if len(v) else (np.nan, np.nan)


def consistency_across_sites(ref_values, batch_values, direction: str) -> dict:
    """Share of batch sites that lie beyond the ordinary-reference range in the direction of the shift (§2.7 A iii)."""
    lo, hi = _ordinary_range(ref_values)
    b = pd.Series(batch_values).dropna().astype(float)
    if direction == "higher":
        beyond = (b > hi)
    elif direction == "lower":
        beyond = (b < lo)
    else:
        beyond = (b > hi) | (b < lo)
    return dict(share=float(beyond.mean()) if len(b) else np.nan, n_beyond=int(beyond.sum()), n=int(len(b)), ref_lo=lo, ref_hi=hi)


# ----------------------------------------------------------------------------- Check A
def check_a(compare: pd.DataFrame, energy: dict, ref_sites: pd.DataFrame, batch_sites: pd.DataFrame,
            ordinary_ref_mask: pd.Series, c2st: dict | None = None, th: Thresholds = Thresholds(),
            primary: list[str] = PRIMARY_KPIS) -> dict:
    """Batch-wide drift on the primary KPIs.

    compare      : tidy table from stats.compare_kpis (one row per KPI; columns kpi, n_ref_usable, n_batch_usable,
                   shift_mad, ci_low, ci_high, p_perm, p_holm, is_primary, direction)
    energy       : dict from stats.energy_distance_test on the primary KPIs ({statistic, p, n_perm})
    ref_sites, batch_sites : site tables (need the primary KPI columns and the flag columns)
    ordinary_ref_mask : boolean Series aligned with ref_sites, True for ordinary reference sites
    c2st         : dict from ml.c2st ({auc, p, coef}) or None
    """
    prim = compare[compare.kpi.isin(primary)].set_index("kpi")
    out = dict(i_beyond_null=bool(energy.get("p", 1.0) < th.alpha), energy_p=float(energy.get("p", np.nan)),
               energy_stat=float(energy.get("statistic", np.nan)), drivers=[], per_kpi={}, abstained_kpis=[])
    for k, r in prim.iterrows():
        usable_ok = (r.n_batch_usable >= th.min_usable_sites) and (r.n_ref_usable >= th.min_usable_sites)
        sig = bool(np.isfinite(r.p_holm) and r.p_holm < th.alpha and abs(r.shift_mad) >= th.min_effect_mad)
        # consistency is judged on USABLE batch sites only (hard reliability flag excludes a site for that KPI)
        hard_flag = KPI_RELIABILITY_FLAG.get(k)
        if hard_flag and hard_flag not in SOFT_FLAGS and hard_flag in batch_sites.columns:
            batch_vals = batch_sites.loc[~batch_sites[hard_flag].astype(bool), k]
        else:
            batch_vals = batch_sites[k]
        cons = consistency_across_sites(ref_sites.loc[ordinary_ref_mask.values, k], batch_vals, r.direction) if usable_ok else dict(share=np.nan, n_beyond=0, n=0)
        out["per_kpi"][k] = dict(shift_mad=float(r.shift_mad), ci=(float(r.ci_low), float(r.ci_high)), p_holm=float(r.p_holm),
                                 n_ref_usable=int(r.n_ref_usable), n_batch_usable=int(r.n_batch_usable), significant=sig,
                                 direction=r.direction, consistency_share=cons["share"], n_beyond=cons["n_beyond"], usable_ok=usable_ok)
        if not usable_ok:
            out["abstained_kpis"].append(k)
        elif sig:
            out["drivers"].append(k)
    out["ii_carried_by_primary"] = len(out["drivers"]) > 0
    shares = [out["per_kpi"][k]["consistency_share"] for k in out["drivers"]]
    out["iii_consistent"] = bool(len(shares) and np.nanmax(shares) >= th.consistency_share)
    out["iv_classifier_corroborates"] = None if c2st is None else bool(c2st.get("p", 1.0) < th.alpha)
    out["all_usable"] = len(out["abstained_kpis"]) < len(primary)
    return out


# ----------------------------------------------------------------------------- Check B
def check_b(local_tables: dict[str, pd.DataFrame], batch_sites: pd.DataFrame, image_reviewed: dict | None = None,
            th: Thresholds = Thresholds(), primary: list[str] | None = None, local_kpis: list[str] = LOCAL_KPIS) -> dict:
    """Localized defect check.

    local_tables : {kpi: DataFrame from stats.local_exceedance with columns site, value, ref_max, exceeds, margin_in_mad}
    batch_sites  : site table with flag columns (bright_low_contrast, grey_pore)
    image_reviewed : {(site, kpi): True/False} supplied by a human after looking at the evidence image. Three states (E21):
                   key absent or dict None → 'unreviewed' (flag stays pending); True → 'confirmed' (credible);
                   False → 'refuted' (the investigation is closed for that site/KPI; the flag is still listed).
    local_kpis   : only KPIs with per-site extreme semantics are considered (LOCAL_KPIS, D28); tables for other KPIs are
                   returned under 'descriptive_flags' and never promote.
    `primary` is accepted for backward compatibility and ignored.
    A flag is 'credible' only with severity margin AND reliable measurement AND image review (plan §2.0 rule 5).
    Without review, the best status is 'credible_pending_review' — the verdict for that is 'investigate — localized anomaly
    (image review pending)', because routing the crop to a reviewer *is* the investigation.
    """
    flags, credible, pending, refuted, descriptive = [], [], [], [], []
    review_word = {None: "unreviewed", True: "confirmed", False: "refuted"}
    bs = batch_sites.set_index("site")
    for k, tab in local_tables.items():
        if tab is None or len(tab) == 0:
            continue
        if k not in local_kpis:
            for _, r in tab[tab.exceeds.astype(bool)].iterrows():
                descriptive.append(dict(site=r.site, kpi=k, value=float(r.value), ref_max=float(r.ref_max), margin_in_mad=float(r.margin_in_mad),
                                        note="batch-mean KPI: outlying site, not a localized defect (see per-site drift)"))
            continue
        rel_flag = KPI_RELIABILITY_FLAG.get(k)
        for _, r in tab[tab.exceeds.astype(bool)].iterrows():
            site = r.site
            unreliable = bool(rel_flag and rel_flag in bs.columns and bs.loc[site, rel_flag] and rel_flag not in SOFT_FLAGS)
            soft = bool(rel_flag and rel_flag in bs.columns and bs.loc[site, rel_flag] and rel_flag in SOFT_FLAGS)
            rv = None if image_reviewed is None else image_reviewed.get((site, k), None)
            rev = None if rv is None else bool(rv)
            rec = dict(site=site, kpi=k, value=float(r.value), ref_max=float(r.ref_max), margin_in_mad=float(r.margin_in_mad),
                       severity_ok=bool(r.margin_in_mad >= th.severity_margin_mad), measurement_reliable=not unreliable,
                       measurement_note="fallback threshold (grey-pore site)" if soft else ("unreliable (low-contrast site)" if unreliable else "ok"),
                       image_reviewed=rev, review_status=review_word[rev])
            rec["promotable"] = k in LOCAL_KPIS_PROMOTE
            flags.append(rec)
            if rec["promotable"] and rec["severity_ok"] and rec["measurement_reliable"]:
                (credible if rev is True else pending if rev is None else refuted).append(rec)
    return dict(flags=flags, credible=credible, credible_pending_review=pending, refuted=refuted, descriptive_flags=descriptive,
                n_sites_flagged=len({f["site"] for f in flags}), n_sites_credible=len({f["site"] for f in credible}),
                n_sites_pending=len({f["site"] for f in pending}), n_sites_refuted=len({f["site"] for f in refuted}),
                max_severity_mad=max([f["margin_in_mad"] for f in credible + pending], default=np.nan))


# ----------------------------------------------------------------------------- quality abstention
def quality_abstention(batch_sites: pd.DataFrame, a: dict, th: Thresholds = Thresholds()) -> dict:
    reasons = []
    n = len(batch_sites)
    if n < th.min_usable_sites:
        reasons.append(f"only {n} sites in batch (< {th.min_usable_sites})")
    if not a.get("all_usable", True):
        reasons.append("fewer than min usable sites on every primary KPI")
    for col, label in [("bright_low_contrast", "low bright-phase contrast"), ("grey_pore", "raised black level / grey pores")]:
        if col in batch_sites.columns and batch_sites[col].astype(bool).mean() > 0.5:
            reasons.append(f"{label} on more than half of the sites")
    return dict(abstain=len(reasons) > 0, reasons=reasons)


# ----------------------------------------------------------------------------- attenuation under acquisition views
ATT_VIEWS = ("stratified", "adjusted")


def attenuation(energy_unadjusted: dict, energy_stratified: dict | None, energy_adjusted: dict | None) -> dict:
    """Share by which the multivariate statistic drops under the stratified / adjusted views (§2.6). Descriptive, not causal.
    ``available`` is True when at least one view produced a finite share; without it a drift reject is withheld (E21)."""
    base = float(energy_unadjusted.get("statistic", np.nan))
    def share(e):
        if e is None or not np.isfinite(base) or base <= 0: return np.nan
        return float(1 - e.get("statistic", np.nan) / base)
    out = dict(stratified=share(energy_stratified), adjusted=share(energy_adjusted))
    out["available"] = bool(any(np.isfinite(out[v]) for v in ATT_VIEWS))
    return out


def _att_normalise(att: dict | None) -> dict:
    """None / missing / all-NaN views → available=False. The verdict code never treats a missing view as 'no attenuation'."""
    out = dict(stratified=np.nan, adjusted=np.nan)
    if att:
        out.update({k: float(att.get(k, np.nan)) if att.get(k) is not None else np.nan for k in ATT_VIEWS})
    out["available"] = bool(any(np.isfinite(out[v]) for v in ATT_VIEWS))
    return out


def _strong_attenuation(att: dict, th: Thresholds) -> bool:
    return any(np.isfinite(att[v]) and att[v] > th.strong_attenuation for v in ATT_VIEWS)


# ----------------------------------------------------------------------------- verdict
def decide(a: dict, b: dict, abst: dict, att: dict | None = None, th: Thresholds = Thresholds()) -> dict:
    """Combine the checks into one verdict plus the three outcome columns and a 'what would move it' sentence.

    att : dict from ``attenuation`` / acquisition.three_views (keys stratified, adjusted). None or all-NaN means the
    acquisition sensitivity views were NOT computed: a drift reject is then withheld (verdict stays 'investigate — drift')
    and the reason says so; a missing view is never read as 'the shift was not attenuated' (E21)."""
    att = _att_normalise(att)
    strong_att = _strong_attenuation(att, th)
    drift_alert = bool(a["i_beyond_null"])
    esc = escalation(b, th)
    local_status = ("credible" if b["n_sites_credible"] else ("pending_review" if esc["escalate"] else "review_routed") if b["n_sites_pending"]
                    else ("flag" if b["n_sites_flagged"] else "none"))

    # reject via Check A requires classifier corroboration to be PRESENT and positive; a missing classifier run cannot
    # count as corroboration (report-agent finding; D28)
    a_full = a["i_beyond_null"] and a["ii_carried_by_primary"] and a["iii_consistent"] and (a["iv_classifier_corroborates"] is True)
    b_reject = b["n_sites_credible"] >= th.reject_min_credible_sites and np.nan_to_num(b["max_severity_mad"]) >= th.reject_min_severity_mad

    if abst["abstain"]:
        # D45: under an abstention the more specific path names the label — a credible or escalated localized finding
        # (confirmed cracks) beats a drift alert that n < 5 cannot support; the three outcome columns carry the full state
        verdict = INVESTIGATE_LOCAL if local_status in ("credible", "pending_review") else INVESTIGATE_DRIFT
        reason = ("quality abstention: " + "; ".join(abst["reasons"])
                  + f" — paths observed despite the abstention: drift alert {'yes' if drift_alert else 'no'}, localized {local_status}")
    elif (a_full and att["available"] and not strong_att) or b_reject:
        verdict = REJECT
        reason = ("batch-wide drift beyond null, carried by primary KPIs %s, consistent across sites, classifier corroborates, "
                  "not attenuated under the stratified / adjusted acquisition views (%s)" %
                  (a["drivers"], ", ".join(f"{v} {att[v]:+.2f}" for v in ATT_VIEWS if np.isfinite(att[v])))) if a_full and att["available"] and not strong_att else \
                 f"{b['n_sites_credible']} sites with credible localized anomalies, max severity {b['max_severity_mad']:.1f} MAD beyond reference max"
    elif local_status in ("credible", "pending_review"):
        verdict = INVESTIGATE_LOCAL
        reason = f"{b['n_sites_credible'] or b['n_sites_pending']} site(s) with a severe localized anomaly" + (" (image review pending)" if local_status == "pending_review" else "")
    elif drift_alert:
        verdict = INVESTIGATE_DRIFT
        why = []
        if not a["ii_carried_by_primary"]: why.append("not carried by a primary KPI")
        if a["ii_carried_by_primary"] and not a["iii_consistent"]: why.append("not consistent across sites")
        if a["iv_classifier_corroborates"] is False: why.append("classifier does not corroborate")
        if a["iv_classifier_corroborates"] is None: why.append("classifier corroboration not available")
        if strong_att: why.append("shift attenuates strongly under acquisition stratification/adjustment")
        if a_full and not att["available"]: why.append("acquisition sensitivity views (stratified / adjusted) not available — reject withheld until they are computed")
        reason = "multivariate shift beyond reference null; " + ("; ".join(why) if why else "see drivers")
    else:
        verdict = CONSISTENT
        reason = "no drift beyond the reference null on primary KPIs and no credible localized anomaly; see MDC for what could not have been detected"
        if local_status == "review_routed":
            reason += ("; %d site(s) routed to image review (%s beyond the ordinary-reference max) — below the single-site escalation margin of %.1f MAD "
                       "with no second site or KPI in agreement, so the verdict is unchanged until a reviewer confirms" %
                       (b["n_sites_pending"], ", ".join(f"{f['site']} {f['kpi']} {f['margin_in_mad']:.1f} MAD" for f in b["credible_pending_review"]), th.single_site_escalate_mad))
        if local_status == "flag":
            n_ref_ = int(b.get("n_sites_refuted", 0))
            reason += (f"; {b['n_sites_flagged']} site(s) exceed the ordinary-reference max without meeting the severity/reliability conditions (expected on ~40 % of clean batches)"
                       + (f", of which {n_ref_} refuted by image review" if n_ref_ else ""))

    return dict(verdict=verdict, reason=reason,
                outcome_columns=dict(drift_alert=drift_alert, localized=local_status, quality_abstention=bool(abst["abstain"])),
                drivers=a["drivers"], attenuation=att, escalation=esc, what_would_move_it=what_would_move(verdict, a, b, abst, att, th),
                thresholds_hash=th.hash(), provenance=th.provenance)


def escalation(b: dict, th: Thresholds) -> dict:
    """Tiered localized rule (D33): does the set of pending (severity-ok, reliable, unreviewed) flags flip the batch verdict?"""
    pend = b.get("credible_pending_review", []) or []
    sites, kpis = {f["site"] for f in pend}, {f["kpi"] for f in pend}
    mx = max([float(f["margin_in_mad"]) for f in pend], default=float("nan"))
    by_single = bool(pend) and mx >= th.single_site_escalate_mad
    by_sites = len(sites) >= th.agreement_min_sites
    by_kpis = len(kpis) >= th.agreement_min_kpis
    return dict(escalate=bool(by_single or by_sites or by_kpis), by_single_site=by_single, by_site_agreement=by_sites, by_kpi_agreement=by_kpis,
                max_pending_margin_mad=mx, n_pending_sites=len(sites), n_pending_kpis=len(kpis),
                rule=f"single site ≥ {th.single_site_escalate_mad} MAD, or ≥ {th.agreement_min_sites} sites / ≥ {th.agreement_min_kpis} KPIs pending at ≥ {th.severity_margin_mad} MAD, or a confirmed review")


def what_would_move(verdict: str, a: dict, b: dict, abst: dict, att: dict, th: Thresholds) -> str:
    if verdict == CONSISTENT and b.get("n_sites_pending") and not b.get("n_sites_credible"):
        return (f"would become 'investigate — localized' if the reviewer confirms the routed crop(s), if any site reached ≥ {th.single_site_escalate_mad} MAD beyond the "
                f"ordinary-reference max, or if ≥ {th.agreement_min_sites} sites or ≥ {th.agreement_min_kpis} promotable KPIs exceeded it by ≥ {th.severity_margin_mad} MAD; "
                f"would close if the reviewer refutes the crop(s)")
    if verdict == CONSISTENT:
        # nearest primary KPI to significance
        cand = sorted(((v["p_holm"], k) for k, v in a["per_kpi"].items() if np.isfinite(v["p_holm"])), key=lambda t: t[0])
        if cand:
            p, k = cand[0]
            return (f"would become 'investigate — drift' if {k} (Holm p = {p:.2f}, shift {a['per_kpi'][k]['shift_mad']:+.1f} MAD) crossed α = {th.alpha} with |shift| ≥ {th.min_effect_mad} MAD; "
                    f"would become 'investigate — localized' if a site exceeded the ordinary-reference max by ≥ {th.single_site_escalate_mad} MAD on crack_frac or pore_max_d, "
                    f"or ≥ {th.agreement_min_sites} sites / both KPIs by ≥ {th.severity_margin_mad} MAD (a single site between the two margins is routed to image review without changing the verdict)")
        return "would become 'investigate' if any primary KPI crossed α or any site showed a credible localized anomaly"
    if verdict == INVESTIGATE_DRIFT:
        if abst["abstain"]:
            # E25: the former 'only' → 'at least' rewrite produced "with at least 3 sites in batch (< 5)"; quote the reasons as recorded
            return "would become decidable once resolved: " + "; ".join(abst["reasons"])
        missing = []
        if not a["ii_carried_by_primary"]: missing.append("a primary KPI crossing α with |shift| ≥ %.1f MAD" % th.min_effect_mad)
        if a["ii_carried_by_primary"] and not a["iii_consistent"]: missing.append("≥ %.0f %% of sites beyond the ordinary-reference range" % (100 * th.consistency_share))
        if a["iv_classifier_corroborates"] is not True: missing.append("classifier corroboration (material-only run, p < α)")
        if not att.get("available", False): missing.append("the acquisition sensitivity views (stratified / adjusted) computed and not attenuating the shift")
        elif _strong_attenuation(att, th): missing.append("the shift surviving acquisition stratification/adjustment")
        return ("would become 'reject' with " + " and ".join(missing)) if missing else "would become 'consistent' if the multivariate shift fell inside the reference null"
    if verdict == INVESTIGATE_LOCAL:
        return f"would become 'reject' with ≥ {th.reject_min_credible_sites} credible sites each ≥ {th.reject_min_severity_mad} MAD beyond the reference max; would become 'consistent' if image review did not confirm the anomaly"
    return "would become 'investigate' if the drift attenuated strongly under acquisition adjustment or if fewer sites carried it"
