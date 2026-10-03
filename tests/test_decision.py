import numpy as np, pandas as pd, pytest
from polaron_qc import PRIMARY_KPIS
from polaron_qc.decision import (Thresholds, check_a, check_b, quality_abstention, attenuation, decide,
                                 CONSISTENT, INVESTIGATE_DRIFT, INVESTIGATE_LOCAL, REJECT)

rng = np.random.default_rng(0)


def sites(n, batch, shift=None, flags=None):
    shift = shift or {}
    df = pd.DataFrame({k: rng.normal(1.0, 0.1, n) + shift.get(k, 0.0) for k in PRIMARY_KPIS})
    df.insert(0, "site", [f"{batch}_s{i}" for i in range(n)]); df.insert(0, "batch", batch)
    df["bright_low_contrast"] = False; df["grey_pore"] = False
    for col, idx in (flags or {}).items():
        df.loc[idx, col] = True
    return df


def compare_table(p_holm=None, shift=None, n_batch=7, n_ref=17, direction="higher"):
    p_holm = p_holm or {}; shift = shift or {}
    rows = []
    for k in PRIMARY_KPIS:
        rows.append(dict(kpi=k, n_ref_usable=n_ref, n_batch_usable=n_batch, shift_mad=shift.get(k, 0.1), ci_low=-0.5, ci_high=0.7,
                         cliffs_delta=0.1, p_perm=p_holm.get(k, 0.6), p_method="exact", n_perm=1000, p_holm=p_holm.get(k, 0.8),
                         is_primary=True, direction=direction if k in shift else "none"))
    return pd.DataFrame(rows)


def local_tables(flags):
    """flags: list of (kpi, site, margin_in_mad)"""
    out = {k: pd.DataFrame(columns=["site", "value", "ref_max", "exceeds", "margin_in_mad"]) for k in set(PRIMARY_KPIS) | {"bright_max_d"}}
    for k, site, m in flags:
        row = pd.DataFrame([dict(site=site, value=2.0, ref_max=1.3, exceeds=True, margin_in_mad=m)])
        out[k] = row if len(out[k]) == 0 else pd.concat([out[k], row], ignore_index=True)
    return out


def test_consistent_path():
    ref = sites(17, "R"); bat = sites(7, "B"); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.1, p=0.6), ref, bat, mask)
    b = check_b(local_tables([]), bat)
    v = decide(a, b, quality_abstention(bat, a))
    assert v["verdict"] == CONSISTENT
    assert v["outcome_columns"] == dict(drift_alert=False, localized="none", quality_abstention=False)
    assert "MDC" in v["reason"]


def test_flag_without_severity_stays_consistent_but_is_reported():
    ref = sites(17, "R"); bat = sites(7, "B"); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.1, p=0.6), ref, bat, mask)
    b = check_b(local_tables([("crack_frac", "B_s1", 0.4)]), bat)
    v = decide(a, b, quality_abstention(bat, a))
    assert v["verdict"] == CONSISTENT and v["outcome_columns"]["localized"] == "flag" and "40 %" in v["reason"]


def test_localized_pending_review_then_credible():
    ref = sites(17, "R"); bat = sites(7, "B"); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.1, p=0.6), ref, bat, mask)
    b = check_b(local_tables([("crack_frac", "B_s1", 2.4)]), bat)             # severity ok (≥ 2 MAD), no review yet
    assert b["n_sites_pending"] == 1 and b["n_sites_credible"] == 0
    v = decide(a, b, quality_abstention(bat, a)); assert v["verdict"] == INVESTIGATE_LOCAL and "pending" in v["reason"]
    b2 = check_b(local_tables([("crack_frac", "B_s1", 2.4)]), bat, image_reviewed={("B_s1", "crack_frac"): True})
    assert b2["n_sites_credible"] == 1
    v2 = decide(a, b2, quality_abstention(bat, a)); assert v2["verdict"] == INVESTIGATE_LOCAL and v2["outcome_columns"]["localized"] == "credible"


def test_localized_unreliable_measurement_is_only_a_flag():
    ref = sites(17, "R"); bat = sites(7, "B", flags={"bright_low_contrast": [1]}); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.1, p=0.6), ref, bat, mask)
    b = check_b(local_tables([("bright_max_d", "B_s1", 3.0)]), bat)
    assert b["n_sites_flagged"] == 1 and b["n_sites_pending"] == 0 and b["flags"][0]["measurement_reliable"] is False
    assert decide(a, b, quality_abstention(bat, a))["verdict"] == CONSISTENT


def test_batch_mean_kpi_exceedance_is_descriptive_not_localized():
    """A site with high bright_frac is an outlying site, not a local defect (D28)."""
    ref = sites(17, "R"); bat = sites(7, "B"); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.1, p=0.6), ref, bat, mask)
    b = check_b(local_tables([("bright_frac", "B_s3", 2.5)]), bat)
    assert b["n_sites_flagged"] == 0 and b["n_sites_pending"] == 0 and len(b["descriptive_flags"]) == 1
    assert decide(a, b, quality_abstention(bat, a))["verdict"] == CONSISTENT


def test_reject_requires_classifier_present():
    ref = sites(17, "R"); bat = sites(7, "B", shift={"crack_frac": 1.0}); mask = pd.Series([True] * 17)
    cmp = compare_table(p_holm={"crack_frac": 0.004}, shift={"crack_frac": 3.2})
    a = check_a(cmp, dict(statistic=0.9, p=0.002), ref, bat, mask, c2st=None)
    v = decide(a, check_b(local_tables([]), bat), quality_abstention(bat, a))
    assert v["verdict"] == INVESTIGATE_DRIFT and "not available" in v["reason"]


def test_consistency_uses_usable_sites_only():
    ref = sites(17, "R"); bat = sites(7, "B", shift={"bright_frac": 1.0}, flags={"bright_low_contrast": [0, 1]}); mask = pd.Series([True] * 17)
    bat.loc[[0, 1], "bright_frac"] = 0.0   # unusable sites carry garbage values that must not count
    cmp = compare_table(p_holm={"bright_frac": 0.004}, shift={"bright_frac": 3.0}, n_batch=5)
    a = check_a(cmp, dict(statistic=0.9, p=0.002), ref, bat, mask)
    assert a["per_kpi"]["bright_frac"]["n_beyond"] == 5 and a["per_kpi"]["bright_frac"]["consistency_share"] == 1.0


def test_grey_pore_is_soft_flag():
    ref = sites(17, "R"); bat = sites(7, "B", flags={"grey_pore": [2]}); mask = pd.Series([True] * 17)
    b = check_b(local_tables([("pore_max_d", "B_s2", 2.0)]), bat)
    assert b["n_sites_pending"] == 1 and "fallback" in b["flags"][0]["measurement_note"]


def test_non_promotable_local_kpi_stays_a_flag():
    ref = sites(17, "R"); bat = sites(7, "B"); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.1, p=0.6), ref, bat, mask)
    b = check_b(local_tables([("bright_max_d", "B_s1", 4.0)]), bat)
    assert b["n_sites_flagged"] == 1 and b["n_sites_pending"] == 0 and b["flags"][0]["promotable"] is False
    assert decide(a, b, quality_abstention(bat, a))["verdict"] == CONSISTENT


def test_margin_below_two_mad_is_not_promoted():
    ref = sites(17, "R"); bat = sites(7, "B"); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.1, p=0.6), ref, bat, mask)
    b = check_b(local_tables([("crack_frac", "B_s1", 1.6)]), bat)
    assert b["n_sites_flagged"] == 1 and b["n_sites_pending"] == 0
    assert decide(a, b, quality_abstention(bat, a))["verdict"] == CONSISTENT


def test_drift_not_carried_by_primary_is_investigate():
    ref = sites(17, "R"); bat = sites(7, "B"); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.9, p=0.01), ref, bat, mask)
    v = decide(a, check_b(local_tables([]), bat), quality_abstention(bat, a))
    assert v["verdict"] == INVESTIGATE_DRIFT and "not carried" in v["reason"] and v["outcome_columns"]["drift_alert"]


def test_reject_path_and_attenuation_downgrade():
    ref = sites(17, "R"); bat = sites(7, "B", shift={"crack_frac": 1.0}); mask = pd.Series([True] * 17)
    cmp = compare_table(p_holm={"crack_frac": 0.004}, shift={"crack_frac": 3.2})
    a = check_a(cmp, dict(statistic=0.9, p=0.002), ref, bat, mask, c2st=dict(auc=0.9, p=0.01))
    assert a["drivers"] == ["crack_frac"] and a["iii_consistent"] and a["iv_classifier_corroborates"]
    v = decide(a, check_b(local_tables([]), bat), quality_abstention(bat, a))
    assert v["verdict"] == REJECT and v["thresholds_hash"] == Thresholds().hash()
    att = attenuation(dict(statistic=0.9), dict(statistic=0.3), dict(statistic=0.8))
    assert att["stratified"] == pytest.approx(1 - 0.3 / 0.9)
    v2 = decide(a, check_b(local_tables([]), bat), quality_abstention(bat, a), att)
    assert v2["verdict"] == INVESTIGATE_DRIFT and "attenuates" in v2["reason"]


def test_reject_from_localized_alone():
    ref = sites(17, "R"); bat = sites(7, "B"); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.1, p=0.6), ref, bat, mask)
    flags = [("crack_frac", "B_s1", 2.5), ("crack_frac", "B_s3", 2.2)]
    b = check_b(local_tables(flags), bat, image_reviewed={("B_s1", "crack_frac"): True, ("B_s3", "crack_frac"): True})
    assert decide(a, b, quality_abstention(bat, a))["verdict"] == REJECT


def test_quality_abstention_column():
    ref = sites(17, "R"); bat = sites(7, "B", flags={"bright_low_contrast": [0, 1, 2, 3]}); mask = pd.Series([True] * 17)
    a = check_a(compare_table(), dict(statistic=0.1, p=0.6), ref, bat, mask)
    q = quality_abstention(bat, a)
    assert q["abstain"] and any("low bright-phase contrast" in r for r in q["reasons"])
    v = decide(a, check_b(local_tables([]), bat), q)
    assert v["outcome_columns"]["quality_abstention"] and v["outcome_columns"]["drift_alert"] is False and v["verdict"].startswith("investigate")


def test_usable_n_abstains_per_kpi():
    ref = sites(17, "R"); bat = sites(7, "B"); mask = pd.Series([True] * 17)
    cmp = compare_table(); cmp.loc[cmp.kpi == "bright_frac", "n_batch_usable"] = 4
    a = check_a(cmp, dict(statistic=0.1, p=0.6), ref, bat, mask)
    assert a["abstained_kpis"] == ["bright_frac"] and a["all_usable"]


def test_thresholds_hash_changes_with_values():
    assert Thresholds().hash() != Thresholds(alpha=0.01).hash()
