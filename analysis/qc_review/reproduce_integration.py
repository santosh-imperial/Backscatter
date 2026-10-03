"""Focused review probes, using synthetic tables rather than material labels.

Run from the repository root:
    /opt/anaconda3/bin/python3 -m analysis.qc_review.reproduce_integration

This records current behaviour; it is not a regression test asserting that the
behaviour is correct. No raw images, feature caches, or implementation are changed.
"""
import json

import numpy as np
import pandas as pd

from polaron_qc import PRIMARY_KPIS
from polaron_qc import acquisition, decision, report, stats


def site_table(n, batch):
    values = np.linspace(0.8, 1.2, n)
    df = pd.DataFrame({k: values.copy() for k in PRIMARY_KPIS})
    df.insert(0, "site", [f"{batch}_new_{i}" for i in range(n)])
    df.insert(0, "batch", batch)
    df["bright_low_contrast"] = False
    df["grey_pore"] = False
    return df


def main():
    ref, batch = site_table(17, "R"), site_table(7, "B")
    batch["crack_frac"] += 1
    compare = pd.DataFrame([
        dict(kpi=k, n_ref_usable=17, n_batch_usable=7,
             shift_mad=3.2 if k == "crack_frac" else 0.0,
             ci_low=2.0 if k == "crack_frac" else -0.5,
             ci_high=4.0 if k == "crack_frac" else 0.5,
             p_holm=0.004 if k == "crack_frac" else 0.8,
             direction="higher" if k == "crack_frac" else "none")
        for k in PRIMARY_KPIS
    ])
    a = decision.check_a(compare, {"statistic": 0.9, "p": 0.002},
                         ref, batch, pd.Series(True, index=ref.index),
                         c2st={"p": 0.01})
    no_local = decision.check_b({}, batch)
    q = decision.quality_abstention(batch, a)
    missing = decision.decide(a, no_local, q)
    attenuated = decision.decide(a, no_local, q,
                                att={"stratified": 0.7, "adjusted": 0.1})

    local = {"crack_frac": pd.DataFrame([
        dict(site=batch.site.iloc[0], value=2.0, ref_max=1.3,
             exceeds=True, margin_in_mad=2.4)
    ])}
    unreviewed = decision.check_b(local, batch)
    refuted = decision.check_b(local, batch,
                              image_reviewed={(batch.site.iloc[0], "crack_frac"): False})

    unseen = site_table(5, "UNSEEN")
    images = pd.DataFrame([
        dict(batch="UNSEEN", site=s, det=det, p1=24 if det == "BSE" else 0,
             p50=100, std=20, empty_bin_frac=0.0, band_top=0, band_bottom=0,
             H=1800)
        for s in unseen.site for det in ("BSE", "ETD", "Inlens")
    ])
    provisional = report.derive_flags(unseen, images)
    derived = acquisition.derive_flags(unseen, images)
    corrected = unseen.drop(columns=["grey_pore"]).merge(
        derived[["site", "grey_pore"]], on="site")

    try:
        stats.mdc(np.arange(17), 18, n_sim=1, shifts=[0], test={"n_mc": 9})
        larger_error = None
    except ValueError as exc:
        larger_error = str(exc)
    equal = stats.mdc(np.arange(7), 7, n_sim=1, shifts=[0], test={"n_mc": 9})

    out = {
        "missing_acquisition": {
            "verdict": missing["verdict"], "reason": missing["reason"],
            "with_strong_attenuation": attenuated["verdict"],
        },
        "negative_image_review": {
            "pending_unreviewed": unreviewed["n_sites_pending"],
            "pending_explicit_false": refuted["n_sites_pending"],
        },
        "unseen_grey_pore": {
            "n_sites": len(unseen),
            "report_raised_black": int(provisional.raised_black_level.sum()),
            "report_grey_pore": int(provisional.grey_pore.sum()),
            "acquisition_grey_pore": int(derived.grey_pore.sum()),
            "report_quality_abstain": decision.quality_abstention(unseen, {"all_usable": True})["abstain"],
            "derived_quality_abstain": decision.quality_abstention(corrected, {"all_usable": True})["abstain"],
            "stats_fallback_count_with_column_only": stats.usable_n(corrected, "pore_frac")["n_fallback"],
        },
        "mdc": {
            "larger_incoming_error": larger_error,
            "equal_size_n_sim_used": equal["n_sim_used"],
            "equal_size_mdc": str(equal["mdc_mad"]),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
