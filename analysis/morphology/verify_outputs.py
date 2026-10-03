"""Validate E18 data contracts and report consistency after rendering."""
import json

import numpy as np
import pandas as pd

from analysis.morphology.run_analysis import OUT, ROOT, MORPH, PRIMARY_KPIS


def main():
    sites = pd.read_csv(OUT / "morphology_sites.csv")
    variants = pd.read_csv(OUT / "threshold_variants.csv")
    comparison = pd.read_csv(OUT / "pairwise_comparisons.csv")
    multi = pd.read_csv(OUT / "multivariate_comparisons.csv")
    manifest = json.loads((OUT / "manifest.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    cached = pd.read_csv(ROOT / "analysis_cache/site_features.csv").set_index("site")
    assert len(sites) == len(cached) == sites.site.nunique()
    assert not sites.duplicated(["batch", "site"]).any()
    assert set(variants.threshold_offset) == {-5, 0, 5}
    assert variants.groupby("site").size().eq(3).all()
    for k in ("pore_frac", "bright_frac"):
        nominal = variants[variants.threshold_offset == 0].set_index("site")
        assert np.allclose(nominal[f"{k}_recomputed"], cached.loc[nominal.index, k], atol=1e-10)
    for k in ("pore_horizontal_alignment", "bright_horizontal_alignment"):
        assert sites[k].dropna().between(-1-1e-12, 1+1e-12).all()
    for k in ("pore_alignment_strength", "bright_alignment_strength"):
        assert sites[k].dropna().between(0, 1+1e-12).all()
    assert np.isfinite(sites[MORPH]).all().all()
    has_band = comparison.dropna(subset=["threshold_delta_low", "threshold_delta_high"])
    assert (has_band.threshold_delta_low <= has_band.median_delta + 1e-10).all()
    assert (has_band.threshold_delta_high >= has_band.median_delta - 1e-10).all()
    assert (comparison.p_holm_screen >= comparison.p_exploratory - 1e-12).all()
    for r in comparison.itertuples():
        assert r.n_ref <= len(sites[sites.batch == r.reference])
        assert r.n_batch <= len(sites[sites.batch == r.batch])
        if r.kpi.startswith(("bright_", "profile_bright")) and r.batch == "Batch_1":
            assert r.n_batch == 5
        if r.view == "ordinary_reference":
            assert r.n_ref == 10
    for r in multi.itertuples():
        if r.view == "quality_matched" and r.reference == "Batch_3":
            assert r.n_ref == 13
        if r.view == "ordinary_reference":
            assert r.n_ref == 10
    assert manifest["frozen_primary_kpis"] == list(PRIMARY_KPIS)
    assert manifest["inputs_changed_during_run"] == []
    assert summary["n_sites"] == len(sites)
    text = (OUT / "report.html").read_text()
    assert text.count('src="data:image/png;base64,') == 4
    assert "not equivalence or acceptance" in text
    assert "not a confidence interval" in text
    assert "probability of correctness" in text
    assert "n_boot=4000" not in text or manifest["n_boot"] == 4000
    print("E18 output contracts, counts, envelopes, provenance and report checks passed.")


if __name__ == "__main__":
    main()
